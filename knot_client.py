# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""A KNOT client with explicit forget/retain sampler phases."""
import json
import logging
from pathlib import Path
import re

from mga_data import IndexSampler, PoisonedDataset, PoisonSpec, select_subset
from mga_protocol import clone_state

from plato.config import Config
from plato.clients import simple

import forgetting_iid
import unlearning_iid


class Client(simple.Client):
    """KNOT client using deleted samples for GA and retained samples for recovery."""

    def process_server_response(self, server_response):
        # Physical workers serve many virtual clients: reset per-dispatch state.
        self.local_epochs = None
        self.mga_phase = server_response.get("mga_phase")
        self._mga_response = dict(server_response)
        self._pending_knot_sampler_mode = None
        self._knot_sampler_mode = None
        if self.mga_phase is not None and self.mga_phase not in ("train", "unlearn", "recover"):
            raise ValueError("Invalid explicit MGA phase.")
        if "unlearning_epochs" in server_response:
            self.unlearning_epochs = int(server_response["unlearning_epochs"])
            logging.info(
                "[%s] Dynamic unlearning_epochs (E)=%s "
                "(server aggregations=%s, client aggregations=%s).",
                self,
                self.unlearning_epochs,
                server_response.get("server_aggregation_count"),
                server_response.get("client_aggregation_count"),
            )

        if "local_epochs" in server_response:
            self.local_epochs = int(server_response["local_epochs"])
            logging.info("[%s] Local recovery epochs=%s.", self, self.local_epochs)

        client_pool = set(Config().clients.clients_requesting_deletion)
        sampler_mode = server_response.get("sampler_mode")
        if self.client_id in client_pool and sampler_mode is not None:
            self._pending_knot_sampler_mode = sampler_mode

    def configure(self):
        super().configure()
        self.trainer.mga_context = None
        self.trainer.mga_reference = None
        self.trainer.mga_reverse_result = None
        self._mga_partition = None
        # Plato configures a physical worker before loading its first datasource.
        if self.sampler is None:
            return
        sampler_mode = getattr(self, "_pending_knot_sampler_mode", None)
        explicit = getattr(self, "mga_phase", None) is not None
        poisoning = bool(getattr(Config().clients, "backdoor_enabled", False))
        if explicit or poisoning:
            original = list(self.sampler.get())
            self._mga_partition = original
            membership = original
            seed = int(getattr(Config().data, "random_seed", 1))
            if sampler_mode in ("forget", "retain"):
                scope = str(getattr(Config().trainer, "mga_forget_scope", "client")) if explicit else "samples"
                if scope not in ("client", "samples"):
                    raise ValueError("mga_forget_scope must be 'client' or 'samples'.")
                if scope == "samples":
                    removed = select_subset(original, Config().clients.deleted_data_ratio, seed, self.client_id)
                    membership = [i for i in original if (i in removed) == (sampler_mode == "forget")]
                elif sampler_mode == "retain":
                    raise ValueError("A fully removed client cannot participate in recovery.")
            self.sampler = IndexSampler(membership, seed + self.current_round * 10007 + self.client_id)
            self._knot_sampler_mode = sampler_mode
            if explicit:
                response = self._mga_response
                dispatch_id = response.get("mga_dispatch_id", "")
                if not re.fullmatch(r"[0-9]+_[0-9]+_[0-9]+", dispatch_id):
                    raise ValueError("Missing or malformed MGA dispatch ID.")
                output = Path(Config().params["result_path"]) / "mga_clients"
                output.mkdir(parents=True, exist_ok=True)
                self.trainer.mga_context = {
                    "phase": self.mga_phase, "client_id": self.client_id,
                    "round": self.current_round, "request_id": response.get("mga_request_id"),
                    "dispatch_id": dispatch_id,
                    "execution_file": str(output / (dispatch_id + ".execution.json")),
                    "iterations": response.get("unlearning_epochs"),
                    "lr": response.get("mga_lr"), "radius": response.get("mga_radius"),
                }
                if Path(self.trainer.mga_context["execution_file"]).exists():
                    raise FileExistsError("Refusing to reuse an earlier dispatch result.")
            self._pending_knot_sampler_mode = None
            return
        if sampler_mode is None:
            return

        if sampler_mode == "forget":
            self.sampler = forgetting_iid.Sampler(
                self.datasource, self.client_id, False
            )
            logging.info(
                "[%s] Forget sampler deployed: samples=%s, deleted_ratio=%.1f%%.",
                self,
                self.sampler.num_samples(),
                Config().clients.deleted_data_ratio * 100,
            )
        elif sampler_mode == "retain":
            self.sampler = unlearning_iid.Sampler(
                self.datasource, self.client_id, False
            )
            logging.info(
                "[%s] Retain sampler deployed: samples=%s, retained_ratio=%.1f%%.",
                self,
                self.sampler.num_samples(),
                (1.0 - Config().clients.deleted_data_ratio) * 100,
            )
        else:
            raise ValueError(f"Unknown sampler mode: {sampler_mode}")

        self._knot_sampler_mode = sampler_mode
        self._pending_knot_sampler_mode = None

    def _allocate_data(self):
        super()._allocate_data()
        if getattr(self, "_mga_partition", None) is None:
            return
        cfg = Config().clients
        spec = PoisonSpec(
            enabled=bool(getattr(cfg, "backdoor_enabled", False)),
            rate=float(getattr(cfg, "backdoor_poison_rate", 0.0)),
            target_label=getattr(cfg, "backdoor_target_label", 1),
            seed=int(getattr(cfg, "backdoor_seed", getattr(Config().data, "random_seed", 1))),
            dataset=Config().data.datasource,
            trigger=str(getattr(cfg, "backdoor_trigger", "invert")),
        )
        retrain = bool(getattr(Config().args, "scratch", False) or getattr(Config().trainer, "knot_retrain", False))
        target = self.client_id in set(cfg.clients_requesting_deletion)
        active = target and not retrain and (self.mga_phase in ("train", "unlearn") or (
            self.mga_phase is None and (self.current_round < cfg.data_deletion_round
                                       or getattr(self, "_knot_sampler_mode", None) == "forget")))
        self.trainset = PoisonedDataset(self.trainset, self._mga_partition, spec, self.client_id, active)
        manifest = self.trainset.manifest()
        manifest.update(phase=self.mga_phase, round=self.current_round,
                        sampled_examples=self.sampler.num_samples())
        logging.info("MGA_POISON_PLAN %s", json.dumps(manifest, sort_keys=True))
        if self.trainer.mga_context is not None:
            context = self.trainer.mga_context
            context["poison_plan"] = manifest
            p = Path(context["execution_file"]).with_suffix(".plan.json")
            with p.open("x", encoding="utf8") as handle:
                json.dump(manifest, handle, indent=2)

    def _load_payload(self, server_payload):
        super()._load_payload(server_payload)
        if getattr(self, "mga_phase", None) == "unlearn":
            self.trainer.mga_reference = clone_state(server_payload)

    def customize_report(self, report):
        report = super().customize_report(report)
        if getattr(self, "mga_phase", None) is None:
            return report
        context = self.trainer.mga_context
        path = Path(context["execution_file"])
        if not path.is_file():
            raise RuntimeError("Training did not write fresh MGA execution evidence.")
        execution = json.loads(path.read_text(encoding="utf8"))
        for key in ("phase", "client_id", "request_id", "dispatch_id", "round"):
            if execution.get(key) != context.get(key):
                raise RuntimeError(f"MGA execution evidence mismatches {key}.")
        report.mga_phase = self.mga_phase
        report.mga_request_id = context["request_id"]
        report.mga_dispatch_id = context["dispatch_id"]
        report.mga_execution = execution
        return report

