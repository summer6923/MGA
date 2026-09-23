# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Request barrier for MGA on the existing asynchronous Plato transport.

Normal training/recovery retain the existing scheduler. At one removal request we
freeze the accepted retained records, drain/discard old jobs, dispatch every target
exactly once against one immutable reference, and wait for every validated return
before recovery. A reverse round is never passed to retained-only FedAvg.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import torch
from plato.config import Config


def enabled_for_config(cfg):
    if bool(getattr(cfg.args, "scratch", False)) or bool(
            getattr(cfg.trainer, "knot_retrain", False)):
        return False
    explicit = getattr(cfg.trainer, "mga_enabled", None)
    if explicit is not None:
        return bool(explicit)
    strategy = str(getattr(cfg.trainer, "unlearning_strategy", "")).lower()
    return strategy == "mga" or (strategy == "knot" and bool(
        getattr(cfg.server, "skip_cluster_rollback_after_deletion", False)))


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def combine_states(template, states, weights, normalize=True):
    if not states or len(states) != len(weights):
        raise ValueError("States and weights must be nonempty and paired.")
    if any(not math.isfinite(float(w)) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Invalid state-combination weights.")
    if any(set(s) != set(template) for s in states):
        raise ValueError("Returned state keys differ from the reference.")
    coeff = [w / sum(weights) for w in weights] if normalize else weights
    result = clone_state(template)
    for key, target in result.items():
        for state in states:
            value = state[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Incompatible model state at {key}.")
            if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite model state at {key}.")
        if not (target.is_floating_point() or target.is_complex()):
            continue  # Integer buffers are not reverse-optimization parameters.
        if normalize:
            target.zero_()
            for state, weight in zip(states, coeff):
                target.add_(state[key].detach().cpu(), alpha=weight)
        else:
            for state, weight in zip(states, coeff):
                target.add_(state[key].detach().cpu() - template[key].cpu(), alpha=weight)
    return result


class PendingTargets:
    def __init__(self, targets, request_id):
        self.targets = tuple(sorted(int(i) for i in targets))
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("MGA requires distinct target clients.")
        self.request_id = request_id
        self.returns = {}

    @property
    def pending(self):
        return [i for i in self.targets if i not in self.returns]

    def accept(self, client_id, request_id, payload):
        if request_id != self.request_id:
            raise ValueError("Stale or wrong MGA request identifier.")
        if client_id not in self.targets:
            raise ValueError("A retained client cannot return a target-local update.")
        if client_id in self.returns:
            raise ValueError("Duplicate MGA return; refusing to count it twice.")
        self.returns[client_id] = clone_state(payload)


class MGAPipelineMixin:
    def _init_mga_pipeline(self):
        cfg = Config()
        self._mga_enabled = enabled_for_config(cfg)
        self._mga_phase = "train"
        self._mga_reverse_round = False
        self._mga_records = {}
        self._mga_busy = False
        self._mga_request = None
        self._mga_normal_per_round = self.clients_per_round
        self._mga_normal_minimum = self.minimum_clients
        if not self._mga_enabled:
            return
        if bool(getattr(cfg.args, "resume", False)):
            raise ValueError("MGA request state is not resumable: use a new isolated run.")
        targets = list(cfg.clients.clients_requesting_deletion)
        if (not targets or len(set(targets)) != len(targets)
                or any(type(i) is not int or i < 1 or i > self.total_clients for i in targets)
                or len(targets) >= self.total_clients):
            raise ValueError("Invalid target-client set; retained clients must remain.")
        request = int(cfg.clients.data_deletion_round)
        batches = math.ceil(len(targets) / self.clients_per_round)
        if request < 2 or int(cfg.trainer.rounds) < request + batches:
            raise ValueError("Need pretraining, all target batches, and at least one recovery round.")
        radius = float(getattr(cfg.trainer, "mga_radius", 0.5))
        timeout = float(getattr(cfg.server, "mga_request_timeout", 600.0))
        if not math.isfinite(radius) or radius <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("MGA radius and request timeout must be finite and positive.")
        self._mga_dir = Path(cfg.params["result_path"])
        self._mga_dir.mkdir(parents=True, exist_ok=True)
        if (self._mga_dir / "mga_status.json").exists():
            raise FileExistsError("MGA requires a fresh output directory; refusing to overwrite a run.")
        self._mga_write_status("training")

    def _mga_event(self, event, **fields):
        record = {"event": event, "round": int(self.current_round),
                  "monotonic_seconds": time.monotonic(), "phase": self._mga_phase,
                  "request_id": self._mga_request.request_id if self._mga_request else None,
                  **fields}
        with (self._mga_dir / "mga_events.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        logging.info("MGA_EVENT %s", json.dumps(record, allow_nan=False))

    def _mga_write_status(self, status, **fields):
        data = {"status": status, "phase": self._mga_phase,
                "request_id": self._mga_request.request_id if self._mga_request else None,
                "pending_targets": self._mga_request.pending if self._mga_request else None,
                "received_targets": sorted(self._mga_request.returns) if self._mga_request else [],
                **fields}
        temp = self._mga_dir / "mga_status.json.tmp"
        temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")
        os.replace(temp, self._mga_dir / "mga_status.json")

    def _mga_capture_retained(self, updates):
        if not self._mga_enabled or self._mga_phase != "train":
            return
        targets = set(Config().clients.clients_requesting_deletion)
        accepted = set(self._last_aggregated_client_ids)
        for update in updates:
            i = int(update.client_id)
            if i in accepted and i not in targets:
                score = max(0, self.server_aggregation_count - self.client_aggregation_counts.get(i, 0))
                self._mga_records[i] = (clone_state(update.payload), score)

    def _mga_begin_request(self):
        self._mga_phase = "drain"
        self._mga_deadline = time.monotonic() + float(getattr(Config().server, "mga_request_timeout", 600.0))
        self._mga_request = PendingTargets(Config().clients.clients_requesting_deletion, uuid.uuid4().hex)
        self._mga_count_snapshot = dict(self.client_aggregation_counts)
        self._mga_server_count = self.server_aggregation_count
        self._mga_before = clone_state(self.trainer.model.state_dict())
        self._mga_drain_finish = self.wall_time
        torch.save(self._mga_before, self._mga_dir / "mga_before_request.pth")
        buffered = list(self.reported_clients)
        for item in buffered:
            if isinstance(item, tuple):
                self._mga_drain_finish = max(self._mga_drain_finish, item[0])
        self.reported_clients = []
        self.updates = []
        self._mga_event("request", targets=list(self._mga_request.targets),
                        discarded_buffered_updates=len(buffered),
                        draining_active_clients=sorted(self.training_clients))
        self._mga_write_status("draining")

    def _mga_finish_drain(self):
        if self.training_clients:
            raise RuntimeError("Cannot construct request barrier while old clients are active.")
        if not self._mga_records:
            raise RuntimeError("No accepted retained-client states exist for the reference.")
        ids = sorted(self._mga_records)
        uniform = str(getattr(Config().trainer, "avg_mode", "")) == "uniform"
        weights = [1.0 if uniform else 1.0 / (1 + self._mga_records[i][1]) for i in ids]
        self._mga_reference = combine_states(
            self._mga_before, [self._mga_records[i][0] for i in ids], weights)
        self._mga_return_weights = {
            i: float(getattr(Config().trainer, "mga_alpha0", 1.0)) /
               (1 + max(0, self._mga_server_count - self._mga_count_snapshot.get(i, 0)))
            for i in self._mga_request.targets
        }
        if any(not math.isfinite(w) or w <= 0 for w in self._mga_return_weights.values()):
            raise ValueError("MGA return weights must be positive and finite.")
        torch.save(self._mga_reference, self._mga_dir / "mga_reference.pth")
        self.wall_time = max(self.wall_time, self._mga_drain_finish)
        self.selected_clients = None
        self.reported_clients = []
        self.updates = []
        self.current_reported_clients = {}
        self.current_processed_clients = {}
        self.training_sids = []
        self._mga_phase = "unlearn"
        self._mga_event("reference_ready", retained_clients=ids,
                        reference_weights=[w / sum(weights) for w in weights],
                        return_weights=self._mga_return_weights,
                        weighting="uniform" if uniform else "recorded_participation_staleness")
        self._mga_write_status("unlearning")

    async def _select_clients(self, for_next_batch=False):
        if not getattr(self, "_mga_enabled", False):
            return await super()._select_clients(for_next_batch=for_next_batch)
        if not for_next_batch and self._mga_phase == "train" and (
                self.current_round + 1 >= int(Config().clients.data_deletion_round)):
            self._mga_normal_minimum = self.minimum_clients
            self._mga_normal_per_round = self.clients_per_round
            self._mga_begin_request()
        if self._mga_phase == "drain":
            if self.training_clients:
                return
            self._mga_finish_drain()
        if not for_next_batch:
            self._mga_reverse_round = self._mga_phase == "unlearn"
            if self._mga_reverse_round:
                self.clients_per_round = min(self._mga_normal_per_round, len(self._mga_request.pending))
                self.minimum_clients = self.clients_per_round
                self.selected_clients = None
                self._mga_batch_reports = {}
            elif self._mga_phase == "recover":
                self.clients_per_round = min(self._mga_normal_per_round,
                                             self.total_clients - len(self._mga_request.targets))
                self.minimum_clients = min(self._mga_normal_minimum, self.clients_per_round)
        return await super()._select_clients(for_next_batch=for_next_batch)

    def _mga_choose_targets(self, clients_pool, clients_count):
        selected = self._mga_request.pending[:clients_count]
        if not selected or not set(selected).issubset(clients_pool):
            raise RuntimeError("Target dispatch blocked; no recovery is allowed before completion.")
        self._mga_event("target_batch_selected", clients=selected)
        return selected

    def _mga_response(self, response, client_id):
        response["mga_phase"] = self._mga_phase
        response["mga_request_id"] = self._mga_request.request_id if self._mga_request else None
        response["mga_dispatch_id"] = f"{os.getpid()}_{self.current_round}_{client_id}"
        if self._mga_phase == "unlearn":
            if client_id not in self._mga_request.pending:
                raise RuntimeError("Attempted to dispatch a non-pending target.")
            cfg = Config().trainer
            fixed = getattr(cfg, "fixed_unlearning_epochs", None)
            minimum, maximum = int(getattr(cfg, "dynamic_e_min", 1)), int(getattr(cfg, "dynamic_e_max", 5))
            count = self._mga_count_snapshot.get(client_id, 0)
            budget = int(fixed) if fixed is not None else min(maximum, minimum + math.floor(math.log2(1 + count)))
            if minimum < 1 or maximum < minimum or budget < 1:
                raise ValueError("Invalid MGA iteration budget; refusing a silent no-op.")
            base_lr = float(Config().parameters.optimizer.lr)
            lr = float(getattr(cfg, "mga_lr", base_lr * float(getattr(cfg, "unlearning_lr_scale", 1.0))))
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("MGA reverse learning rate must be positive and finite.")
            response.update(unlearning_epochs=budget, sampler_mode="forget",
                            mga_radius=float(getattr(cfg, "mga_radius", 0.5)), mga_lr=lr,
                            client_aggregation_count=count)
            self._mga_event("reverse_dispatch", client_id=client_id, budget=budget, lr=lr,
                            radius=response["mga_radius"], dispatch_id=response["mga_dispatch_id"])
        elif self._mga_phase == "recover":
            if client_id in self._mga_request.targets:
                raise RuntimeError("Target clients are excluded from ordinary recovery.")
            response["local_epochs"] = int(getattr(Config().server, "retained_recovery_epochs_after_deletion", Config().trainer.epochs))
        return response

    def customize_server_payload(self, payload):
        if getattr(self, "_mga_enabled", False) and self._mga_phase == "unlearn":
            return clone_state(self._mga_reference)
        return super().customize_server_payload(payload)

    async def _process_clients(self, client_info):
        if not getattr(self, "_mga_enabled", False) or self._mga_phase in ("train", "recover"):
            return await super()._process_clients(client_info)
        if self._mga_phase == "drain":
            self._mga_drain_finish = max(self._mga_drain_finish, client_info[0])
            self._mga_event("discard_pre_request_return", client_id=client_info[2]["client_id"])
            self.reported_clients = []
            if not self.training_clients:
                await self._select_clients()
            return
        client = client_info[2]
        i = int(client["client_id"])
        if i not in self.selected_clients or i in self._mga_batch_reports:
            raise RuntimeError("Unsolicited or duplicate return in MGA batch.")
        self._mga_batch_reports[i] = client_info
        if len(self._mga_batch_reports) == len(self.selected_clients):
            self._mga_busy = True
            try:
                self.wall_time = max(self.wall_time, max(x[0] for x in self._mga_batch_reports.values()))
                self.updates = [SimpleNamespace(client_id=i, report=x[2]["report"],
                               payload=x[2]["payload"], staleness=0)
                               for i, x in sorted(self._mga_batch_reports.items())]
                self.reported_clients = []
                await self._process_reports()
                await self.wrap_up()
                await self._select_clients()
            finally:
                self._mga_busy = False
        elif (hasattr(Config().trainer, "max_concurrency") and
              len(self.current_reported_clients) >= len(self.trained_clients)):
            await self._select_clients(for_next_batch=True)

    def _mga_accept_returns(self, updates):
        for update in updates:
            report = update.report
            context = getattr(report, "mga_execution", None)
            if (getattr(report, "mga_phase", None) != "unlearn" or not context
                    or context.get("request_id") != self._mga_request.request_id
                    or context.get("dispatch_id") != getattr(report, "mga_dispatch_id", None)
                    or context.get("client_id") != update.client_id):
                raise RuntimeError("Missing or mismatched reverse-execution evidence.")
            reverse = context.get("reverse")
            if not reverse or reverse.get("gradient_examples", 0) < 1:
                raise RuntimeError("Target-local reverse optimization did not execute.")
            self._mga_request.accept(update.client_id, getattr(report, "mga_request_id", None), update.payload)
            self._mga_event("reverse_return", client_id=update.client_id, reverse=reverse,
                            poison_reads=context.get("poison_reads", 0))
        self._mga_write_status("unlearning")
        if self._mga_request.pending:
            return clone_state(self._mga_before)
        ids = list(self._mga_request.targets)
        result = combine_states(self._mga_reference,
                                [self._mga_request.returns[i] for i in ids],
                                [self._mga_return_weights[i] for i in ids], normalize=False)
        self.algorithm.load_weights(result)
        for cluster in range(self.num_clusters):
            self.algorithm.load_weights(result, cluster_id=cluster)
        self._mga_phase = "recover"
        torch.save(result, self._mga_dir / "mga_after_unlearning.pth")
        self._mga_event("unlearning_complete", targets=ids)
        self._mga_write_status("recovering")
        return result

    async def _periodic_task(self):
        if getattr(self, "_mga_enabled", False) and (
                self._mga_phase in ("drain", "unlearn") or self._mga_busy):
            if time.monotonic() > self._mga_deadline:
                self._mga_write_status("failed", reason="request_timeout")
                await self._close()
            return
        return await super()._periodic_task()

    async def _close(self):
        if getattr(self, "_mga_enabled", False):
            complete = (self._mga_phase == "recover" and not self._mga_request.pending
                        and self.current_round >= int(Config().trainer.rounds))
            self._mga_write_status("completed" if complete else "failed",
                                   final_round=int(self.current_round))
            if not complete:
                self._mga_event("aborted", reason="closed_before_all_stages_completed")
                await self._close_connections()
                os._exit(2)
        return await super()._close()
