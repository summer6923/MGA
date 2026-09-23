# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Isolated, seeded entrypoint used by run_mga.py, including spawned workers."""
import hashlib
import io
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from plato.config import Config

CFG = Config()
RUN = Path(CFG.params["base_path"]).resolve()
SEED = int(getattr(CFG.trainer, "random_seed", getattr(CFG.data, "random_seed", 1)))
random.seed(SEED)
np.random.seed(SEED % (2**32 - 1))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(int(os.environ.get("MGA_NUM_THREADS", "2")))

ORIGINAL_LOAD = torch.load

def audited_load(source, *args, **kwargs):
    if isinstance(source, (str, os.PathLike)):
        resolved = Path(source).resolve()
        if not resolved.is_relative_to(RUN):
            raise RuntimeError(f"Scratch run refuses an external checkpoint: {resolved}")
        with (RUN / f"checkpoint_reads_{os.getpid()}.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps({"path": str(resolved), "pid": os.getpid()}) + "\n")
    elif not isinstance(source, io.BytesIO):
        raise RuntimeError("Unknown model-load source; only current-run files or tensor IPC are permitted.")
    return ORIGINAL_LOAD(source, *args, **kwargs)

torch.load = audited_load

from knot_server import Server
from knot_client import Client
from knot_algorithm import Algorithm
from knot_trainer import Trainer


class ScratchServer(Server):
    def init_trainer(self):
        super().init_trainer()
        state = {k: v.detach().cpu().clone() for k, v in self.trainer.model.state_dict().items()}
        torch.save(state, RUN / "initial_random_state.pth")
        h = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            h.update(name.encode()); h.update(str(tensor.dtype).encode())
            h.update(tensor.contiguous().numpy().tobytes())
        (RUN / "initialization.json").write_text(json.dumps({
            "seed": SEED, "resume": bool(CFG.args.resume),
            "initial_model_state_path": getattr(CFG.server, "initial_model_state_path", None),
            "tensor_sha256": h.hexdigest(), "tensors": len(state),
            "torch_version": torch.__version__, "model_module": type(self.trainer.model).__module__,
        }, indent=2), encoding="utf8")


if __name__ == "__main__":
    if CFG.args.resume or getattr(CFG.server, "initial_model_state_path", None):
        raise ValueError("This entrypoint is for fresh random-weight training only.")
    client = Client(algorithm=Algorithm, trainer=Trainer)
    server = ScratchServer(algorithm=Algorithm, trainer=Trainer)
    server.run(client)
