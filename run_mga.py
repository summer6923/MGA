# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Run a fresh MGA experiment without overwriting code, models, or results.

Linux: python run_mga.py --config configs/mga_mnist_smoke.yml \
  --data-path /path/to/cached/data --output /new/output/directory --gpu 0
"""
from pathlib import Path
import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time

import yaml


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0", help="CUDA device index, or empty string for CPU")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("This process-group launcher requires Linux/POSIX.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = Path(__file__).resolve().parent
    source_config = Path(args.config).resolve()
    cfg = yaml.safe_load(source_config.read_text(encoding="utf8"))
    if cfg.get("server", {}).get("initial_model_state_path"):
        parser.error("A pretrained initialization is not allowed in a scratch run.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "original_config.yml").write_bytes(source_config.read_bytes())
    cfg.setdefault("general", {})["base_path"] = str(output)
    cfg["data"]["data_path"] = str(Path(args.data_path).resolve())
    cfg["server"]["model_path"] = "models"
    cfg["server"]["checkpoint_path"] = "checkpoints"
    cfg.setdefault("results", {})["result_path"] = "results"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg["server"]["port"] = sock.getsockname()[1]
    cfg["server"]["address"] = "127.0.0.1"
    resolved = output / "config.yml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("config_file", None)
    start = time.monotonic()
    reason = None
    command = [sys.executable, "-B", "-u", str(root / "mga_entry.py"), "-c", str(resolved), "-l", "info"]
    process = None
    try:
        with (output / "train.log").open("x", encoding="utf8") as log:
            process = subprocess.Popen(command, cwd=output, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            last_print = start
            while process.poll() is None:
                now = time.monotonic()
                if now - start > args.timeout:
                    reason = "absolute_runtime_timeout"
                    break
                if now - last_print >= 30:
                    status = output / "results/mga_status.json"
                    print("MGA_PROGRESS", status.read_text() if status.exists() else "starting", flush=True)
                    last_print = now
                time.sleep(1)
    finally:
        if process is not None:
            stop_group(process)
    status_file = output / "results/mga_status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    complete = process.returncode == 0 and status.get("status") == "completed" and reason is None
    result = {"returncode": process.returncode, "completed": complete, "reason": reason,
              "actual_runtime_seconds": time.monotonic() - start, "status": status,
              "run": str(output), "command": command}
    (output / "run_outcome.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("MGA_OUTCOME", json.dumps(result), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
