# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _locked(lock_path):
    lock_file = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _atomic_save(path, values):
    path = Path(path)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    np.save(str(temporary_path), values)
    os.replace(str(temporary_path) + ".npy", str(path))


def _load_or_initialize(path, total_clients):
    path = Path(path)
    expected_size = int(total_clients) + 1
    try:
        values = np.load(str(path))
        if values.ndim != 1:
            raise ValueError("pth_rec must be one-dimensional")
    except (FileNotFoundError, ValueError, EOFError):
        values = np.zeros(expected_size, dtype=np.float64)

    if values.size < expected_size:
        expanded = np.zeros(expected_size, dtype=np.float64)
        expanded[: values.size] = values
        values = expanded
    return values


def ensure_pth_rec(path, total_clients):
    path = Path(path)
    with _locked(str(path) + ".lock"):
        values = _load_or_initialize(path, total_clients)
        _atomic_save(path, values)


def load_pth_rec(path, total_clients):
    path = Path(path)
    ensure_pth_rec(path, total_clients)
    return _load_or_initialize(path, total_clients)


def update_pth_rec(path, client_id, current_round, total_clients):
    path = Path(path)
    with _locked(str(path) + ".lock"):
        values = _load_or_initialize(path, total_clients)
        values[int(client_id)] = float(current_round)
        _atomic_save(path, values)
