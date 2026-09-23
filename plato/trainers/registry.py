# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging

from plato.config import Config

from plato.trainers import basic

registered_trainers = {
    "basic": basic.Trainer,
}


def get(model=None, callbacks=None):
    """Get the trainer with the provided name."""
    trainer_name = Config().trainer.type
    logging.info("Trainer: %s", trainer_name)

    if trainer_name in registered_trainers:
        return registered_trainers[trainer_name](model=model, callbacks=callbacks)
    else:
        raise ValueError(f"No such trainer: {trainer_name}")
