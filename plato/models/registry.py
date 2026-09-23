# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
The registry for machine learning models.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
from typing import Union

from plato.config import Config

from plato.models import lenet5, multilayer, resnet

registered_models = {"lenet5": lenet5.Model, "multilayer": multilayer.Model}
registered_factories = {"resnet": resnet.Model}


def get(**kwargs: Union[str, dict]):
    """Get the model with the provided name."""
    model_name = (
        kwargs["model_name"] if "model_name" in kwargs else Config().trainer.model_name
    )

    model_type = (
        kwargs["model_type"]
        if "model_type" in kwargs
        else (
            Config().trainer.model_type
            if hasattr(Config().trainer, "model_type")
            else model_name.split("_")[0]
        )
    )

    model_params = (
        kwargs["model_params"]
        if "model_params" in kwargs
        else (
            Config().parameters.model._asdict()
            if hasattr(Config().parameters, "model")
            else {}
        )
    )

    if model_type in registered_models:
        registered_model = registered_models[model_type]
        return registered_model(**model_params)

    if model_type in registered_factories:
        return registered_factories[model_type].get(
            model_name=model_name, **model_params
        )

    raise ValueError(f"No such model: {model_name}")
