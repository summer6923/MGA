# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Data sources used by MGA's image and tabular experiments."""
import logging
from plato.config import Config
from plato.datasources import mnist, cifar10, purchase

registered_datasources = {"MNIST": mnist, "CIFAR10": cifar10, "Purchase": purchase}


def get(client_id=0, **kwargs):
    name = kwargs.get("datasource_name", Config().data.datasource)
    if name not in registered_datasources:
        raise ValueError(f"Unsupported MGA data source: {name}")
    logging.info("Data source: %s", name)
    return registered_datasources[name].DataSource(**kwargs)


def get_input_shape():
    name = Config().data.datasource
    if name not in registered_datasources:
        raise ValueError(f"Unsupported MGA data source: {name}")
    return registered_datasources[name].DataSource.input_shape()
