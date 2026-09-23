# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
This registry for Processors contains framework-specific implementations of
Processors for data payloads.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from typing import Tuple

from plato.config import Config
from plato.processors import pipeline

from plato.processors import base

registered_processors = {"base": base.Processor}


def get(
    user: str, processor_kwargs=None, **kwargs
) -> Tuple[pipeline.Processor, pipeline.Processor]:
    """Get an instance of the processor."""
    outbound_processors = []
    inbound_processors = []

    assert user in ("Server", "Client")

    if user == "Server":
        config = Config().server
    else:
        config = Config().clients

    if hasattr(config, "outbound_processors") and isinstance(
        config.outbound_processors, list
    ):
        outbound_processors = config.outbound_processors

    if hasattr(config, "inbound_processors") and isinstance(
        config.inbound_processors, list
    ):
        inbound_processors = config.inbound_processors

    for processor in outbound_processors:
        logging.info("%s: Using Processor for sending payload: %s", user, processor)
    for processor in inbound_processors:
        logging.info("%s: Using Processor for receiving payload: %s", user, processor)

    def map_f(name):
        if processor_kwargs is not None and name in processor_kwargs:
            this_kwargs = {**kwargs, **(processor_kwargs[name])}
        else:
            this_kwargs = kwargs

        if name not in registered_processors:
            raise ValueError(f"Unsupported MGA payload processor: {name}")
        return registered_processors[name](name=name, **this_kwargs)

    outbound_processors = list(map(map_f, outbound_processors))
    inbound_processors = list(map(map_f, inbound_processors))

    return pipeline.Processor(outbound_processors), pipeline.Processor(
        inbound_processors
    )
