# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
The registry for samplers designed to partition the dataset across the clients.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""
import logging

from plato.config import Config

from plato.samplers import iid, dirichlet

registered_samplers = {"iid": iid.Sampler, "noniid": dirichlet.Sampler}


def get(datasource, client_id, testing=False, **kwargs):
    """Get an instance of the sampler."""

    sampler_type = (
        kwargs["sampler_type"]
        if "sampler_type" in kwargs
        else Config().data.testset_sampler
        if testing and hasattr(Config().data, "testset_sampler")
        else Config().data.sampler
    )
    if testing:
        logging.info("[Client #%d] Test set sampler: %s", client_id, sampler_type)
    else:
        logging.info("[Client #%d] Sampler: %s", client_id, sampler_type)
        
    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](
            datasource, client_id, testing=testing
        )
    else:
        raise ValueError(f"No such sampler: {sampler_type}")

    return registered_sampler
