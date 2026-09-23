# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
Implements a Processor for applying local differential privacy using gaussian mechanism.
"""
import math

from plato.processors import feature_additive_noise


class Processor(feature_additive_noise.Processor):
    """
    Implements a Processor for applying local differential privacy using gaussian mechanism.
    """
    def __init__(self,
                 epsilon=None,
                 delta=None,
                 sensitivity=None,
                 **kwargs) -> None:

        scale = 2 * math.log(1.25 / delta) * sensitivity**2 / epsilon**2
        super().__init__(method="gaussian", scale=scale, **kwargs)
