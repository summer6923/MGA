# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
Implements a Processor for dequantizing model parameters.
"""

import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for dequantizing model parameters.
    """

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """Quantizes each individual layer of the model."""

        return layer.to(torch.float32)
