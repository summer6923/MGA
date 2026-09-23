# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
Obtaining a model from the PyTorch Hub.
"""

import torch


class Model:
    """
    The model loaded from PyTorch Hub.

    We will soon be using the get_model() method for torchvision 0.14 when it is released.
    """

    @staticmethod
    # pylint: disable=unused-argument
    def get(model_name=None, **kwargs):
        """Returns a named model from PyTorch Hub."""
        return torch.hub.load("pytorch/vision", model_name, **kwargs)
