# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
The Multi-Layer Perception model for PyTorch.
The model follows the previous work to use tanh as activation
Reference: https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2019.pdf
"""
import collections

import torch.nn as nn

from plato.config import Config
from einops import rearrange



class Model(nn.Module):
    """The Multi-Layer Perception model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """

    def __init__(self, input_dim=600, num_classes=100,d_model=512,num_heads=8,num_layers=6):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024), nn.Tanh())

        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.Tanh())

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.fc5 = nn.Linear(128, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict["fc1"] = self.fc1
        self.layerdict["fc2"] = self.fc2
        self.layerdict["fc3"] = self.fc3
        self.layerdict["fc4"] = self.fc4
        self.layerdict["fc5"] = self.fc5

        self.layers.append("fc1")
        self.layers.append("fc2")
        self.layers.append("fc3")
        self.layers.append("fc4")
        self.layers.append("fc5")


        self.output_dim = num_classes
        self.embed_layer = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """Forward pass."""
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        x = x.unsqueeze(1)
        B,T,D = x.size()
        x_emb = self.embed_layer(x)
        x_emb = rearrange(x_emb, 'b m d -> m b d')
        x_trans = self.transformer_encoder(x_emb) 
        x_trans = rearrange(x_trans, 'm b d -> b m d') 
        x_out = self.fc(x_trans)
        x_out = self.norm(x_out)
        return x_out[:, -1, :]

        return x

    # def forward_to(self, x, cut_layer):
    #     """Forward pass, but only to the layer specified by cut_layer."""
    #     layer_index = self.layers.index(cut_layer)

    #     for i in range(0, layer_index + 1):
    #         x = self.layerdict[self.layers[i]](x)

    #     return x

    # def forward_from(self, x, cut_layer):
    #     """Forward pass, starting from the layer specified by cut_layer."""
    #     layer_index = self.layers.index(cut_layer)

    #     for i in range(layer_index + 1, len(self.layers)):
    #         x = self.layerdict[self.layers[i]](x)

    #     return x

    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        if hasattr(Config().trainer, "num_classes"):
            return Model(
                input_dim=Config().trainer.input_dim,
                num_classes=Config().trainer.num_classes,
                d_model=Config().trainer.d_model
            )
        return Model()
