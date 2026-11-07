"""Encoder/decoder ablations for scNODE, installed under scNODE/optim/.

Extends the unchanged original model.layer classes so dynamic_model.py's
isinstance(latent_encoder, LinearVAENet) check continues to pass.

Alternatives (apply ONE at a time, to encoder/decoder hidden layers only):
- layernorm: Linear -> LayerNorm -> original activation (Ba et al., 2016)
- dropout: Linear -> original activation -> Dropout (Srivastava et al., 2014)
- gelu: replace hidden-layer activation by GELU (Hendrycks & Gimpel, 2016)

The final projection/activation, VAE mu/std heads, and ODE are unchanged.
The original VAE standard deviation parameterization is also unchanged.
"""

import torch.nn as nn
from model.layer import LinearNet, LinearVAENet

_VALID = {"layernorm", "dropout", "gelu"}


def _change_hidden_layers(net, variant, dropout_p):
    if variant not in _VALID:
        raise ValueError("variant must be one of {}".format(sorted(_VALID)))
    if not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must be >= 0 and < 1")

    layers = list(net.net)  # preserve repeated activation modules
    linear_positions = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]
    if len(linear_positions) < 2:
        raise ValueError("Architecture ablation requires at least one hidden layer")
    output_linear_position = linear_positions[-1]
    modified = []
    for idx, layer in enumerate(layers):
        if isinstance(layer, nn.Linear) and idx != output_linear_position:
            modified.append(layer)
            if variant == "layernorm":
                modified.append(nn.LayerNorm(layer.out_features))
        elif idx > 0 and isinstance(layers[idx - 1], nn.Linear) and (idx - 1) != output_linear_position:
            # This is the activation immediately following a hidden Linear.
            modified.append(nn.GELU() if variant == "gelu" else layer)
            if variant == "dropout":
                modified.append(nn.Dropout(p=dropout_p))
        else:
            modified.append(layer)
    net.net = nn.Sequential(*modified)


class AblationVAENet(LinearVAENet):
    """Original variational encoder, with one hidden-layer ablation."""

    def __init__(self, input_dim, latent_size_list, output_dim, act_name,
                 variant="layernorm", dropout_p=0.1):
        super().__init__(input_dim, latent_size_list, output_dim, act_name)
        _change_hidden_layers(self, variant, dropout_p)


class AblationDecoder(LinearNet):
    """Original observation decoder, with one hidden-layer ablation."""

    def __init__(self, input_dim, latent_size_list, output_dim, act_name,
                 variant="layernorm", dropout_p=0.1):
        super().__init__(input_dim, latent_size_list, output_dim, act_name)
        _change_hidden_layers(self, variant, dropout_p)
