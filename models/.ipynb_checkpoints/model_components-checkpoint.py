import torch
import torch.nn as nn 

def make_fc_layers(in_dim: int,
                   hidden_dim: int = 128,
                   num_layers: int = 2,
                   activation: str = "relu"
                   ):
    """
    Generate fully
    """
    activations = {"tanh":nn.Tanh, "lrelu":nn.LeakyReLU, "relu":nn.ReLU}
    fc_layers = []
    for i in range(num_layers):
        hidden_dim_ = in_dim if i == 0 else hidden_dim
        fc_layers.extend(
            [nn.Linear(hidden_dim_, hidden_dim),
            activations[activation]()])
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers