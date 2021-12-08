import torch
import torch.nn as nn 
from .types_ import *

activations = dict(tanh=nn.Tanh, lrelu=nn.LeakyReLU, relu=nn.ReLU, none=nn.Identity)

def make_fc_layers_no_out(in_dim: int,
                   hidden_dims: List[int] = [128,128],
                   activation: str = "relu",
                   **kwargs
                   ):
    """
    (version 1: the output is the last 'hidden dim')
    Generate a fully connected network. Take input size in_dim, expand to the 
    size of first hidden layer, and then keep passing through. 

    The output layer is the last layer of `hidden_dims`. For example if we have 
    in_dim=2, and hidden_dims=[128,128], then we would have one layer (2,128), 
    followed by nonlinearity, then a layer (128,128), followed by a nonlinearity.
    """
    fc_layers = []
    h_dim_ = in_dim

    for h_dim in hidden_dims:
        fc_layers.extend(
            [nn.Linear(h_dim_, h_dim),
            activations[activation]()])
        h_dim_ = h_dim

    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers

def make_fc_layers(in_dim: int,
                   out_dim: int,
                   hidden_dims: List[int] = [128,128],
                   activation: str = "relu",
                   **kwargs
                   ):
    """
    (version 2: the out_dim is specified, and the hidden_dims are acutally hidden layers)
    Generate a fully connected network. Take input size in_dim, expand to the 
    size of first hidden layer, and then keep passing through. 

    The output layer is the last layer of `hidden_dims`. For example if we have 
    in_dim=2, and hidden_dims=[128,128], then we would have one layer (2,128), 
    followed by nonlinearity, then a layer (128,128), followed by a nonlinearity.
    """
    fc_layers = []
    h_dim_ = in_dim

    for h_dim in hidden_dims:
        fc_layers.extend(
            [nn.Linear(h_dim_, h_dim),
            activations[activation]()]
            )
        h_dim_ = h_dim
    fc_layers.extend(
            [nn.Linear(h_dim_, out_dim),
            activations[activation]()]
            )
            

    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers

def make_cnn_layers(in_dim=(1,36,36), in_channels=1,
                    hidden_dims=[32,64,128,256], activation='relu',
                    do_norm_layer=True,
                    **kwargs,
                   ):
    """
    """
    if len(in_dim)==2: in_dim=(1,*in_dim)
    NormLayer = nn.BatchNorm2d if do_norm_layer else nn.Identity
    ActivationLayer = activations[activation]

    modules = []
    for h_dim in hidden_dims:
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
            NormLayer(h_dim),
            ActivationLayer(),
        ))
        in_channels = h_dim

    cnn_layer = nn.Sequential(*modules)
    return cnn_layer

def make_de_cnn_layers(out_dims=1, hidden_dims=[256,128,64,32],
                    activation='relu', do_norm_layer=True,
                    **kwargs,):
    """
    Out dims is the cnn out channel number
    """
    NormLayer = nn.BatchNorm2d if do_norm_layer else nn.Identity
    ActivationLayer = activations[activation]

    modules = []
    for i in range(len(hidden_dims)):
        hidden_dim_in = hidden_dims[i]
        hidden_dim_out = hidden_dims[i+1] if i+1<len(hidden_dims) else out_dims
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dim_in, hidden_dim_out,kernel_size=3,
                            stride = 2, padding=1, output_padding=1),
            NormLayer(hidden_dim_out),
            ActivationLayer(),
        ))
    de_cnn_layer = nn.Sequential(*modules)

    return de_cnn_layer

