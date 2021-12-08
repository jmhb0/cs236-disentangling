from .base_vae import BaseVAE
from .types_ import *

class VAE(BaseVAE):
    def __init__(self, 
                 img_shape,
                 z_dim=4,
                 encoder_kwargs=dict(type='fc', hidden_dims=200, hidden_dim=2),
                 decoder_kwargs=dict(type='fc', hidden_dims=200, hidden_dim=2),
                ):
        x=super(VAE, self).__init__()

    
    def forward(self, *inputs) -> Tensor:
        pass

    def loss_function(self, *inputs):
        pass