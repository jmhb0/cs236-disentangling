import torch
import torch.nn as nn
from . import base_vae 
from .types_ import *
from . import model_components as mc 
import importlib
importlib.reload(base_vae)

class VAE(base_vae.BaseVAE):
    def __init__(self, 
                 img_shape,
                 img_shape_original,
                 z_dim=4,
                 loss_kwargs: dict = dict(bs=None, N=None, beta=1, recon_key='bce', M_N=1),
                 normalize_out: bool = True,
                 encoder_kwargs=dict(net_type='fc', hidden_dims=[128,128], activation='lrelu'),
                 decoder_kwargs=dict(net_type='fc', hidden_dim=[128,128], activation='lrelu'),
                ):

        # call base class init to initialize standard things
        super().__init__(
                img_shape=img_shape, img_shape_original=img_shape_original, 
                ndim_transforms=0, z_dim=z_dim, 
                loss_kwargs=loss_kwargs, normalize_out=normalize_out,
                )

        # create encoder network
        encoder_net_type=encoder_kwargs.get('net_type','') 
        if encoder_net_type=='fc': 
            self.encoder_net = mc.make_fc_layers(in_dim=self.img_shape[0], **encoder_kwargs) 
            hidden_dims=encoder_kwargs['hidden_dims']
            self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)
        elif encoder_net_type=='cnn': raise NotImplementedError()
        else: raise ValueError()
    
        # create decoder network 
        decoder_net_type=decoder_kwargs.get('net_type','') 
        if decoder_net_type=='fc': 
            self.decoder_net = mc.make_fc_layers(in_dim=z_dim, **decoder_kwargs) 
            self.last_layer = nn.Linear(hidden_dims[-1], self.img_shape[0])
        elif decoder_net_type=='cnn': raise NotImplementedError()
        else: raise ValueError()

        if normalize_out: 
            self.last_layer = nn.Sequential(self.last_layer, nn.Sigmoid())

        # check the input image has the right params
        if encoder_net_type=='fc': assert len(img_shape)==1
        elif encoder_net_type=='cnn': assert len(img_shape)==3

    def encode(self, x):
        x = self.encoder_net(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        return self.last_layer(self.decoder_net(z))

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def forward(self, x) -> Tensor:
        """ y is image output, mu and logvar are parameters of the sampling"""
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        y = self.decode(z)
        return x, y, mu, logvar

    def loss_function(self, x, y, mu, logvar):
        """
        x is data, y is reconstruction
        Paramaters are saved to `self` in call to parent class __init__
        """
        loss_recon= self.recon_loss_func(y,x)*y.size(1)
        loss_kl= -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = loss_recon+ self.M_N*self.beta*loss_kl

        return dict(loss=loss, loss_recon=loss_recon, loss_kl=loss_kl)
    


