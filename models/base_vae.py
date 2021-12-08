from .types_ import *
import torch
from torch import nn
from abc import ABC, abstractmethod
from collections import OrderedDict


class BaseVAE(nn.Module):
    #@property
    #@abstractmethod
    def __init__(self, 
                 img_shape: tuple=None,
                 img_shape_original: tuple=None,
                 device: str = 'cpu',
                 ndim_transforms: int = 0,
                 z_dim: int=2, 
                 loss_kwargs: dict = dict(beta=1, bs=None, N=None, recon_key='BCE'),
                 normalize_out: bool = True,
                 seed: int=0,
                ):
        """
        Args: 
            z_dim: dimension, not including the representation needed for the 
                any transform representations
            img_shape: tuple shape of a single data point. Data should be flat 
                if applicabale             
            ndim_transforms: the number of latent space variables reserved for 
                encoding transformations. 

        loss_kwargs
            bs: batch size 
            N: number of data points
        """
        super().__init__()
        self.device=device
        """
        self.img_shape=img_shape
        self.img_shape_original=img_shape_original
        #assert len(img_shape_original)==3
        """
        self.loss_kwargs=loss_kwargs
        self.z_dim=z_dim
        self.ndim_transforms = ndim_transforms

        if seed is not None: 
            self.set_deterministic_mode(seed)

        # check normalizing out 
        loss_key = loss_kwargs['recon_key']
        if not normalize_out and loss_key=='BCE':
            raise ValueError()
        
        # set up loss function kwargs 
        loss_funcs_lookup = dict(bce=nn.functional.binary_cross_entropy, 
                                mse=nn.functional.mse_loss)
        self.recon_loss_func = loss_funcs_lookup[loss_key]
        self.beta = loss_kwargs['beta']
        self.M_N = loss_kwargs['M_N']

    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs) -> Tensor:
        pass

    def set_deterministic_mode(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


