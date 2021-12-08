from .types_ import *
from torch import nn
from abc import ABC, abstractmethod
from collections import OrderedDict

class BaseVAE(nn.Module, ABC):
    @property
    @abstractmethod
    def __init__(self, 
                 img_shape: tuple,
                 z_dim: int, 
                 loss_kwargs: dict =dict(M_N=1, beta=1, key='BCE'),
                ):
        """
        Args: 
            z_dim: dimension, not including the representation needed for the 
                any transform representations
            img_shape: tuple shape of a single data point. Data should be flat 
                if applicabale             
            
        """
        x=super(BaseVAE, self).__init__()
        print(x)
        self.data_transforms, self.img_shape = None, None, None
        # the number of params needed to parameterize transformations
        self.ndim_transforms = 0 
        
    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs) -> Tensor:
        pass