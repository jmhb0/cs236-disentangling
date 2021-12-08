import torch
import torch.nn as nn
from . import base_vae 
from .types_ import *
from . import model_components as mc 
import numpy as np 
import importlib
importlib.reload(base_vae)

class SpatialVAE(base_vae.BaseVAE):
    def __init__(self, 
                 img_shape,
                 img_shape_original,
                 device='cpu',
                 z_dim=4,
                 loss_kwargs: dict = dict(bs=None, N=None, beta=1, recon_key='BCE',
                     theta_prior=np.pi, dx_prior=0.1),
                 spatial_transforms=dict(rot=True, trans=True),
                 normalize_out: bool = True,
                 encoder_kwargs=dict(net_type='fc', hidden_dims=[128,128], activation='lrelu'),
                 #coord_decoder_kwargs=dict(net_type='fc', activation='lrelu'),
                 decoder_kwargs=dict(net_type='fc', hidden_dim=[128,128], activation='lrelu'),
                ):
        """
        loss_kwargs
            theta_prior & dx_prior: standard deviations of the priors (assume mean 0)
        """
        # call base class init to initialize standard things
        super(SpatialVAE, self).__init__(
                img_shape=img_shape, img_shape_original=img_shape_original,
                ndim_transforms=0, z_dim=z_dim, device=device,
                loss_kwargs=loss_kwargs, normalize_out=normalize_out,
                )
        # create the x_grid coordinates to be transformed & consumed
        self.x_grid = self.imcoordgrid(img_shape_original[1:]).to(self.device)

        # record which transforms are applied 
        self.spatial_transforms = spatial_transforms 
        self.do_transform_rot = spatial_transforms.get('rot',False)
        self.do_transform_trans = spatial_transforms.get('trans',False)
        self.ndim_trans = 0
        if self.do_transform_rot: self.ndim_trans+=1
        if self.do_transform_trans: self.ndim_trans+=2

        # create encoder network
        encoder_net_type=encoder_kwargs.get('net_type','') 
        if encoder_net_type=='fc': 
            self.encoder_net = mc.make_fc_layers(in_dim=self.img_shape[0], **encoder_kwargs) 
            hidden_dims=encoder_kwargs['hidden_dims']
            self.fc_mu = nn.Linear(hidden_dims[-1], z_dim+self.ndim_trans)
            self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim+self.ndim_trans)
        elif encoder_net_type=='cnn': raise NotImplementedError()
        else: raise ValueError()
    
        # create decoder network 
        decoder_net_type=decoder_kwargs.get('net_type','')
        if decoder_net_type=='fc':
            hidden_dims= decoder_kwargs['hidden_dims']
            self.coord_latent = CoordLatent(z_dim, hidden_dims[0], activation=decoder_kwargs.get('activation'))
            self.decoder_net = mc.make_fc_layers(in_dim=hidden_dims[0], **decoder_kwargs)
            # spatialVAE does 1 pixel at a time so out-dim is 1
            self.last_layer = nn.Linear(hidden_dims[-1], 1)
            if normalize_out:
                self.last_layer=nn.Sequential(self.last_layer, nn.Sigmoid())


        # check the input image has the right params
        if encoder_net_type=='fc': assert len(img_shape)==1
        elif encoder_net_type=='cnn': assert len(img_shape)==3

        # put the model on GPU if applicable 
        if self.device.type=='cuda': self.cuda()

    def grid2xy(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """ 
        util: given a meshgrid style output (a grid where x0 is coord set
        0 and x1 is coord set 1, flatten so that element 0 is coord (0,0), element
        1 is coord (0,1) and so on.
        The final shape will then be (n_coords, 2)
        """
        X = torch.cat((X1[None], X2[None]), 0)
        d0, d1 = X.shape[0], X.shape[1] * X.shape[2]
        X = X.reshape(d0, d1).T
        return X

    def imcoordgrid(self, im_dim: Tuple) -> torch.Tensor:
        """
        util: given image dimension, get the coordinates of the pixels, so we 
        have shape (n_pixels, 2), and the indices corresponds to the standard
        flatten() operation.
        The coordinates are in the range [-1,1] in x and y. 
        """
        xx = torch.linspace(-1, 1, im_dim[0])
        yy = torch.linspace(1, -1, im_dim[1])
        x0, x1 = torch.meshgrid(xx, yy)
        return self.grid2xy(x0, x1)

    def transform_coordinates(self, coord: torch.Tensor,
                          phi: Union[torch.Tensor, float] = 0,
                          coord_dx: Union[torch.Tensor, float] = 0,
                          ) -> torch.Tensor:
        assert coord.ndim==3, "Must expand coords, where 0th dimension is " \
                        "batch level"
        if torch.sum(phi) == 0: 
            phi = coord.new_zeros(coord.shape[0])
        phi = phi[:,0]
        rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
        rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
        rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
        coord = torch.bmm(coord, rotmat)

        if coord_dx.ndim==2: coord_dx=coord_dx.unsqueeze(1)
        if torch.sum(coord_dx)!=0:
            coord += coord_dx
        return coord

    def encode(self, x):
        x = self.encoder_net(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z, x_grid,):
        bs = len(z)
        y = self.coord_latent(x_grid, z)
        y = self.decoder_net(y)
        y = self.last_layer(y)
        y = y.view(bs,*self.img_shape)
        return y

    def sample_gaussian(self, mu, logvar):
        """ 
        Sample n times from a Gaussian with diagonal covariance matrix of 
        d-dimensionsm where we specify different parameters fro each sample.
        Input and shape is (n_sampes, d).  
        Args
            mu: the mean vectors to sample from, shape (n_samples, d)
            logvar: log of the per-element variances, whose corresponding stdDev 
            is the diagonal of the covariance matrix, shape (n_samples, d)
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def split_latent(self, z):
        bs = len(z)
        phi, dx = torch.Tensor(0), torch.Tensor(0)

        # rotation and translation
        if self.do_transform_rot and self.do_transform_trans:
            phi = z[:,[0]]
            dx = z[:,1:3]
            z = z[:,3:]
        # rotation only 
        elif self.do_transform_rot:
            phi = z[:,[0]]
            z = z[:,1:]
        # translation only 
        elif self.do_transform_trans:
            dx = z[:,:2]
            z = z[:,2:]
            pass

        assert z.shape[1]==self.z_dim
        return z, phi, dx

    def forward(self, x) -> Tensor:
        bs = x.size(0)
        mu, logvar = self.encode(x)
        z = self.sample_gaussian(mu, logvar)
        z, phi, dx = self.split_latent(z)

        # create a copy of x_grid for each sample and then transform
        x_grid = self.x_grid.expand(x.shape[0], *self.x_grid.shape)
        x_grid = self.transform_coordinates(x_grid, phi, dx)
        
        # decode  the part of z that is not transformed + the x_grid
        y = self.decode(z, x_grid,)

        return x, y, mu, logvar

    def loss_function(self, x, y, z_mu, z_logvar):
        """
        """
        # reconstructon loss
        log_p_x_g_z = -self.recon_loss_func(y,x)*y.size(1)
        
        # distribution loss 
        kl_div = 0
        z_logstd = (1/2)*z_logvar
        z_std = torch.exp(z_logstd)
        
        if self.do_transform_rot: 
            # extract theta values and remove them from 
            theta_mu, theta_logstd, theta_std = z_mu[:,0], z_logstd[:,0], z_std[:,0]
            z_mu, z_logstd, z_std = z_mu[:,1:], z_logvar[:,1:], z_std[:,1:]
            # compute kl loss    
            sigma = self.loss_kwargs['theta_prior']
            kl_div_theta = -theta_logstd + np.log(sigma) + (theta_std**2 + theta_mu**2)/2/sigma**2 - 0.5
            kl_div += kl_div_theta
        else:
            kl_div_theta=0
        
        # unit normal prior on rest of z (includes translation)
        z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
        z_kl = torch.sum(z_kl, 1)
        kl_div += z_kl
        kl_div = kl_div.mean()
        
        # the elbo is the loss 
        elbo = log_p_x_g_z - kl_div
        elbo_beta = log_p_x_g_z - kl_div*self.loss_kwargs['beta']

        #### Now we do a bunch of variable renaming 
        # we wish to max the elbo, so negate it or the loss
        loss = -elbo_beta
        loss_recon = -log_p_x_g_z
        loss_kl = kl_div
        loss_kl_z = torch.mean(z_kl)
        
                    
        return dict(loss=loss, loss_recon=loss_recon, loss_kl=loss_kl,
                    # add these other terms later if I want.
#                    kl_div_theta=kl_div_theta
                   )
    
class CoordLatent(nn.Module):
    """
    Intermediate module for the latent decoder. 
    The following is applied for each batch: 
    Each forward pass takes the latent representation, z, and a single x_coord. 
    They are each projected to h_dim size with a linear layer, and then added
    together. 
    The dimension 0 will be bs*x_coord.shape[1]
    """
    def __init__(self, latent_dim, out_dim, activation='lrelu', do_inter_activation=False):
        super(CoordLatent, self).__init__()
        self.fc_coord = nn.Linear(2, out_dim)
        self.fc_latent = nn.Linear(latent_dim, out_dim, bias=False)
        self.activation = mc.activations[activation]()

    def forward(self, x_coord, z):
        assert len(x_coord)==len(z)
        batch_dim, n = x_coord.size()[:2]
        x_coord = x_coord.reshape(batch_dim * n, -1)
        h_x = self.fc_coord(x_coord)
        h_x = h_x.reshape(batch_dim, n, -1)
        h_z = self.fc_latent(z)
        h = h_x.add(h_z.unsqueeze(1))
        h = h.reshape(batch_dim * n, -1)
        if self.activation is not None:
            h = self.activation(h)
        return h

