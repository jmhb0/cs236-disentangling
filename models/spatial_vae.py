import torch
import torch.nn as nn
import kornia
import os
os.listdir()
from . import loss_utils as lu
from . import base_vae 
from .types_ import *
from . import model_components as mc 
import numpy as np 
import importlib
import sys
import util

importlib.reload(base_vae)
importlib.reload(mc)

class SpatialVAE(base_vae.BaseVAE):
    def __init__(self, 
                 m, n,
                 device='cpu',
                 z_dim=4,
                 prior="standard",
                 prior_z="standard",
                 k_for_z_gmm=4, # only if actually learning a prior_z="gmm_learnable"
                 do_consistency=False,
                 loss_kwargs: dict = dict(bs=None, N=None, beta=1, recon_key='BCE',
                     theta_prior=np.pi, dx_prior=0.1, no_theta_mu_inference=False),
                 spatial_transforms=dict(rot=True, trans=True),
                 normalize_out: bool = True,
                 encoder_kwargs=dict(net_type='fc', hidden_dims=[128,128], activation='lrelu'),
                 #coord_decoder_kwargs=dict(net_type='fc', activation='lrelu'),
                 decoder_kwargs=dict(net_type='fc', hidden_dim=[128,128], activation='lrelu'),
                 coord_latent_kwargs=dict(znet_type='simple', xnet_type='simple', activation='relu'),
                 **kwargs
                ):
        """
        loss_kwargs
            theta_prior & dx_prior: standard deviations of the priors (assume mean 0)
        """
        # call base class init to initialize standard things
        super(SpatialVAE, self).__init__(
                ndim_transforms=0, z_dim=z_dim, device=device,
                loss_kwargs=loss_kwargs, normalize_out=normalize_out,
                )
        self.do_consistency = do_consistency 
        self.encoder_kwargs, self.decoder_kwargs, self.coord_latent_kwargs, self.loss_kwargs = \
                encoder_kwargs, decoder_kwargs, coord_latent_kwargs, loss_kwargs
        self.m, self.n = m, n
        self.prior=prior
        self.prior_z=prior_z
        if self.prior=="gmm_learnable":
            self.k=self.gmm_k
            # learnable parameters for the k gaussian mixtures
            self.theta_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, 1)    
                                        / np.sqrt(self.k * 1))                   # unsure about the normalization choice
            '''
            theta_pre_mu = torch.linspace(-np.pi, np.pi, self.k)
            theta_pre_std = torch.ones(self.k)
            self.theta_pre = torch.cat((theta_pre_mu, theta_pre_std))
            self.theta_pre = torch.nn.Parameter(self.theta_pre.unsqueeze(1).unsqueeze(0))
            '''
            # gmm weighting - prior is uniform
            self.pi = torch.nn.Parameter(torch.ones(self.k) / self.k, requires_grad=False)

        if self.prior_z=="gmm_learnable":
            self.k_for_z_gmm = k_for_z_gmm
            self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k_for_z_gmm, self.z_dim)    
                                        / np.sqrt(self.k_for_z_gmm * 1))                   # unsure about the normalization choice

        # create the x_grid coordinates to be transformed & consumed
        self.x_grid = self.imcoordgrid((m,n)).to(self.device).contiguous()

        # record which transforms are applied 
        self.spatial_transforms = spatial_transforms 
        self.do_transform_rot = spatial_transforms.get('rot',False)
        self.do_transform_trans = spatial_transforms.get('trans',False)
        self.ndim_trans = (1 if self.do_transform_rot else 0) \
                        + (2 if self.do_transform_trans else 0)

        # create encoder network
        self.encoder_net_type=encoder_kwargs.get('net_type','') 
        if self.encoder_net_type=='fc': 
            self.in_dim = m*n
            self.encoder_net = mc.make_fc_layers_no_out(in_dim=self.m*self.n, **encoder_kwargs) 
            hidden_dims=encoder_kwargs['hidden_dims']
            self.fc_mu = nn.Linear(hidden_dims[-1], z_dim+self.ndim_trans)
            self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim+self.ndim_trans)
        elif self.encoder_net_type=='cnn': 
            self.dim_last_featuremap = 2 # HARDCODE
            hidden_dims = encoder_kwargs['hidden_dims']
            self.encoder_net = mc.make_cnn_layers(**encoder_kwargs)
            self.fc_mu = nn.Linear(self.dim_last_featuremap**2*hidden_dims[-1], z_dim+self.ndim_trans) 
            self.fc_logvar = nn.Linear(self.dim_last_featuremap**2*hidden_dims[-1], z_dim+self.ndim_trans) 

        else: raise ValueError()

        coord_latent_out_dim = coord_latent_kwargs['out_dim']
        self.coord_latent = CoordLatent(z_dim=self.z_dim, **coord_latent_kwargs)

        # create decoder network 
        self.decoder_net_type=decoder_kwargs.get('net_type','')
        if self.decoder_net_type=='fc':
            hidden_dims= decoder_kwargs['hidden_dims']
            # dimension determined by what comes out of coord_latent. Out dimension is 1 always.
            self.decoder_net = mc.make_fc_layers(in_dim=coord_latent_out_dim, out_dim=1, **decoder_kwargs)
            self.last_layer = nn.Linear(1,1)
            if normalize_out:
                self.last_layer = nn.Sequential(self.last_layer, nn.Sigmoid())
        
        elif self.decoder_net_type=='cnn':
            hidden_dims= decoder_kwargs['hidden_dims']
            self.dim_first_featuremap=2 # HARDCODE
            self.decoder_in_channels = hidden_dims[0]
            coord_latent_out_nodes = self.dim_first_featuremap**2 * self.decoder_in_channels
            self.coord_latent_out_nodes = coord_latent_out_nodes
            self.coord_latent = CoordLatent(z_dim, coord_latent_out_nodes, activation=decoder_kwargs.get('activation'))
            self.decoder_net = mc.make_de_cnn_layers(**decoder_kwargs)
            self.last_layer = nn.Linear(32**2*1,1)
            if normalize_out:
                self.last_layer=nn.Sequential(self.last_layer, nn.Sigmoid())

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
        b = x.size(0)
        if self.encoder_net_type=="fc":
            x = x.view(b, -1)
        elif self.encoder_net_type=='cnn':
            x = x.view(b, 1, self.m, self.n)

        x = self.encoder_net(x)
        if self.encoder_net_type=='cnn':
            x = x.view(x.size(0),-1)

        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z, x_grid,):
        bs = len(z)
        y = self.coord_latent(x_grid, z)
        if self.decoder_net_type=='cnn':
            c, m = self.decoder_in_channels, self.dim_first_featuremap
            y = y.reshape(len(y), c, m, m).contiguous()

        y = self.decoder_net(y)
        if self.decoder_net_type=='cnn':
            y = y.reshape(len(y), -1)

        y = self.last_layer(y)
        y = y.view(bs, self.m*self.n)
        if self.decoder_net_type=='cnn': y=y.unsqueeze(1)
        return y

    def sample_gaussian(self, mu, logvar):
        """ 
        Sample n times from a Gaussian with diagonal covariance matrix of 
        d-dimensionsm where we specify different parameters for each sample.
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

    def forward(self, x, style_inference_only=False, no_consistency=False) -> Tensor:
        """
        Full reconstruct, returning the intermediate encoding variables.
        Args: 
            x: original image shape (b,-1), a batch of flattened images. 
            style_inference_only (bool): if True, then set theta and dx=0, to 
                reconstruct pose only. This should not be used during training.
            no_consistency: don't generate cosistency training stuff, even if the model's flag self.do_consistency 
            is true. This is needed for eval.
        """
        # If doing consistency training, then create a randomly-rotated image for 
        # each image. Concatenate the result. This doubles the number of samples
        # The loss function will handle this in the case that self.do_consistency=True.
        if self.do_consistency and not no_consistency:
            bs = x.size(0)
            rotations = (torch.rand(bs)*360).to(x.device) # random rotation
            x = x.view(bs,1,self.m,self.n)
            x_rotated = kornia.geometry.transform.rotate(x, rotations, mode='bilinear').to(x.device)
            x = torch.cat((x, x_rotated))
            x = x.view(x.size(0), -1)

        bs = x.size(0)
        mu, logvar = self.encode(x)
        z = self.sample_gaussian(mu, logvar)
        z, phi, dx = self.split_latent(z)

        # create a copy of x_grid for each sample and then transform
        x_grid = self.x_grid.expand(x.shape[0], *self.x_grid.shape).contiguous()
        # transofrm coordinates unless we want to reconstruct the style vecotr only
        if not style_inference_only:
            x_grid = self.transform_coordinates(x_grid, phi, dx)
        
        # decode  the part of z that is not transformed + the x_grid
        y = self.decode(z, x_grid,)

        return x, y, mu, logvar

    def loss_kl_theta(self, *args):
        """ 
        Compute divergence loss for the angle, depending on the prior that 
        was chosen with model initialization. 
        Returns: KL loss along the batch, shape (bs,)
        """
        if self.prior=="standard":
            return self.loss_kl_theta_normal_0_1(*args)
        if self.prior=="standard_mc":
            return self.loss_kl_theta_normal_0_1_monte_carlo(*args)
        if self.prior=="gmm_learnable":
            return self.loss_kl_theta_gmm_learnable(*args)
        else: 
            raise NotImplementedError(f"Prior f{self.prior} not implemented")
        pass

    def loss_kl_theta_normal_0_1(self, x, y, theta_mu, theta_logstd, theta_std):
        sigma = self.loss_kwargs['theta_prior']
        if not self.loss_kwargs['no_theta_mu_inference']:
            kl_div_theta = -theta_logstd + np.log(sigma) + (theta_std**2 + theta_mu**2)/2/sigma**2 - 0.5
        else:
            kl_div_theta = -theta_logstd + np.log(sigma) + (theta_std**2               )/2/sigma**2 - 0.5
        return kl_div_theta

    def loss_kl_theta_normal_0_1_monte_carlo(self, x, y, theta_mu, theta_logstd, theta_std):
        """
        Bc theta_mu etc are coming in with loss like
        """
        theta_mu, theta_logstd, theta_std = theta_mu[:,None], theta_logstd[:,None], theta_std[:,None]
        theta_var = theta_std**2
        theta_logvar = torch.log(theta_var)

        # sample
        theta_sample = self.sample_gaussian(theta_mu, theta_logvar)

        # compute q(z|x)
        log_q_z_x = lu.log_normal(theta_sample, theta_mu, theta_var)

        # compute p(z)
        m_0, v_1 = torch.Tensor([0]).to(self.device), torch.Tensor([1]).to(self.device)
        log_p_z = lu.log_normal(theta_sample, m_0, v_1, device=self.device)
        
        # estimator 
        kl_div_theta = (log_q_z_x-log_p_z)

        return kl_div_theta

    def loss_kl_theta_gmm_learnable(self, x, y, theta_mu, theta_logstd, theta_std):
        """
        Let the prior for the angle be a GMM with learable parameters
        """
        bs = theta_mu.size(0)
        # getting the z samples is the same
        theta_mu, theta_logstd, theta_std = theta_mu[:,None], theta_logstd[:,None], theta_std[:,None]
        theta_var = theta_std**2
        theta_logvar = torch.log(theta_var)

        # sample 
        theta_sample = self.sample_gaussian(theta_mu, theta_logvar)

        # log jhkq(z|x) is the same as the last version (since we have the same assumptions on the encoder)
        log_q_z_x = lu.log_normal(theta_sample, theta_mu, theta_var)

        # log p(z)
        prior_theta_mus, prior_theta_logvars = self.theta_pre[:,:self.k], self.theta_pre[:,self.k:]
        prior_theta_vars = torch.exp(prior_theta_logvars) ### alternative is to use softplus + 1e-5
        prior_theta_mus = lu.duplicate(prior_theta_mus, bs)
        prior_theta_vars = lu.duplicate(prior_theta_vars, bs)

        log_p_z = lu.log_normal_mixture(theta_sample, prior_theta_mus, prior_theta_vars)
        
        kl_div_theta = (log_q_z_x-log_p_z)
        return kl_div_theta        

    def loss_kl_z(self,*args):
        if self.prior_z=="standard":
            return self.loss_kl_z_normal_0_1(*args)
        elif self.prior_z=="gmm_learnable":
            return self.loss_kl_z_gmm(*args)

    def loss_kl_z_normal_0_1(self, x, y, z_mu, z_logstd, z_std):
        z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
        z_kl = torch.sum(z_kl, 1)  # shape 128
        return z_kl

    def loss_kl_z_gmm(self, x, y, z_mu, z_logstd, z_std):
        bs = z_mu.size(0)
        z_logvar = 2*z_logstd
        z_var = torch.exp(z_logvar)
        z_sample = self.sample_gaussian(z_mu, z_logvar)

        # log q(z|x) is the same as the last version (since we have the same assumptions on the encoder)
        log_q_z_x = lu.log_normal(z_sample, z_mu, z_var)

        # get and reshape the prior. Duplicate it for each element of the batch.
        prior_z_mus, prior_z_logvars = self.z_pre[:,:self.k_for_z_gmm], self.z_pre[:,self.k_for_z_gmm:]
        prior_z_vars = torch.exp(prior_z_logvars)
        prior_z_mus = lu.duplicate(prior_z_mus, bs)
        prior_z_vars = lu.duplicate(prior_z_vars, bs)

        # compute the divergence 
        log_p_z = lu.log_normal_mixture(z_sample, prior_z_mus, prior_z_vars)
        
        z_kl = (log_q_z_x-log_p_z)
        return z_kl 
        
    def loss_function(self, x, y, z_mu, z_logvar, epoch):
        """
        All losses kept with shape (bs,) then added, then meaned at the end
        """
        if self.decoder_net_type=='cnn':
            x, y = x.view(len(x),-1), y.view(len(y), -1)

        if self.do_consistency:
            b = int(x.size(0)/2)
            # split all vars into regular and rotated components (so that the regular loss is only computed on the original input)
            x, x_rotated = torch.split(x, b, 0)
            y, y_rotated = torch.split(y, b, 0)
            z_mu, z_mu_rotated = torch.split(z_mu, b, 0)
            z_logvar, z_logvar_rotated = torch.split(z_logvar, b, 0)

            z, z_rotated = z_mu[:, self.ndim_trans:], z_mu_rotated[:, self.ndim_trans:]
            normalizer_std = torch.linalg.norm(torch.std(z, dim=0), ord=2)
            dist = torch.linalg.norm(z-z_rotated, ord=2, dim=1)
            loss_consistency = dist / normalizer_std # shape (bs,)
            loss_consistency = loss_consistency.mean(0) # shape (0,), mean over batches
        else: 
            loss_consistency = 0

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
            # compute kl loss from funcs (depending on self.prior variable)
            # has shape (bs,)
            kl_div_theta = self.loss_kl_theta(x, y, theta_mu, theta_logstd, theta_std)
            kl_div_theta = kl_div_theta
            kl_div += kl_div_theta
        else:
            kl_div_theta=0

        if self.do_transform_trans: 
            z_mu, z_logstd, z_std, z_logstd = z_mu[:,2:], z_logvar[:,2:], z_std[:,2:], z_logstd[:,2:]
            # todo. Compute KL divergence if implemented
        else:
            pass
        
        # unit normal prior on rest of z (includes translation)
        z_kl = self.loss_kl_z(x, y, z_mu, z_logstd, z_std)
        # add the z_kl to the existing kl divergence
        kl_div += z_kl
        kl_div = kl_div.mean() # shape (0,)
        
        # the elbo is the loss 
        elbo = log_p_x_g_z - kl_div

        # load and update consistency weight
        consistecy_weight = self.loss_kwargs['consistency_weight'] if self.do_consistency else 0

        epoch_start_decreasing=10
        decay_factor=1
        if epoch>=epoch_start_decreasing:
            consistecy_weight= consistecy_weight* decay_factor**(epoch-epoch_start_decreasing+1)
        elbo_beta = log_p_x_g_z - kl_div*self.loss_kwargs['beta'] - loss_consistency*consistecy_weight

        #### Now we do a bunch of variable renaming (more for back-compatibility reasons)
        # we wish to max the elbo, so negate it or the loss
        loss = -elbo_beta
        loss_recon = -log_p_x_g_z
        loss_kl = kl_div
        loss_kl_z = torch.mean(z_kl)
     
        kl_div_theta = torch.mean(kl_div_theta)
        
        return dict(loss=loss, loss_recon=loss_recon, loss_kl=loss_kl, elbo=elbo,
                loss_kl_theta=kl_div_theta, loss_kl_z=loss_kl_z, beta=self.loss_kwargs['beta'],
                consistency_weight=consistecy_weight,
                loss_consistency=loss_consistency,
                    # add these other terms later if I want.
#                    kl_div_theta=kl_div_theta
                   )
    def to(self, device):
        """ 
        Overwrite original self. Calls the original to() method which puts 
        parameters on device, but also puts x_grid on device.
        """
        super().to(device)
        self.x_grid = self.x_grid.to(device).contiguous()
        return self
    
class CoordLatent(nn.Module):
    """
    Intermediate module for the latent decoder. 
    Define a network for z->h_z (style vector) and x_coord->h_x (specifying a 
    single coordinate) and combine them to a hidden representation h, that is 
    passed to the rest of the generator. 

    Args: 
        out_dim

    """
    def __init__(self, z_dim, out_dim=128, **coord_latent_kwargs):
        super(CoordLatent, self).__init__()
        self.combine_z_x = coord_latent_kwargs.get("combine_z_x", "add")
        znet_type = coord_latent_kwargs.get("znet_type", "simple")
        xnet_type = coord_latent_kwargs.get("xnet_type", "simple")
        self.znet_type, self.xnet_type = znet_type, xnet_type
        self.softmax_xvec = coord_latent_kwargs.get("softmax_xvec", "False")

        self.z_dim=z_dim
        if self.combine_z_x=='concat':
            out_dim = outdim//2
        # build x_net
        if xnet_type=="simple":
            self.xnet = nn.Linear(2, out_dim)
        elif xnet_type=="fc":
            x_hidden_dims = coord_latent_kwargs.get("x_hidden_dims", [128])
            self.xnet = mc.make_fc_layers(2, hidden_dims=x_hidden_dims, out_dim=out_dim)
        else: 
            raise ValueError()
        #if self.softmax_xvec:
        #    self.xnet = nn.Sequential(self.xnet, nn.Softmax(dim=1))


        if znet_type=="simple": 
            self.znet = nn.Linear(z_dim, out_dim)
        elif znet_type=="fc":
            z_hidden_dims = coord_latent_kwargs.get("z_hidden_dims", [128,128])
            activation = coord_latent_kwargs.get("activation", "relu")
            self.znet = mc.make_fc_layers(in_dim=z_dim, out_dim=out_dim,
                    hidden_dims=z_hidden_dims, activation=activation)
        elif znet_type=='cnn':
            z_hidden_dims = coord_latent_kwargs.get("z_hidden_dims", [128,64,32,   16])
            self.z_hidden_dims = z_hidden_dims
            self.znet_num_cnn_layers = len(z_hidden_dims)
            activation = coord_latent_kwargs.get("activation", "relu")
            do_norm_layer = coord_latent_kwargs.get("do_norm_layer", True)

            # layer to send the z dim to vector size that is ready to be reshaped 
            # into a featuremap
            featmap_in_dim = 2*2*z_hidden_dims[0]
            self.znet_z_to_featmap = mc.make_fc_layers(in_dim=self.z_dim, out_dim=featmap_in_dim, hidden_dims=[]) 

            # the cnn layers 
            cnn_out_dims=1 # HARDCODE
            self.znet_cnn_layers = mc.make_de_cnn_layers(out_dims=cnn_out_dims, hidden_dims=z_hidden_dims, 
                    activation=activation)
            # layer putting the cnn output to the same dim as h_x (the arg `out_dim`)
            featmap_out_dim = (2**(len(z_hidden_dims)+1))**2*cnn_out_dims
            self.znet_last_layer = mc.make_fc_layers(featmap_out_dim, out_dim=out_dim, hidden_dims=[])
        else: raise ValueError()
        
        activate_out = coord_latent_kwargs.get("activate_out","relu")
        self.activate_out_layer = mc.activations[activate_out]()

    
    def decode_z(self, z):
        """ the z only part, which is conditionl on the type of znet"""
        if self.znet_type in ("simple", "fc"):
            return self.znet(z)
        elif self.znet_type == 'cnn':
            b, zdim = z.shape
            z = self.znet_z_to_featmap(z)
            z = z.view(b, self.z_hidden_dims[0], 2, 2)
            z = self.znet_cnn_layers(z)
            z = z.view(b, -1)
            z = self.znet_last_layer(z)
            return z

        else:
            raise ValueError()
    def decode_x(self, x_coord):
        h_x = self.xnet(x_coord)
        assert h_x.ndim==2
        if self.softmax_xvec:
            h_x = torch.softmax(h_x, dim=1)
        return h_x

    def forward(self, x_coord, z):
        assert len(x_coord)==len(z)
        batch_dim, n = x_coord.size()[:2]

        # Put each x_coord into its own batch and pass each through its own network
        # Then shape back to (b,n,-1)
        x_coord = x_coord.view(batch_dim * n, 2)
        h_x = self.decode_x(x_coord)
        h_x = h_x.view(batch_dim, n, -1)

        # Put the style vector through its network
        h_z = self.decode_z(z)

        # Combine x and z representations. We unsqueeze h_z, so the same h_z 
        # representation is used within a given batch
        if self.combine_z_x=='add':
            h = h_x.add(h_z.unsqueeze(1))
        elif self.combine_z_x=='multiply':
            h = h_x.multiply(h_z.unsqueeze(1))

        # Put each x_coord back into its own batch before leaving this module
        h = h.view(batch_dim*n, -1)
        h = self.activate_out_layer(h)

        return h


