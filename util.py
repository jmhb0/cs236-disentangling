import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import time 
from pathlib import Path
from sklearn.manifold import TSNE

def print_mem_gbs():
    al = torch.cuda.memory_allocated()/1024**3
    cac = torch.cuda.memory_reserved()/1024**3
    print(f"####### Mem stats: {al:.6f}", f"{cac:.6f}")

def test_dataloader_time(loader, epochs=5, verbose=True, load_device=None, max_iters=None):
    """
    Test the dataloader time.
    We iterate over the dataloader, save the data to a variable and send to device. 

    Do this `epoch` times, and average the time over all epochs.
    If load_device is not None, then we also load each batch to device.
    """
    print(f"Initial devices: data {next(iter(loader))[0].device}, label {next(iter(loader))[1].device}")
    if load_device is None: print("No device loading" )
    else: print(f"Loading data and labels at each iteration to {load_device}")
    print(f"Loader num_workers {loader.num_workers}")
    iters_per_run = max_iters if max_iters is not None else len(loader)
    print(f"Loading {iters_per_run} batches. Dataloader has {len(loader)} batches")
    print("\n")

    last = time.time()
    duration_lst=[]
    for _ in range(epochs):
        i=0
        while 1:
            if max_iters is not None and max_iters<i: break
            i+=1
            for i, (data, label) in enumerate(loader): 
                if max_iters is not None and max_iters<i: break
                if load_device: 
                    data, label = data[0].to(load_device), label[0].to(load_device)

        duration = time.time()-last
        if verbose: print(duration)
        duration_lst.append(duration)
        last = time.time()

    mean_duration = sum(duration_lst)/len(duration_lst)
    if verbose:
        print(f"mean duration {mean_duration}")
    return mean_duration, duration_lst

def create_logging_dir(dir_project, msg='', verbose=0, flush_sec=10):
    ts_for_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ts_for_dir = datetime.datetime.now().strftime("%Y%m%d") # day, not minutes
    dir_log = f"{dir_project}/tb_logs/{ts_for_dir}/{ts_for_file}_{msg}"
    Path(dir_log).mkdir(parents=True, exist_ok=True)
    if verbose: print(f"Tensorboard logdir \n{dir_log}")
    writer = SummaryWriter(dir_log, flush_secs=flush_sec)
    return writer, dir_log

def display_grid_of_samples(dataset, nrow=10, figsize=(9,9)):
    """ 
    Plot nrow**2 number of samles in a grid 
    """
    x = dataset[:nrow**2][0].cpu()
    assert len(dataset.img_shape_original)==3
    x = x.view(len(x), *dataset.img_shape_original)
    print(x.shape)
    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=0)
    f, axs = plt.subplots(figsize=figsize)
    axs.imshow(grid[0], cmap='gray')
    plt.close()
    return f

def reconstruct(model, y, style_inference_only=False, device='cpu'):
    """ 
    Reimplementation of reconstruct (from model.forward) but with the option 
    to do style inference only. This was created to work on saved models that
    were saved before style_inference_only flag was available. 
    """
    model.to(device)
    #if model.decoder_net_type=='cnn':
    #    y = y.reshape(len(y),1,*model.img_shape).contiguous()
    mu, logvar = model.encode(y.to(device))
    z = mu # instead of sampling
    z, phi, dx = model.split_latent(z)
    x_grid = model.x_grid.expand(y.shape[0], *model.x_grid.shape)
    if not style_inference_only:
        x_grid = model.transform_coordinates(x_grid, phi, dx)
    # the model decoder does not enforce contiguity as a space-saver
    y_hat = model.decode(z.contiguous(), x_grid.contiguous(),)
    return y, y_hat

def plot_reconstructions(model, dataset, nrows=10, figsize=(9,9), device='cpu', 
        close_figure=True):
    nsamples = (nrows**2)//2
    model.to(device)
    x = dataset[:nsamples][0].to(device)
    x, y, _, _ = model(x, no_consistency=True)
    x, y = x.to('cpu'), y.to('cpu')
    x = x.view(nsamples, *dataset.img_shape_original)
    y = y.view(nsamples, *dataset.img_shape_original)
    # interleave the samples
    f = plot_pairs_of_images_grid(x,y, close_figure=close_figure)
    return f

def make_grid_from_columns(col1, col2, col3, m=72,n=72):
    """
    Given 3 tensors of images with shape (b,m*n), make a grid arranged so that
    col1 is 1st column, etc. Involves reshaping the flattened array to mXn.
    Suitable for plotting imshow.
    TODO: should be pretty easy to generalize this to arbitrary column number.
    """
    n_theta = len(col1)
    grid = torch.zeros(3*n_theta, 1, m, n)
    grid[0::3,0] = col1
    grid[1::3,0] = col2
    grid[2::3,0] = col3
    grid = torchvision.utils.make_grid(grid, nrow=3, pad=1)[0]
    return grid

def plot_pairs_of_images_grid(x, y, nrows=10, figsize=(9,9), close_figure=True):
    """
    Given two lists of images x,y, interleave them in a grid so that x[i] is 
    next to y[i]. For example x could be original image, and y could be recon.
    """
    nsamples = (nrows**2)//2
    x, y = x[:nsamples], y[:nsamples]
    grid = torch.empty(nsamples*2, *x.shape[1:])
    grid[0::2]=x
    grid[1::2]=y
    if grid.ndim==3:
        grid = grid.unsqueeze(1)
    grid = torchvision.utils.make_grid(grid, nrow=nrows, padding=2, pad_value=1)
    f, axs=plt.subplots(figsize=figsize)
    axs.set_axis_off()
    axs.imshow(grid[0], cmap='gray')
    if close_figure: plt.close()
    return f

def plot_latent_2d_spatialVAE(model, dataset, nrows=6, figsize=(9,9), std_range=1.5):
    """
    Plot two standard devations wrt the embedded test set.
    """
    assert model.z_dim == 2
    # first find test set standard deviation
    n_sample=500
    x = dataset[:n_sample][0].to(model.device)

    _, _, mu, logvar = model(x)
    mu, _, _ = model.split_latent(mu)
    mu = mu.to('cpu')
    std_of_mu = torch.std(mu, axis=0).detach().numpy()

    # generate the z values in the right range
    xrange = torch.linspace(-std_range*std_of_mu[0], +std_range*std_of_mu[0], nrows,)
    yrange = torch.linspace(-std_range*std_of_mu[1], +std_range*std_of_mu[1], nrows)
    from itertools import product
    z = torch.tensor(list(product(xrange, yrange))).to(model.device)

    # get the x-grid, that will be untransformed
    bs = len(z)
    x_grid = model.x_grid.to(model.device)
    x_grid = x_grid.expand(bs, *x_grid.shape)

    # run the generator for simulated images, put on the grid and then plit
    y = model.decode(z.contiguous(), x_grid.contiguous())
    y = y.reshape(len(y), *model.img_shape_original).to('cpu').detach()
    y_grid = torchvision.utils.make_grid(y, nrow=nrows)
    f, axs = plt.subplots(figsize=figsize)
    axs.imshow(y_grid[0], cmap='gray')
    plt.close()
    return f

def check_rotations_imgs(model, x, rotations=[0,-30,30,-60,60,-90,90,-120,120,-150,150,180], figsize=(10,10)):
    """
    Given a single image x, apply a set of rotations, and then reconstruct
    them.
    Also compute the distance in z.
    """
    rotations=[0,-30,30,-60,60,-90,90,-120,120,-150,150,180]
    assert len(x.shape)==3
    n = len(rotations)
    x_rotated = torch.zeros(n, *x.shape)
    for i in range(n):
        x_rotated[i] = torch.Tensor(transform.rotate(x[0], rotations[i]))

    # do reconstruction and put it in a grid
    y_no_rotation, y_rotation = reconstruct_batch(model, x_rotated)
    y_no_rotation, y_rotation = y_no_rotation.unsqueeze(1).cpu().detach(), y_rotation.unsqueeze(1).cpu().detach()
    img_grid = make_grid_from_n_img_lists(x_rotated, y_rotation, y_no_rotation)[0]

    # compute embedding distances from the first image
    dist_embedding = get_embedding_distances(model, x.unsqueeze(0), x_rotated)

    # make axis labels for embedding distanfes
    l = img_grid.shape[0]
    dx = l / len(y_no_rotation)
    yticks = np.arange(0.5*dx, l, dx)
    f, axs = plt.subplots(figsize=figsize)
    axs.imshow(img_grid, cmap='gray')
    axs.set_yticks(yticks);
    yticklabels = [f"{d:.3f}" for d in dist_embedding]
    axs.set_yticklabels(yticklabels)
    axs.xaxis.set_major_locator(plt.NullLocator())
    plt.close()

    return dict(f=f, img_grid=img_grid, x_rotated=x_rotated, dist_embedding=dist_embedding,
                yticklabels=yticklabels, yticks=yticks)

def do_batch_of_rotations(y, m=None, n=None, thetas=[0,30,60,90,120,150,180,210,240,270,300,330]):
    """
    Given a batch of images, and list of rotations, return a batch where each item is a
    batch of images that have done rotation by the angle specified in list thetas.
    Args:
        y: eiteher shape (b,-1) or (b,y_pix,x_pix). A batch of images
        m, n: if y.shape==2, then m,n must be supplied, and corresond to the image shape
    Returns:
        y_rotated: shape (b,n_theta,m*n) where n_theta is the length of the
            variable thetas.
    """
    b = len(y)
    if y.ndim==2:
        assert m is not None and n is not None, "m and n must be supplied if y is 2-dim"
        assert np.product(y.shape)==b*m*n, "m and n don't match dimensions, .view() will fail"
        assert y.ndim==2
    #else: assert y.ndim==3
    y = y.view(b,m,n)
    n_thetas = len(thetas)
    y_rotated = torch.zeros(b, n_thetas, m, n)

    # iterate over angles and rotate the batch 1-by-1
    for i, theta in enumerate(thetas):
        f_rotation = torchvision.transforms.RandomRotation([theta,theta])
        y_rotated[:,i] = f_rotation(y)
    # flatten the images again which is the standard data format for this model
    y_rotated = y_rotated.view(b,n_thetas, -1)
    return y_rotated

def get_distance_matrices(y_rotated, model, device='cuda'):
    """
    Given input of shape (b, n_theta, -1), get a batch of distance matrices
    in the input space
    """
    rotate, translate = model.do_transform_rot, model.do_transform_trans
    y_rotated=y_rotated.to(device)
    model.to(device)
    assert y_rotated.ndim==3
    b, n_theta = y_rotated.shape[:2]
    y_rotated = y_rotated.view(b*n_theta,-1)
    if model.decoder_net_type=='cnn':
        y_rotated = y_rotated.reshape(b*n_theta,1,*model.img_shape).contiguous()
    z_mu, z_logstd = model.encode(y_rotated)
    z = z_mu
    if rotate:
        theta = z[:,0]
        z = z[:,1:]
    if translate:
        dx = z[:,1:3]
        z = z[:,2:]
    # put the remaining z back into batches of rotations
    z = z.view(b, n_theta, z.size(1))
    z_dim = z.size(1)
    dist_matrix = torch.zeros(b, z_dim, z_dim)
    for i in range(b):
        dist_matrix[i] = torch.cdist(z[i], z[i]).detach()
    return dist_matrix

def infer_theta_rotation_tensor(y_rotated, model, device='cuda'):
    """
    For the special case of an input tensor with shape (b,n_thetas,-1),
    infer rotation with `model.encode`, and return it with shape (b,n_thetas,1).
    """
    model.to(device)
    b, n_theta = y_rotated.shape[:2]
    y_rotated = y_rotated.view(b*n_theta,-1).to(device)
    # inference
    if model.decoder_net_type=='cnn':
        y_rotated = y_rotated.reshape(b*n_theta,1,*model.img_shape).contiguous()
    z_mu, z_logstd = model.encode(y_rotated)
    rot_mu, rot_std = z_mu[:,0], torch.exp(z_logstd[:,0])
    # shape back
    rot_mu, rot_std = rot_mu.view(b, n_theta, 1), rot_std.view(b, n_theta, 1)
    return rot_mu, rot_std

def rotation_tests_batch(y, model,
        thetas=[0,30,60,90,120,150,180,210,240,270,300,330],
        distance_normalizer=1, show_dist_matrix=False,
        nrows=3, ncols=3, figsize=(20,20), device='cuda'):
    """
    Take a batch of images and a list of rotations, and produce a set of plots.
    The plots are aranged on a grid with `nrows` rows and `nrows` cols.

    Each plot is a grid where the rows are the original rotations, column
    1 is the original image, column 2 is the reconstruction, and column 3 is the
    reconstruction with rotation angle set to 0 - the style only.
    The y tick labels are the embedding distance (by q_net) from the original image,
    normalized by `distance_normalizer`, which could be, for example the output
    of the function get_stds_embedding.
    Side-effect: puts the model and data on cpu. Puts the model in eval().
    Args
        y: shape (b, m*n)
    Returns:
    """
    """
    if model.do_consistency: 
        b = y.size(0)//2
        y, _ = torch.split(y,2,0)
    """
    try: m, n = model.img_shape_original[-2:]
    except: m, n = model.m, model.n
    model.to(device)
    y = y.to(device)
    b, n_theta = nrows*ncols, len(thetas)
    assert b<=len(y)
    y = y[:b]
    # get the set of rotated images.
    y_rotated = do_batch_of_rotations(y, m, n, thetas=thetas)
    # get distance matrix. Let `dists` be the distance of each thing from index 0
    dist_matrices = get_distance_matrices(y_rotated, model, device=device)
    dist_matrices = dist_matrices / distance_normalizer
    dists = dist_matrices[:,0,:]
    # get the inferred rotation parameters
    if model.decoder_net_type=='cnn':
        y_rotated = y_rotated.reshape(b*n_theta,1,*model.img_shape).contiguous()
    rot_mu, rot_std = infer_theta_rotation_tensor(y_rotated, model, device=device)

    # do reconstructions. One set with and one without pose inference
    y_rotated = y_rotated.view(b*n_theta, -1)
    y_rotated, y_rotated_hat = reconstruct(model, y_rotated, device=device)
    y_rotated, y_rotated_hat_style = reconstruct(model, y_rotated, device=device, style_inference_only=True)

    # put the images to their display dimensions and put to cpu for plotting
    y_rotated = y_rotated.view(b,n_theta,m,n).cpu()
    y_rotated_hat = y_rotated_hat.view(b,n_theta,m,n).cpu()
    y_rotated_hat_style = y_rotated_hat_style.view(b,n_theta,m,n).cpu()

    if show_dist_matrix: ncols_plot=ncols*2
    else: ncols_plot=ncols
    f, axs = plt.subplots(nrows, ncols_plot, figsize=figsize)

    def add_dist_and_rots_yticks(axs, grid, dists, rot_mu, rot_std):
        # set up a tick for the midpoint of each image
        l = grid.shape[0]
        dx = l / n_theta
        yticks = np.arange(0.5*dx, l, dx)
        # axis labels for distances
        yticklabels = [f"{d:.3f}" for d in dists]
        axs.set_yticks(yticks)
        axs.set_yticklabels(yticklabels)
        # axis labels for rotations
        rot_mu = rot_mu *180/np.pi
        #rot_mu = rot_mu - rot_mu[0,0] # set first rotation to 0
        rot_std = rot_std *180/np.pi
        yticklabels = [f"mu:{rot_mu[i,0]:.0f}, std:{rot_std[i,0]:.2f}" for i in range(len(rot_mu))]

        secax_y = axs.secondary_yaxis('right')
        secax_y.set_yticks(yticks)
        secax_y.set_yticklabels(yticklabels)

        axs.xaxis.set_major_locator(plt.NullLocator())

    # iterate over the batches and make grids
    for i in range(nrows):
        for j in range(ncols):
            col_mult = 2 if show_dist_matrix else 1
            k = i*ncols+j   # data index
            grid = make_grid_from_columns(y_rotated[k], y_rotated_hat[k], y_rotated_hat_style[k], m=m, n=n)
            axs[i,j*col_mult].imshow(grid, cmap='gray')
            if show_dist_matrix:
                # plot distance matrix
                add_dist_and_rots_yticks(axs[i,j*2], grid, dists[k], rot_mu[k], rot_std[k])
                heatmap = axs[i,j*2+1].imshow(dist_matrices[k].detach(), cmap='gray')
                f.colorbar(heatmap, ax=axs[i,j*2+1], orientation='vertical', shrink=0.45) # the shrink part is a hack
    plt.close()
    return f

def get_stds_embedding(y, model, device='cuda'):
    """
    Get the standard deviation of the embedded data in the latent space, both
    elemntwise, and as the L2 value of that vector.
    Args:
        y: input dataset
    """
    model.to(device)
    y = y.to(device)
    assert len(y)>99, "need more samples to reasonably estimate embedding standard deviation"
    rotate, translate = model.do_transform_rot, model.do_transform_trans
    zdim_skip = (1 if rotate else 0)+(2 if translate else 0)
    z_mu, _ = model.encode(y)
    z_mu = z_mu[:,zdim_skip:]
    stds = torch.std(z_mu, axis=0)
    stds_l2 = torch.linalg.norm(stds, ord=2)
    return stds, stds_l2.item()

def interpolate_2_imgs(model, y1, y2, n_steps=9, device='cpu'):
    """
    Given 2 images y1 and y2, interpolate between those two images in the 
    latent space 
    """
    rotate, translate = model.do_transform_rot, model.do_transform_trans
    m, n = model.img_shape_original[-2:]
    zskip_dims = (1 if rotate else 0) + (2 if translate else 0)
    y1, y2 = y1.to(device), y2.to(device)
    model.to(device)

    z1, _ = model.encode(y1.cuda().view(1,-1))
    z2, _ = model.encode(y2.cuda().view(1,-1))
    z1, z2 = z1[:,zskip_dims:], z2[:,zskip_dims:]
    z_fractions = torch.linspace(0,1,n_steps+2)[1:-1]

    # create the interpolated set of z's
    z_fractions = torch.linspace(0,1,n_steps+2)[1:-1]
    z_samples = torch.cat([z1+(z2-z1)*z_frac for z_frac in z_fractions]).to(device)

    # generate the non-rotated or translated grid
    x = model.imcoordgrid((m,n))
    x = x.expand(n_steps, *x.shape[-2:]).contiguous().to(device)
    # generate the new samples
    y_hat = model.decode(z_samples, x).view(n_steps,m,n).to(device)
    y1, y2 = y1.view(1,m,n), y2.view(1,m,n)
    y_interpolation_grid = torch.vstack((y1, y_hat, y2)).detach()

    return y_interpolation_grid.unsqueeze(1)

def interpolate_2_imgs_plot(model, y1, y2, n_steps=9, figsize=(12,12), device='cpu'):
    """
    Wrapper around interpolate_2_imgs that also produces the figure.
    """
    grid = interpolate_2_imgs(model, y1, y2, n_steps=9, device='cuda')
    grid = torchvision.utils.make_grid(grid, nrow=len(grid))[0].cpu()
    f, axs=plt.subplots(figsize=(12,12))
    plt.imshow(grid,cmap='gray')
    plt.close()
    return f

def save_model_checkpoint(model, DIR_tb_log, epoch, verbose=1):
    save_dir = Path(f"{DIR_tb_log}/saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"model_epoch_{epoch}.sav"
    if verbose: print(f"Saving model checkpoint {fname}")
    torch.save(model, fname)

def plot_img_cnn_coord_layer_out(model, y, bs=10, m=32,n=32, figsize_mult=2, 
        device='cuda', sigmoid_out=False):
    y=y.to(device)
    assert bs>=len(y), "dataset too smal for chosen batch size)"
    y = y[:bs]
    model.to(device)
    figsize = np.array((3,bs)) * figsize_mult
    z, _ = model.encode(y)
    drop_zdim = 3

    # remove the layer things
    drop_zdim = (1 if model.spatial_transforms['rot'] else 0) \
          + (2 if model.spatial_transforms['trans'] else 0)
    z = z[:,drop_zdim:]
    # embedding
    z = model.coord_latent.znet_z_to_featmap(z)
    # get the post-cnn layers
    first_cnn_channel = model.coord_latent_kwargs['z_hidden_dims'][0]
    z_in_featmap = z.view(bs, first_cnn_channel, 2,2)
    z_post_cnn_layers = model.coord_latent.znet_cnn_layers(z_in_featmap).detach().cpu()
    if sigmoid_out: 
        z_post_cnn_layers = torch.sigmoid(z_post_cnn_layers)
    # do standard reconstruction
    y, y_hat = reconstruct(model, y, device='cuda')
    y, y_hat = y.cpu(), y_hat.cpu().detach()

    f, axs = plt.subplots(bs,3, figsize=figsize)
    for i in range(bs):
        axs[i,0].imshow(y[i].view(m,n), cmap='gray')
        axs[i,1].imshow(y_hat[i].view(m,n), cmap='gray')
        axs[i,2].imshow(z_post_cnn_layers[i].view(m,n), cmap='gray')
        for ax in axs[i]: ax.set_axis_off()
    plt.close()
    return f

def print_hx_mappings_side_by_side(model, coords_list = [[0,0], [0.5,0.5], [-0.5,-0.5], [1,1], [0.7,-0.3]],
                                   device='cuda', do_softmax=False):
    """
    Run the xnet to h_x and see what it produces for a range of coordinates. Print them side-by side, so
    somethign like
        2.7571	0.0000	2.9505	0.0000	0.0000
        0.0000	0.0000	0.0000	0.0000	0.0000
        0.0000	0.0000	0.0000	0.0000	0.0000
        0.0000	0.0000	0.0000	0.0000	0.0000
        0.9916	0.2839	0.3468	0.0000	0.0000
    """
    model.to(device)
    x_coords = [torch.Tensor(x).to(device) for x in coords_list]
    h_x = [model.coord_latent.xnet(x_coord).detach() for x_coord in x_coords]
    if do_softmax:
        h_x = [torch.softmax(x, dim=0) for x in h_x]
    print_str = f""
    for i in range(len(h_x[0])):
        line = ""
        for j in range(len(coords_list)):
            line += f"{h_x[j][i].item():.4f}\t"
        print_str+=line
        print_str+=f"\n"
    return print_str

def dump_dict_to_yaml(d, fname):
    import yaml
    with open(fname, 'w') as f:
        yaml.dump(d, f, sort_keys=False)
def read_yaml_to_dict(fname):
    import yaml
    with open(fname, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d

transform_lookup={
    "rotate-180" : transforms.RandomRotation(180),
    "horizontal-flip-0.5" : transforms.RandomHorizontalFlip(0.5)
}
def get_transforms_from_config(params):
    """ For reading in the model, convert a string of transforms to actual transforms """
    def get_transforms_from_list(transform_str_lst):
        transform_lst = []
        for t in transform_str_lst:
            if t not in transform_lookup.keys(): 
                raise ValueError(f"Transform key must be one of: {transform_lookup.keys()}")
            transform_lst.append(transform_lookup[t])

        return transforms.Compose(transform_lst)
    
    train_lst = params['data_params']['dataset_params']['train_transforms']
    test_lst =  params['data_params']['dataset_params']['test_transforms']
    return get_transforms_from_list(train_lst), get_transforms_from_list(test_lst)

def manifold_2d_spatial_vae(model, dataset, nrows=12, stds_range=1.5, 
        device='cuda', return_domain=False):
    """
    For 2d latent spaces, return a grid that can be plotted directly with
    imshow(). The range of the space is computed by: (i) embedding `dataset` into
    z and comuting that std dev. Taking a linspace() sampling over by 1.5X std's
    (if param stds_range=1.5).
    """
    bs=10000
    assert model.z_dim==2
    m, n = model.m, model.n
    y = dataset[:bs][0].cuda()
    model.cuda()
    z_mu, z_logvar = model.encode(y)
    z_std = torch.exp(z_logvar)**0.5
    z_mu_style, z_std_style, z_logvar = z_mu[:,-2:].detach().cpu().numpy(), z_std[:,-2:].detach().cpu().numpy(), z_logvar.detach().cpu().numpy()

    mean_of_z_mu = np.mean(z_mu_style, axis=0)
    print(mean_of_z_mu)
    stds_of_z_mu = np.std(z_mu_style, axis=0)

    stds_0 = torch.linspace(-stds_range*stds_of_z_mu[0], stds_range*stds_of_z_mu[0], nrows)
    stds_0 -= mean_of_z_mu[0]
    stds_1 = torch.linspace(-stds_range*stds_of_z_mu[1], stds_range*stds_of_z_mu[1], nrows)
    stds_1 -= mean_of_z_mu[1]
    z_samples = torch.cartesian_prod(stds_0, stds_1).cuda()
    n_samples = len(z_samples)

    x_grid = model.x_grid
    x_grid = x_grid.expand(n_samples, x_grid.size(0), x_grid.size(1)).contiguous().cuda()

    y_hat = model.decode(z_samples, x_grid).detach().cpu()
    y_hat = y_hat.view(y_hat.size(0), 1, m, n)
    grid = torchvision.utils.make_grid(y_hat, nrow=nrows)[0] 

    if not return_domain: return grid
    else: return grid, stds_0, stds_1, z_mu_style

def get_TSNE_embedding(X, perplexity=30, n_components=2, random_state=None):
    # Check whether it's a torch tensor Variable that must be detached
    if hasattr(X, 'requires_grad') and X.requires_grad: X_ = X.detach().cpu()
    else: X_ = X
    print(f"Doing {n_components}-dim TSNE for perplexity {perplexity}")
    reducer = TSNE(perplexity=perplexity, n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(X_)
    return embedding

def get_TSNE_img_map(embedding, data, n_yimgs=70, n_ximgs=70,
            xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Given an embedding, plot a TSNE grid of the images.
    A grid is generated, then each image is assigned to a rectangles if it's
    enclosed by that rectangle. The image nearest the rectangle centroid (L2
    distance) is assigned that rectangle, and plotted in the image.
    Args
      embedding: Array shape (n_imgs, 2,) holding the tsne embedding coordinates
        of the imgs stored in data
      data: original data set of images. len(data)==len(embedding). To be plotted
        on the TSNE grid.
    Returns
        Torch tensor that, if plotted with imshow, does the embedding we want.
    """
    assert len(data)==len(embedding)
    img_shape = data.shape[-2:]
    ylen, xlen = data.shape[-2:]
    if xmin is None: xmin=embedding[:,0].min()
    if xmax is None: xmax=embedding[:,0].max()
    if ymin is None: ymin=embedding[:,1].min()
    if ymax is None: ymax=embedding[:,1].max()
    data=data.cpu()

    # Define grid corners
    ycorners, ysep = np.linspace(ymin, ymax, n_yimgs, endpoint=False, retstep=True)
    xcorners, xsep = np.linspace(xmin, xmax, n_ximgs, endpoint=False, retstep=True)
    # centroids of the grid
    ycentroids=ycorners+ysep/2
    xcentroids=xcorners+xsep/2

    # determine which point in the grid each embedded point belongs
    img_grid_indxs = (embedding - np.array([xmin, ymin])) // np.array([xsep,ysep])
    img_grid_indxs = img_grid_indxs.astype(dtype=int)

    #  Array that will hold each points distance to the centroid
    img_dist_to_centroids = np.zeros(len(embedding))

    # array to hold the final set of images
    img_plot=torch.zeros(n_yimgs*img_shape[0], n_ximgs*img_shape[1])

    # array that will give us the returnedindices
    object_indices=torch.zeros((n_ximgs, n_yimgs), dtype=torch.int)

    # Iterate over the grid
    for i in range(n_ximgs):
        for j in range(n_yimgs):
            ## Get indices of points that are in this box
            indxs=indxs = np.where(
                    np.all(img_grid_indxs==np.array([i,j])
                    ,axis=1)
                )[0]

            ## calculate distance to centroid for each point
            centroid=np.array([xcentroids[i],ycentroids[j]])
            img_dist_to_centroids[indxs] = np.linalg.norm(embedding[indxs] - centroid, ord=2, axis=1)

            ## Find the nearest image to the centroid
            # if there are no imgs in this box, then skip
            if len(img_dist_to_centroids[indxs])==0:
                indx_nearest=-1
            # else find nearest
            else:
                # argmin over the distances to centroid (is over a restricted subset)
                indx_subset = np.argmin(img_dist_to_centroids[indxs])
                indx_nearest = indxs[indx_subset]
                # Put image in the right spot in the larger image
                xslc = slice(i*xlen, i*xlen+xlen)
                yslc = slice(j*ylen, j*ylen+ylen)
                img_plot[xslc, yslc] = torch.Tensor(data[int(indx_nearest)])

            # save the index
            object_indices[i,j] = indx_nearest

    return img_plot.cpu(), object_indices

