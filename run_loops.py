import torch
import util
import time
import tqdm

def train_epoch(epoch, model, optimizer, train_loader, start_time, tb_writer=None, 
                 print_loss_interval=1, device='cpu', loss_is_avgd=True, testing=False):
    epoch_start_time = time.time()
    model.to(device)
    #N = len(train_loader.dataset)
    model.train()
    train_loss, train_loss_recon, train_loss_kl, train_elbo, train_loss_kl_theta, train_loss_kl_z, train_loss_consistency = 0, 0, 0, 0, 0, 0, 0
    
    N=0
    t = tqdm.tqdm(enumerate(train_loader))
    for batch_idx, (data, label) in t:
        N+=len(data)
        #if batch_idx %10==0:print(batch_idx)
        if testing and batch_idx>2: break
        data = data.to(device)
        optimizer.zero_grad()
        x, y, mu, logvar = model(data)
        loss_dict = model.loss_function(x, y, mu, logvar, epoch)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        ## The rest of the code is for loggig only
        # If the above losses are all batch-wise means, then we multiply the losses to totals for logging purposes only
        if loss_is_avgd: b = data.size(0)
        else: b=1
        train_loss += b*loss.item()
        train_loss_recon += b*loss_dict['loss_recon'].item()
        train_loss_kl += b*loss_dict['loss_kl'].item()
        train_elbo += b*loss_dict['elbo'].item()
        train_loss_kl_theta += b*loss_dict['loss_kl_theta'].item()
        train_loss_kl_z += b*loss_dict['loss_kl_z'].item()
        train_loss_consistency += b*loss_dict['loss_consistency']
        beta = loss_dict['beta']
        consistency_weight = loss_dict['consistency_weight']

        t.set_description(f"Loss {train_loss/N:.3f} recon {train_loss_recon/N:.3f} kl {train_loss_kl/N:.3f}, kl_theta {train_loss_kl_theta/N:.3f}, beta {beta}, consistency {train_loss_consistency/N:.3f}, consistency_weight={consistency_weight:.3f}")

    train_loss_avg, train_loss_recon_avg, train_loss_kl_avg, train_elbo, train_loss_kl_theta_avg, train_loss_kl_z_avg= train_loss/N, train_loss_recon/N, train_loss_kl/N, train_elbo/N, train_loss_kl_theta/N, train_loss_kl_z/N
    
    if print_loss_interval is not None and  epoch % print_loss_interval == 0:
        print(f'====> Epoch: {epoch} Train loss: {train_loss_avg:.8f}') 
    if tb_writer is not None:
        tb_writer.add_scalar("train_loss", train_loss_avg, epoch)
        tb_writer.add_scalar("train_recon_loss", train_loss_recon_avg, epoch)
        tb_writer.add_scalar("train_kl_loss", train_loss_kl_avg, epoch)
        tb_writer.add_scalar("train_elbo", train_elbo, epoch)
        tb_writer.add_scalar("train_loss_kl_theta", train_loss_kl_theta_avg, epoch)
        tb_writer.add_scalar("train_loss_kl_z", train_loss_kl_z_avg, epoch)
        tb_writer.add_scalar("train_epoch_duration_mins", (time.time()-epoch_start_time)/60, epoch)
        tb_writer.add_scalar("train_total_duration_mins", (time.time()-start_time)/60, epoch)
        tb_writer.add_scalar("train_consistency_loss", train_loss_consistency/N, epoch)
        tb_writer.add_scalar("consistency_weight", consistency_weight, epoch)

def test_epoch(epoch, model, test_loader, start_time, tb_writer=None, 
                tb_plot_interval=5, print_loss_interval=1, device='cpu', loss_is_avgd=True, 
                testing=False, skip_dist_matrix=False):
    epoch_start_time = time.time()
    model.to(device)
    #N = len(test_loader.dataset)
    model.eval()
    test_loss, test_loss_recon, test_loss_kl, test_elbo, test_loss_kl_theta, test_loss_kl_z, test_loss_consistency = 0, 0, 0, 0, 0, 0, 0
    
    N=0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            N+=len(data)
            #if batch_idx %10==0:print(batch_idx)
            if testing and batch_idx>2: break
            data = data.to(device)
            x, y, mu, logvar = model(data)
            loss_dict = model.loss_function(x, y, mu, logvar, epoch)
            loss = loss_dict['loss']

            # If the above losses are all batch-wise means, then we multiply the losses to totals for logging purposes only
            if loss_is_avgd: b = data.size(0)
            else: b=1
            test_loss += b*loss.item()
            test_loss_recon += b*loss_dict['loss_recon'].item()
            test_loss_kl += b*loss_dict['loss_kl'].item()
            test_elbo += b*loss_dict['elbo'].item()
            test_loss_kl_theta += b*loss_dict['loss_kl_theta'].item()
            test_loss_consistency += b*loss_dict['loss_consistency']
            test_loss_kl_z += b*loss_dict['loss_kl_z'].item()
            consistency_weight = loss_dict['consistency_weight']
        # avg the losses 
        test_loss_avg, test_loss_recon_avg, test_loss_kl_avg, test_loss_kl_theta_avg, test_loss_kl_z_avg = test_loss/N, test_loss_recon/N, test_loss_kl/N, test_loss_kl_theta/N, test_loss_kl_z/N
    
    if print_loss_interval is not None and  epoch % print_loss_interval == 0:
        print(f'====> Epoch: {epoch} Test loss: {test_loss_avg:.8f}') 
    if tb_writer is not None:
        tb_writer.add_scalar("test_loss", test_loss_avg, epoch)
        tb_writer.add_scalar("test_recon_loss", test_loss_recon_avg, epoch)
        tb_writer.add_scalar("test_kl_loss", test_loss_kl_avg, epoch)
        tb_writer.add_scalar("test_elbo", test_elbo, epoch)
        tb_writer.add_scalar("test_loss_kl_theta", test_loss_kl_theta_avg, epoch)
        tb_writer.add_scalar("test_loss_kl_z", test_loss_kl_z_avg, epoch)
        tb_writer.add_scalar("test_epoch_duration_mins", (time.time()-epoch_start_time)/60, epoch)
        tb_writer.add_scalar("test_total_duration_mins", (time.time()-start_time)/60, epoch)
        tb_writer.add_scalar("test_loss_consistency", test_loss_consistency/N, epoch)
        tb_writer.add_scalar("consistency_weight", consistency_weight, epoch)

    if tb_writer is not None and epoch % tb_plot_interval==0:
        print("Doing plots ...", end=" ")
        # reconstruction plot
        f = util.plot_reconstructions(model, test_loader.dataset, nrows=10)
        tb_tag=f"reconstructions_{epoch}"
        tb_writer.add_figure(tb_tag, f, close=True)

        # rotation embedding distance matrix plot. Use l2 of stds as a normalizer 
        # for matrix distances 
        if not skip_dist_matrix:
            plot_device='cpu'
            model.x_grid = model.x_grid.to(plot_device) # todo: delete later 
            y = torch.clone(test_loader.dataset[:100][0]).to(plot_device)# some test data 
            stds, stds_l2 = util.get_stds_embedding(y, model)
            distance_normalizer = stds_l2
            f = util.rotation_tests_batch(y, model,
                thetas=[0,30,60,90,120,150,180,210,240,270,300,330],
                distance_normalizer=distance_normalizer, show_dist_matrix=True,
                nrows=7, ncols=3, figsize=(40,40), device=plot_device)
            tb_tag=f"rotations_distance_{epoch}"
            tb_writer.add_figure(tb_tag, f, close=True)
            model.x_grid = model.x_grid.to('cuda')# todo: delete later
        print("Done")


if __name__=="__main__":
    from data import datasets 
    from models import spatial_vae, vae, base_vae
    import util
    import numpy as np

    DIR_PROJECT = "/pasteur/u/jmhb/vae-transform-invariance"
    DEFAULT_DATASETS_DIR=f"{DIR_PROJECT}/data/datasets"
    DATASET=datasets.MNIST
    flatten, binarize = True, False
    bs, num_workers = 512, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    tb_plot_interval=5

    dataset_train = DATASET(train=True, flatten=flatten, binarize=binarize)
    dataset_test = DATASET(train=False, flatten=flatten, binarize=binarize)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=num_workers,
                            batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, num_workers=num_workers,
                            batch_size=bs, shuffle=True)


    img_shape = dataset_train.img_shape
    img_shape_original = dataset_train.img_shape_original
    z_dim = 4
    loss_kwargs = dict(
        recon_key='bce',
        beta=0.0001,
        M_N=train_loader.batch_size/len(train_loader.dataset),
        theta_prior=np.pi,
        dx_prior=0.1,
    )
    encoder_kwargs=dict(net_type='fc',
                        hidden_dims=[200,200],
                        activation='lrelu')
    decoder_kwargs=dict(net_type='fc',
                        hidden_dims=[200,200],
                        activation='lrelu')
    spatial_transforms=dict(rot=True, trans=False)
    lr=5e-4

    #model = vae.VAE(img_shape=img_shape, img_shape_original = img_shape_original
    #    , z_dim=z_dim, loss_kwargs=loss_kwargs,encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)
    model = spatial_vae.SpatialVAE(img_shape=img_shape, img_shape_original=img_shape_original, z_dim=z_dim,
                                   loss_kwargs=loss_kwargs,encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs,
                                   spatial_transforms=spatial_transforms)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tb_writer = util.create_logging_dir(dir_project=DIR_PROJECT, verbose=1)

    print("Training")
    epochs=2
    for epoch in range(epochs):
        train_epoch(epoch=epoch, model=model, optimizer=optimizer, train_loader=train_loader,
            tb_writer=tb_writer, device=device)
        test_epoch(epoch=epoch, model=model, test_loader=test_loader, tb_writer=tb_writer,
                tb_plot_interval=tb_plot_interval, print_loss_interval=1, device=device)

