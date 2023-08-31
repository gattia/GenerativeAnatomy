import torch

from .utils import (
    compute_chamfer,
    adjust_learning_rate
)

from .recon_evaluation import compute_recon_loss

from .predictive_validation_class import Regress

from GenerativeAnatomy.sdf.datasets import (
    read_mesh_get_sampled_pts, 
    get_pts_center_and_scale,
    read_meshes_get_sampled_pts
)
from GenerativeAnatomy.sdf.mesh import create_mesh

from .reconstruct_latent_S3 import reconstruct_latent_S3
from .reconstruct_diffusion_sdf import reconstruct_mesh_diffusion_sdf

# from GenerativeAnatomy.sdf.testing import get_mean_errors
import numpy as np
import sys
import os
import copy
import pymskt as mskt
import wandb
import time

try:
    from pytorch3d.loss import chamfer_distance
    __chamfer__ = True
except:
    print('Error importing `chamfer_distance` from pytorch3d.loss')
    __chamfer__ = False

try:
    from GenerativeAnatomy.dependencies import sinkhorn
    __emd__ = True
except:
    print('Error importing `sinkhorn` from GenerativeAnatomy.dependencies')
    __emd__ = False





def reconstruct_latent(
    decoders,
    num_iterations,
    latent_size,
    xyz, #Nx3
    sdf_gt, #Nx1 or list of Nx1
    loss_type='l1',
    lr=5e-4,
    loss_weight=1.0,
    max_batch_size=30000,
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence='num_iterations',
    convergence_patience=50,
    log_wandb=False,
    log_wandb_step=10,
    verbose=False,
    optimizer_name='adam',
    n_samples=None,
    max_n_samples=None,#100000,
    n_steps_sample_ramp=None, #200,
    difficulty_weight=None,
    pts_surface=None,
):
    if type(sdf_gt) in (torch.Tensor, np.ndarray):
        sdf_gt = [sdf_gt]
    elif type(sdf_gt) in (list, tuple):
        pass
    elif type(sdf_gt) in (str):
        raise Exception(
            'Must provided xyz/sdf from mesh - resconstruct latent will not load mesh' +
            'from file. Try reconstruct_mesh instead.'
        )
    else:
        raise Exception('Invalid sdf_gt type')
    
    if isinstance(pts_surface, (list, tuple)):
        pts_surface = torch.tensor(pts_surface).cuda()
    elif isinstance(pts_surface, np.ndarray):
        pts_surface = torch.from_numpy(pts_surface).cuda()

    if not isinstance(decoders, (list, tuple)):
        decoders = [decoders,]
    
    # Setup n LR updates
    if (n_lr_updates == 0) or (n_lr_updates is None): 
        adjust_lr_every = num_iterations + 1
    else:
        adjust_lr_every = num_iterations // n_lr_updates

    # Setup n_samples, if not specified. 
    if n_samples is None:
        n_samples = xyz.shape[0]
    
    if (max_n_samples is not None) and (n_steps_sample_ramp is not None):
        print('Ramping up number of samples')
        n_samples_init = n_samples
    else:
        n_samples_init = None
    
    # Set a clamp (maximum) distance to "model"
    for sdf_idx, sdf in enumerate(sdf_gt):
        if clamp_dist is not None:
            sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
        # Move to GPU
        sdf_gt[sdf_idx] = sdf.cuda()

    # Initialize random latent vector directly on GPU
    latent = torch.ones(1, latent_size, device=torch.device('cuda')).normal_(mean=latent_init_mean, std=latent_init_std)
    latent.requires_grad = True
    latent_input = latent.expand(n_samples, -1)

    # Initialize optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam([latent], lr=lr)
    elif optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS([latent])

    # Initialize loss
    if loss_type == 'l1':
        loss_fn = torch.nn.L1Loss(reduction='none')
    elif loss_type == 'l2':
        loss_fn = torch.nn.MSELoss(reduction='none')

    # Initialize convergence tracking
    patience = 0
    loss = 100
    recon_loss = 100
    
    # MOVE DECODERS TO GPU
    # SET DECODERS TO EVAL SO NO BATCH NORM ETC.
    for decoder in decoders:
        decoder.cuda()
        decoder.eval()
    
    # PASS XYZ TO GPU
    xyz = xyz.cuda()
    
    for step in range(num_iterations):

        #update LR
        if optimizer_name == 'adam':
            adjust_learning_rate(
                initial_lr=lr,
                optimizer=optimizer,
                iteration=step, 
                decreased_by=lr_update_factor, 
                adjust_lr_every=adjust_lr_every
            )

        def step_():
            global recon_loss_
            global latent_loss_
            global loss_

            recon_loss_ = 0

            optimizer.zero_grad()

            if n_samples_init is not None:
                n_samples_ = n_samples_init + int((max_n_samples - n_samples_init) * min(1.0, (step / n_steps_sample_ramp)))
                print(n_samples_)
            else:
                n_samples_ = n_samples
                print('not changing... ', n_samples_)

            if n_samples_ != xyz.shape[0]:
                # Below if/else is just to get a list of indices to sample
                if len(sdf_gt) > 1:
                    # get equal number of samples from each surface
                    # the list pts_surface is a list that indicates
                    # which surface each point in xyz belongs to
                    n_samples_per = n_samples_ // len(sdf_gt)
                    # pre allocate array to store random samples
                    rand_samp = torch.empty(n_samples_, dtype=torch.int64, device=torch.device('cuda'))
                    for idx in range(len(sdf_gt)):
                        # get the locations of the points that belong to the current surface
                        pts_ = (pts_surface == idx).nonzero(as_tuple=True)[0]
                        # get a random permutation of the points
                        perm = torch.randperm(pts_.shape[0])
                        # get the randomly permuted indices from this surface
                        pts_ = pts_[perm[:n_samples_per]]
                        # store the points in the pre-allocated rand_samp array
                        rand_samp[idx*n_samples_per:(idx+1)*n_samples_per] = pts_
                    
                    if len(rand_samp) < n_samples_:
                        # if we don't have enough points, then just take random points
                        perm = torch.randperm(xyz.shape[0])
                        _idx_ = perm[:n_samples_-len(rand_samp)]
                        rand_samp = torch.cat([rand_samp, _idx_], dim=0)
                else:
                    rand_samp = torch.randperm(xyz.shape[0])[:n_samples_]

                # Use rand_samp indices to get xyz and sdf_gt
                xyz_input = xyz[rand_samp, ...]
                sdf_gt_ = [x[rand_samp, ...] for x in sdf_gt]
            else:
                # if n_samples == xyz.shape[0], then just use all of the xyz points and sdf_gt
                xyz_input = xyz
                sdf_gt_ = sdf_gt
            
            if n_samples_init is not None:
                # if n_samples_init is not None, then we are ramping up the number of samples
                # so we need to update the latent_input
                latent_input_ = latent.expand(n_samples_, -1)
            else:
                latent_input_ = latent_input
                
            # concat latent and xyz that will be inputted into decoder. 
            inputs = torch.cat([latent_input_, xyz_input], dim=1)

            #TODO: potentially store each decoder's loss and return it to track in wandb?

            # Iterate over the decoders (if there are multiple)
            for decoder_idx, decoder in enumerate(decoders):
                current_pt_idx = 0

                # Iterate over points in xyz_input 
                while current_pt_idx < inputs.shape[0]:
                    # Get batch size & predict sdf
                    current_batch_size = min(max_batch_size, inputs.shape[0] - current_pt_idx)
                    pred_sdf = decoder(inputs[current_pt_idx:current_pt_idx + current_batch_size, ...])
                    
                    # initialize loss as zeros
                    _loss_ = torch.zeros(current_batch_size, device=torch.device('cuda'))

                    # Apply clamping distance - to ignore points that are too far away
                    if clamp_dist is not None:
                        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
                    
                    # Compute loss
                    if pred_sdf.shape[1] == 1:
                        # if only one surface - then just loss_fn (l1/l2) between pred_sdf and sdf_gt
                        if difficulty_weight is not None:
                            raise NotImplementedError
                        _loss_ += loss_fn(pred_sdf.squeeze(), sdf_gt_[decoder_idx][current_pt_idx:current_pt_idx + current_batch_size, ...].squeeze()) * loss_weight
                        
                    else:
                        # if multiple surfaces - then compute loss for each surface and weight them
                        for sdf_idx in range(pred_sdf.shape[1]):
                            if difficulty_weight is not None:
                                error_sign = torch.sign(sdf_gt_[sdf_idx][current_pt_idx:current_pt_idx + current_batch_size, ...].squeeze() - pred_sdf[:, sdf_idx].squeeze())
                                sdf_gt_sign = torch.sign(sdf_gt_[sdf_idx][current_pt_idx:current_pt_idx + current_batch_size, ...].squeeze())
                                sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                            else:
                                sample_weights = torch.ones_like(pred_sdf[:,sdf_idx].squeeze())
                            _loss_ += loss_fn(
                                pred_sdf[:,sdf_idx].squeeze(), 
                                sdf_gt_[sdf_idx][current_pt_idx:current_pt_idx + current_batch_size, ...].squeeze()
                            ) * loss_weight * sample_weights
                    
                    # send gradient from each batch of SDF values to latent
                    _loss_ = torch.mean(_loss_)
                    _loss_.backward()
                    # update the global loss
                    recon_loss_ += _loss_

                    current_pt_idx += current_batch_size
            
            # Compute latent loss - used to constrain new predictions to be close to zero (mean)
            # penalizing "abnormal" shapes
            if l2reg is True:
                latent_loss_ = latent_reg_weight * torch.mean(latent ** 2)
                latent_loss_.backward()
            else:
                latent_loss_ = 0        
            
            loss_ = recon_loss_ + latent_loss_

            # if optimizer_name =='adam':
            #     return 


            # loss_.backward()
            return loss_
        
        # Run the step_ (defined above) and the appropriate optimizer
        if optimizer_name == 'adam':
            step_()
            optimizer.step()
        elif optimizer_name == 'lbfgs':
            print('LBFGS step:', step)
            optimizer.step(step_)

        # Print progress/loss as appropriate
        if step % 50 == 0:
            print('Step: ', step, 'Loss: ', loss_.item())
            if verbose is True:
                print('\tLatent norm: ', latent.norm)
        
        # Log to wandb as appropriate
        if (log_wandb is True) and (step % log_wandb_step == 0):
            wandb.log({
                'total_loss': loss_.item(),
                'l1_loss': loss_.item(),
                'recon_loss': recon_loss_.item(),
                'latent_loss': latent_loss_.item() if l2reg is True else np.nan,
                'latent_norm': latent.norm().item()
            })

        # Handle end of loop accounting of loss/latent based on convergence criteria
        if convergence == 'overall_loss':
            if loss_ < loss:
                loss = loss_
                latent_ = torch.clone(latent)
                patience = 0
            else:
                patience += 1
            
            if patience > convergence_patience:
                print('Converged!')
                print('Step: ', step)
                break
        elif convergence == 'recon_loss':
            if recon_loss_ < recon_loss:
                recon_loss = recon_loss_
                latent_ = torch.clone(latent)
                patience = 0
            else:
                patience += 1
                
            if patience > convergence_patience:
                print('Converged!')
                print('Step: ', step)
                break
        else:
            loss = loss_
            latent_ = torch.clone(latent)
            
    return loss, latent_

def reconstruct_mesh(
    path,
    decoders,
    latent_size,
    num_iterations=1000,
    lr=5e-4,
    batch_size=32**3,
    batch_size_latent_recon=3*10**4,
    loss_weight=1.0,
    loss_type='l1',
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    calc_symmetric_chamfer=False,
    calc_assd=False,
    calc_emd=False,
    return_unscaled=True,
    n_pts_per_axis=256,
    log_wandb=False,
    return_latent=False,
    convergence='num_iterations',
    convergence_patience=50,
    fit_similarity=False,
    scale_jointly=False,
    register_similarity=False,
    n_pts_per_axis_mean_mesh=128,
    scale_all_meshes=True, #whether when scaling a model it should be on all points in all meshes or not
    mesh_to_scale=0, # PRETTY MUCH ASSUME ALWAYS SCALING FIRST MESH
    decoder_to_scale=0, # PRETTY MUCH ASSUME ALWAYS SCALING FIRST DECODER
    scale_method='max_rad',
    verbose=False,
    objects_per_decoder=1,
    latent_optimizer_name='adam',
    get_rand_pts=False,
    n_pts_random=100000,
    sigma_rand_pts = 0.001,
    n_samples_chamfer=None,
    n_samples_latent_recon=10000,
    max_n_samples_latent_recon=None,#100000,
    n_steps_sample_ramp_latent_recon=None, #200,
    difficulty_weight_recon=None,
    chamfer_norm=2,
    func=None,
):
    """
    Reconstructs mesh at path using decoders. 

    NOTES: 
    Assumes that length of path = sum(objects_per_decoder)
    That is, 
        path0_mesh = decoder0_mesh0
        path1_mesh = decoder0_mesh1 OR decoder1_mesh0
        etc. 
    """

    # Check if path is a single mesh or a list of meshes & set multi_object flag
    if isinstance(path, str):
        multi_object = False
    elif isinstance(path, (list, tuple)):
        multi_object = True
        # appropriately set the number of random points for multi-object reconstructions
        if isinstance(n_pts_random, (int, float)):
            n_pts_random = [n_pts_random,] * len(path)
        if isinstance(sigma_rand_pts, (int, float)):
            sigma_rand_pts = [sigma_rand_pts,] * len(path)
    else:
        raise ValueError('path must be a string or a list/tuple of strings')
    
    # make decoders a list so that it can be iterated over (make agnostic to number of decoders)
    if not isinstance(decoders,(list, tuple)):
        decoders = [decoders,]
    
    # make objects_per_decoder a list so that it can be iterated over
    if isinstance(objects_per_decoder, (list, tuple)):
        assert len(objects_per_decoder) == len(decoders), 'If objects_per_decoder is a list, it must be the same length as decoders'
    elif isinstance(objects_per_decoder, int):
        # if single int, assume that all decoders have the same number of objects
        objects_per_decoder = [objects_per_decoder,] * len(decoders)

    tic = time.time()

    if (fit_similarity is True) and (register_similarity is True):
        raise ValueError('Cannot fit similarity and register similarity at the same time')
    
    if (scale_jointly) or (register_similarity is True):
        # if register first, then register new mesh to the mean of the decoder (zero latent vector)
        # create mean mesh of only mesh, or "mesh_to_scale" if more than one.
        mean_latent = torch.zeros(1, latent_size)
        # create mean mesh, assume that using decoder_0 & mesh_0, but
        # technically this can be specified.
        mean_mesh = create_mesh(
            decoder=decoders[decoder_to_scale].cuda(),
            latent_vector=mean_latent.cuda(),
            n_pts_per_axis=n_pts_per_axis_mean_mesh,
            objects=objects_per_decoder[decoder_to_scale],
            batch_size=batch_size,
            verbose=verbose,
        )

        if objects_per_decoder[decoder_to_scale] > 1:
            mean_mesh = mean_mesh[mesh_to_scale]

        if mean_mesh is None:
            # Mean mesh is None if the zero latent vector is not well defined/learned
            # yet. In this case, the results will be very poor, might as well skip.
            result = {
                'mesh': [None, ] * sum(objects_per_decoder),
            }
            if calc_symmetric_chamfer:
                for idx in range(sum(objects_per_decoder)):
                    result[f'chamfer_{idx}'] = np.nan
            if calc_assd:
                for idx in range(sum(objects_per_decoder)):
                    result[f'assd_{idx}'] = np.nan
            if calc_emd:
                for idx in range(sum(objects_per_decoder)):
                    result['emd_{idx}'] = np.nan
            if return_latent:
                result['latent'] = mean_latent
            return result
    else:
        mean_mesh = None

    toc = time.time()
    time_load_mean = toc - tic
    tic = time.time()
    if verbose is True:
        print(f'Loaded mean mesh in {time_load_mean:.2f} seconds')

    # read in mesh(es) and get sampled points for fitting decoder too
    # handle single or multiple meshes appropriately. 
    if multi_object is False:
        result_ = read_mesh_get_sampled_pts(
            path,
            sigma=sigma_rand_pts,
            center_pts= not scale_jointly,
            norm_pts= not scale_jointly,
            scale_method=scale_method,
            get_random=get_rand_pts,
            register_to_mean_first=True if register_similarity else False,
            mean_mesh=mean_mesh if register_similarity else None,
            n_pts_random=n_pts_random,
            include_surf_in_pts=get_rand_pts
        )
    elif multi_object is True:
        result_ = read_meshes_get_sampled_pts(
            paths=path,
            mean=[0,0,0],
            sigma=sigma_rand_pts,
            center_pts= not scale_jointly,
            norm_pts= not scale_jointly,
            scale_all_meshes=scale_all_meshes,
            mesh_to_scale=mesh_to_scale,
            scale_method=scale_method,
            get_random=get_rand_pts,
            register_to_mean_first=True if register_similarity else False,
            mean_mesh=mean_mesh,
            n_pts_random=n_pts_random,
            include_surf_in_pts=get_rand_pts
        )
    else:
        raise ValueError('multi_object must be True or False')

    xyz = result_['pts']
    sdf_gt = result_['sdf']
    pts_surface = result_['pts_surface']

    # ensure all data are torch tensors and have the correct shape
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz).float()
    if multi_object is True:
        for sdf_idx, sdf_gt_ in enumerate(sdf_gt):
            if not isinstance(sdf_gt_, torch.Tensor):
                sdf_gt[sdf_idx] = torch.from_numpy(sdf_gt_).float()

            if len(sdf_gt[sdf_idx].shape) == 1:
                sdf_gt[sdf_idx] = sdf_gt[sdf_idx].unsqueeze(1)
    elif multi_object is False:
        if not isinstance(sdf_gt, torch.Tensor):
            sdf_gt = torch.from_numpy(sdf_gt).float()

        if len(sdf_gt.shape) == 1:
            sdf_gt = sdf_gt.unsqueeze(1)

    toc = time.time()
    time_load_mesh = toc - tic
    if verbose is True:
        print(f'Loaded mesh in {time_load_mesh:.2f} seconds')

    tic = time.time()

    # FIT THE LATENT CODE TO THE MESH
    # specify general reconstruction parameters that apply to
    # all recon methods.
    reconstruct_inputs = {
        'decoders': decoders,
        'num_iterations': num_iterations,
        'latent_size': latent_size,
        'sdf_gt': sdf_gt,
        'xyz': xyz,
        'lr': lr,
        'loss_weight': loss_weight,
        'loss_type': loss_type,
        'l2reg': l2reg,
        'latent_init_std': latent_init_std,
        'latent_init_mean': latent_init_mean,
        'clamp_dist': clamp_dist,
        'latent_reg_weight': latent_reg_weight,
        'n_lr_updates': n_lr_updates,
        'lr_update_factor': lr_update_factor,
        'log_wandb': log_wandb,
        'convergence': convergence,
        'convergence_patience': convergence_patience,
        'verbose': verbose,
    }

    # specify latent recon functions, and extra parameters if needed
    if fit_similarity is False:
        reconstruct_function = reconstruct_latent
        recon_params_ = {
            'max_batch_size': batch_size_latent_recon,
            'optimizer_name': latent_optimizer_name,
            'n_samples': n_samples_latent_recon,
            'difficulty_weight': difficulty_weight_recon,
            'pts_surface': pts_surface,
            'max_n_samples': max_n_samples_latent_recon, 
            'n_steps_sample_ramp': n_steps_sample_ramp_latent_recon
        }
 
    elif fit_similarity is True:
        reconstruct_function = reconstruct_latent_S3
        recon_params_ = {
            'soft_contrain_scale': True,
            'scale_constrain_deviation': 0.05,
            'soft_constrain_theta': True,
            'theta_constrain_deviation': torch.pi/36,
            'soft_constrain_translation': True,
            'translation_constrain_deviation': 0.05,
            'init_scale_method': 'max_rad',
            'transform_update_patience': 500
        }
    else:
        raise ValueError('fit_similarity must be True or False')

    reconstruct_inputs.update(recon_params_)

    loss, latent = reconstruct_function(**reconstruct_inputs)

    toc = time.time()
    time_recon_latent = toc - tic
    if verbose is True:
        print(f'Reconstructed latent in {time_recon_latent:.2f} seconds')
    tic = time.time()

    # if fit_similarity is True, then set transform parameters
    # for reconstructed mesh from the returned latent
    if fit_similarity is True:
        R = latent['R']
        t = latent['t']
        s = latent['s']
        latent = latent['latent']

    else:
        R = None
        t = None
        s = None

    if verbose is True:
        print(result_['icp_transform'])

    # create mesh(es) from latent
    meshes = []
    for decoder_idx, decoder in enumerate(decoders):
        # pass alignment parameters to return mesh to original position
        # pass number of objects in case decoder is a multi-object decoder
        mesh = create_mesh(
            decoder=decoder.cuda(),
            latent_vector=latent.cuda(),
            n_pts_per_axis=n_pts_per_axis,
            path_original_mesh=None,
            offset=result_['center'],
            scale=result_['scale'],
            icp_transform=result_['icp_transform'],
            R=R,
            t=t,
            s=s,
            objects=objects_per_decoder[decoder_idx],
            verbose=verbose,
        )
        if objects_per_decoder[decoder_idx] > 1:
            # append sequentially so they match the order of meshes at "path"
            for mesh_ in mesh:
                meshes.append(mesh_)
        else:
            meshes.append(mesh)

    toc = time.time()
    time_create_mesh = toc - tic
    if verbose is True:
        print(f'Created mesh in {time_create_mesh:.2f} seconds')
    tic = time.time()
    
    if func is not None:
        func_results = func(orig_mesh=result_['orig_mesh'], recon_meshes=meshes)

    toc = time.time()
    time_calc_recon_funcs = toc - tic
    if verbose is True:
        print(f'metrics in {time_calc_recon_funcs:.2f} seconds')
    tic = time.time()

    if calc_emd or calc_symmetric_chamfer or calc_assd or return_latent or (func is not None):
        result = {'mesh': meshes}

        if calc_emd or calc_symmetric_chamfer or calc_assd:
            result_ = compute_recon_loss(
                meshes=meshes,
                orig_pts=result_['orig_pts'],
                n_samples_chamfer=n_samples_chamfer,
                chamfer_norm=chamfer_norm,
                calc_symmetric_chamfer=calc_symmetric_chamfer,
                calc_assd=calc_assd,
                calc_emd=calc_emd,
            )
            toc = time.time()
            time_calc_recon_loss = toc - tic
            if verbose is True:
                print(f'metrics in {time_calc_recon_loss:.2f} seconds')

            result.update(result_)

        if return_latent:
            result['latent'] = latent
        
        if func is not None:
            result.update(func_results)      

        if log_wandb is True:
            result_ = copy.copy(result)
            del result_['mesh']
            wandb.log(result_)

        return result
    else:
        return meshes

def tune_reconstruction(
    model,
    config,
    use_wandb=True
):
    """
    Tune reconstruction parameters using wandb for logging.
    """
    if use_wandb is True:
        wandb.login(key=os.environ['WANDB_KEY'])
        # wandb.init(
        #     # Set the project where this run will be logged
        #     project=config["project_name"], # "diffusion-net-predict-sex",
        #     entity=config["entity_name"], # "bone-modeling",
        #     # Track hyperparameters and run metadata
        #     config=config,
        #     name=config['run_name'],
        #     tags=config['tags']
        # )

    
    dict_loss = get_mean_errors(
        mesh_paths=config['mesh_paths'],
        decoders=model,
        num_iterations=config['num_iterations'],
        register_similarity=True,
        latent_size=config['latent_size'],
        lr=config['lr'],
        loss_weight=config['loss_weight'],
        loss_type=config['loss_type'],
        l2reg=config['l2reg'],
        latent_init_std=config['latent_init_std'],
        latent_init_mean=config['latent_init_mean'],
        clamp_dist=config['clamp_dist'],
        latent_reg_weight=config['latent_reg_weight'],
        n_lr_updates=config['n_lr_updates'],
        lr_update_factor=config['lr_update_factor'],
        calc_symmetric_chamfer=config['chamfer'],
        calc_assd=config['assd'],
        calc_emd=config['emd'],
        convergence=config['convergence'],
        convergence_patience=config['convergence_patience'],
        log_wandb=use_wandb,
        verbose=config['verbose'],
        objects_per_decoder=config['objects_per_decoder'],
        batch_size_latent_recon=config['batch_size_latent_recon'],
        get_rand_pts=config['get_rand_pts_recon'],
        n_pts_random=config['n_pts_random_recon'],
        sigma_rand_pts=config['sigma_rand_pts_recon'],
        n_samples_latent_recon=config['n_samples_latent_recon'],
        difficulty_weight_recon=config['difficulty_weight_recon'],
        chamfer_norm=config['chamfer_norm'],
        config=config
    )

def get_mean_errors(
    mesh_paths,
    decoders,
    latent_size,
    calc_symmetric_chamfer=False,
    calc_assd=False,
    calc_emd=False,
    log_wandb=False,
    num_iterations=1000,
    n_pts_per_axis=256,
    lr=5e-4,
    loss_weight=1.0,
    loss_type='l1',
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence='num_iterations',
    convergence_patience=50,
    config=None,
    register_similarity=False,
    scale_all_meshes=True,
    model_type='deepsdf',
    point_cloud_size=1024,
    verbose=False,
    objects_per_decoder=1,
    mesh_to_scale=0,
    decoder_to_scale=0,
    batch_size_latent_recon=3*10**4,
    latent_optimizer_name='adam',
    get_rand_pts=False,
    n_pts_random=100000,
    sigma_rand_pts=0.01,
    n_samples_latent_recon=10000,
    max_n_samples_latent_recon=None,#100000,
    n_steps_sample_ramp_latent_recon=None, #200,
    difficulty_weight_recon=None,
    chamfer_norm=2,
    recon_func=None,
    predict_val_variables=None,

    scale_jointly=False,
):
    """
    Reconstruct meshes & compute errors    
    """

    loss = {}
    
    reconstruct_inputs = {
        'latent_size':latent_size,
        'calc_symmetric_chamfer':calc_symmetric_chamfer,
        'calc_assd':calc_assd,
        'calc_emd':calc_emd,
        'register_similarity':register_similarity,
        'scale_jointly':scale_jointly,
        'scale_all_meshes':scale_all_meshes,
        'return_latent': True
    }

    if model_type == 'deepsdf':
        reconstruct_inputs_ = {
            'decoders':decoders,
            'log_wandb':log_wandb,
            'num_iterations':num_iterations,
            'n_pts_per_axis':n_pts_per_axis,
            'lr':lr,
            'loss_weight':loss_weight,
            'loss_type':loss_type,
            'l2reg':l2reg,
            'latent_init_std':latent_init_std,
            'latent_init_mean':latent_init_mean,
            'clamp_dist':clamp_dist,
            'latent_reg_weight':latent_reg_weight,
            'n_lr_updates':n_lr_updates,
            'lr_update_factor':lr_update_factor,
            'convergence':convergence,
            'convergence_patience':convergence_patience,
            'register_similarity':register_similarity,
            'objects_per_decoder':objects_per_decoder,
            'batch_size_latent_recon': batch_size_latent_recon,
            'verbose': verbose,
            'latent_optimizer_name': latent_optimizer_name,
            'get_rand_pts': get_rand_pts,
            'n_pts_random': n_pts_random,
            'sigma_rand_pts': sigma_rand_pts,
            'n_samples_latent_recon': n_samples_latent_recon,
            'max_n_samples_latent_recon': max_n_samples_latent_recon,
            'n_steps_sample_ramp_latent_recon': n_steps_sample_ramp_latent_recon,
            'difficulty_weight_recon': difficulty_weight_recon,
            'chamfer_norm': chamfer_norm,
            'func': recon_func,
            
        }

        recon_fx = reconstruct_mesh
    elif model_type == 'diffusion':
        reconstruct_inputs_ = {
            'model': decoders,
            'n_encode_iterations': num_iterations,
            'point_cloud_size': point_cloud_size,
            'scale_to_original_mesh': True,
        }
        recon_fx = reconstruct_mesh_diffusion_sdf
    else:
        raise ValueError(f'model_type must be either "deepsdf" or "diffusion"m received {model_type}')

    reconstruct_inputs.update(reconstruct_inputs_)

    if predict_val_variables is not None:
        reg = Regress(
            list_factors=predict_val_variables, 
            list_paths=mesh_paths
        )
        

    for idx, mesh_path in enumerate(mesh_paths):
        if log_wandb is True:
            config_ = config.copy()
            config_['mesh_path'] = mesh_path
            config_['mesh_idx'] = idx
            wandb.init(
                # Set the project where this run will be logged
                project=config["project_name"], # "diffusion-net-predict-sex",
                entity=config["entity_name"], # "bone-modeling",
                # Track hyperparameters and run metadata
                config=config_,
                name=config['run_name'],
                tags=config['tags']
            )
        reconstruct_inputs['path'] = mesh_path
        result_ = recon_fx(
            **reconstruct_inputs
        )
        if verbose is True:
            print('result_', result_)

        if predict_val_variables is not None:
            reg.add_latent(result_)

        for mesh_idx in range(len(result_['mesh'])):
            if calc_symmetric_chamfer:
                if idx == 0:
                    loss[f'chamfer_{mesh_idx}'] = []
                loss[f'chamfer_{mesh_idx}'].append(result_[f'chamfer_{mesh_idx}'])
            if calc_emd:
                if idx == 0:
                    loss[f'emd_{mesh_idx}'] = []
                loss[f'emd_{mesh_idx}'].append(result_[f'emd_{mesh_idx}'])
            if calc_assd:
                if idx == 0:
                    loss[f'assd_{mesh_idx}'] = []
                loss[f'assd_{mesh_idx}'].append(result_[f'assd_{mesh_idx}'])
        
        # if a function was given - append its results. 
        if (recon_func is not None):
            for key, val in result_.items():
                if 'func_' == key[:5]:
                    if idx == 0:
                        loss[key[5:]] = []
                    loss[key[5:]].append(val)

        if log_wandb is True:
            wandb.finish()

    if verbose is True:
        print('loss', loss)
    loss_ = {}

    if predict_val_variables is not None:
        predictive_results = reg.calc_r2()
        loss_.update(predictive_results)
        
    for key, item in loss.items():
        mean = np.mean(item)
        std = np.std(item)
        median = np.median(item)
        hist = wandb.Histogram(item)
        loss_[key] = mean
        loss_[f'{key}_std'] = std
        loss_[f'{key}_mean'] = mean
        loss_[f'{key}_median'] = median
        loss_[f'{key}_hist'] = hist

    return loss_
