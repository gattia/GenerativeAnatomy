from GenerativeAnatomy.sdf.sdf_utils import (
    get_learning_rate_schedules, 
    adjust_learning_rate, 
    save_latent_vectors,
    save_model,
    get_optimizer,
    get_latent_vecs,
    get_checkpoints,
)
from GenerativeAnatomy.sdf.reconstruct import get_mean_errors, compare_cart_thickness

from GenerativeAnatomy.sdf.train.utils import (
    get_kld,
    cyclic_anneal_linear,
    calc_weight,
    add_plain_lr_to_config,
    NoOpProfiler,
    get_profiler,
)

import wandb
import os
import torch 
import time
import numpy as np


DICT_VALIDATION_FUNCS = {
    'compare_cart_thickness': compare_cart_thickness,
    None: None
}

loss_l1 = torch.nn.L1Loss(reduction='none')

def train_deep_sdf(
    config, 
    model,
    sdf_dataset, 
    use_wandb=False):    

    config = add_plain_lr_to_config(config)
    config['checkpoints'] = get_checkpoints(config)
    config['lr_schedules'] = get_learning_rate_schedules(config)

    model = model.to(config['device'])

    if use_wandb is True:
        wandb.login(key=os.environ['WANDB_KEY'])
        wandb.init(
            # Set the project where this run will be logged
            project=config["project_name"], # "diffusion-net-predict-sex",
            entity=config["entity_name"], # "bone-modeling",
            # Track hyperparameters and run metadata
            config=config,
            name=config['run_name'],
            tags=config['tags']
        )
        wandb.watch(model, log='all')

    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config['objects_per_batch'],
        shuffle=True,
        num_workers=config['num_data_loader_threads'],
        drop_last=False,
        prefetch_factor=config['prefetch_factor'],
        pin_memory=True 
    )

    latent_vecs = get_latent_vecs(len(data_loader.dataset), config).cuda()

    optimizer = get_optimizer(model, latent_vecs, lr_schedules=config['lr_schedules'], optimizer=config["optimizer"], weight_decay=config["weight_decay"])

    # profiler that runs if config['profiler'] is True, else a dummy profiler is used and should have no effect
    with get_profiler(config) as profiler:

        for epoch in range(1, config['n_epochs'] + 1):
            # not passing latent_vecs because presumably they are being tracked by the
            # and updated in memory? 
            log_dict = train_epoch(model, data_loader, latent_vecs, optimizer=optimizer, config=config, epoch=epoch, return_loss=True)
            
            if epoch in config['checkpoints']:
                save_latent_vectors(
                    config=config,
                    epoch=epoch,
                    latent_vec=latent_vecs,
                )
                save_model(
                    config=config,
                    epoch=epoch,
                    decoder=model,
                )
                if ('val_paths' in config) & (config['val_paths'] is not None):
                    
                    torch.cuda.empty_cache()

                    # TODO: Change this to just accept the config? 
                    # or... update all parameters to be the same in the config and the function call?
                    # this will just allow unpacking of the config dict.
                    dict_loss = get_mean_errors(
                        mesh_paths=config['val_paths'],
                        decoders=model,
                        num_iterations=config['num_iterations_recon'],
                        register_similarity=True,
                        latent_size=config['latent_size'],
                        lr=config['lr_recon'],
                        # loss_weight
                        # loss_type
                        l2reg=config['l2reg_recon'],
                        # latent_init_std
                        # latent_init_mean
                        clamp_dist=config['clamp_dist_recon'],
                        # latent_reg_weight
                        n_lr_updates=config['n_lr_updates_recon'],
                        lr_update_factor=config['lr_update_factor_recon'],
                        calc_symmetric_chamfer=config['chamfer'],
                        calc_assd=config['assd'],
                        calc_emd=config['emd'],
                        convergence=config['convergence_type_recon'],
                        convergence_patience=config['convergence_patience_recon'],
                        # log_wandb
                        verbose=config['verbose'],
                        objects_per_decoder=2,
                        batch_size_latent_recon=config['batch_size_latent_recon'],
                        get_rand_pts=config['get_rand_pts_recon'],
                        n_pts_random=config['n_pts_random_recon'],
                        sigma_rand_pts=config['sigma_rand_pts_recon'],
                        n_samples_latent_recon=config['n_samples_latent_recon'], 
                        # difficulty_weight_recon
                        # chamfer_norm
                        scale_all_meshes=True,
                        recon_func=None if (('recon_val_func_name' not in config)) else DICT_VALIDATION_FUNCS[config['recon_val_func_name']],
                        predict_val_variables=None if ('predict_val_variables' not in config) else config['predict_val_variables'],

                        scale_jointly=config['scale_jointly'],
                    )

                    log_dict.update(dict_loss)

            if use_wandb is True:                    
                wandb.log(log_dict)
            
            profiler.step()
        
    return

def train_epoch(
    model, 
    data_loader,
    latent_vecs,
    optimizer,
    config, 
    epoch, 
    return_loss=True,
    verbose=False,
    n_surfaces=2,
):
    # n_surfaces = len(models)
    start = time.time()
    # for model in models:
    model.train()

    adjust_learning_rate(config['lr_schedules'], optimizer, epoch)
    
    step_losses = 0
    step_l1_loss = 0
    step_code_reg_loss = 0
    step_l1_losses = [0. for _ in range(n_surfaces)]

        # if config['code_regularization_type_prior'] == 'kld_diagonal':
        #     kld_loss = get_kld(latent_vecs)


    for sdf_data, indices in data_loader:
        if config['verbose'] is True:
            print('sdf index size:', indices.size())
            print('xyz data size:', sdf_data['xyz'].size())
            print('sdf gt size:', sdf_data['gt_sdf'].size())
                
        xyz = sdf_data['xyz'].cuda()
        xyz = xyz.reshape(-1, 3)

        num_sdf_samples = xyz.shape[0]
        xyz.requires_grad = False

        indices = indices.cuda()        
        
        sdf_gt = []
        for surf_idx in range(n_surfaces):
            sdf_gt_ = sdf_data['gt_sdf'][:, :, surf_idx].reshape(-1, 1)
            if config['enforce_minmax'] is True:
                sdf_gt_ = torch.clamp(sdf_gt_, -config['clamp_dist'], config['clamp_dist'])
            elif config['hard_sample_difficulty_weight'] is not None:
                sdf_gt_ = torch.clamp(sdf_gt_, -1, 1)
            sdf_gt_.requires_grad = False
            sdf_gt.append(sdf_gt_)
        
        if config['verbose'] is True:
            print('sdf gt size:', sdf_gt[0].size(), sdf_gt[1].size())

        xyz = torch.chunk(xyz, config['batch_split'])
        indices = torch.chunk(
            indices.unsqueeze(-1).repeat(1, config['samples_per_object_per_batch']).view(-1), #repeat the index for every sample
            config['batch_split'], # split the data into the appropriate number of batches - so can fit in ram. 
        )

        for surf_idx in range(n_surfaces):
            sdf_gt[surf_idx] = torch.chunk(sdf_gt[surf_idx], config['batch_split'])
        
        if config['verbose'] is True:
            print('len sdf_gt', len(sdf_gt))
            print('len sdf_gt chunks', len(sdf_gt[0]), len(sdf_gt[1]))
            print('len xyz chunks', len(xyz))

        batch_loss = 0.0
        batch_l1_loss = 0.0
        batch_l1_losses = [0.0 for _ in range(n_surfaces)]
        batch_code_reg_loss = 0.0

        optimizer.zero_grad()

        for split_idx in range(config['batch_split']):
            if config['verbose'] is True:
                print('Split idx: ', split_idx)
            
            batch_vecs = latent_vecs(indices[split_idx])
            if 'variational' in config and config['variational'] is True:
                mu = batch_vecs[:, :config['latent_size']]
                logvar = batch_vecs[:, config['latent_size']:]
                std = torch.exp(0.5 * logvar)
                err = torch.randn_like(std)
                batch_vecs = std * err + mu

            inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
            # inputs = inputs.to(config['device'])

            # pred_sdfs = []
            # for model in models:
            pred_sdf = model(inputs, epoch=epoch)
            if config['enforce_minmax'] is True:
                pred_sdf = torch.clamp(pred_sdf, -config['clamp_dist'], config['clamp_dist'])
            elif config['hard_sample_difficulty_weight'] is not None:
                pred_sdf = torch.clamp(pred_sdf, -1, 1)

            if config['verbose'] is True:
                print('len pred_sdf', pred_sdf.shape)
                print('split idx', split_idx)
            l1_losses = []
            for surf_idx in range(n_surfaces):
                if config['verbose'] is True:
                    print('surf idx', surf_idx)
                    print(len(sdf_gt))
                    print(len(sdf_gt[surf_idx]))
                    print('pred_sdf shape', pred_sdf.shape)
                    print('unsqueezed pred_sdf shape', pred_sdf[:, surf_idx].shape)
                    print('sdf_gt shape', sdf_gt[surf_idx][split_idx].shape)
                l1_losses.append(loss_l1(pred_sdf[:, surf_idx], sdf_gt[surf_idx][split_idx].squeeze(1).cuda()))

            if 'multi_object_overlap' in config and config['multi_object_overlap'] is True:
                raise Exception('Not implemented yet')
                # Should add some weighted penalty to the l1 loss
                # this is similar to surface_accuracy_e below
                # The idea being - the signs of the objects should never have 2 objects as (-) becuase it 
                # means one object is inside the other.
                # However, we dont want there to be any gaps between them - so we need to be intelligent about 
                # how to penalize this so that it doesnt end up with weird artefacts.
            
            # curriculum SDF equation 5
            # progressively fine-tune the regions of surface cared about by the network. 
            if config['surface_accuracy_e'] is not None:
                weight_schedule = 1 - calc_weight(epoch, config['n_epochs'], config['surface_accuracy_schedule'], config['surface_accuracy_cooldown'])
                for l1_idx, l1_loss in enumerate(l1_losses):
                    l1_losses[l1_idx] = torch.maximum(
                        l1_loss - (weight_schedule * config['surface_accuracy_e']),
                        torch.zeros_like(l1_loss)
                    )
            
            # curriculum SDF equation 6
            # progressively fine-tune the regions of surface cared about by the network.
            # weighting gives higher preference to regions closer to surface / with opposite sign. 
            if config['sample_difficulty_weight'] is not None:
                weight_schedule = calc_weight(epoch, config['n_epochs'], config['sample_difficulty_weight_schedule'], config['sample_difficulty_cooldown'])
                difficulty_weight = weight_schedule * config['sample_difficulty_weight']
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    # Weights points independently
                    # so, if hard for one surface - then we weight it heavily, but if
                    # easy for another surface - then we weight it less.
                    error_sign = torch.sign(surf_gt_[split_idx].squeeze(1).cuda() - pred_sdf[:, surf_idx])
                    sdf_gt_sign = torch.sign(surf_gt_[split_idx].squeeze(1).cuda())
                    sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                    l1_losses[surf_idx] = l1_losses[surf_idx] * sample_weights

            elif config['sample_difficulty_lx'] is not None:
                weight_schedule = calc_weight(epoch, config['n_epochs'], config['sample_difficulty_lx_schedule'], config['sample_difficulty_lx_cooldown'])
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    difficulty_weight = 1 / (l1_losses[surf_idx] ** config['sample_difficulty_lx']  + config['sample_difficulty_lx_epsilon'])
                    difficulty_weight = difficulty_weight * weight_schedule
                    l1_losses[surf_idx] = l1_losses[surf_idx] * difficulty_weight
                    
            elif config['hard_sample_difficulty_power'] is not None:
                weight_schedule = calc_weight(epoch, config['n_epochs'], config['hard_sample_difficulty_schedule'], config['hard_sample_difficulty_cooldown'])

                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    weight = 1 + config['hard_sample_difficulty_alpha'] * (-1 * surf_gt_[split_idx].squeeze(1).cuda() * pred_sdf[:, surf_idx])
                    weight = torch.clamp(weight, min=0)
                    weight = weight ** (config['hard_sample_difficulty_power'] * weight_schedule)
                    l1_losses[surf_idx] = l1_losses[surf_idx] * weight
                    
            # Weight each surface loss by the number of samples it has
            # so that the sum of them all is the same as the mean loss. 
            for idx, l1_loss_ in enumerate(l1_losses):
                l1_losses[idx] = l1_loss_ / num_sdf_samples
            
            
            # l1_loss = l1_loss / num_sdf_samples

            # Comput total loss for each mesh (which is the same as the 
            # mean).
            l1_loss = 0

            # Create weights for each surface
            if ('surface_weighting' in config) & (isinstance(config['surface_weighting'] , (list, tuple))):
                assert len(config['surface_weighting']) == n_surfaces
                weights_total = n_surfaces
                weights_sum = sum(config['surface_weighting'])
                weights = []
                for weight in config['surface_weighting']:
                    weights.append(weight / weights_sum * weights_total)
            else:
                weights = [1,] * n_surfaces

            for l1_idx, l1_loss_ in enumerate(l1_losses):
                l1_loss += l1_loss_.sum() * weights[l1_idx]
            
            # Normalize by number of surfaces 
            l1_loss = l1_loss / len(l1_losses)

            if config['verbose'] is True:
                print(f'l1 losses: {[l1_loss_.sum().item() for l1_loss_ in l1_losses]}')
                print(f'l1 loss: {l1_loss.item()}')
            
            batch_l1_loss += l1_loss.item()
            for l1_idx, l1_loss_ in enumerate(l1_losses):
                batch_l1_losses[l1_idx] += l1_loss_.sum().item()
            chunk_loss = l1_loss

            if config['code_regularization'] is True:
                if 'variational' in config and config['variational'] is True:
                    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
                    reg_loss = kld
                    code_reg_norm = 1
                else:
                    if config['code_regularization_type_prior'] == 'spherical':
                        # spherical prior
                        # all latent vectors should have the same unit length
                        # therefore, the latent dimensions will be correlated
                        # with one another - this is as opposed to PCA (and below). 
                        reg_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    elif config['code_regularization_type_prior'] == 'identity':
                        # independently penalize each dimension/value of latent code
                        # therefore latent code ends up having identity covariance matrix
                        reg_loss = torch.sum(torch.square(batch_vecs))
                    elif config['code_regularization_type_prior'] == 'kld_diagonal':
                        reg_loss = get_kld(batch_vecs)
                    else:
                        raise ValueError(f'Unknown code regularization type prior: {config["code_regularization_type_prior"]}')
                    code_reg_norm = num_sdf_samples

                reg_loss = (
                    config['code_regularization_weight'] * min(1, epoch/config['code_regularization_warmup']) * reg_loss
                ) / code_reg_norm

                if config['code_cyclic_anneal'] is True:
                    anneal_weight = cyclic_anneal_linear(epoch=epoch, n_epochs=config['n_epochs'])
                    reg_loss = reg_loss * anneal_weight

                chunk_loss = chunk_loss + reg_loss.cuda()
                batch_code_reg_loss += reg_loss.item()
            
            chunk_loss.backward()

            batch_loss += chunk_loss.item()

        step_losses += batch_loss
        step_l1_loss += batch_l1_loss
        step_code_reg_loss += batch_code_reg_loss
        for l1_idx, l1_loss_ in enumerate(batch_l1_losses):
            step_l1_losses[l1_idx] += l1_loss_ # l1_loss_
        
        if config['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['grad_clip']
            )
        
        optimizer.step()
    end = time.time()

    seconds_elapsed = end - start

    save_loss = step_losses / len(data_loader)
    save_l1_loss = step_l1_loss / len(data_loader)
    save_code_reg_loss = step_code_reg_loss / len(data_loader) 
    save_l1_losses = [l1_loss_ / len(data_loader) for l1_loss_ in step_l1_losses]   
    
    print('save loss: ', save_loss)
    print('\t save l1 loss: ', save_l1_loss)
    print('\t save code loss: ', save_code_reg_loss)
    print('\t save l1 losses: ', save_l1_losses)

    log_dict = {
        'loss': save_loss,
        'epoch_time_s': seconds_elapsed,
        'l1_loss': save_l1_loss,
        'latent_code_regularization_loss': save_code_reg_loss,
    }
    for l1_idx, l1_loss_ in enumerate(save_l1_losses):
        log_dict['l1_loss_{}'.format(l1_idx)] = l1_loss_

    if config['log_latent'] is not None:
        vecs = latent_vecs.weight.data.cpu().numpy()
        for latent_idx in range(config['log_latent']):
            log_dict[f'latent_{latent_idx}'] = wandb.Histogram(vecs[:, latent_idx])
            log_dict[f'latent_{latent_idx}_mean'] = vecs[:, latent_idx].mean()
            log_dict[f'latent_{latent_idx}_std'] = vecs[:, latent_idx].std()


    return log_dict

