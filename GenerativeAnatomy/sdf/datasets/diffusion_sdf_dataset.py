import os
from datetime import datetime
import torch
import numpy as np

from .sdf_dataset import SDFSamples, unpack_numpy_data, read_mesh_get_sampled_pts

today_date = datetime.now().strftime("%b_%d_%Y")

class DiffusionSDFSamples(SDFSamples):
    def __init__(
        self,
        list_mesh_paths,
        subsample=None,
        print_filename=False,
        n_pts=500000,
        p_near_surface=0.7,
        p_further_from_surface=0.2,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        axis_align=False,
        norm_pts=False,
        scale_method='max_rad',
        loc_save=os.environ['LOC_SDF_CACHE'],
        include_seed_in_hash=True,
        save_cache=True,
        random_seed=None,
        verbose=False,
        load_cache=True,
        point_cloud_size=1024,
        equal_pos_neg=True,
        fix_mesh=True,
    ):
        self.list_mesh_paths = list_mesh_paths
        self.subsample = subsample
        self.n_pts = n_pts
        self.p_near_surface = p_near_surface
        self.p_further_from_surface = p_further_from_surface
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.rand_function = rand_function
        self.center_pts = center_pts
        self.axis_align = axis_align
        self.norm_pts = norm_pts
        self.scale_method = scale_method
        self.loc_save = loc_save
        self.include_seed_in_hash = include_seed_in_hash
        self.random_seed = random_seed
        self.verbose = verbose
        self.point_cloud_size = point_cloud_size
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh

        self.list_hash_params = [
            'diffusion_sdf', #just to make unique for this data class
            self.n_pts,
            self.p_near_surface, self.sigma_near,
            self.p_further_from_surface, self.sigma_far,
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.fix_mesh
        ]

        if save_cache is True:
            cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(cache_folder, exist_ok=True)

        n_p_near_surface = int(n_pts * p_near_surface)
        n_p_further_from_surface = int(n_pts * p_further_from_surface)
        n_p_random = n_pts - n_p_near_surface - n_p_further_from_surface

        pt_sample_combos = [
            [n_p_near_surface, sigma_near],
            [n_p_further_from_surface, sigma_far],
            [n_p_random, None]
        ]       

        self.data = []
        for loc_mesh in list_mesh_paths:
            if print_filename is True:
                print(loc_mesh)

            # Create hash and filename 
            file_hash = self.create_hash(loc_mesh)
            cached_file = self.find_hash(filename=f'{file_hash}.npz')
            #preallocate torch array
            pts_array = torch.zeros((n_pts, 4))

            if (len(cached_file) > 0) and (load_cache is True):
                # if hashed file exists, load it. 
                data_ = np.load(cached_file[0])
                data = unpack_numpy_data(data_, point_cloud=True)
                
            else:
                # otherwise, load the mesh and create SDF samples. 
                print('Creating SDF Samples')
                data = {
                    'xyz': torch.zeros((n_pts, 3)),
                    'gt_sdf': torch.zeros((n_pts)),
                }
                pts_idx = 0
                for n_pts_, sigma_ in pt_sample_combos:
                    result_ = read_mesh_get_sampled_pts(
                        loc_mesh, 
                        mean=[0,0,0], 
                        sigma=sigma_, 
                        n_pts=n_pts_, 
                        rand_function=rand_function, 
                        center_pts=center_pts,
                        # axis_align=axis_align,
                        norm_pts=norm_pts,
                        scale_method=scale_method,
                        get_random=True,
                        return_orig_mesh=False,
                        return_new_mesh=False,
                        return_orig_pts=False,
                        return_point_cloud=True,
                        fix_mesh=self.fix_mesh,
                    )
                    xyz_ = result_['xyz']
                    sdfs_ = result_['gt_sdf']

                    data['xyz'][pts_idx:pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
                    data['gt_sdf'][pts_idx:pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
                    pts_idx += n_pts_
                data['point_cloud'] = torch.from_numpy(result_['point_cloud']).float()
                if save_cache is True:
                    # if want to cache, and new... then save. 
                    filepath = os.path.join(cache_folder, f'{file_hash}.npz')
                    np.savez(filepath, xyz=data['xyz'], gt_sdf=data['gt_sdf'], point_cloud=data['point_cloud'])

            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data['pos_idx'] = pos_idx
            data['neg_idx'] = neg_idx
            data['surf_idx'] = surf_idx

            self.data.append(data)
    
    def sdf_pos_neg_idx(self, data):
        pos_idx = (data['gt_sdf'] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data['gt_sdf'] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data['gt_sdf'] == 0).nonzero(as_tuple=True)[0]

        return pos_idx, neg_idx, surf_idx
    # def sdf_pos_neg(self, data):
    #     # print('data', data.shape)
    #     pos_idx = data['gt_sdf'] > 0
    #     neg_idx = data['gt_sdf'] < 0
    #     surf_idx = data['gt_sdf'] == 0

    #     # print('pos_idx', pos_idx.shape)
    #     # print('neg_idx', neg_idx.shape)

    #     pos_data = {
    #         'xyz': data['xyz'][pos_idx, :],
    #         'gt_sdf': data['gt_sdf'][pos_idx],
    #         'point_cloud': data['point_cloud']
    #     }

    #     neg_data = {
    #         'xyz': data['xyz'][neg_idx, :],
    #         'gt_sdf': data['gt_sdf'][neg_idx],
    #         'point_cloud': data['point_cloud']
    #     }

    #     return pos_data, neg_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        if self.subsample is not None:
            # print('Subsampling', self.subsample)

            if self.equal_pos_neg is True:
                samples_per_sign = int(self.subsample/2)
                
                idx_pos = data_['pos_idx'].repeat(data_['pos_idx'].size(0)//samples_per_sign + 1)
                perm_pos = torch.randperm(idx_pos.size(0))
                idx_pos = perm_pos[:samples_per_sign]

                idx_neg = data_['neg_idx'].repeat(data_['neg_idx'].size(0)//samples_per_sign + 1)
                perm_neg = torch.randperm(idx_neg.size(0))
                idx_neg = perm_neg[:samples_per_sign]

                idx_ = torch.cat((idx_pos, idx_neg), dim=0)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
            
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[:self.subsample]

            
            pt_cloud_perm = torch.randperm(data_['point_cloud'].size(0))
            idx_pc = pt_cloud_perm[:self.point_cloud_size]

            # print('idx_pc', idx_pc.shape)
            # print('data_ point cloud', data_['point_cloud'].shape)

            # print('idx_', idx_.shape)
            # print('data_ xyz', data_['xyz'].shape)

            data_ = {
                'xyz': data_['xyz'][idx_, :],
                'gt_sdf': data_['gt_sdf'][idx_],
                'point_cloud': data_['point_cloud'][idx_pc, :],
            }

            # print('min xyz', data_['xyz'].min(axis=0))
            # print('max xyz', data_['xyz'].max(axis=0))
            # print('min sdf', data_['gt_sdf'].min())
            # print('max sdf', data_['gt_sdf'].max())
            # print('min pc', data_['point_cloud'].min(axis=0))
            # print('max pc', data_['point_cloud'].max(axis=0))

            # print(data_)
        return data_, idx