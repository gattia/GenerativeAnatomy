import os
import pymskt as mskt
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import torch
import hashlib
from datetime import datetime
import warnings
import time
import point_cloud_utils as pcu


today_date = datetime.now().strftime("%b_%d_%Y")

def get_rand_uniform_pts(n_pts, mins=(-1, -1, -1), maxs=(1, 1, 1)):
    """
    Given a set of points, returns a set of points that are randomly sampled
    from a uniform distribution from min(s) to max(s).

    Args:
        n_pts (int): Number of points to sample
        mins (tuple, optional): Minimum value of the distribution. Defaults to (-1, -1, -1).
        maxs (tuple, optional): Maximum value of the distribution. Defaults to (1, 1, 1).

    Returns:
        np.ndarray: (n_pts, 3) array of points
    """
    rand_gen = np.random.uniform

    pts = np.zeros((n_pts, len(mins)))
    mins = np.tile(mins, [n_pts, 1])
    pts[:,:] = rand_gen(mins, maxs)

    return pts

def get_pts_center_and_scale(
    pts,
    center=True,
    scale=True,
    scale_method='max_rad',
    return_pts=False,
    pts_center=None
):
    """
    Given a set of points, returns the center and scale of the points.

    Args:
        pts (np.ndarray): (n_pts, 3) array of points
        center (bool, optional): Whether to center the points. Defaults to True.
        scale (bool, optional): Whether to scale the points. Defaults to True.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        return_pts (bool, optional): Whether to return the centered and scaled points. Defaults to False.
        pts_center (np.ndarray, optional): (n_pts, 3) array of points to center the points on. Defaults to None.
    
    Returns:
        tuple: (center, scale) of the points

    Raises:
        NotImplementedError: If scale_method is not implemented

    Notes:
        If pts_center is not None, then use that to center the points
        and use all of the points in pts to scale. This is used for
        the bone only, and then scaling based on bone + cartilage
    """

    if pts_center is None:
        center = np.mean(pts, axis=0)
    else:
        center = np.mean(pts_center, axis=0)
    pts -= center

    if scale_method == 'max_rad':
        scale = np.max(np.linalg.norm(pts, axis=-1), axis=-1)
        pts /= scale
    else:
        raise NotImplementedError(f'Scale Method ** {scale_method} ** Not Implemented')
   
    if return_pts is True:
        return center, scale, pts
    
    return center, scale

def meshfix(mesh, assert_=False, assert_error=0.01):
    """
    Fixes a mesh using meshfix.
    
    Args:
        mesh (mskt.mesh.Mesh): Mesh to fix
    
    Notes:
        Fixes the mesh in place.
    """
    n_pts_orig = mesh.point_coords.shape[0]
    mesh.fix_mesh()
    n_pts_fixed = mesh.point_coords.shape[0]
    # Asserting that no more than 1% of the mesh points were removed
    print(f'Fixed mesh, {n_pts_orig} -> {n_pts_fixed} ({(n_pts_fixed - n_pts_orig) / n_pts_orig * 100:.2f}%)')
    if assert_ is True:
        assert (n_pts_orig - n_pts_fixed) < (assert_error * n_pts_orig), f'Mesh dropped too many points, {n_pts_orig} -> {n_pts_fixed}'

def get_cube_mins_maxs(pts):
    """
    Given a set of points, returns the mins and maxs of the points
    in a cube.
    
    Args:
        pts (np.ndarray): (n_pts, 3) array of points
    
    Returns:
        tuple: (mins, maxs) of the points
    """

    mean = np.mean(pts, axis=0)
    norm_pts = pts - mean
    radial_max = np.max(np.linalg.norm(norm_pts, axis=-1))
    mins = mean - radial_max
    maxs = mean + radial_max
    
    return mins, maxs

def read_mesh_get_sampled_pts(
    path, 
    mean=[0,0,0], 
    sigma=1, 
    n_pts=200000, 
    rand_function='normal', 
    center_pts=True,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    register_to_mean_first=False,
    mean_mesh=None,
    fix_mesh=True,
    include_surf_in_pts=False,
    # Single mesh specific
    return_point_cloud=False,
    **kwargs,
):
    """
    Function to read in and sample points from a single mesh.
    
    Args:
        path (str): Path to mesh
        mean (list, optional): Mean to apply for random sample(s). Defaults to [0,0,0].
        sigma (float, optional): Standard deviation/scale to apply for random sample(s). Defaults to 1.
        n_pts (int, optional): Number of points to sample. Defaults to 200000.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Whether to center the points. Defaults to True.
        norm_pts (bool, optional): Whether to normalize the points. Defaults to False.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        get_random (bool, optional): Whether to sample random points. Defaults to True.
        return_orig_mesh (bool, optional): Whether to return the original mesh. Defaults to False.
        return_new_mesh (bool, optional): Whether to return the new mesh. Defaults to False.
        return_orig_pts (bool, optional): Whether to return the original points. Defaults to False.
        register_to_mean_first (bool, optional): Whether to register the mesh to the mean mesh first. Defaults to False.
        mean_mesh (vtkPolyData or mskt.mesh.Mesh, optional): Mean mesh to register to. Defaults to None.
        return_point_cloud (bool, optional): Whether to return the point cloud. Defaults to False.
        fix_mesh (bool, optional): Whether to fix the mesh (using meshfix). Defaults to True.
        include_surf_in_pts (bool, optional): Whether to include the surface points in the random points. Defaults to False.
    
    Returns:
        dict: Dictionary of results
    
    Notes:
    """
    list_deprecated = ['return_scale', 'return_center', 'return_orig_pts', 'return_orig_mesh', 'return_new_mesh']
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f'{kwarg} is deprecated and not used in this function - always True')
    
    results = {}

    # if mesh path does not exist, return None (skipping)
    if os.path.exists(path) is False:
        print(f'Skipping ... path does not exist, {path}')
        return None

    # read in mesh & "fix" using meshfix if requested
    orig_mesh = mskt.mesh.Mesh(path)
    if fix_mesh is True:
        meshfix(orig_mesh)
    
    # return orig_pts expanded dims for compatibility when storing
    # multiple meshes in the same dictionary
    results['orig_pts'] = [orig_mesh.point_coords]

    if (register_to_mean_first is True):
        # Rigid + scaling alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.

        if mean_mesh is None:
            raise Exception('Must provide mean mesh to register to')
        icp_transform = orig_mesh.rigidly_register(
            other_mesh=mean_mesh,
            as_source=True,
            apply_transform_to_mesh=True,
            return_transformed_mesh=False,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode='similarity',
            return_transform=True,
        )
        results['icp_transform'] = icp_transform
    else:
        results['icp_transform'] = None
    
    if (center_pts is True) or (norm_pts is True):
        center, scale, new_pts = get_pts_center_and_scale(
            np.copy(orig_mesh.point_coords),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=True
        )
    
    else:
        new_pts = np.copy(orig_mesh.point_coords)
        scale = 1
        center = np.zeros(3)
    
    new_mesh = orig_mesh.copy()
    new_mesh.point_coords = new_pts

    results['new_pts'] = [new_pts]

    if get_random is True:
        if sigma is not None:
            rand_pts = new_mesh.rand_pts_around_surface(n_pts=n_pts, surface_method='bluenoise', distribution=rand_function, sigma=sigma)
        else:
            mins, maxs = get_cube_mins_maxs(new_pts)
            rand_pts = get_rand_uniform_pts(n_pts, mins=mins, maxs=maxs)
        
        if include_surf_in_pts is True:
            rand_pts = np.concatenate([rand_pts, new_pts], axis=0)

        if norm_pts is True:
            rand_pts = np.clip(rand_pts, -1, 1)

        rand_sdf = new_mesh.get_sdf_pts(pts=rand_pts, method='pcu')

        results['xyz'] = rand_pts
        results['sdf'] = rand_sdf
        results['pts_surface'] = [0] * rand_pts.shape[0]
    else:
        results['pts'] = new_pts
        results['sdf'] = np.zeros(new_pts.shape[0])
        results['pts_surface'] = [0] * new_pts.shape[0]
        
    
    if return_point_cloud is True:
        results['point_cloud'] = new_pts

    # Theres no reason we shouldnt include these... they are always there and take up effectively no space. 
    results['scale'] = scale
    results['center'] = center

    results['orig_mesh'] = [orig_mesh]
    results['new_mesh'] = [new_mesh]
    
    return results

def unpack_pts(data, pts_name='orig_pts'):
    # get original points...
    pts = []
    pts_arrays = [x for x in data.files if f'{pts_name}_' in x]
    if len(pts_arrays) > 0:
        for pts_idx in range(len(pts_arrays)):
            pts.append(data[f'{pts_name}_{pts_idx}'])
        
    return pts

def unpack_numpy_data(
    data_,
    point_cloud=False
):
    data = {}

    # Get points / xyz coords
    if 'pts' in data_:
        data['xyz'] = torch.from_numpy(data_['pts']).float()
    elif 'xyz' in data_:
        data['xyz'] = torch.from_numpy(data_['xyz']).float()
    else:
        raise ValueError('No pts or xyz in cached file')
    
    # Get SDFs of the original points
    if 'sdfs' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['sdfs']).float()
    elif 'gt_sdf' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['gt_sdf']).float()
    elif 'sdf' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['sdf']).float()
    else:
        raise ValueError('No sdfs or gt_sdf or sdf in cached file')

    # Get random point cloud surface points... this was used for Diffusion SDFs model
    if point_cloud is True:
        data['point_cloud'] = torch.from_numpy(data_['point_cloud']).float()

    # get original points...
    orig_pts= unpack_pts(data_, pts_name='orig_pts')
    data['orig_pts'] = orig_pts
    # get new points...
    new_pts = unpack_pts(data_, pts_name='new_pts')
    data['new_pts'] = new_pts

    return data

def read_meshes_get_sampled_pts(
    paths, 
    mean=[0,0,0], 
    sigma=[1, 1], 
    n_pts=[200000, 200000], 
    rand_function='normal', 
    center_pts=True,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    register_to_mean_first=False,
    mean_mesh=None,
    fix_mesh=True,
    include_surf_in_pts=False,
    # Multiple mesh specific
    scale_all_meshes=True,
    center_all_meshes=False,
    mesh_to_scale=0,
    verbose=False,
    icp_transform=None,
    **kwargs,
):
    """
    Function to read in and sample points from multiple meshes.
    """
    tic = time.time()
    list_deprecated = ['return_scale', 'return_center', 'return_orig_pts', 'return_orig_mesh', 'return_new_mesh']
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f'{kwarg} is deprecated and not used in this function - always True')
    
    # preallocate results dictionary
    results = {}

    # Read all meshes and store in list
    orig_meshes = []
    orig_pts = []
    for path in paths:
        if os.path.exists(path) is False:
            print(f'Skipping ... path does not exist, {path}')
            return None
        mesh = mskt.mesh.Mesh(path)
        # fixing meshes ensures they are not degenerate
        # degenerate meshes will cause issues fitting SDFs. 
        if fix_mesh is True:
            meshfix(mesh)
        orig_meshes.append(mesh)
        orig_pts.append(mesh.point_coords)
    
    # return orig_pts
    results['orig_pts'] = orig_pts

    toc = time.time()
    print(f'Finished reading meshes in {toc - tic:.3f}s')
    tic = time.time()

    # Copy all meshes & points to new lists
    new_meshes = []
    new_pts = []
    for mesh_idx, orig_mesh in enumerate(orig_meshes):
        new_mesh_ = orig_mesh.copy()
        new_pts.append(new_mesh_.point_coords)
        new_meshes.append(new_mesh_)

    if (register_to_mean_first is True):
        # Rigid + scaling (similarity) alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.
        print('Registering meshes to mean mesh')
        if mean_mesh is None:
            raise ValueError('Must provide mean mesh to register to')

        if icp_transform is None:
            icp_transform = orig_meshes[mesh_to_scale].rigidly_register(
                other_mesh=mean_mesh,
                as_source=True,
                apply_transform_to_mesh=False,
                return_transformed_mesh=False,
                max_n_iter=100,
                n_landmarks=1000,
                reg_mode='similarity',
                return_transform=True,
            )
        results['icp_transform'] = icp_transform

        # apply transform to all meshes
        for idx, new_mesh in enumerate(new_meshes):
            new_mesh.apply_transform_to_mesh(icp_transform)
            new_pts[idx] = new_mesh.point_coords

    else:
        results['icp_transform'] = None
    
    toc = time.time()
    print(f'Finished registering meshes in {toc - tic:.3f}s')
    tic = time.time()

    if (center_pts is True) or (norm_pts is True):
        print('Scaling and centering meshes')
        if scale_all_meshes is True:
            pts_ = np.concatenate(new_pts, axis=0)
            if center_all_meshes is True:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None
            else:
                # set specific points to center becuase they are not the same
                # for centering as they are for scaling (pts_)
                pts_center = new_pts[mesh_to_scale]
        else:
            pts_ = new_pts[mesh_to_scale]
            if center_all_meshes is True:
                # set specific points to center because scale/center are not on
                # the same data
                pts_center = np.concatenate(new_pts, axis=0)
            else:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None

        center, scale = get_pts_center_and_scale(
            np.copy(pts_),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=False,
            pts_center=pts_center,
        )

        for pts_idx, new_pts_ in enumerate(new_pts):
            new_pts[pts_idx] = (new_pts_ - center)/scale
    else:
        # Do nothing to the points because they are left the same.
        scale = 1
        center = np.zeros(3)
    
    toc = time.time()
    print(f'Finished centering and scaling meshes in {toc - tic:.3f}s')
    tic = time.time()

    for mesh_idx, new_mesh in enumerate(new_meshes):
        new_mesh.point_coords = new_pts[mesh_idx]
    
    results['new_pts'] = new_pts

    if get_random is True:
        rand_pts = []
        rand_sdf = []
        pts_surface = []

        if None in sigma:
            pts_cube = np.concatenate(new_pts, axis=0)
            mins, maxs = get_cube_mins_maxs(pts_cube)

        for new_pts_idx, new_mesh_ in enumerate(new_meshes):
            if n_pts[new_pts_idx]  > 0:
                if sigma[new_pts_idx] is not None:
                    rand_pts_ = new_mesh_.rand_pts_around_surface(n_pts=n_pts[new_pts_idx], surface_method='bluenoise', distribution=rand_function, sigma=sigma[new_pts_idx])
                else:
                    rand_pts_ = get_rand_uniform_pts(n_pts[new_pts_idx], mins=mins, maxs=maxs)
                
                if include_surf_in_pts is True:
                    rand_pts_ = np.concatenate([rand_pts_, new_pts_], axis=0)

                rand_pts.append(rand_pts_)
                pts_surface.append([new_pts_idx] * rand_pts_.shape[0])
            else:
                rand_pts.append(np.zeros((0,3)))
                pts_surface.append(np.zeros((0,3)))
        
        rand_pts = np.concatenate(rand_pts, axis=0)
        pts_surface = np.concatenate(pts_surface, axis=0)

        
        for new_mesh in new_meshes:
            tic_ = time.time()
            print(rand_pts.shape, new_mesh.point_coords.shape)
            print(rand_pts.dtype, new_mesh.point_coords.dtype)
            print(type(rand_pts))
            rand_sdf.append(new_mesh.get_sdf_pts(pts=rand_pts, method='pcu'))
            toc_ = time.time()
            print(f'Finished calculating SDFs in {toc_ - tic_:.3f}s')

        results['pts'] = rand_pts
        results['sdf'] = rand_sdf
        results['pts_surface'] = pts_surface
    else:
        sdfs = []
        # Need to set SDFs for the same mesh to be 0
        # but need to actually calculate the SDFs for the other
        # meshes.
        for mesh_idx, new_mesh in enumerate(new_meshes):
            sdfs_ = []

            for pts_idx, new_pts_ in enumerate(new_pts):
                if verbose is True:
                    print('mesh_idx, new_mesh point_coords shape', mesh_idx, new_mesh.point_coords.shape)
                if pts_idx == mesh_idx:
                    if verbose is True:
                        print('adding zeros new_pts_ shape (zero)', new_pts_.shape)
                    # same mesh, set SDFs to 0
                    sdfs_.append(np.zeros(new_pts_.shape[0]))
                else:
                    # different mesh, calculate SDFs
                    _sdfs_ = new_mesh.get_sdf_pts(pts=new_pts_, method='pcu')
                    if verbose is True:
                        print('caculating SDFs for new_pts_ ', _sdfs_.shape)
                    sdfs_.append(_sdfs_)

            sdfs.append(np.concatenate(sdfs_, axis=0))

        pts_surface = []
        for pts_idx, new_pts_ in enumerate(new_pts):
            pts_surface.append([pts_idx] * new_pts_.shape[0])
        pts_surface = np.concatenate(pts_surface, axis=0)

        results['pts'] = np.concatenate(new_pts, axis=0)
        results['sdf'] = sdfs
        results['pts_surface'] = pts_surface
    
    toc = time.time()
    print(f'Finished getting random points and SDFs in {toc - tic:.3f}s')


    results['new_pts'] = new_pts

    results['scale'] = scale
    results['center'] = center

    results['orig_mesh'] = orig_meshes
    results['new_mesh'] = new_meshes

    return results


class SDFSamples(torch.utils.data.Dataset):
    """
    Dataset class for sampling SDFs from meshes.
    
    Args:
        list_mesh_paths (list): List of paths to meshes
        subsample (int, optional): Number of points to subsample. Defaults to None.
        n_pts (int, optional): Number of points to sample. Defaults to 500000.
        p_near_surface (float, optional): Proportion of points to sample near the surface. Defaults to 0.4.
        p_further_from_surface (float, optional): Proportion of points to sample further from the surface. Defaults to 0.4.
        sigma_near (float, optional): Standard deviation/scale of the distribution for points near the surface. Defaults to 0.01.
        sigma_far (float, optional): Standard deviation/scale of the distribution for points further from the surface. Defaults to 0.1.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Whether to center the points. Defaults to True.
        axis_align (bool, optional): Whether to axis align the points. Defaults to False.
        norm_pts (bool, optional): Whether to normalize the points. Defaults to False.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        loc_save (str, optional): Location to save the cached files. Defaults to os.environ['LOC_SDF_CACHE'].
        include_seed_in_hash (bool, optional): Whether to include the random seed in the hash. Defaults to True.
        save_cache (bool, optional): Whether to save the cached files. Defaults to True.
        load_cache (bool, optional): Whether to load the cached files. Defaults to True.
        random_seed (int, optional): Random seed. Defaults to None.
        reference_mesh (vtkPolyData or mskt.mesh.Mesh, optional): Reference mesh to register to. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        equal_pos_neg (bool, optional): Whether to have equal positive and negative SDFs. Defaults to True.
        fix_mesh (bool, optional): Whether to fix the meshes (sing meshfix). Defaults to True.
        print_filename (bool, optional): Whether to print the filename when loading. Defaults to False.
        
        Notes:
            If reference_mesh is not None, then all meshes will be registered to the reference mesh.
            If equal_pos_neg is True, then the number of positive and negative SDFs will be equal.
            If fix_mesh is True, then the meshes will be fixed using meshfix.
            If print_filename is True, then the filename will be printed when loading.    
        """
    
    def __init__(
        self,
        list_mesh_paths,
        subsample=None,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        axis_align=False,
        norm_pts=False,
        scale_method='max_rad',
        scale_jointly=False,
        loc_save=os.environ['LOC_SDF_CACHE'],
        include_seed_in_hash=True,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
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
        self.scale_jointly = scale_jointly
        self.loc_save = loc_save
        self.include_seed_in_hash = include_seed_in_hash
        self.random_seed = random_seed
        self.reference_mesh = reference_mesh
        self.verbose = verbose
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.print_filename = print_filename

        # set defaults so can use same 'norm_and_scale_all_meshes' function
        # for single and multiple meshes
        if not hasattr(self, 'reference_object'):
            self.reference_object = 0
        if not hasattr(self, 'n_meshes'):
            self.n_meshes = 1

        # preprocess inputs before proceeding
        self.preprocess_inputs()

        self.list_hash_params = self.get_hash_params()

        if save_cache is True:
            self.cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(self.cache_folder, exist_ok=True)

        # get the combinations of points and sigmas to sample
        self.pt_sample_combos = self.get_pt_sample_combos()

        if self.reference_mesh is not None:
            self.load_reference_mesh()

        self.data = []
        for loc_mesh in list_mesh_paths:
            if self.verbose is True:
                print('Loading mesh:', loc_mesh)
                
            data = self.get_sample_data_dict(loc_mesh)
            
            if data is not None:
                self.data.append(data)
            else:
                print('Skipping mesh:', loc_mesh)
                print('Error in loading')
        
        if self.scale_jointly is True:
            self.norm_and_scale_all_meshes()
    
    def norm_and_scale_all_meshes(self):
        """
        Normalize and scale all of the meshes.

        Take the average of the center of each mesh and uses it to center all of the meshes.
        Then, takes the max radius of all of the meshes (after centering) and uses it to 
        scale all of the meshes.

        Now, all of the meshes are centered and scaled jointly so the anatomical surfaces should
        roughly be aligned, removing this as a source of variation.
        """
        # get the center of all of the meshes
        centers = []
        for data in self.data:
            # center around the reference object
            xyz = data['new_pts'][self.reference_object]
            center = np.mean(xyz, axis=0)
            centers.append(center)
        centers = np.stack(centers, axis=0)
        center = np.mean(centers, axis=0).astype(np.float32)

        # subtract the center from all of the meshes
        for idx, data in enumerate(self.data):
            self.data[idx]['xyz'] -= center
            # iterate over all of the meshes and subtract the center
            for mesh_idx in range(self.n_meshes):
                self.data[idx]['new_pts'][mesh_idx] -= center
        
        # get the max radius of all of the meshes
        max_radii = 0
        for data in self.data:
            for mesh_idx in range(self.n_meshes):
                xyz = data['new_pts'][mesh_idx]
                max_radius = np.max(np.linalg.norm(xyz, axis=-1)).astype(np.float32)
                if max_radius > max_radii:
                    max_radii = max_radius
        
                
        # divide all of the meshes by the max radius
        for idx, data in enumerate(self.data):
            self.data[idx]['xyz'] /= max_radii
            # do the same for the sdf of each point
            self.data[idx]['gt_sdf'] /= max_radii
            # do the same for the original points
            for mesh_idx in range(self.n_meshes):
                self.data[idx]['new_pts'][mesh_idx] /= max_radii

        
    def preprocess_inputs(self):
        """
        Preprocess inputs to ensure they are in the correct format.
        """

        if self.scale_jointly is True:
            if self.center_pts is True:
                raise ValueError('Scale jointly assumes centering at end... so center should be False')
            if self.norm_pts is True:
                raise ValueError('Scale jointly assumes normalizing at end... so norm should be False')
    
    def get_dict_pts(self, data, pts_name):
        dict_pts = {}
        for idx_, orig_pts_ in enumerate(data[pts_name]):
            dict_pts[f'{pts_name}_{idx_}'] = orig_pts_
        return dict_pts

    def save_data_to_cache(self, data, file_hash):
        """
        Save the data to the cache.
        
        Args:
            data (dict): Dictionary of data to save
            file_hash (str): Hash of the file
        """
        # if want to cache, and new... then save. 
        filepath = os.path.join(self.cache_folder, f'{file_hash}.npz')
        dict_pts = {}
        dict_pts.update(self.get_dict_pts(data, 'orig_pts'))
        dict_pts.update(self.get_dict_pts(data, 'new_pts'))

        np.savez(filepath, pts=data['xyz'], sdfs=data['gt_sdf'], **dict_pts)

    def get_sample_data_dict(self, loc_mesh):
        """
        Given a mesh path, return a dictionary of the sampled points and SDFs.

        Args:
            loc_mesh (str): Path to mesh

        Returns:
            dict: Dictionary of sampled points and SDFs
        """

        # Create hash and filename
        file_hash = self.create_hash(loc_mesh)
        cached_file = self.find_hash(filename=f'{file_hash}.npz')

        if (len(cached_file) > 0) and (self.load_cache is True):
            # if hashed file exists, load it. 
            data_ = np.load(cached_file[0])
            data = unpack_numpy_data(data_)
            
        else:
            # otherwise, load the mesh and create SDF samples. 
            print('Creating SDF Samples')
            if  self.print_filename is True:
                print(loc_mesh)
            data = {
                'xyz': torch.zeros((self.n_pts, 3)),
                'gt_sdf': torch.zeros((self.n_pts)),
            }
            pts_idx = 0
            for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
                result_ = read_mesh_get_sampled_pts(
                    loc_mesh, 
                    mean=[0,0,0], 
                    sigma=sigma_, 
                    n_pts=n_pts_, 
                    rand_function=self.rand_function, 
                    center_pts=self.center_pts,
                    norm_pts=self.norm_pts,
                    scale_method=self.scale_method,
                    get_random=True,
                    return_orig_mesh=False,
                    return_new_mesh=False,
                    fix_mesh=self.fix_mesh,
                    register_to_mean_first=False if self.reference_mesh is None else True,
                    mean_mesh=self.reference_mesh,
                )

                if result_ is None:
                    return None
            
                xyz_ = result_['pts'] if 'pts' in result_ else result_['xyz']
                sdfs_ = result_['sdf'] if 'sdf' in result_ else result_['gt_sdf']

                data['xyz'][pts_idx:pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
                data['gt_sdf'][pts_idx:pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
                pts_idx += n_pts_

                if idx_ == 0:
                    data['orig_pts'] = torch.from_numpy(result_['orig_pts']).float()
                    data['new_pts'] = torch.from_numpy(result_['new_pts']).float()

            if self.save_cache is True:
                self.save_data_to_cache(data, file_hash)

        pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
        data['pos_idx'] = pos_idx
        data['neg_idx'] = neg_idx
        data['surf_idx'] = surf_idx

        return data
    
    def get_pt_sample_combos(self):
        """
        Get the combinations of points and sigmas to sample from each mesh.

        Returns:
            list: List of lists of [n_pts, sigma] to sample
        """

        n_p_near_surface = int(self.n_pts * self.p_near_surface)
        n_p_further_from_surface = int(self.n_pts * self.p_further_from_surface)
        n_p_random = self.n_pts - n_p_near_surface - n_p_further_from_surface

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, None]
        ]

        return pt_sample_combos
    
    def sdf_pos_neg_idx(self, data):
        """
        Get the indices of the positive, negative, and surface SDFs.
        
        Args:
            data (dict): Dictionary of sampled points and SDFs
        
        Returns:
            tuple: (pos_idx, neg_idx, surf_idx) of indices
        """

        pos_idx = (data['gt_sdf'] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data['gt_sdf'] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data['gt_sdf'] == 0).nonzero(as_tuple=True)[0]

        # Repeat +/- indices if either of them does not have enough for a full batch. 
        samples_per_sign = int(self.subsample/2)
        pos_idx = pos_idx.repeat(samples_per_sign//pos_idx.size(0) + 1)
        neg_idx = neg_idx.repeat(samples_per_sign//neg_idx.size(0) + 1)
        
        return pos_idx, neg_idx, surf_idx
        
    def find_hash(self, filename='hashed_filename.npz'):
        """
        Find the hashed filename in the cache.
        
        Args:
            filename (str, optional): Hashed filename. Defaults to 'hashed_filename.npz'.
        
        Returns:
            list: List of paths to the hashed filename
        """

        files = []
        for p, d, f in os.walk(self.loc_save):
            for filename_ in f:
                if filename_ == filename:
                    files.append(os.path.join(p, filename_))
                    print('File found in cache:', files[-1])
                    return files
                    
        return files

    def load_reference_mesh(self):
        """
        Load the reference mesh.
        
        Raises:
            TypeError: If reference mesh is not a string, list of strings, or mesh.Mesh object
        """

        if issubclass(type(self.reference_mesh), mskt.mesh.Mesh):
            pass
        elif isinstance(self.reference_mesh, int):
            if isinstance(self.list_mesh_paths[0], (str, mskt.mesh.Mesh)):
                mesh = self.list_mesh_paths[self.reference_mesh]
            elif isinstance(self.list_mesh_paths[0], (list, tuple)):
                
                mesh = self.list_mesh_paths[self.reference_mesh][self.mesh_to_scale]
            else:
                raise TypeError('provided list_meshes wrong type')
            self.reference_mesh = mskt.mesh.Mesh(mesh)
        elif isinstance(self.reference_mesh, str):
            self.reference_mesh = mskt.mesh.Mesh(self.reference_mesh)
        elif isinstance(self.reference_mesh, list):
            # below will throw error in SDFSamples, but will work in MultiSurfaceSDFSamples
            # where self.mesh_to_scale is defined & a list/tuple type likely
            # TODO: Why is reference_object different from mesh_to_scale?
            self.reference_mesh = mskt.mesh.Mesh(self.reference_mesh[self.reference_object])
        else:
            raise TypeError('Reference mesh must be a string, list of strings, or mesh.Mesh object, not', type(self.reference_mesh))

    def get_hash_params(self):
        """
        Get the parameters to hash for saving/loading the cache.
        
        Returns:
            list: List of parameters to hash
        """

        list_hash_params = [
            self.n_pts,
            self.p_near_surface, self.sigma_near,
            self.p_further_from_surface, self.sigma_far,
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.reference_mesh,
            self.fix_mesh,
            self.scale_jointly
        ]

        return list_hash_params
    
    def create_hash(self, loc_mesh):
        """
        Create the hash for the cache.
        
        Args:
            loc_mesh (str): Path to mesh
            
        Returns:
            str: Hashed string
        """

        list_hash_params = self.list_hash_params.copy()
        if isinstance(loc_mesh, str):
            list_hash_params.insert(0, loc_mesh)
        elif isinstance(loc_mesh, (list, tuple)):
            for path in loc_mesh:
                if self.verbose is True:
                    print(loc_mesh)
                list_hash_params.insert(0, path)
    
        if (self.include_seed_in_hash is True) and (self.random_seed is not None):
            list_hash_params.append(self.random_seed) # random seed state
        if self.verbose is True:
            print('List Params', list_hash_params)
        list_hash_params = [str(x) for x in list_hash_params]
        file_params_string = '_'.join(list_hash_params)
        hash_str = hashlib.md5(file_params_string.encode()).hexdigest()
        return hash_str
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Length of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the item at the index.
        
        Args:
            idx (int): Index of the item
        
        Returns:
            dict: Dictionary of the item
            idx (int): Index of the item
        """
        data_ = self.data[idx]
        if self.subsample is not None:
            if self.equal_pos_neg is True:
                samples_per_sign = int(self.subsample/2)
                
                # idx_pos = data_['pos_idx'].repeat(data_['pos_idx'].size(0)//samples_per_sign + 1)
                # perm_pos = torch.randperm(idx_pos.size(0))
                perm_pos = torch.randperm(data_['pos_idx'].size(0))[:samples_per_sign]
                idx_pos = data_['pos_idx'][perm_pos]

                # idx_neg = data_['neg_idx'].repeat(data_['neg_idx'].size(0)//samples_per_sign + 1)
                # perm_neg = torch.randperm(idx_neg.size(0))
                # idx_neg = perm_neg[:samples_per_sign]
                perm_neg = torch.randperm(data_['neg_idx'].size(0))[:samples_per_sign]
                idx_neg = data_['neg_idx'][perm_neg]

                idx_ = torch.cat((idx_pos, idx_neg), dim=0)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
            
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[:self.subsample]

            data_ = {
                'xyz': data_['xyz'][idx_, :],
                'gt_sdf': data_['gt_sdf'][idx_],
            }
        
        return data_, idx
    
class MultiSurfaceSDFSamples(SDFSamples):
    def __init__(
        self,
        list_mesh_paths,
        subsample=None,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        axis_align=False,
        norm_pts=False,
        scale_method='max_rad',
        scale_jointly=False,
        loc_save=os.environ['LOC_SDF_CACHE'],
        include_seed_in_hash=True,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,    
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,

        # Multi surface specific 
        scale_all_meshes=True,                  
        center_all_meshes=False,                
        mesh_to_scale=0,
        reference_object=0,                   
    ):
        # Multi surface specific
        self.mesh_to_scale = mesh_to_scale
        self.total_n_pts = sum(n_pts)
        self.scale_all_meshes = scale_all_meshes
        self.center_all_meshes = center_all_meshes
        self.n_meshes = len(list_mesh_paths[0])
        self.reference_object = reference_object

        super().__init__(
            list_mesh_paths=list_mesh_paths,
            subsample=subsample,
            n_pts=n_pts,
            p_near_surface=p_near_surface,
            p_further_from_surface=p_further_from_surface,
            sigma_near=sigma_near,
            sigma_far=sigma_far,
            rand_function=rand_function,
            center_pts=center_pts,
            axis_align=axis_align,
            norm_pts=norm_pts,
            scale_method=scale_method,
            scale_jointly=scale_jointly,
            loc_save=loc_save,
            include_seed_in_hash=include_seed_in_hash,
            save_cache=save_cache,
            load_cache=load_cache,
            random_seed=random_seed,
            reference_mesh=reference_mesh,
            verbose=verbose,
            equal_pos_neg=equal_pos_neg,
            fix_mesh=fix_mesh,
            print_filename=print_filename,
        )
    
    def preprocess_inputs(self):
        super().preprocess_inputs()

        if isinstance(self.list_mesh_paths[0], (list, tuple)):
            self.n_meshes = len(self.list_mesh_paths[0])
        elif isinstance(self.list_mesh_paths[0], (str, mskt.mesh.Mesh)):
            self.n_meshes = len(self.list_mesh_paths)
        
        if not isinstance(self.p_near_surface, (list, int)):
            self.p_near_surface = [self.p_near_surface] * self.n_meshes
        if not isinstance(self.p_further_from_surface, (list, int)):
            self.p_further_from_surface = [self.p_further_from_surface] * self.n_meshes
        if not isinstance(self.sigma_near, (list, int)):
            self.sigma_near = [self.sigma_near] * self.n_meshes
        if not isinstance(self.sigma_far, (list, int)):
            self.sigma_far = [self.sigma_far] * self.n_meshes
        if not isinstance(self.n_pts, (list, int)):
            self.n_pts = [self.n_pts] * self.n_meshes
    
    def get_sample_data_dict(self, loc_meshes):
        if type(loc_meshes) not in (tuple, list):
            loc_meshes = [loc_meshes]
        
        # Create hash and filename 
        file_hash = self.create_hash(loc_meshes)
        cached_file = self.find_hash(filename=f'{file_hash}.npz')
        
        if (len(cached_file) > 0) and (self.load_cache is True):
            # if hashed file exists, load it. 
            data_ = np.load(cached_file[0])
            data = unpack_numpy_data(data_)
            
        else:
            # otherwise, load the mesh and create SDF samples. 
            print('Creating SDF Samples')
            if self.print_filename is True:
                print(loc_meshes)

            data = {
                'xyz': torch.zeros((sum(self.n_pts), 3)),
                'gt_sdf': torch.zeros((sum(self.n_pts), len(loc_meshes))),
            }
            pts_idx = 0
            icp_transform = None
            for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
                tic = time.time()
                result_ = read_meshes_get_sampled_pts(
                    loc_meshes, 
                    mean=[0,0,0], 
                    sigma=sigma_, 
                    n_pts=n_pts_, 
                    rand_function=self.rand_function, 
                    center_pts=self.center_pts,
                    norm_pts=self.norm_pts,
                    scale_method=self.scale_method,
                    get_random=True,
                    fix_mesh=self.fix_mesh,
                    register_to_mean_first=False if self.reference_mesh is None else True,  #
                    mean_mesh=self.reference_mesh,  # 
                    
                    # Multi surface specific
                    mesh_to_scale=self.mesh_to_scale,
                    scale_all_meshes=self.scale_all_meshes,
                    center_all_meshes=self.center_all_meshes,

                    icp_transform=icp_transform,
                )

                if result_ is None:
                    return None
                
                icp_transform = result_['icp_transform']

                toc = time.time()
                print(f'{idx_} - {sigma_}: {toc - tic}s')
                
                if idx_ == 0:
                    data['orig_pts'] = result_['orig_pts']
                    data['new_pts'] = result_['new_pts']
                
                xyz_ = result_['pts'] if 'pts' in result_ else result_['xyz']
                sdfs_ = result_['sdf'] if 'sdf' in result_ else result_['gt_sdf']

                data['xyz'][pts_idx:pts_idx + sum(n_pts_), :] = torch.from_numpy(xyz_).float()

                for mesh_idx, _sdfs_ in enumerate(sdfs_):
                    data['gt_sdf'][pts_idx:pts_idx + sum(n_pts_), mesh_idx] = torch.from_numpy(_sdfs_).float()
                pts_idx += sum(n_pts_)
            
            if (data is not None) and (self.save_cache is True):
                self.save_data_to_cache(data, file_hash)
        
        # Drop points that have are labeled as being inside
        # 2 objects - clearly this is an error. 
        if data['gt_sdf'].shape[1] == 2:
            sdf_ = data['gt_sdf'].clone()
            sdf_[sdf_ < 0] = -1
            sdf_[sdf_ > 0] = 1
            sdf_[sdf_ == 0] = 0
            total = torch.sum(sdf_, axis=1)

            out_out = torch.sum(total == 2)
            out_in = torch.sum(total == 0)
            in_in = torch.sum(total == -2)

            # Drop overlapping points
            data['gt_sdf'] = data['gt_sdf'][total != -2, :]
            data['xyz'] = data['xyz'][total != -2, :]


            if self.verbose is True:
                print('total shape', total.shape)
                print('total', total)
                print('out_out', out_out)
                print('out_in', out_in)
                print('in_in', in_in)

        if self.equal_pos_neg is True:
            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data['pos_idx'] = pos_idx
            data['neg_idx'] = neg_idx
            data['surf_idx'] = surf_idx

        return data

    def get_pt_sample_combos(self):
        n_p_near_surface = [int(n_pts_ * p_near) for n_pts_, p_near in zip(self.n_pts, self.p_near_surface)]
        n_p_further_from_surface = [int(n_pts_ * p_far) for n_pts_, p_far in zip(self.n_pts, self.p_further_from_surface)]
        n_p_random = [n_pts_ - n_p_near - n_p_far for n_pts_, n_p_near, n_p_far in zip(self.n_pts, n_p_near_surface, n_p_further_from_surface)]

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, [None,] * self.n_meshes]
        ]

        return pt_sample_combos

    def get_hash_params(self):
        list_hash_params = [
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.scale_all_meshes,
            self.center_all_meshes,
            self.reference_mesh,
            self.reference_object,
            False,
            self.fix_mesh,
            self.scale_jointly
        ]

        for n_pts_ in self.n_pts:
            list_hash_params.append(n_pts_)
        for p_near in self.p_near_surface:
            list_hash_params.append(p_near)
        for p_far in self.p_further_from_surface:
            list_hash_params.append(p_far)
        for sigma_near in self.sigma_near:
            list_hash_params.append(sigma_near)
        for sigma_far in self.sigma_far:
            list_hash_params.append(sigma_far)
        
        return list_hash_params


    def sdf_pos_neg_idx(self, data):
        '''
        - iterate over each mesh
        - get number of points for that mesh and get: 
            - points positive (outside mesh)
            - points negative (inside mesh)
        - return list of indices
        '''

        self.samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]

        pos_idx = []
        neg_idx = []
        surf_idx = []
        pts_idx_ = 0
        if self.verbose is True:
            print('data', data['xyz'].shape, data['gt_sdf'].shape)
        
        self.samples_per_sign = []
        for mesh_idx, subsample_ in enumerate(self.samples_per_mesh):
            
            samples_per_sign = int(subsample_/2)
            print(samples_per_sign)
            self.samples_per_sign.append(samples_per_sign)

            # BELOW NEEDS LOGIC TO UNPACK  1/2 pos/neg pts for each mesh
            # mesh_sdfs = data['gt_sdf'][pts_idx_:pts_idx_ + n_pts_, mesh_idx]
            mesh_sdfs = data['gt_sdf'][:, mesh_idx].clone()
            pos_idx_ = (mesh_sdfs > 0).nonzero(as_tuple=True)[0] #+ pts_idx_
            neg_idx_ = (mesh_sdfs < 0).nonzero(as_tuple=True)[0] #+ pts_idx_
            surf_idx_ = (mesh_sdfs == 0).nonzero(as_tuple=True)[0] #+ pts_idx_

            # print('pre repeat:')
            # print('pos_idx_', pos_idx_.shape)
            # print('neg_idx_', neg_idx_.shape)

            # Repeat +/- indices if either of them does not have enough for a full batch. 
            pos_idx_ = pos_idx_.repeat(samples_per_sign//pos_idx_.size(0) + 1)
            neg_idx_ = neg_idx_.repeat(samples_per_sign//neg_idx_.size(0) + 1)

            # print('pos_idx_', pos_idx_.shape)
            # print('neg_idx_', neg_idx_.shape)

            pos_idx.append(pos_idx_)
            neg_idx.append(neg_idx_)
            surf_idx.append(surf_idx_)
        
        return pos_idx, neg_idx, surf_idx

    def __getitem__(self, idx):
        #TODO: get rid of pts_array from above
        # replace with self.data[idx] = [pts, sdfs_]
        # this will simplify everything downstream because this is something that we are
        # constantly undoing/re-doing elsewhere in the code - and it even lookts like
        # the sdfs/pts are stroed separately in the npy files. 
        data_ = self.data[idx]
        if self.subsample is not None:
            if self.equal_pos_neg is True:
                # get number of points for each mesh
                # this is weighted by the number of points in the mesh 
                # relative to the total number of points in the dataset
                # samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]
                idx_ = []
                for mesh_idx, samples_per_sign in enumerate(self.samples_per_sign):
                    # get number of positive and negative points for this mesh
                    # samples_per_sign = int(subsample_/2)
                    if self.verbose is True:
                        print('samples_per_sign', samples_per_sign)
                    
                    if samples_per_sign == 0:
                        continue
                    # get random indices for positive and negative points
                    # idx_pos = data_['pos_idx'][mesh_idx].repeat(data_['pos_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_pos = torch.randperm(data_['pos_idx'][mesh_idx].size(0))[:samples_per_sign]
                    idx_pos = data_['pos_idx'][mesh_idx][perm_pos]

                    # idx_neg = data_['neg_idx'][mesh_idx].repeat(data_['neg_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_neg = torch.randperm(data_['neg_idx'][mesh_idx].size(0))[:samples_per_sign]
                    idx_neg = data_['neg_idx'][mesh_idx][perm_neg]

                    # combine positive and negative indices
                    idx_ += [idx_pos, idx_neg]
                    
                
                # combine indices for all meshes
                idx_ = torch.cat(idx_, dim=0)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[: self.subsample]
            
            if self.verbose is True:
                print('idx_ size:', idx_.size(), 'idx_ min:', idx_.min(),'idx_ max:', idx_.max())
                print('equal neg pos', self.equal_pos_neg)

            data_ = {
                'xyz': data_['xyz'][idx_, :],
                'gt_sdf': data_['gt_sdf'][idx_, :],
            }
        
        return data_, idx
