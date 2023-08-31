from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import torch

ROTATION_RANGE = 1.0


def vtk_to_torch(mesh, edges=True, remove_faces=False):
    r"""Converts a :obj:`vtk.vtkPolydata` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (vtk.vtkPolydata): A :obj:`vtkPolydata` mesh.
        edges (bool, optional): If set to :obj:`True`, create
            graph edges from faces. (default: :obj:`True`)
        remove_faces (bool, optional): If set to :obj:`True`,
            and :obj:`edges` is :obj:`True` then remove graph
            faces after computing edges. Otherwise, if 
            :obj:`edges` is :obj:`True` and :obj:`remove_faces`
            is set to :obj:`False` then leave both faces and
            edges. (default: :obj:`False`)
    .. note::
        :obj:`vtkPolydata` mesh must be a triangular mesh. 
        Current implementation assumes trian
    """
    faces = vtk_to_numpy(mesh.GetPolys().GetData())
    faces = faces.reshape(-1, 4)
    faces = np.delete(faces, 0, 1)
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    
    points = torch.from_numpy(points).to(torch.float)
    faces = torch.from_numpy(faces).contiguous()

    return points, faces


def setup_scheduler(config, optimizer):
    if config['scheduler'] == 'cosine_anneal':
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['Tmax'], 
            eta_min=config['lr_min'], 
            last_epoch=-1, 
            verbose=False
        )
    elif config['scheduler'] == 'cosine_anneal_warm_restarts':
        scheduler_ = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config['Tmax'], 
            T_mult=config['Tmult'], 
            eta_min=config['lr_min'], 
            last_epoch=- 1, 
            verbose=False
        )
    elif config['scheduler'] == 'cyclic_lr':
        scheduler_ = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=config['lr'], 
            max_lr=config['lr_max'], 
            step_size_up=config['cyclic_lr_step_up'], 
            step_size_down=None, 
            mode=config['cyclic_lr_code'], 
            gamma=1.0, 
            scale_fn=None, 
            scale_mode='cycle', 
            cycle_momentum=True, 
            base_momentum=0.8, 
            max_momentum=0.9, 
            last_epoch=- 1, 
            verbose=False
        )
    elif config['scheduler'] == 'one_cycle_lr':
        scheduler_ = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config['lr_max'], 
            total_steps=config['one_cycle_lr_total_steps'], 
            epochs=None, 
            steps_per_epoch=None, 
            pct_start=0.3, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95, 
            div_factor=25.0, 
            final_div_factor=10000.0, 
            three_phase=False, 
            last_epoch=- 1, 
            verbose=False
        )
    else:
        return None
    
    return scheduler_

def setup_optimizer(config, model):
    if config['optimizer'] == 'adam':
        optimizer_ = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2_norm'])
    elif config['optimizer'] == 'sgd':
        optimizer_ = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['l2_norm'])
    
    return optimizer_

def random_scale(pos, min_=0.1, max_=10, center_first=False, log=True):

    if min_ > max_:
        raise ArithmeticError(f'minimum random scale should be less than maximum. Min is : {min_}, max is: {max_}')

    centroid = torch.mean(pos, dim=-2, keepdim=True)
    if center_first is True:
        pos = (pos - centroid)

    rand_sample = torch.rand(1, device=pos.device, dtype=pos.dtype)

    if log is True:
        min_log = torch.log(min_)
        max_log = torch.log(max_)
        range_log = max_log - min_log
        rand_scale = torch.exp(min_log + rand_sample * range_log)
    elif log is False:
        range_ = max_ - min_
        rand_scale = min_ + rand_sample * range_

    pos = pos * rand_scale

    if center_first is True:
        pos = pos + centroid
    
    return pos



def normalize_mesh(pos, method='mean', scale_method='max_rad', ref_scale=None, center_only=False):
    # center and unit-scale positions
        
    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if center_only is False:
        if ref_scale is not None:
            pos = pos / ref_scale
        elif scale_method == 'max_rad':
            scale = torch.max(torch.norm(pos, dim=-1), dim=-1, keepdim=True).values.unsqueeze(-1)
            pos = pos / scale
        elif scale_method == 'area': 
            if faces is None:
                raise ValueError("must pass faces for area normalization")
            coords = pos[faces]
            vec_A = coords[:, 1, :] - coords[:, 0, :]
            vec_B = coords[:, 2, :] - coords[:, 0, :]
            face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
            total_area = torch.sum(face_areas)
            scale = (1. / torch.sqrt(total_area))
            pos = pos * scale
        else:
            raise ValueError("unrecognized scale method")
    else:
        pass
    return pos


# BELOW WORKS BUT IS TOO SLOW - 12 seconds per iteration. 
# def assd(points_1, points_2, n_rand_pts=1000):
#     if (len(points_1.shape) != 2) & (len(points_1.shape) != 2):
#         raise Exception('points must be 2 dimensional and are: points_1 {points_1.shape} & points_2 {points_2.shape}')
    
#     min_dist_1 = 0
#     min_dist_2 = 0

#     for pt_idx in range(points_1.shape[0]):
#         min_dist_1 += torch.min(torch.sqrt(torch.sum(torch.square(points_1[pt_idx, :] - points_2), dim=-1)))
    
#     for pt_idx in range(points_2.shape[0]):
#         min_dist_2 += torch.min(torch.sqrt(torch.sum(torch.square(points_2[pt_idx, :] - points_1), dim=-1)))

#     assd = (min_dist_1 + min_dist_2) / (points_1.shape[0] + points_2.shape[0])

#     return assd

def __assd__(points_1, points_2):
    if len(points_1.shape) == len(points_2.shape):
        points_1 = torch.unsqueeze(points_1, dim=-2)
    dist = torch.sqrt(torch.sum(torch.square(points_1 - points_2), dim=-1))
    min_1, _ = torch.min(dist, dim=0)
    min_2, _ = torch.min(dist, dim=1)
    assd_ = (torch.sum(min_1) + torch.sum(min_2)) / (len(min_1) + len(min_2))

    return assd_

def sum_min_distances(points_1, points_2):
    """
    Sum of the minimum distances from each point in points_1 to all points in points_2
        - Run the same function twice, with the point sets order switched then you will have minimum
        distances in both ways. 
    """
    mins, _ = torch.min(torch.sqrt(torch.sum(torch.square(points_1[:, None, :] - points_2), dim=-1)), axis=-1)
    return torch.sum(mins)

def get_min_distances(points_1, points_2, pts_per_block):
    n_full_blocks = points_1.shape[0] // pts_per_block
    min_dist = 0
    for block_idx in range(n_full_blocks):
        points_1_ = points_1[block_idx*pts_per_block:(block_idx+1)*pts_per_block, :]
        min_dist += sum_min_distances(points_1_, points_2)
    
    if points_1.shape[0] % pts_per_block != 0:
        last_block_size = points_1.shape[0] % pts_per_block
        points_1_ = points_1[-last_block_size:, :]
        min_dist += sum_min_distances(points_1_, points_2)
    return min_dist


def assd(points_1, points_2, rand_pts=False, n_pts=100):
    if len(points_1.shape) == 3:
        points_1 = points_1[0,:,:]
    if len(points_2.shape) == 3:
        points_2 = points_2[0,:,:]
        
    if (len(points_1.shape) != 2) & (len(points_1.shape) != 2):
        raise Exception(f'points must be 2 dimensional and are: points_1 {points_1.shape} & points_2 {points_2.shape}')
    
    if rand_pts is True:
        rand_1 = torch.randint(low=0, high=points_1.shape[0], size=(n_pts,))
        rand_2 = torch.randint(low=0, high=points_2.shape[0], size=(n_pts,))

        points_1 = points_1[rand_1, :]
        points_2 = points_2[rand_2, :]

        assd_ = __assd__(points_1, points_2)
    else:
        min_dist_1 = get_min_distances(points_1, points_2, n_pts)
        min_dist_2 = get_min_distances(points_2, points_1, n_pts)

        assd_ = (min_dist_1 + min_dist_2) / (points_1.shape[0] + points_2.shape[0])

    return assd_
    

def update_config_param(config, param, epoch):
    initial_value = config[f'{param}_initial_value']
    if f'{param}_secondary_value' in config:
        secondary_value = config[f'{param}_secondary_value']

    # DEFINE PARAMETERS FOR WARMUP
    if f'{param}_warmup' in config:
        warmup = config[f'{param}_warmup']
        if warmup is True:
            warmup_patience = config[f'{param}_warmup_patience']
    else:
        warmup = False
    
    # DEFINE PARAMETERS FOR COOLDOWN
    if f'{param}_cooldown' in config:
        cooldown = config[f'{param}_cooldown']
        if cooldown is True:
            cooldown_patience = config[f'{param}_cooldown_patience']
            cooldown_value = config[f'{param}_cooldown_value']
            cooldown_start = config['n_epochs'] - cooldown_patience
    else:
        cooldown = False

    # DEFINE PARAMETERS FOR CYCLIC_ANNEAL_LINEAR
    if f'{param}_cyclic_anneal_linear' in config:
        cyclic_anneal_linear = config[f'{param}_cyclic_anneal_linear']
        cyclic_anneal_linear_epochs = config[f'{param}_cyclic_anneal_linear_epochs']
        cyclic_anneal_linear_proportion_increasing = config[f'{param}_cyclic_anneal_linear_prop_increase']
        cyclic_anneal_linear_proportion_plateau_after = config[f'{param}_cyclic_anneal_linear_prop_plateau_after']
        cyclic_anneal_linear_n_steps_increasing = int(cyclic_anneal_linear_epochs * cyclic_anneal_linear_proportion_increasing)
        cyclic_anneal_linear_n_steps_plateau = cyclic_anneal_linear_epochs - cyclic_anneal_linear_n_steps_increasing
        cyclic_anneal_linear_n_steps_plateau_after = int(cyclic_anneal_linear_proportion_plateau_after * cyclic_anneal_linear_n_steps_plateau)
        cyclic_anneal_linear_n_steps_plateau_before = cyclic_anneal_linear_n_steps_plateau - cyclic_anneal_linear_n_steps_plateau_after 
        cyclic_anneal_linear_step_size = (secondary_value - initial_value) / cyclic_anneal_linear_n_steps_increasing
    else:
        cyclic_anneal_linear = False
    
    # TEST IF IN WARMUP
    if warmup is True:
        # if within the warmup period still then skip the rest
        if epoch < warmup_patience:
            return initial_value
    
    # TEST IF IN COOLDOWN
    if cooldown is True:
        # if within the cooldown period then skip the rest
        if epoch > cooldown_start:
            return cooldown_value

    # LOGIC FOR THE LINEAR CYCLIC ANNEALING (IF WE ARE DOING IT)
    if cyclic_anneal_linear is True:
        # if warmup then adjust the epoch to pretend like the end
        # of warmup is the first epoch
        if warmup is True:
            epoch_ = epoch - warmup_patience
        else:
            epoch_ = epoch
        # figure out how many steps into this particular cycle we are
        steps_into_cycle = epoch_ % cyclic_anneal_linear_epochs

        if steps_into_cycle < cyclic_anneal_linear_n_steps_plateau_before:
            return initial_value
        
        if steps_into_cycle > cyclic_anneal_linear_n_steps_increasing + cyclic_anneal_linear_n_steps_plateau_before:
            # if into the plateau section of cycle then just return the secondary value
            return secondary_value
        else:
            # otherwise, calculate the value based on the initial value
            # the step size and how many steps we are into this cycle
            return initial_value + (cyclic_anneal_linear_step_size * (steps_into_cycle-cyclic_anneal_linear_n_steps_plateau_before))

    else:
        # If we get to here it means that we arent using any of
        # the other scheduler steps, so return the original param value. 
        return config[param]   


def l1_l2_reg(model, order=2, norm=None):
    '''
    Should use weight_decay option in optimizer instead. 
    '''
    for W in model.parameters():
        if norm is None:
            # makes l2_reg be the correct torch type
            norm = W.norm(order)
        else:
            norm += W.norm(order)
    
    return norm


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None, rotation_range_degrees=None, rotation_range_percent=None):
    R = random_rotation_matrix(randgen, rotation_range_degrees=rotation_range_degrees, rotation_range_percent=rotation_range_percent) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotation_matrix(randgen=None, rotation_range_degrees=None, rotation_range_percent=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if rotation_range_degrees is not None:
        rotation_range = np.abs(rotation_range_degrees/360)
    elif rotation_range_percent is not None:
        rotation_range = np.abs(rotation_range_percent)
    else:
        rotation_range = ROTATION_RANGE

    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi * rotation_range  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0  * rotation_range # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

