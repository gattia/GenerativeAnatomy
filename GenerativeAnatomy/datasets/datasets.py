from torch.utils.data import Dataset
import pymskt as mskt
from GenerativeAnatomy.utils import vtk_to_torch, normalize_mesh
import diffusion_net
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

OP_CACHE_DIR= os.getenv('OP_CACHE_DIR')

class Bone(Dataset):
    """
    This is based off of the data-set loader by nick sharp in diffusion-net
    """

    def __init__(
        self, 
        list_meshfiles, 
        list_outcomes, 
        k_eig, 
        op_cache_dir=None, 
        normalize=False,
        center_only=False, 
        path_ref_mesh=None, 
        norm_pts=True, 
        mean_pts=None, 
        std_pts=None):
        
        self.list_meshfiles = list_meshfiles
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.normalize = normalize
        self.center_only = center_only

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.outcomes_list = []

        if (self.normalize is True) & (norm_pts is True):
            raise Exception('It is currently not advised to normalize (to unit sphere) & whitten points (subtract mean divide by STD) for each point individually')

        if path_ref_mesh is not None:
            self.ref_mesh = mskt.mesh.io.read_vtk(path_ref_mesh)
            self.ref_verts, self.ref_faces = vtk_to_torch(self.ref_mesh)
            self.ref_scale = torch.max(torch.norm(self.ref_verts, dim=-1), dim=-1, keepdim=True).values.unsqueeze(-1)
        else:
            self.ref_scale = None
        
        for idx, filepath in enumerate(list_meshfiles):
            mesh = mskt.mesh.io.read_vtk(filepath)
            verts, faces = vtk_to_torch(mesh)
            # center and unit scale
            if self.normalize is True:
                verts = normalize_mesh(verts, ref_scale=self.ref_scale, center_only=self.center_only)
            
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.outcomes_list.append(list_outcomes[idx])

        for idx, outcome in enumerate(self.outcomes_list):
            self.outcomes_list[idx] = torch.tensor(outcome)
        
        if mean_pts is not None:
            self.mean_verts = mean_pts
        else:
            self.mean_verts = torch.mean(torch.stack(self.verts_list), dim=0)
        
        if std_pts is not None:
            self.std_verts = std_pts
        else:
            self.std_verts = torch.std(torch.stack(self.verts_list), dim=0, unbiased=True)
        
        if norm_pts is True:
            for idx, verts in enumerate(self.verts_list):
                verts = verts - self.mean_verts
                verts = verts / self.std_verts
                self.verts_list[idx] = verts
            
    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        verts = self.verts_list[idx]
        faces = self.faces_list[idx]
        outcome = self.outcomes_list[idx]
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, outcome
    

def load_data(
    df_dataset,
    fraction_test=0.2,
    op_cache_dir=OP_CACHE_DIR,
    k_eig=128,
    equal_sexes_test=True,
    sexes_normalized_train_test=False,
    normalize_meshes=False,
    center_only=False,
    df_outcome_column_name='sex_1_M_2_F',
    df_mesh_path_column_name='filepath',
    sampling_sequential=True,
    updated_generic_mesh_name=None,
    path_ref_mesh=None,
    return_mean_shape=False,
    return_faces=True,
    whitten_inputs=False,
    batch_size=1,
):
    if df_outcome_column_name == 'sex_1_M_2_F':
        outcome = np.asarray(df_dataset[df_outcome_column_name], dtype=int) - 1
    elif df_outcome_column_name == 'age_y':
        outcome = np.asarray(df_dataset[df_outcome_column_name], dtype=float)
    elif df_outcome_column_name == 'kl':
        outcome = np.asarray(df_dataset[df_outcome_column_name], dtype=int)

    if updated_generic_mesh_name is not None:
        df_dataset[df_mesh_path_column_name] = df_dataset[df_mesh_path_column_name].apply(lambda x: os.path.join(x[:x.rfind('/')], updated_generic_mesh_name))
    
    n_test = int(fraction_test * len(df_dataset))
    if (df_outcome_column_name == 'sex_1_M_2_F') & ((equal_sexes_test is True) or (sexes_normalized_train_test is True)) & (sampling_sequential is True):
        if equal_sexes_test is True:
            n_test_male = n_test//2
            n_test_female = n_test_male # n_test - n_test_male -> better keep sexes same, vs. using specified fraction.

            # Getting male indices
            male_indices = np.where(outcome == 0)[0]
            test_indices_male = male_indices[:n_test_male]
            train_indices_male = male_indices[n_test_male:]

            # Getting female indices
            female_indices = np.where(outcome == 1)[0]
            test_indices_female = female_indices[:n_test_female]
            train_indices_female = female_indices[n_test_female:]

            # Getting test/train indices
            test_indices = test_indices_female.tolist() + test_indices_male.tolist()
            train_indices = train_indices_female.tolist() + train_indices_male.tolist()
        elif sexes_normalized_train_test is True:
            male_indices = np.where(outcome == 0)[0]
            n_male = len(male_indices)
            n_test_male = int(fraction_test * n_male)
            test_indices_male = male_indices[:n_test_male]
            train_indices_male = male_indices[n_test_male]

            female_indices = np.where(outcome == 1)[0]
            n_female = len(female_indices)
            n_test_female = int(fraction_test * n_female)
            test_indices_female = female_indices[:n_test_female]
            train_indices_female = female_indices[n_test_female]

            # Getting test/train indices
            test_indices = test_indices_female.tolist() + test_indices_male.tolist()
            train_indices = train_indices_female.tolist() + train_indices_male.tolist()
    elif sampling_sequential is False:
        raise NotImplementedError('Currently only sequential sampling is used for development/consistency during training')
    else:
        indices = np.arange(len(df_dataset), dtype=int)
        train_indices = indices[:n_test]
        test_indices = indices[n_test:]
    
    train_bone_dataset = Bone(
        list_meshfiles=df_dataset[df_mesh_path_column_name][train_indices],
        list_outcomes=outcome[train_indices],
        k_eig=k_eig,
        op_cache_dir=op_cache_dir,
        normalize=normalize_meshes,
        center_only=center_only,
        path_ref_mesh=path_ref_mesh,
        norm_pts=whitten_inputs,
    )
    
    test_bone_dataset = Bone(
        list_meshfiles=df_dataset[df_mesh_path_column_name][test_indices],
        list_outcomes=outcome[test_indices],
        k_eig=k_eig,
        op_cache_dir=op_cache_dir,
        normalize=normalize_meshes,
        center_only=center_only,
        path_ref_mesh=path_ref_mesh,
        norm_pts=whitten_inputs,
        mean_pts=train_bone_dataset.mean_verts,
        std_pts=train_bone_dataset.std_verts
    )

    train_loader = DataLoader(train_bone_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_bone_dataset, batch_size=batch_size, shuffle=False)

    dict_return = {
        'train_loader': train_loader,
        'test_loader': test_loader,
    }
    if return_mean_shape is True:
        dict_return['mean_train_shape'] = train_bone_dataset.mean_verts
    if return_faces is True:
        dict_return['faces_train_shape'] = train_bone_dataset.faces_list[0]

    return dict_return
    # if return_mean_shape is False:
    #     return train_loader, test_loader
    # elif return_mean_shape is True:
    #     return train_loader, test_loader, train_bone_dataset.mean_verts
