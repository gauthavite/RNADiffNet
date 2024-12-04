import os
import numpy as np
import torch

from torch.utils.data import Dataset
from diffusion_utils import normalize_positions, get_all_operators, read_mesh, get_operators

import pymeshlab

class RNAMeshDataset(Dataset):
    """RNA Mesh Dataset
    """

    def __init__(self, root_dir, train, num_eig, op_cache_dir=None, target_num_faces=None, point_cloud=False, num_points=None):
        # , n_classes=260):
        
        """
            root_dir (string): Directory with all the meshes.
            train (bool): If True, use the training set, else use the test set.
            num_eig (int): Number of eigenvalues to use.
            op_cache_dir (string): Directory to cache the operators.
            n_classes (int): Number of classes.
            target_num_faces (int, optional): If specified, downsample the mesh to have approximately this number of faces.
            point_cloud (bool, optional): If specified, remove the faces.
            point_cloud (bool, optional): If specified, choose num_points random points.
        """

        self.train = train  # bool
        self.root_dir = root_dir
        self.num_eig = num_eig
        self.op_cache_dir = op_cache_dir
        self.target_num_faces = target_num_faces
        self.point_cloud = point_cloud
        if self.point_cloud and (target_num_faces is not None):
            raise ValueError("Cannot have arguments point_cloud=True and target_num_faces!=None at the same time")
        
        elif self.point_cloud and (self.op_cache_dir is not None):
            self.op_cache_dir = os.path.join(op_cache_dir, f"point_cloud")
            os.makedirs(self.op_cache_dir, exist_ok=True)

        elif (target_num_faces is not None) and (self.op_cache_dir is not None):
            self.op_cache_dir = os.path.join(op_cache_dir, f"downsampled_{target_num_faces}")
            os.makedirs(self.op_cache_dir, exist_ok=True)

        if (not point_cloud) and (num_points is not None):
            raise ValueError("Cannot have point_cloud=False and num_points!=None")
        # self.n_class = n_classes # (includes -1)
        
        if self.op_cache_dir is not None:
            os.makedirs(self.op_cache_dir, exist_ok=True)
            
        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list_og = []  # per-vertex
        self.labels_list = []  # per-vertex new indices

        label_map = np.loadtxt(os.path.join(self.root_dir, 'label_map'), dtype=int)
        label_map = {k: v for k, v in label_map}

        self.n_classes = len(label_map)

        # Load the meshes & labels
        if self.train:
            with open(os.path.join(self.root_dir, "train.txt")) as f:
                this_files = [line.rstrip() for line in f]
        else:
            with open(os.path.join(self.root_dir, "test.txt")) as f:
                this_files = [line.rstrip() for line in f]

        print("loading {} files: {}".format(len(this_files), this_files))

        # Load the actual files

        off_path = os.path.join(root_dir, "off")
        label_path = os.path.join(root_dir, "labels")
        for f in this_files:
            off_file = os.path.join(off_path, f)
            label_file = os.path.join(label_path, f[:-4] + ".txt")

            verts, faces = read_mesh(off_file)
            labels_og = np.loadtxt(label_file).astype(int) + 1 # shift -1 --> 0
        
            if self.target_num_faces is not None:
                ms = pymeshlab.MeshSet()
                # Ensure correct data types and shapes
                verts = np.asarray(verts, dtype=np.float64)
                faces = np.asarray(faces, dtype=np.int32)
                labels_og = np.asarray(labels_og, dtype=np.float64).reshape(-1, 1)  # Ensure it's a column vector

                # Create a mesh with per-vertex labels as scalar attributes
                m = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces, v_scalar_array=labels_og)
                ms.add_mesh(m)
                # Simplify the mesh while preserving scalar attributes
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=self.target_num_faces,
                    preservenormal=True,
                    preserveboundary=True,
                    preservetopology=False,
                    optimalplacement=True,
                    planarquadric=True,
                    qualitythr=0.3,
                    autoclean=True
                )                
                mesh = ms.current_mesh()
                verts = mesh.vertex_matrix()
                faces = mesh.face_matrix()
                labels_og = mesh.vertex_scalar_array()
                labels_og = labels_og.astype(int)

            if num_points is not None:
                # Randomly sample num_points points
                indices = np.random.choice(len(verts), size=num_points, replace=False)
                verts = verts[indices]
                labels_og = labels_og[indices]

            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces) if not self.point_cloud else torch.zeros((0, 3), dtype=torch.int32)
            
            labels = torch.tensor([label_map[l] for l in labels_og])
            labels_og = torch.tensor(labels_og)

            

            # center and unit scale
            verts = normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)
            self.labels_list_og.append(labels_og)

        # Precompute operators
        if self.op_cache_dir is not None:
            self.frames_list, self.vertex_area_list, self.L_list, self.evals_list, self.evecs_list,\
            self.gradX_list, self.gradY_list = get_all_operators(self.verts_list, self.faces_list, k_eig=self.num_eig, op_cache_dir=self.op_cache_dir) if num_points is None else (np.zeros_like(self.verts_list) for _ in range(7))

    def __len__(self):
        return len(self.verts_list)
    
    def __getitem__(self, idx):
        if self.op_cache_dir is not None:
        #     return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.vertex_area_list[idx], self.L_list[idx], \
        # self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx]
            data= dict(vertices=self.verts_list[idx],
                        faces=self.faces_list[idx],
                        frames=self.frames_list[idx],
                        vertex_area=self.vertex_area_list[idx],
                        L=self.L_list[idx],
                        evals=self.evals_list[idx],
                        evecs=self.evecs_list[idx],
                        gradX=self.gradX_list[idx],
                        gradY=self.gradY_list[idx],
                        labels=self.labels_list[idx],
                        labels_og=self.labels_list_og[idx])

        else:
            vertices, faces, labels, labels_og = self.verts_list[idx], self.faces_list[idx], self.labels_list[idx], self.labels_list_og[idx]

            geometry_values = get_operators(vertices, faces, self.num_eig, None)
            frames, vertex_area, L, evals, evecs, gradX, gradY = geometry_values
            data = dict(vertices=vertices,
                        faces=faces,
                        frames=frames,
                        vertex_area=vertex_area,
                        L=L,
                        evals=evals,
                        evecs=evecs,
                        gradX=gradX,
                        gradY=gradY,
                        labels=labels,
                        labels_og=labels_og)

        return  data
