import torch
from dataclasses import dataclass

@dataclass
class Config:
    # General
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 200

    # Data
    data_basepath = "/Data/rna/RNADataset"
    cache_dir = "/Data/rna/cache"
    
    # Mesh resampling
    target_num_faces = None # 2000
    train_point_cloud = False # Remove the faces to convert the mesh into a point cloud
    val_point_cloud = False # Remove the faces to convert the mesh into a point cloud

    # Model
    model_name = "DiffusionNet" # DiffusionNet or PointNetDenseCls
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    batch_size = 32 # For PointNet
    num_points = 14000 # For PointNet

    # DiffusionNet parameters
    inp_feat = "xyz"  # Type of input Features (xyz, HKS, WKS)
    num_eig = 128  # Number of eigenfunctions to use for Spectral Diffusion
    p_in = 3  # Number of input features
    # p_out = 1  # Number of output features
    n_block = 4  # Number of DiffusionNetBlock
    n_channels = 128  # Width of the network
    outputs_at = "vertices"

    # Save dir
    save_dir = "/Data/rna/models"
    figures_dir = "/Data/rna/figures"

    # Wandb log
    log_wandb = True
    project_name = 'RNADiffusionNet'
    run_name = 'Classic'
