from rna_config import Config
from rna_dataset import RNAMeshDataset

from torch.utils.data import DataLoader

from trainer import Trainer

print("Creating dataloaders...")

train_dataset = RNAMeshDataset(
    Config.data_basepath,
    train=True,
    num_eig=Config.num_eig,
    op_cache_dir=Config.cache_dir,
    target_num_faces=None, #Config.target_num_faces,
    point_cloud=Config.train_point_cloud,
    num_points=None if Config.model_name == "DiffusionNet" else Config.num_points,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=None if Config.model_name == "DiffusionNet" else Config.batch_size,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
)

valid_dataset = RNAMeshDataset(
    Config.data_basepath,
    train=False,
    num_eig=Config.num_eig,
    op_cache_dir=Config.cache_dir,
    target_num_faces=Config.target_num_faces,
    point_cloud=Config.val_point_cloud,
    num_points=None if Config.model_name == "DiffusionNet" else Config.num_points,
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=None if Config.model_name == "DiffusionNet" else Config.batch_size,
    num_workers=0, 
    persistent_workers=False
)


model_cfg = {
    "inp_feat": Config.inp_feat,
    "num_eig": Config.num_eig,
    "p_in": Config.p_in,
    "p_out": train_dataset.n_classes,
    "N_block": Config.n_block,
    "n_channels": Config.n_channels,
    "outputs_at": Config.outputs_at,
}

my_trainer = Trainer(
    Config.model_name,
    model_cfg,
    train_loader,
    valid_loader,
    device=Config.device,
    save_dir=Config.save_dir,
    figures_dir=Config.figures_dir,
    log_wandb=Config.log_wandb,
    num_epochs=Config.num_epochs,
    loss=Config.loss,
    project_name=Config.project_name,
    run_name=Config.run_name,
    drag=False
)

my_trainer.run()
