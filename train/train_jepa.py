import os
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from train.scaler_M import Scaler
from train.M_dataset import MambaSequenceDataset
from train.mamba_jepa import MambaJEPA
from train.mamba_policy import MambaConfig
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

class TrajectoryBatchSampler(Sampler):
    def __init__(self, cum_lengths, shuffle=False):
        self.cum_lengths = cum_lengths
        self.shuffle = shuffle
        self.traj_indices = list(range(len(cum_lengths) - 1))

    def __iter__(self):
        indices = self.traj_indices
        if self.shuffle:
            indices = np.random.permutation(self.traj_indices).tolist()
        for i in indices:
            yield list(range(self.cum_lengths[i], self.cum_lengths[i+1]))

    def __len__(self):
        return len(self.traj_indices)

class LitMambaJEPA(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.jepa = MambaJEPA(config)
        self.scaler = scaler
        
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.total_steps = 0
        self.max_training_steps = 5000 

    def _prepare_batch(self, batch):
        rgb = batch['rgb']
        lowdim = batch['lowdim']
        
        # Noise injection (for robustness in representation learning)
        noise_agl = 0.02
        for key in lowdim:
            if 'act' not in key:
                lowdim[key] += torch.randn_like(lowdim[key]) * noise_agl * 0.1

        lowdim_norm = self.scaler.normalize(lowdim)
        
        concat_lowdim = torch.cat([
            lowdim_norm['agl_1'], lowdim_norm['agl_2'], lowdim_norm['agl_3'], lowdim_norm['agl_4'], 
            lowdim_norm['agl_5'], lowdim_norm['agl_6'], lowdim_norm['gripper_pos'],
            lowdim_norm['agl2_1'], lowdim_norm['agl2_2'], lowdim_norm['agl2_3'], 
            lowdim_norm['agl2_4'], lowdim_norm['agl2_5'], lowdim_norm['agl2_6'], 
            lowdim_norm['gripper_pos2']
        ], dim=-1).unsqueeze(0)

        rgb_seq = {cam: rgb[cam].unsqueeze(0) for cam in rgb}
        return rgb_seq, concat_lowdim

    def training_step(self, batch, batch_idx):
        rgb, lowdim = self._prepare_batch(batch)
        loss = self.jepa(rgb, lowdim, num_context_points=20)
        
        self.jepa.update_ema(self.total_steps, self.max_training_steps)
        self.total_steps += 1
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, lowdim = self._prepare_batch(batch)
        loss = self.jepa(rgb, lowdim, num_context_points=10)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

def limit_dataset(dataset, max_trajs):
    if len(dataset.records) > max_trajs:
        dataset.records = dataset.records[:max_trajs]
        dataset.lengths = dataset.lengths[:max_trajs]
        dataset.cum_lengths = np.cumsum([0] + dataset.lengths)
    return dataset

def main():
    seed_everything(42)
    config = MambaConfig()
    config.camera_names = ['top']
    config.img_size = (640, 480)
    root_path = "/home/jeonchanwook/MTIL/transfer.100"
    
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    val_dataset = MambaSequenceDataset(root_dir=root_path, mode="test", selected_cameras=config.camera_names)
    
    # 10 trajectories for quick testing
    train_dataset = limit_dataset(train_dataset, 10) 
    val_dataset = limit_dataset(val_dataset, 2)
    
    scaler = Scaler(lowdim_dict=train_dataset.lowdim_shapes)
    scaler.load('/home/jeonchanwook/MTIL/scaler_params.pth')
    train_dataset.scaler = scaler; val_dataset.scaler = scaler
    
    train_sampler = TrajectoryBatchSampler(train_dataset.cum_lengths, shuffle=True)
    val_sampler = TrajectoryBatchSampler(val_dataset.cum_lengths, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    
    model = LitMambaJEPA(config, scaler)
    csv_logger = CSVLogger("lightning_logs", name="jepa")
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=1,
        logger=csv_logger,
        precision=32,
        enable_checkpointing=True
    )
    
    print(f"Pure JEPA Training (Representation Only): {len(train_sampler)} train trajs.")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
