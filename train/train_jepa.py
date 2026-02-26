import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from train.scaler_M import Scaler
from train.mamba_jepa import MambaJEPA
from train.mamba_policy import MambaConfig
from train.M_dataset import MambaSequenceDataset

class LitMambaJEPA(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.jepa = MambaJEPA(config)
        self.scaler = scaler
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.total_steps = 0
        self.max_training_steps = 100000 

    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']  
        lowdim = batch['lowdim']
        lowdim = self.scaler.normalize(lowdim)
        concat_lowdim = torch.cat([
            lowdim['agl_1'], lowdim['agl_2'], lowdim['agl_3'], lowdim['agl_4'], lowdim['agl_5'], lowdim['agl_6'], lowdim['gripper_pos'],
            lowdim['agl2_1'], lowdim['agl2_2'], lowdim['agl2_3'], lowdim['agl2_4'], lowdim['agl2_5'], lowdim['agl2_6'], lowdim['gripper_pos2']
        ], dim=1)

        # Reshape to sequence [B=1, L=BatchSize, ...]
        rgb_seq = {cam: rgb[cam].unsqueeze(0) for cam in rgb} 
        lowdim_seq = concat_lowdim.unsqueeze(0) 
        
        loss = self.jepa(rgb_seq, lowdim_seq)
        
        self.log("jepa_loss", loss, prog_bar=True)
        
        # Update EMA
        self.jepa.update_ema(self.total_steps, self.max_training_steps)
        self.total_steps += 1
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.jepa.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

def main():
    seed_everything(42)
    config = MambaConfig()
    config.camera_names = ['top']
    config.img_size = (640, 480)
    
    root_path = "/home/jeonchanwook/MTIL/transfer.100"
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    
    lowdim_dict = {
        'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
        'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
        'gripper_pos': 1, 'gripper_pos2': 1,
        'agl_1_act': (16,1), 'agl_2_act': (16,1), 'agl_3_act': (16,1),
        'agl_4_act': (16,1), 'agl_5_act': (16,1), 'agl_6_act': (16,1),
        'agl2_1_act': (16,1), 'agl2_2_act': (16,1), 'agl2_3_act': (16,1),
        'agl2_4_act': (16,1), 'agl2_5_act': (16,1), 'agl2_6_act': (16,1),
        'gripper_act':(16,1), 'gripper_act2':(16,1)
    }
    scaler = Scaler(lowdim_dict=lowdim_dict)
    
    # Correcting scaler path to root
    scaler_path = '/home/jeonchanwook/MTIL/scaler_params.pth'
    if not os.path.exists(scaler_path):
        scaler_path = 'scaler_params.pth'
    
    scaler.load(scaler_path)
    train_dataset.scaler = scaler
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    model = LitMambaJEPA(config, scaler)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=50,
        precision=32
    )
    
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
