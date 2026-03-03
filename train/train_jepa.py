import os
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from M_dataset_jepa import MambaSequenceDataset
from mamba_jepa import MambaJEPA
from mamba_policy import MambaConfig
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class LitMambaJEPA(pl.LightningModule):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = MambaJEPA(config)
        self.total_steps = 0
        self.max_training_steps = 10000 

    def training_step(self, batch, batch_idx):
        images = batch['rgb'] 
        loss = self.model(images)
        
        self.model.update_ema(self.total_steps, self.max_training_steps)
        self.total_steps += 1
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['rgb']
        loss = self.model(images)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.05)
        return optimizer

def main():
    seed_everything(42)
    # Tensor Core 최적화 (RTX 3090용)
    torch.set_float32_matmul_precision('high')
    
    config = MambaConfig()
    config.num_blocks = 4 
    config.camera_names = ['top']
    config.img_size = (640, 480)
    config.max_t = 10  # 10 프레임 (2초 분량)으로 설정
    
    root_path = "/home/pilab/workspace/MTIL/transfer.50"
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", num_frames=10, base_hz=50, target_hz=4)
    val_dataset = MambaSequenceDataset(root_dir=root_path, mode="test", num_frames=10, base_hz=50, target_hz=4)
    
    print(f"JEPA dataset windows - train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = LitMambaJEPA(config)
    csv_logger = CSVLogger(
        "lightning_logs",
        name="jepa_pretrain_final",
        flush_logs_every_n_steps=1,
    )
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=-1,
        strategy='deepspeed_stage_2', # Zero Redundancy Optimizer(ZeRO-2) 활성화로 GPU 메모리 OOM 원천 차단
        logger=csv_logger,
        log_every_n_steps=1,
        precision="16-mixed", # 16비트 혼합 정밀도 사용 (메모리 절반 절약)
        accumulate_grad_batches=2,
        gradient_clip_val=1.0
    )
    
    print(f"Starting Highly-Optimized JEPA Training (DDP, precision=16-mixed, Dataloader RAM Caching)")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
