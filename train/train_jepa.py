import os
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from train.M_dataset_jepa import MambaSequenceDataset
from train.mamba_jepa import MambaJEPA
from train.mamba_policy import MambaConfig
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
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
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
    
    root_path = "/home/jeonchanwook/MTIL/transfer.100"
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", window_size=16)
    val_dataset = MambaSequenceDataset(root_dir=root_path, mode="test", window_size=16)
    
    print(f"JEPA dataset windows - train: {len(train_dataset)}, val: {len(val_dataset)}")

    # OOM 방지를 위해 Batch Size 1로 축소
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = LitMambaJEPA(config)
    csv_logger = CSVLogger(
        "lightning_logs",
        name="jepa_pretrain_final",
        flush_logs_every_n_steps=1,
    )
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=1,
        logger=csv_logger,
        log_every_n_steps=1,
        precision="16-mixed", # 16비트 혼합 정밀도 사용 (메모리 절반 절약)
        accumulate_grad_batches=4, # 4번의 배치를 모아서 업데이트 (실질적 Batch Size 4 효과)
        gradient_clip_val=1.0
    )
    
    print(f"Starting JEPA Memory-Optimized Training (Precision: 16-mixed, Grad Accum: 4)")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
