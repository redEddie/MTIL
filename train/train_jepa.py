import os
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler, BatchSampler
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from train.metric_M import my_Metric
from train.scaler_M import Scaler
from train.M_dataset import MambaSequenceDataset
from train.mamba_jepa import MambaJEPA
from train.mamba_policy import MambaConfig
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

class TrajectoryBatchSampler(Sampler):
    """
    데이터셋의 cum_lengths를 이용하여 궤적 단위로 인덱스를 묶어서 반환하는 샘플러.
    이를 통해 DataLoader가 1회 호출될 때 하나의 전체 궤적을 반환하게 함.
    """
    def __init__(self, cum_lengths, shuffle=False):
        self.cum_lengths = cum_lengths
        self.shuffle = shuffle
        self.traj_indices = list(range(len(cum_lengths) - 1))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.traj_indices)
        for i in self.traj_indices:
            yield list(range(self.cum_lengths[i], self.cum_lengths[i+1]))

    def __len__(self):
        return len(self.traj_indices)

class LitMambaJEPA(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.jepa = MambaJEPA(config)
        self.scaler = scaler
        self.metric = my_Metric()
        
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.total_steps = 0
        self.max_training_steps = 100000 

        # Standard deviations for noise injection (from train.py)
        self.std_agl_1 = 0.0036; self.std_agl_2 = 0.5280; self.std_agl_3 = 0.1980
        self.std_agl_4 = 0.0164; self.std_agl_5 = 0.3592; self.std_agl_6 = 0.5998
        self.std_agl2_1 = 0.1084; self.std_agl2_2 = 0.5019; self.std_agl2_3 = 0.4448
        self.std_agl2_4 = 0.1414; self.std_agl2_5 = 0.3066; self.std_agl2_6 = 0.2251
        self.std_grip1 = 0.2553; self.std_grip2 = 0.2475

    def _prepare_batch(self, batch):
        # batch: DataLoader가 TrajectoryBatchSampler를 통해 묶어준 [L, ...] 형태의 데이터
        rgb = batch['rgb']
        lowdim = batch['lowdim']
        
        # Noise injection (Sequence level)
        noise_agl = 0.02
        for key, std in [('agl_1', self.std_agl_1), ('agl_2', self.std_agl_2), ('agl_3', self.std_agl_3),
                         ('agl_4', self.std_agl_4), ('agl_5', self.std_agl_5), ('agl_6', self.std_agl_6),
                         ('agl2_1', self.std_agl2_1), ('agl2_2', self.std_agl2_2), ('agl2_3', self.std_agl2_3),
                         ('agl2_4', self.std_agl2_4), ('agl2_5', self.std_agl2_5), ('agl2_6', self.std_agl2_6)]:
            if key in lowdim:
                lowdim[key] += torch.randn_like(lowdim[key]) * noise_agl * std

        lowdim_norm = self.scaler.normalize(lowdim)
        
        # Concat lowdim [1, L, 14] -> Batch size is 1 trajectory
        concat_lowdim = torch.cat([
            lowdim_norm['agl_1'], lowdim_norm['agl_2'], lowdim_norm['agl_3'], lowdim_norm['agl_4'], 
            lowdim_norm['agl_5'], lowdim_norm['agl_6'], lowdim_norm['gripper_pos'],
            lowdim_norm['agl2_1'], lowdim_norm['agl2_2'], lowdim_norm['agl2_3'], 
            lowdim_norm['agl2_4'], lowdim_norm['agl2_5'], lowdim_norm['agl2_6'], 
            lowdim_norm['gripper_pos2']
        ], dim=-1).unsqueeze(0)

        # RGB Dict transformation: {cam: [1, L, 3, H, W]}
        rgb_seq = {cam: rgb[cam].unsqueeze(0) for cam in rgb}

        # Target actions [1, L, 16, 14]
        gt_actions = torch.cat([
            lowdim_norm['agl_1_act'], lowdim_norm['agl_2_act'], lowdim_norm['agl_3_act'],
            lowdim_norm['agl_4_act'], lowdim_norm['agl_5_act'], lowdim_norm['agl_6_act'],
            lowdim_norm['gripper_act'],
            lowdim_norm['agl2_1_act'], lowdim_norm['agl2_2_act'], lowdim_norm['agl2_3_act'],
            lowdim_norm['agl2_4_act'], lowdim_norm['agl2_5_act'], lowdim_norm['agl2_6_act'],
            lowdim_norm['gripper_act2']
        ], dim=-1).unsqueeze(0)

        return rgb_seq, concat_lowdim, gt_actions

    def training_step(self, batch, batch_idx):
        rgb, lowdim, _ = self._prepare_batch(batch)
        
        loss, _ = self.jepa(rgb, lowdim)
        
        self.jepa.update_ema(self.total_steps, self.max_training_steps)
        self.total_steps += 1
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("ema_momentum", self.jepa.current_momentum, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, lowdim, gt_actions = self._prepare_batch(batch)
        
        loss, pred_actions = self.jepa(rgb, lowdim)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        # Metrics update
        ctx_len = pred_actions.shape[1]
        pred_actions_denorm = self.denormalize(pred_actions.reshape(-1, 16, 14))
        gt_actions_denorm = self.denormalize(gt_actions[:, :ctx_len].reshape(-1, 16, 14))
        self.metric.update(pred_actions_denorm, gt_actions_denorm)
        
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def denormalize(self, actions):
        arm1_dict = {
            'agl_1_act': actions[..., 0:1], 'agl_2_act': actions[..., 1:2], 'agl_3_act': actions[..., 2:3],
            'agl_4_act': actions[..., 3:4], 'agl_5_act': actions[..., 4:5], 'agl_6_act': actions[..., 5:6],
            'gripper_act': actions[..., 6:7]
        }
        arm2_dict = {
            'agl2_1_act': actions[..., 7:8], 'agl2_2_act': actions[..., 8:9], 'agl2_3_act': actions[..., 9:10],
            'agl2_4_act': actions[..., 10:11], 'agl2_5_act': actions[..., 11:12], 'agl2_6_act': actions[..., 12:13],
            'gripper_act2': actions[..., 13:14]
        }
        arm1_denorm = self.scaler.denormalize(arm1_dict)
        arm2_denorm = self.scaler.denormalize(arm2_dict)
        return torch.cat([
            arm1_denorm['agl_1_act'], arm1_denorm['agl_2_act'], arm1_denorm['agl_3_act'],
            arm1_denorm['agl_4_act'], arm1_denorm['agl_5_act'], arm1_denorm['agl_6_act'],
            arm1_denorm['gripper_act'],
            arm2_denorm['agl2_1_act'], arm2_denorm['agl2_2_act'], arm2_denorm['agl2_3_act'],
            arm2_denorm['agl2_4_act'], arm2_denorm['agl2_5_act'], arm2_denorm['agl2_6_act'],
            arm2_denorm['gripper_act2']
        ], dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def limit_dataset(dataset, max_trajs):
    """기존 데이터셋 객체의 내부 리스트를 슬라이싱하여 궤적 수를 제한함"""
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
    
    # 1. 데이터셋 생성 (기존 MambaSequenceDataset 그대로 사용)
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    val_dataset = MambaSequenceDataset(root_dir=root_path, mode="test", selected_cameras=config.camera_names)
    
    # 2. 궤적 수 제한 (200개로 제한)
    train_dataset = limit_dataset(train_dataset, 200)
    val_dataset = limit_dataset(val_dataset, 50)
    
    scaler = Scaler(lowdim_dict=train_dataset.lowdim_shapes)
    scaler.load('/home/jeonchanwook/MTIL/scaler_params.pth')
    train_dataset.scaler = scaler; val_dataset.scaler = scaler
    
    # 3. Trajectory-based Batch Sampling 설정
    # 1 Iteration = 1 Trajectory가 되도록 설정
    train_sampler = TrajectoryBatchSampler(train_dataset.cum_lengths, shuffle=True)
    val_sampler = TrajectoryBatchSampler(val_dataset.cum_lengths, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    
    model = LitMambaJEPA(config, scaler)
    csv_logger = CSVLogger("lightning_logs", name="jepa")
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_cb = ModelCheckpoint(monitor='val_loss', mode='min', save_last=True, save_top_k=2, filename="jepa-{epoch}")

    trainer = pl.Trainer(
        max_epochs=5, # 5 Epoch 설정
        accelerator='gpu',
        devices=1,
        logger=csv_logger,
        callbacks=[lr_monitor, ckpt_cb],
        precision=32,
        gradient_clip_val=1.0
    )
    
    print(f"Training started: {len(train_sampler)} trajectories per epoch.")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
