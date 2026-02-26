import os
import torch
from torch import nn
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from train.metric_M import my_Metric
from train.scaler_M import Scaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from train.M_dataset import MambaSequenceDataset
from train.mamba_jepa import MambaJEPA
from train.mamba_policy import MambaConfig
from pytorch_lightning.loggers import CSVLogger

class LitMambaJEPA(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.jepa = MambaJEPA(config)
        self.scaler = scaler
        self.metric = my_Metric()
        
        self.lr = 1e-4
        self.weight_decay = 1e-4
        
        self.prev_traj_idx = -1
        self.train_sequence_loss = 0.0
        self.val_sequence_loss = 0.0
        
        self.train_total_loss = 0.0
        self.train_total_steps = 0
        self.val_total_loss = 0.0
        self.val_total_steps = 0
        
        self.automatic_optimization = False
        
        # Buffer to store trajectory data
        self.current_traj_rgb = []
        self.current_traj_lowdim = []
        self.current_traj_actions = []
        
        self.total_steps = 0
        self.max_training_steps = 100000 

        # Standard deviations for noise injection (from train.py)
        self.std_agl_1 = 0.0036; self.std_agl_2 = 0.5280; self.std_agl_3 = 0.1980
        self.std_agl_4 = 0.0164; self.std_agl_5 = 0.3592; self.std_agl_6 = 0.5998
        self.std_agl2_1 = 0.1084; self.std_agl2_2 = 0.5019; self.std_agl2_3 = 0.4448
        self.std_agl2_4 = 0.1414; self.std_agl2_5 = 0.3066; self.std_agl2_6 = 0.2251
        self.std_grip1 = 0.2553; self.std_grip2 = 0.2475

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        rgb = batch['rgb']
        lowdim = batch['lowdim']
        traj_idx = batch['traj_idx'].item()

        # Noise injection (from train.py)
        noise_agl = 0.02
        with torch.no_grad():
            for key, std in [('agl_1', self.std_agl_1), ('agl_2', self.std_agl_2), ('agl_3', self.std_agl_3),
                             ('agl_4', self.std_agl_4), ('agl_5', self.std_agl_5), ('agl_6', self.std_agl_6),
                             ('agl2_1', self.std_agl2_1), ('agl2_2', self.std_agl2_2), ('agl2_3', self.std_agl2_3),
                             ('agl2_4', self.std_agl2_4), ('agl2_5', self.std_agl2_5), ('agl2_6', self.std_agl2_6)]:
                if key in lowdim:
                    lowdim[key] += torch.randn_like(lowdim[key]) * noise_agl * std

        # Handle trajectory change
        if traj_idx != self.prev_traj_idx and self.prev_traj_idx != -1:
            self.process_trajectory(optimizer, mode='train')
            self.prev_traj_idx = traj_idx
        elif self.prev_traj_idx == -1:
            self.prev_traj_idx = traj_idx

        # Data preprocessing
        lowdim_norm = self.scaler.normalize(lowdim)
        concat_lowdim = torch.cat([
            lowdim_norm['agl_1'], lowdim_norm['agl_2'], lowdim_norm['agl_3'], lowdim_norm['agl_4'], 
            lowdim_norm['agl_5'], lowdim_norm['agl_6'], lowdim_norm['gripper_pos'],
            lowdim_norm['agl2_1'], lowdim_norm['agl2_2'], lowdim_norm['agl2_3'], 
            lowdim_norm['agl2_4'], lowdim_norm['agl2_5'], lowdim_norm['agl2_6'], 
            lowdim_norm['gripper_pos2']
        ], dim=1)

        # Collect target actions for metrics
        actions = torch.cat([
            lowdim_norm['agl_1_act'], lowdim_norm['agl_2_act'], lowdim_norm['agl_3_act'],
            lowdim_norm['agl_4_act'], lowdim_norm['agl_5_act'], lowdim_norm['agl_6_act'],
            lowdim_norm['gripper_act'],
            lowdim_norm['agl2_1_act'], lowdim_norm['agl2_2_act'], lowdim_norm['agl2_3_act'],
            lowdim_norm['agl2_4_act'], lowdim_norm['agl2_5_act'], lowdim_norm['agl2_6_act'],
            lowdim_norm['gripper_act2']
        ], dim=2) # [1, 16, 14]

        # Accumulate (moving to CPU to save GPU memory)
        self.current_traj_rgb.append({cam: rgb[cam].cpu() for cam in rgb})
        self.current_traj_lowdim.append(concat_lowdim.cpu())
        self.current_traj_actions.append(actions.cpu())

        return None

    def process_trajectory(self, optimizer, mode='train'):
        if not self.current_traj_rgb:
            return

        L = len(self.current_traj_rgb)
        # Move to GPU for processing
        rgb_seq = {}
        for cam in self.current_traj_rgb[0].keys():
            rgb_seq[cam] = torch.stack([d[cam] for d in self.current_traj_rgb], dim=1).to(self.device)
        
        lowdim_seq = torch.stack(self.current_traj_lowdim, dim=1).to(self.device) 
        gt_actions_seq = torch.stack(self.current_traj_actions, dim=1).to(self.device) # [1, L, 16, 14]

        # Forward JEPA
        if mode == 'train':
            jepa_loss, pred_actions = self.jepa(rgb_seq, lowdim_seq)
            if jepa_loss.requires_grad:
                self.manual_backward(jepa_loss)
                torch.nn.utils.clip_grad_norm_(self.jepa.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            self.jepa.update_ema(self.total_steps, self.max_training_steps)
            self.total_steps += 1
            
            # 컬럼명을 jepa_loss로 변경하고, ema_momentum을 추가 기록
            self.log("jepa_loss", jepa_loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log("ema_momentum", self.jepa.current_momentum, on_step=True, on_epoch=False, batch_size=1)
            
            self.train_total_loss += jepa_loss.item() * L
            self.train_total_steps += L
        else:
            with torch.no_grad():
                jepa_loss, pred_actions = self.jepa(rgb_seq, lowdim_seq)
            self.log("val_loss", jepa_loss.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            self.val_total_loss += jepa_loss.item() * L
            self.val_total_steps += L
            
            # Update metrics (only on context part where actions are predicted)
            # pred_actions: [1, context_len, 16, 14]
            # gt_actions_seq: [1, L, 16, 14]
            ctx_len = pred_actions.shape[1]
            # Denormalize for metrics
            pred_actions_denorm = self.denormalize(pred_actions.view(-1, 16, 14))
            gt_actions_denorm = self.denormalize(gt_actions_seq[:, :ctx_len].contiguous().view(-1, 16, 14))
            self.metric.update(pred_actions_denorm, gt_actions_denorm)

        # Reset
        self.current_traj_rgb = []
        self.current_traj_lowdim = []
        self.current_traj_actions = []

    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        lowdim = batch['lowdim']
        traj_idx = batch['traj_idx'].item()

        if traj_idx != self.prev_traj_idx and self.prev_traj_idx != -1:
            self.process_trajectory(None, mode='val')
            self.prev_traj_idx = traj_idx
        elif self.prev_traj_idx == -1:
            self.prev_traj_idx = traj_idx

        lowdim_norm = self.scaler.normalize(lowdim)
        concat_lowdim = torch.cat([
            lowdim_norm['agl_1'], lowdim_norm['agl_2'], lowdim_norm['agl_3'], lowdim_norm['agl_4'], 
            lowdim_norm['agl_5'], lowdim_norm['agl_6'], lowdim_norm['gripper_pos'],
            lowdim_norm['agl2_1'], lowdim_norm['agl2_2'], lowdim_norm['agl2_3'], 
            lowdim_norm['agl2_4'], lowdim_norm['agl2_5'], lowdim_norm['agl2_6'], 
            lowdim_norm['gripper_pos2']
        ], dim=1)
        
        actions = torch.cat([
            lowdim_norm['agl_1_act'], lowdim_norm['agl_2_act'], lowdim_norm['agl_3_act'],
            lowdim_norm['agl_4_act'], lowdim_norm['agl_5_act'], lowdim_norm['agl_6_act'],
            lowdim_norm['gripper_act'],
            lowdim_norm['agl2_1_act'], lowdim_norm['agl2_2_act'], lowdim_norm['agl2_3_act'],
            lowdim_norm['agl2_4_act'], lowdim_norm['agl2_5_act'], lowdim_norm['agl2_6_act'],
            lowdim_norm['gripper_act2']
        ], dim=2)

        self.current_traj_rgb.append({cam: rgb[cam].cpu() for cam in rgb})
        self.current_traj_lowdim.append(concat_lowdim.cpu())
        self.current_traj_actions.append(actions.cpu())
        return None

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        if self.current_traj_rgb:
            self.process_trajectory(optimizer, mode='train')
        
        # Step the scheduler
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
            
        self.log("train_epoch_loss", self.train_total_loss / self.train_total_steps if self.train_total_steps > 0 else 0.0, prog_bar=True)
        self.train_total_loss = 0.0; self.train_total_steps = 0

    def on_validation_epoch_end(self):
        if self.current_traj_rgb:
            self.process_trajectory(None, mode='val')
        self.log_dict(self.metric.compute(), prog_bar=True)
        self.metric.reset()
        self.log("val_epoch_loss", self.val_total_loss / self.val_total_steps if self.val_total_steps > 0 else 0.0, prog_bar=True)
        self.val_total_loss = 0.0; self.val_total_steps = 0

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
        ], dim=2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.jepa.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    seed_everything(42)
    config = MambaConfig()
    config.camera_names = ['top']
    config.img_size = (640, 480)
    root_path = "/home/jeonchanwook/MTIL/transfer.100"
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    val_dataset = MambaSequenceDataset(root_dir=root_path, mode="test", selected_cameras=config.camera_names)
    
    scaler = Scaler(lowdim_dict=train_dataset.lowdim_shapes)
    scaler.load('/home/jeonchanwook/MTIL/scaler_params.pth')
    train_dataset.scaler = scaler; val_dataset.scaler = scaler
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    
    model = LitMambaJEPA(config, scaler)
    
    # Configure CSVLogger
    csv_logger = CSVLogger("lightning_logs", name="jepa")
    
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(monitor='val_epoch_loss', mode='min', save_last=True, save_top_k=2, filename="jepa-{epoch}")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=1,
        # 빠른 테스트를 위한 설정 (필요에 따라 주석 해제하거나 수치를 조절하세요)
        limit_train_batches=2000,  # 200 프레임만 학습 (약 1~2개 궤적)
        limit_val_batches=400,     # 50 프레임만 검증
        logger=csv_logger,
        callbacks=[lr_monitor, ckpt_cb],
        precision=32
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
