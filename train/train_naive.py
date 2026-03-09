"""
train_naive.py - Naive Action Chunking 학습

기존 train.py와의 차이:
  - MambaSequenceDatasetNaive 사용: 매 K 스텝(= future_steps)마다 관측 샘플링
  - 학습 시 Mamba 시퀀스 = [o_0, o_K, o_2K, ...] → 추론과 동일한 패턴
  - chunk_size = Mamba에 공급할 query 횟수 (기존은 raw 프레임 수)
  - 나머지 학습 로직(forward_seq, loss, optimizer)은 동일
"""
import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning import seed_everything

from metric_M import my_Metric
from scaler_M import Scaler
from M_dataset_naive import MambaSequenceDatasetNaive
from mamba_policy import MambaPolicy, MambaConfig


class LitMambaModelNaive(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler, future_steps: int = 16):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.metric = my_Metric()
        self.prev_traj_idx = -1
        self.future_steps = future_steps
        self.train_sequence_loss = 0.0
        self.val_sequence_loss = 0.0

        print("Starting training (naive chunking)...")
        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            sum_camera_feats=config.sum_camera_feats,
            num_blocks=config.num_blocks,
            future_steps=future_steps,
            img_size=config.img_size,
            mamba_cfg=config
        )
        print("Model initialized.")

        self.scaler = scaler
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy.to(device)
        self.scaler.to(device)

        self.lr = 2e-4
        self.weight_decay = 5e-4
        self.train_total_loss = 0.0
        self.train_total_steps = 0
        self.val_total_loss = 0.0
        self.val_total_steps = 0

        self.automatic_optimization = False

        self.std_agl_1 = 0.0036
        self.std_agl_2 = 0.5280
        self.std_agl_3 = 0.1980
        self.std_agl_4 = 0.0164
        self.std_agl_5 = 0.3592
        self.std_agl_6 = 0.5998
        self.std_agl2_1 = 0.1084
        self.std_agl2_2 = 0.5019
        self.std_agl2_3 = 0.4448
        self.std_agl2_4 = 0.1414
        self.std_agl2_5 = 0.3066
        self.std_agl2_6 = 0.2251
        self.std_grip1 = 0.2553
        self.std_grip2 = 0.2475

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        # batch: rgb [B, L, C, H, W], lowdim [B, L, 1], actions [B, L, K, 1]
        # L = query 횟수 (chunk_size), K = future_steps
        rgb = batch['rgb']
        lowdim = batch['lowdim']
        traj_idx = batch['traj_idx'].item()
        is_first_chunk = batch['is_first_chunk'].item()

        # 새 trajectory 시작 -> gradient 적용
        if is_first_chunk and self.prev_traj_idx != -1:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            self.train_sequence_loss = 0.0

        self.prev_traj_idx = traj_idx

        # noise augmentation
        noise_agl = 0.02
        with torch.no_grad():
            for key, scale in [
                ('agl_1', noise_agl * self.std_agl_1), ('agl_2', noise_agl * self.std_agl_2),
                ('agl_3', noise_agl * self.std_agl_3), ('agl_4', noise_agl * self.std_agl_4),
                ('agl_5', noise_agl * self.std_agl_5), ('agl_6', noise_agl * self.std_agl_6),
                ('agl2_1', noise_agl * self.std_agl2_1), ('agl2_2', noise_agl * self.std_agl2_2),
                ('agl2_3', noise_agl * self.std_agl2_3), ('agl2_4', noise_agl * self.std_agl2_4),
                ('agl2_5', noise_agl * self.std_agl2_5), ('agl2_6', noise_agl * self.std_agl2_6),
            ]:
                if key in lowdim:
                    lowdim[key] = lowdim[key] + torch.randn_like(lowdim[key]) * scale

        lowdim = self.scaler.normalize(lowdim)

        # concat_lowdim: [B, L, 14]
        concat_lowdim = torch.cat([
            lowdim['agl_1'], lowdim['agl_2'], lowdim['agl_3'], lowdim['agl_4'],
            lowdim['agl_5'], lowdim['agl_6'], lowdim['gripper_pos'],
            lowdim['agl2_1'], lowdim['agl2_2'], lowdim['agl2_3'], lowdim['agl2_4'],
            lowdim['agl2_5'], lowdim['agl2_6'], lowdim['gripper_pos2']
        ], dim=2)  # [B, L, 14]

        # forward_seq: [B, L, K, 14]
        # L 차원 = query 횟수, K 차원 = future_steps
        pred_action = self.policy.forward_seq(concat_lowdim, rgb)  # [B, L, K, 14]

        # target: [B, L, K, 14]
        actions = torch.cat([
            lowdim['agl_1_act'], lowdim['agl_2_act'], lowdim['agl_3_act'],
            lowdim['agl_4_act'], lowdim['agl_5_act'], lowdim['agl_6_act'],
            lowdim['gripper_act'],
            lowdim['agl2_1_act'], lowdim['agl2_2_act'], lowdim['agl2_3_act'],
            lowdim['agl2_4_act'], lowdim['agl2_5_act'], lowdim['agl2_6_act'],
            lowdim['gripper_act2']
        ], dim=3)  # [B, L, K, 14]

        loss = F.mse_loss(pred_action, actions)
        self.manual_backward(loss)

        self.train_sequence_loss += loss.item()
        self.train_total_loss += loss.item()
        self.train_total_steps += 1

        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=False,
                 sync_dist=False, batch_size=1)

        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        lowdim = batch['lowdim']
        traj_idx = batch['traj_idx'].item()
        is_first_chunk = batch['is_first_chunk'].item()

        if is_first_chunk and self.prev_traj_idx != -1 and self.val_sequence_loss > 0.0:
            self.log("val_loss", self.val_sequence_loss, prog_bar=True, sync_dist=False, batch_size=1)
            self.val_sequence_loss = 0.0

        self.prev_traj_idx = traj_idx

        lowdim = self.scaler.normalize(lowdim)

        concat_lowdim = torch.cat([
            lowdim['agl_1'], lowdim['agl_2'], lowdim['agl_3'], lowdim['agl_4'],
            lowdim['agl_5'], lowdim['agl_6'], lowdim['gripper_pos'],
            lowdim['agl2_1'], lowdim['agl2_2'], lowdim['agl2_3'], lowdim['agl2_4'],
            lowdim['agl2_5'], lowdim['agl2_6'], lowdim['gripper_pos2']
        ], dim=2)

        pred_action = self.policy.forward_seq(concat_lowdim, rgb)  # [B, L, K, 14]

        actions = torch.cat([
            lowdim['agl_1_act'], lowdim['agl_2_act'], lowdim['agl_3_act'],
            lowdim['agl_4_act'], lowdim['agl_5_act'], lowdim['agl_6_act'],
            lowdim['gripper_act'],
            lowdim['agl2_1_act'], lowdim['agl2_2_act'], lowdim['agl2_3_act'],
            lowdim['agl2_4_act'], lowdim['agl2_5_act'], lowdim['agl2_6_act'],
            lowdim['gripper_act2']
        ], dim=3)

        loss = F.mse_loss(pred_action, actions)
        self.val_sequence_loss += loss.item()
        self.val_total_loss += loss.item()
        self.val_total_steps += 1

        pred_action = self.denormalize(pred_action)
        actions = self.denormalize(actions)
        self.metric.update(pred_action, actions)

        return loss

    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        if self.train_sequence_loss > 0.0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            self.log("train_loss", self.train_sequence_loss, prog_bar=True, sync_dist=False, batch_size=1)
            self.train_sequence_loss = 0.0
            self.lr_scheduler_obj.step()

        self.log("train_epoch_loss",
                 self.train_total_loss / self.train_total_steps if self.train_total_steps > 0 else 0.0,
                 prog_bar=True, sync_dist=True)
        self.train_total_loss = 0.0
        self.train_total_steps = 0

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
        optimizer = torch.optim.AdamW(
            list(self.policy.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=0.5e-6
        )
        self.lr_scheduler_obj = scheduler_obj
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler_obj,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_start(self):
        if hasattr(self.trainer, 'optimizers') and len(self.trainer.optimizers) > 0:
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = self.lr
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")

    def on_validation_epoch_end(self):
        self.log_dict(self.metric.compute(), prog_bar=True, sync_dist=True)
        self.metric.reset()

        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")

        self.log("val_epoch_loss",
                 self.val_total_loss / self.val_total_steps if self.val_total_steps > 0 else 0.0,
                 prog_bar=True, sync_dist=True)
        self.val_total_loss = 0.0
        self.val_total_steps = 0


def main():
    TASK = "transfer20"
    FUTURE_STEPS = 16      # K: action chunk 크기 = query 주기 (추론과 반드시 일치)
    CHUNK_SIZE = 10        # L: 한 번에 처리할 query 횟수 (실제 프레임 수 = L * K = 160)
    RESUME_CKPT = None     # resume할 체크포인트 경로 (없으면 None)

    seed_everything(42)

    config = MambaConfig()
    config.camera_names = ['top']
    config.freeze_backbone = True
    config.embed_dim = 2048
    config.lowdim_dim = 14
    config.d_model = 2048
    config.action_dim = 14
    config.sum_camera_feats = False
    config.num_blocks = 4
    config.img_size = (640, 480)

    train_dataset = MambaSequenceDatasetNaive(
        root_dir=f"dataset/{TASK}",
        mode="train",
        resize_hw=(640, 480),
        use_pose10d=True,
        selected_cameras=config.camera_names,
        future_steps=FUTURE_STEPS,
        chunk_size=CHUNK_SIZE
    )
    val_dataset = MambaSequenceDatasetNaive(
        root_dir=f"dataset/{TASK}",
        mode="test",
        resize_hw=(640, 480),
        use_pose10d=True,
        selected_cameras=config.camera_names,
        future_steps=FUTURE_STEPS,
        chunk_size=CHUNK_SIZE
    )

    lowdim_dict = {
        'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
        'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
        'gripper_pos': 1, 'gripper_pos2': 1,
        'agl_1_act': (FUTURE_STEPS, 1), 'agl_2_act': (FUTURE_STEPS, 1),
        'agl_3_act': (FUTURE_STEPS, 1), 'agl_4_act': (FUTURE_STEPS, 1),
        'agl_5_act': (FUTURE_STEPS, 1), 'agl_6_act': (FUTURE_STEPS, 1),
        'agl2_1_act': (FUTURE_STEPS, 1), 'agl2_2_act': (FUTURE_STEPS, 1),
        'agl2_3_act': (FUTURE_STEPS, 1), 'agl2_4_act': (FUTURE_STEPS, 1),
        'agl2_5_act': (FUTURE_STEPS, 1), 'agl2_6_act': (FUTURE_STEPS, 1),
        'gripper_act': (FUTURE_STEPS, 1), 'gripper_act2': (FUTURE_STEPS, 1)
    }

    scaler = Scaler(lowdim_dict=lowdim_dict)
    scaler.load('scaler_params.pth')
    train_dataset.scaler = scaler
    val_dataset.scaler = scaler

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=20, pin_memory=True, drop_last=False,
        prefetch_factor=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=20, pin_memory=True, drop_last=False,
        prefetch_factor=4, persistent_workers=True
    )

    lit_model = LitMambaModelNaive(config, scaler=scaler, future_steps=FUTURE_STEPS)

    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger

    csv_logger = CSVLogger(
        save_dir=f"./logs/naive/{TASK}",
        name="lightning_logs",
        flush_logs_every_n_steps=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        monitor='val_epoch_loss', mode='min',
        save_last=True, save_top_k=5,
        filename="{epoch}-{val_loss:.4f}"
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=200,
        default_root_dir=f"./logs/naive/{TASK}",
        logger=csv_logger,
        log_every_n_steps=1,
        callbacks=[lr_monitor, ckpt_cb],
        precision=32
    )

    import torch.serialization
    torch.serialization.add_safe_globals([MambaConfig])
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=RESUME_CKPT)


if __name__ == "__main__":
    main()
