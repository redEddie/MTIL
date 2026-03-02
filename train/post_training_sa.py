import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from M_dataset_post_training_sa import SAPostTraiingDataset
from mamba_jepa import MambaJEPA
from mamba_policy import MambaConfig, Mamba2, Block

class SAMambaPolicyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, future_steps: int, action_dim: int, config: MambaConfig):
        super().__init__()
        self.future_steps = future_steps
        self.action_dim = action_dim
        
        # 1. Project input to d_model
        self.in_proj = nn.Linear(in_dim, config.d_model)
        
        # 2. Mamba Blocks for sequence modeling
        def mixer_fn(dim):
            return Mamba2(
                d_model=dim,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                headdim=config.headdim,
                ngroups=config.ngroups,
                A_init_range=config.A_init_range,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
                dt_init_floor=config.dt_init_floor,
                dt_limit=config.dt_limit,
                chunk_size=config.chunk_size,
                use_mem_eff_path=config.use_mem_eff_path,
            )

        def mlp_fn(dim):
            hidden_mlp_dim = 4 * dim
            return nn.Sequential(
                nn.Linear(dim, hidden_mlp_dim),
                nn.GELU(),
                nn.Linear(hidden_mlp_dim, dim),
            )
            
        # Add 2 Mamba blocks for the policy head
        self.blocks = nn.ModuleList([
            Block(
                dim=config.d_model,
                mixer_cls=mixer_fn,
                mlp_cls=mlp_fn,
                norm_cls=nn.LayerNorm,
            )
            for _ in range(2) 
        ])
        
        # 3. Output projection to predict future actions
        self.out_proj = nn.Linear(config.d_model, future_steps * action_dim)

    def forward(self, x):
        # x is [B, in_dim]
        # Reshape to [B, L, D] where L=1 for sequence processing
        x = self.in_proj(x).unsqueeze(1) # [B, 1, d_model]
        
        for blk in self.blocks:
            x, _ = blk(x, residual=None)
            
        # x is [B, 1, d_model]
        x = x.squeeze(1) # [B, d_model]
        
        # Predict future actions
        out = self.out_proj(x)
        return out.view(-1, self.future_steps, self.action_dim)


class LitPostTraiingSA(pl.LightningModule):
    def __init__(
        self,
        config: MambaConfig,
        jepa_ckpt_path: str,
        future_steps: int = 16,
        action_dim: int = 14,
        state_dim: int = 14,
        policy_input_mode: str = "sz",
        lr: float = 2e-4,
        weight_decay: float = 5e-4,
        min_valid_weight: float = 0.25,
        min_prev_action_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.future_steps = future_steps
        self.action_dim = action_dim
        self.policy_input_mode = policy_input_mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_valid_weight = min_valid_weight
        self.min_prev_action_weight = min_prev_action_weight
        if self.policy_input_mode not in {"sz", "saz"}:
            raise ValueError(f"Unsupported policy_input_mode: {self.policy_input_mode}")

        self.jepa = MambaJEPA(config)
        self._load_jepa_weights(jepa_ckpt_path)
        self.jepa.eval()
        for p in self.jepa.parameters():
            p.requires_grad_(False)

        head_in_dim = state_dim + config.d_model
        if self.policy_input_mode == "saz":
            head_in_dim += action_dim
            
        # 맘바 블록이 들어간 새로운 헤드 사용
        self.head = SAMambaPolicyHead(
            in_dim=head_in_dim,
            hidden_dim=1024,
            future_steps=future_steps,
            action_dim=action_dim,
            config=config,
        )

    def _load_jepa_weights(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"JEPA checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        mapped_sd = {}
        for k, v in raw_sd.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model."):]
            if nk.startswith("jepa."):
                nk = nk[len("jepa."):]
            mapped_sd[nk] = v

        msg = self.jepa.load_state_dict(mapped_sd, strict=False)
        loaded_keys = set(mapped_sd.keys()) - set(msg.unexpected_keys)
        essential_prefixes = ["backbone.", "patch_proj.", "context_encoder."]
        loaded_essential = {
            p: sum(1 for k in loaded_keys if k.startswith(p)) for p in essential_prefixes
        }
        if loaded_essential["backbone."] == 0 or loaded_essential["context_encoder."] == 0:
            raise RuntimeError(
                f"JEPA checkpoint loaded but essential modules were not mapped correctly: {loaded_essential}"
            )
        print(f"[post_traiing_sa] Loaded JEPA checkpoint: {ckpt_path}")
        print(f"[post_traiing_sa] Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        print(f"[post_traiing_sa] Loaded essential key counts: {loaded_essential}")

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        state = batch["state"]
        prev_action = batch["prev_action"]
        prev_action_valid = batch["prev_action_valid"].float().to(state.device)
        action = batch["action"]
        rgb_hist = batch["rgb_hist"]
        valid_ratio = batch["valid_ratio"].float().to(state.device)

        with torch.no_grad():
            z = self.jepa.encode_z(rgb_hist)

        if self.policy_input_mode == "saz":
            head_input = torch.cat([state, prev_action, z], dim=-1)
        else:
            head_input = torch.cat([state, z], dim=-1)
        pred_action = self.head(head_input)

        # per_sample_loss: [B]
        per_sample_loss = F.smooth_l1_loss(pred_action, action, reduction="none").mean(dim=(1, 2))
        
        # Calculate per-joint L1 error (absolute error) across batch and future_steps: [action_dim]
        # pred_action, action: [B, future_steps, action_dim]
        with torch.no_grad():
            joint_errors = F.l1_loss(pred_action, action, reduction="none").mean(dim=(0, 1))

        weights = torch.clamp(valid_ratio, min=self.min_valid_weight)
        if self.policy_input_mode == "saz":
            prev_weights = torch.where(
                prev_action_valid > 0.5,
                torch.ones_like(prev_action_valid),
                torch.full_like(prev_action_valid, self.min_prev_action_weight),
            )
            weights = weights * prev_weights
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_valid_ratio", valid_ratio.mean(), on_step=False, on_epoch=True, prog_bar=False)
        if self.policy_input_mode == "saz":
            self.log(f"{stage}_prev_action_valid", prev_action_valid.mean(), on_step=False, on_epoch=True, prog_bar=False)
            
        # Log per-joint errors
        for idx in range(self.action_dim):
            self.log(f"{stage}_joint_{idx}_err", joint_errors[idx], on_step=(stage == "train"), on_epoch=True, prog_bar=False)
            
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-7)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


def resolve_jepa_ckpt(user_path: str):
    if user_path:
        return user_path

    # Prefer explicit JEPA pretrain outputs first.
    pretrain_globs = [
        "lightning_logs/jepa_pretrain_final/version_*/checkpoints/*.ckpt",
        "lightning_logs/jepa_pretrain/version_*/checkpoints/*.ckpt",
        "lightning_logs/jepa_pretrain_spatiotemporal/version_*/checkpoints/*.ckpt",
        "lightning_logs/jepa/version_*/checkpoints/*.ckpt",
    ]
    discovered = []
    for pattern in pretrain_globs:
        discovered.extend(glob(pattern))
    if discovered:
        discovered.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return discovered[0]

    raise FileNotFoundError("Could not find a JEPA checkpoint. Pass --jepa_ckpt_path explicitly.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/home/jeonchanwook/MTIL/transfer.50")
    parser.add_argument("--jepa_ckpt_path", type=str, default="")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--min_valid_weight", type=float, default=0.25)
    parser.add_argument("--min_prev_action_weight", type=float, default=0.5)
    parser.add_argument("--policy_input_mode", type=str, default="sz", choices=["sz", "saz"])
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()

    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    config = MambaConfig()
    config.num_blocks = 4
    config.camera_names = ["top"]
    config.img_size = (640, 480)
    config.max_t = 10  # 10 프레임 (2초 분량) 유지

    root_path = args.root_path
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    jepa_ckpt_path = resolve_jepa_ckpt(args.jepa_ckpt_path)

    train_dataset = SAPostTraiingDataset(
        root_dir=root_path,
        mode="train",
        history_frames=10,
        frame_skip=10,
        future_steps=16,
        selected_cameras=config.camera_names,
    )
    val_dataset = SAPostTraiingDataset(
        root_dir=root_path,
        mode="test",
        history_frames=10,
        frame_skip=10,
        future_steps=16,
        selected_cameras=config.camera_names,
    )
    print(f"[post_traiing_sa] dataset frames - train: {len(train_dataset)}, val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = LitPostTraiingSA(
        config=config,
        jepa_ckpt_path=jepa_ckpt_path,
        future_steps=16,
        action_dim=14,
        state_dim=14,
        policy_input_mode=args.policy_input_mode,
        min_valid_weight=args.min_valid_weight,
        min_prev_action_weight=args.min_prev_action_weight,
    )

    csv_logger = CSVLogger(
        "lightning_logs",
        name=f"post_traiing_sa_{args.policy_input_mode}",
        flush_logs_every_n_steps=1,
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="post-training-sa-step{step:06d}-val_loss{val_loss:.4f}",
        save_top_k=5,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=csv_logger,
        callbacks=[ckpt_cb],
        precision=args.precision,
        log_every_n_steps=1,
        val_check_interval=1000,
        limit_val_batches=50,  # 추가: 검증 시간을 대폭 줄이기 위해 50개 배치만 검증
    )
    trainer.fit(model, train_loader, val_loader)

    # 저장된 last.ckpt 복사해서 last_{val_loss}.ckpt 로 이름 변경
    last_ckpt_path = ckpt_cb.last_model_path
    if last_ckpt_path and os.path.exists(last_ckpt_path):
        import shutil
        final_val_loss = trainer.callback_metrics.get("val_loss", 0.0)
        if isinstance(final_val_loss, torch.Tensor):
            final_val_loss = final_val_loss.item()
        
        new_last_path = os.path.join(
            os.path.dirname(last_ckpt_path), 
            f"last_{final_val_loss:.4f}.ckpt"
        )
        shutil.copy2(last_ckpt_path, new_last_path)
        print(f"[post_traiing_sa] Created final checkpoint copy: {new_last_path}")


if __name__ == "__main__":
    main()
