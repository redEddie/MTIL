import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from train.scaler_M import Scaler
from train.mamba_policy import MambaPolicy, MambaConfig
from train.M_dataset import MambaSequenceDataset
from train.train import LitMambaModel # 기존 MTIL 학습 로직 재사용

def limit_dataset(dataset, max_trajs):
    """
    데이터셋의 궤적(trajectory) 수를 제한하여 학습 속도를 높입니다.
    """
    if len(dataset.records) > max_trajs:
        print(f"✂️ Limiting dataset from {len(dataset.records)} to {max_trajs} trajectories.")
        dataset.records = dataset.records[:max_trajs]
        dataset.lengths = dataset.lengths[:max_trajs]
        dataset.cum_lengths = np.cumsum([0] + dataset.lengths)
    return dataset

def main():
    seed_everything(42)
    config = MambaConfig()
    config.camera_names = ['top']
    
    # 1. JEPA 체크포인트 경로 설정
    jepa_ckpt_path = "lightning_logs/jepa/version_9/checkpoints/epoch=4-step=50.ckpt" 
    if not os.path.exists(jepa_ckpt_path):
        jepa_ckpt_path = "lightning_logs/version_1/checkpoints/last.ckpt"

    # 2. 데이터셋 설정
    root_path = os.path.abspath("transfer.100")
    if not os.path.exists(root_path):
        root_path = "/home/jeonchanwook/MTIL/transfer.100"

    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    val_dataset = MambaSequenceDataset(root_dir=root_path, mode="test", selected_cameras=config.camera_names)
    
    # 데이터 양 제한 (1/3 수준)
    train_dataset = limit_dataset(train_dataset, 33)
    val_dataset = limit_dataset(val_dataset, 7)
    
    scaler = Scaler(lowdim_dict=train_dataset.lowdim_shapes)
    scaler_path = '/home/jeonchanwook/MTIL/scaler_params.pth'
    if not os.path.exists(scaler_path):
        scaler_path = 'scaler_params.pth'
    
    scaler.load(scaler_path)
    train_dataset.scaler = scaler
    val_dataset.scaler = scaler
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 3. 모델 초기화
    model = LitMambaModel(config, scaler=scaler)
    model.lr = 2e-5  # 파인튜닝 학습률

    # 4. JEPA 가중치 이식
    if os.path.exists(jepa_ckpt_path):
        print(f"🧬 Loading Pre-trained JEPA weights from {jepa_ckpt_path}")
        checkpoint = torch.load(jepa_ckpt_path, weights_only=False)
        state_dict = checkpoint['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'jepa.context_model.' in k:
                new_k = k.replace('jepa.context_model.', 'policy.')
                new_state_dict[new_k] = v
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights with message: {msg}")
    else:
        print(f"⚠️ JEPA checkpoint not found at {jepa_ckpt_path}.")

    # 5. 로거 및 콜백 설정 (상세 정보 기록용)
    csv_logger = CSVLogger("lightning_logs", name="finetune")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_loss',
        filename='{epoch}-{val_epoch_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    # 6. 미세 조정 학습 시작
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=5,
        precision=32,
        logger=csv_logger,
        callbacks=[lr_monitor, checkpoint_callback]
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
