import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from train.scaler_M import Scaler
from train.mamba_policy import MambaPolicy, MambaConfig
from train.M_dataset import MambaSequenceDataset
from train.train import LitMambaModel # 기존 MTIL 학습 로직 재사용

def main():
    seed_everything(42)
    config = MambaConfig()
    config.camera_names = ['top']
    
    # 1. JEPA 체크포인트 경로 설정 (최근 생성된 경로로 수정)
    jepa_ckpt_path = "lightning_logs/version_1/checkpoints/last.ckpt" 
    
    # 2. 데이터셋 및 스케일러 설정
    root_path = os.path.abspath("transfer.100")
    if not os.path.exists(root_path):
        root_path = "/home/jeonchanwook/MTIL/transfer.100"

    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    
    scaler = Scaler(lowdim_dict=train_dataset.lowdim_shapes)
    
    # Corrected scaler path
    scaler_path = '/home/jeonchanwook/MTIL/scaler_params.pth'
    if not os.path.exists(scaler_path):
        scaler_path = 'scaler_params.pth'
    
    scaler.load(scaler_path)
    train_dataset.scaler = scaler
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 3. 모델 초기화
    model = LitMambaModel(config, scaler=scaler)

    # 4. JEPA 가중치 이식 (중요!)
    if os.path.exists(jepa_ckpt_path):
        print(f"🧬 Loading Pre-trained JEPA weights from {jepa_ckpt_path}")
        checkpoint = torch.load(jepa_ckpt_path)
        state_dict = checkpoint['state_dict']
        
        # JEPA의 context_model 가중치만 추출하여 MambaPolicy에 주입
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'jepa.context_model.' in k:
                # 'jepa.context_model.xxx' -> 'policy.xxx' 로 이름 변경
                new_k = k.replace('jepa.context_model.', 'policy.')
                new_state_dict[new_k] = v
        
        # 가중치 로드 (액션 레이어인 out_proj 등은 제외하고 로드됨)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights with message: {msg}")
    else:
        print(f"⚠️ JEPA checkpoint not found at {jepa_ckpt_path}. Check lightning_logs path.")

    # 5. 미세 조정 학습 시작
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=100,
        precision=32
    )
    
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
