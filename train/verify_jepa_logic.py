import torch
import os
import sys

# root(MTIL)에서 python -m train.verify_jepa_logic 로 실행할 때 내부 임포트가 작동하도록 함
from train.mamba_jepa import MambaJEPA
from train.mamba_policy import MambaConfig
from train.M_dataset import MambaSequenceDataset
from train.scaler_M import Scaler

def verify():
    print("🚀 Starting Mamba-JEPA Logic Verification (Package Mode)...")
    config = MambaConfig()
    config.camera_names = ['top']
    
    # 1. 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaJEPA(config).to(device)
    
    # 2. 데이터셋 로드
    root_path = os.path.abspath("transfer.100")
    if not os.path.exists(root_path):
        root_path = "/home/jeonchanwook/MTIL/transfer.100"

    print(f"Loading dataset from: {root_path}")
    train_dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=config.camera_names)
    
    # 스케일러 로드
    scaler = Scaler(lowdim_dict=train_dataset.lowdim_shapes)
    # Corrected scaler path
    scaler_path = '/home/jeonchanwook/MTIL/scaler_params.pth'
    if not os.path.exists(scaler_path):
        scaler_path = 'scaler_params.pth'
    
    try:
        scaler.load(scaler_path)
        train_dataset.scaler = scaler
        print(f"Successfully loaded scaler: {scaler_path}")
    except Exception as e:
        print(f"Warning: Could not load scaler from {scaler_path}: {e}")
    
    # 3. 데이터 준비 (배치 사이즈 4)
    samples = [train_dataset[i] for i in range(4)]
    
    # 이미지 시퀀스: [B=1, L=4, C, H, W]
    rgbs = {}
    for cam in config.camera_names:
        cam_imgs = torch.stack([s['rgb'][cam] for s in samples]).unsqueeze(0).to(device)
        rgbs[cam] = cam_imgs
    
    # Lowdim 시퀀스: [B=1, L=4, D]
    lowdim_list = []
    for s in samples:
        ld = s['lowdim']
        ld = scaler.normalize(ld)
        concat = torch.cat([
            ld['agl_1'], ld['agl_2'], ld['agl_3'], ld['agl_4'], ld['agl_5'], ld['agl_6'], ld['gripper_pos'],
            ld['agl2_1'], ld['agl2_2'], ld['agl2_3'], ld['agl2_4'], ld['agl2_5'], ld['agl2_6'], ld['gripper_pos2']
        ], dim=0) 
        lowdim_list.append(concat)
    lowdim_seq = torch.stack(lowdim_list).unsqueeze(0).to(device)

    print(f"Input Image Shape: {rgbs['top'].shape}")
    print(f"Input Lowdim Shape: {lowdim_seq.shape}")

    # 4. Forward Pass
    try:
        loss = model(rgbs, lowdim_seq)
        print("\n✅ Logic Verification Success!")
        print(f"Initial JEPA Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
