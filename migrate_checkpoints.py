import torch
import sys
import os
import glob

# 1. 패키지 구조 연결 (변환을 위해 임시로 매핑)
# 이 매핑을 통해 예전 이름으로 저장된 객체를 읽어올 때 새로운 train.xxx 경로의 클래스로 연결됩니다.
import train.mamba_policy
import train.scaler_M
import train.M_dataset
import train.metric_M
import train.train

sys.modules['mamba_policy'] = train.mamba_policy
sys.modules['scaler_M'] = train.scaler_M
sys.modules['M_dataset'] = train.M_dataset
sys.modules['metric_M'] = train.metric_M
sys.modules['train_mamba'] = train.train 

def migrate(ckpt_path):
    try:
        print(f"Migrating {ckpt_path}...")
        # weights_only=False로 로드하여 sys.modules 매핑을 활성화
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # 다시 저장하면 객체의 현재 클래스 경로(train.xxx)가 파일에 기록됨
        torch.save(checkpoint, ckpt_path)
        print(f"  -> Successfully migrated: {os.path.basename(ckpt_path)}")
    except Exception as e:
        print(f"  -> Failed to migrate {ckpt_path}: {e}")

if __name__ == "__main__":
    # 1. lightning_logs 하위의 모든 .ckpt 파일 탐색
    all_checkpoints = glob.glob('lightning_logs/**/*.ckpt', recursive=True)
    
    # 2. 루트 디렉토리의 .ckpt 파일 추가
    root_checkpoints = glob.glob('*.ckpt')
    
    targets = all_checkpoints + root_checkpoints
    
    if not targets:
        print("No checkpoint files found.")
    else:
        print(f"Found {len(targets)} checkpoints to migrate.")
        for ckpt in targets:
            migrate(ckpt)
        print("\nAll migrations completed.")
