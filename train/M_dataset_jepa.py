import os
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import trange, tqdm
from scaler_M import Scaler
import h5py

class MambaSequenceDataset(Dataset):
    """
    윈도우 기반 스마트 로더: 
    궤적 전체를 로드하지 않고, frame_skip 간격으로 num_frames 단위의 윈도우를 인덱싱하여 로드함.
    """
    def __init__(self, root_dir: str, mode: str = "train", num_frames: int = 10, 
                 base_hz: int = 50, target_hz: int = 10,
                 selected_cameras: List[str] = None, resize_hw=(640,480)):
        super().__init__()
        self.dataset_dir = os.path.join(root_dir, mode)
        self.num_frames = num_frames
        
        # 목표 Hz에 맞게 frame_skip 자동 계산 (예: 50Hz 원본 -> 4Hz 타겟이면 50//4 = 12프레임 건너뜀)
        self.frame_skip = max(1, base_hz // target_hz)
        
        # 실제 데이터셋에서 차지하는 전체 윈도우의 길이 (예: 10프레임 * 10간격 = 91프레임 필요)
        self.window_size = (self.num_frames - 1) * self.frame_skip + 1
        
        self.selected_cameras = selected_cameras or ['top']
        self.resize_hw = resize_hw
        
        self.patch_size = 14
        H, W = self.resize_hw[1], self.resize_hw[0]
        self.aligned_H = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        self.aligned_W = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        self.records = []
        self.valid_indices = [] # (record_idx, frame_start_idx) 튜플 저장
        
        # 1. 유효한 윈도우 시작점들 미리 맵핑 (Index Mapping)
        record_items = sorted(os.listdir(self.dataset_dir))
        for r_idx, item in enumerate(record_items):
            record_path = os.path.join(self.dataset_dir, item)
            self.records.append(record_path)
            with h5py.File(record_path, 'r') as root:
                qpos_len = root['/observations/qpos'].shape[0]
                # 궤적 경계를 넘지 않는 범위 내에서 시작점 생성
                if qpos_len >= self.window_size:
                    for f_idx in range(qpos_len - self.window_size + 1):
                        self.valid_indices.append((r_idx, f_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if getattr(self, 'file_handles', None) is None:
            self.file_handles = {}

        record_idx, frame_start = self.valid_indices[idx]
        record_path = self.records[record_idx]
        
        if record_path not in self.file_handles:
            self.file_handles[record_path] = h5py.File(record_path, 'r', swmr=True)
            
        root = self.file_handles[record_path]
        end_idx = frame_start + self.window_size
        
        rgb_dict = {}
        for cam in self.selected_cameras:
            # 시작점부터 끝점까지 frame_skip 간격으로 가져오기
            img_seq = root[f'observations/images/{cam}'][frame_start:end_idx:self.frame_skip] # [num_frames, H, W, 3]
            
            resized_seq = []
            for img in img_seq:
                if (img.shape[1], img.shape[0]) != (self.aligned_W, self.aligned_H):
                    img = cv2.resize(img, (self.aligned_W, self.aligned_H), interpolation=cv2.INTER_LINEAR)
                resized_seq.append(img)
                
            img_tensor = np.stack(resized_seq, axis=0).astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (0, 3, 1, 2)) # [num_frames, 3, H, W]
            rgb_dict[cam] = torch.tensor(img_tensor, dtype=torch.float32)
                
        return {
            'rgb': rgb_dict,
            'traj_idx': record_idx,
            'frame_start': frame_start
        }
