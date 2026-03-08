import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import trange, tqdm
from scaler_M import Scaler
import h5py

class MambaSequenceDataset(Dataset):
    """
    "각 record 폴더"를 "하나의 시계열 궤적"으로 간주합니다.
    chunk_size > 1일 때, 매번 L프레임의 연속 데이터를 반환합니다 (시퀀스 병렬 학습용).
    chunk_size=1일 때는 기존의 프레임별 반환 모드와 동일합니다.
    """
    def __init__(self, root_dir: str, mode: str = "train", use_pose10d: bool = True,
                 resize_hw=(640,480), selected_cameras: List[str] = None,
                 scaler: Optional[Scaler] = None,
                 future_steps=16,
                 chunk_size: int = 1):  # <-- chunk 크기 (L)
        super().__init__()
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.dataset_dir = os.path.join(root_dir, mode)
        self.use_pose10d = use_pose10d
        self.resize_hw = resize_hw
        self.future_steps = future_steps
        self.chunk_size = chunk_size
        self.selected_cameras = selected_cameras
        if self.selected_cameras is None:
            self.selected_cameras = ['top']
        # 모든 궤적 경로를 로드하고 각 궤적의 길이를 기록
        self.records = []
        self.lengths = []  # 각 궤적의 길이를 기록
        for item in sorted(os.listdir(self.dataset_dir)):
            record_path = os.path.join(self.dataset_dir, item)
            self.records.append(record_path)
            with h5py.File(record_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                self.lengths.append(qpos.shape[0])

        # chunk 인덱스 빌드: (traj_idx, start_frame, end_frame)
        self.chunks = []
        for traj_idx, traj_len in enumerate(self.lengths):
            for start in range(0, traj_len, chunk_size):
                end = min(start + chunk_size, traj_len)
                self.chunks.append((traj_idx, start, end))

        # 누적 궤적 길이 (scaler fitting용으로 유지)
        self.cum_lengths = np.cumsum([0] + self.lengths)

        # 저차원(low-dim) 데이터의 키와 형상 정의
        self.lowdim_keys = [
            'agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
            'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6',
            'gripper_pos', 'gripper_pos2',
            'agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act',
            'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act',
            'gripper_act', 'gripper_act2'
        ]
        self.lowdim_shapes = {
            'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
            'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
            'gripper_pos': 1,
            'gripper_pos2': 1,
            'agl_1_act': (future_steps, 1), 'agl_2_act': (future_steps, 1), 'agl_3_act': (future_steps, 1),
            'agl_4_act': (future_steps, 1), 'agl_5_act': (future_steps, 1), 'agl_6_act': (future_steps, 1),
            'agl2_1_act': (future_steps, 1), 'agl2_2_act': (future_steps, 1), 'agl2_3_act': (future_steps, 1),
            'agl2_4_act': (future_steps, 1), 'agl2_5_act': (future_steps, 1), 'agl2_6_act': (future_steps, 1),
            'gripper_act': (future_steps, 1),
            'gripper_act2': (future_steps, 1)
        }

        # Scaler 초기화
        self.scaler = scaler
        if self.scaler is None and mode == "train":
            # 如果没有提供 Scaler 且是训练模式，则初始化一个 Scaler
            self.scaler = Scaler(lowdim_dict=self.lowdim_shapes)
            self.fitting = False  # 标志是否在拟合
        else:
            self.fitting = False

    def __len__(self):
        return len(self.chunks)

    def fit_scaler(self, batch_size=64, num_workers=4):
        """
        计算归一化参数。
        """
        if not self.scaler:
            raise ValueError("Scaler is not initialized.")

        print("Fitting scaler on dataset...")
        self.fitting = True  # 开始拟合，禁用归一化
        data_cache = {key: [] for key in self.scaler.lowdim_dict.keys()}
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        for batch in tqdm(dataloader, desc='Fitting scaler'):
            lowdim = batch['lowdim']
            for key in self.scaler.lowdim_dict.keys():
                data_cache[key].append(lowdim[key])
        self.fitting = False  # 拟合完成，启用归一化

        # 将所有批次的数据拼接起来
        for key in data_cache.keys():
            data_cache[key] = torch.cat(data_cache[key], dim=0)
        # 计算最小值和最大值
        self.scaler.fit(data_cache)
        print("Scaler fitted.")
        return self.scaler


    def save_scaler(self, filepath: str):
        """
        保存 Scaler 的归一化参数到文件。
        """
        if self.scaler:
            self.scaler.save(filepath)
            print(f"Scaler saved to {filepath}.")
        else:
            raise ValueError("Scaler is not initialized.")

    def load_scaler(self, filepath: str):
        """
        从文件加载 Scaler 的归一化参数。
        """
        if self.scaler:
            self.scaler.load(filepath)
            print(f"Scaler loaded from {filepath}.")
        else:
            raise ValueError("Scaler is not initialized.")

    def __getitem__(self, chunk_idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start_frame, end_frame = self.chunks[chunk_idx]
        record_path = self.records[traj_idx]
        traj_len = self.lengths[traj_idx]

        with h5py.File(record_path, 'r') as root:
            qpos = root['/observations/qpos'][()]   # [T, 14]
            act  = root['/action'][()]              # [T, 14]

        # state chunk: [L, 1] per dimension
        q = qpos[start_frame:end_frame]   # [L, 14]

        # 벡터화된 future-shift: frame_offsets[i, s] = min(start+i+s, T-1)
        frame_offsets = (
            np.arange(start_frame, end_frame)[:, None] + np.arange(self.future_steps)[None, :]
        )  # [L, F]
        frame_offsets = np.clip(frame_offsets, 0, traj_len - 1)

        def ft(arr):
            return torch.from_numpy(arr.astype(np.float32))

        # state tensors: [L, 1]
        agl_1        = ft(q[:, 0:1])
        agl_2        = ft(q[:, 1:2])
        agl_3        = ft(q[:, 2:3])
        agl_4        = ft(q[:, 3:4])
        agl_5        = ft(q[:, 4:5])
        agl_6        = ft(q[:, 5:6])
        gripper_pos  = ft(q[:, 6:7])
        agl2_1       = ft(q[:, 7:8])
        agl2_2       = ft(q[:, 8:9])
        agl2_3       = ft(q[:, 9:10])
        agl2_4       = ft(q[:, 10:11])
        agl2_5       = ft(q[:, 11:12])
        agl2_6       = ft(q[:, 12:13])
        gripper_pos2 = ft(q[:, 13:14])

        # action tensors: [L, future_steps, 1]
        agl_1_act    = ft(act[frame_offsets, 0:1])
        agl_2_act    = ft(act[frame_offsets, 1:2])
        agl_3_act    = ft(act[frame_offsets, 2:3])
        agl_4_act    = ft(act[frame_offsets, 3:4])
        agl_5_act    = ft(act[frame_offsets, 4:5])
        agl_6_act    = ft(act[frame_offsets, 5:6])
        gripper_act  = ft(act[frame_offsets, 6:7])
        agl2_1_act   = ft(act[frame_offsets, 7:8])
        agl2_2_act   = ft(act[frame_offsets, 8:9])
        agl2_3_act   = ft(act[frame_offsets, 9:10])
        agl2_4_act   = ft(act[frame_offsets, 10:11])
        agl2_5_act   = ft(act[frame_offsets, 11:12])
        agl2_6_act   = ft(act[frame_offsets, 12:13])
        gripper_act2 = ft(act[frame_offsets, 13:14])

        lowdim_dict = {
            'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4,
            'agl_5': agl_5, 'agl_6': agl_6,
            'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4,
            'agl2_5': agl2_5, 'agl2_6': agl2_6,
            'gripper_pos': gripper_pos, 'gripper_pos2': gripper_pos2,
            'agl_1_act': agl_1_act, 'agl_2_act': agl_2_act, 'agl_3_act': agl_3_act,
            'agl_4_act': agl_4_act, 'agl_5_act': agl_5_act, 'agl_6_act': agl_6_act,
            'agl2_1_act': agl2_1_act, 'agl2_2_act': agl2_2_act, 'agl2_3_act': agl2_3_act,
            'agl2_4_act': agl2_4_act, 'agl2_5_act': agl2_5_act, 'agl2_6_act': agl2_6_act,
            'gripper_act': gripper_act, 'gripper_act2': gripper_act2
        }

        if getattr(self, 'fitting', False):
            rgb_dict = {}
        else:
            rgb_dict = {}
            for cam in self.selected_cameras:
                with h5py.File(record_path, 'r') as root:
                    # L 프레임을 연속 슬라이스로 한 번에 읽음 -> I/O 효율 향상
                    imgs = root[f'observations/images/{cam}'][start_frame:end_frame]  # [L, H, W, C]
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.transpose(imgs, (0, 3, 1, 2))  # [L, C, H, W]
                rgb_dict[cam] = torch.from_numpy(imgs)

        return {
            'lowdim': lowdim_dict,
            'rgb': rgb_dict,
            'traj_idx': traj_idx,
            'is_first_chunk': int(start_frame == 0)
        }


def main():
    root_dir = "insert_data200"  # 데이터셋 디렉토리 경로
    dataset = MambaSequenceDataset(root_dir=root_dir, mode="train", use_pose10d=True)

    # 정규화 파라미터 계산
    scaler = dataset.fit_scaler(batch_size=64, num_workers=0)
    # 정규화 파라미터 저장
    dataset.save_scaler('scaler_params.pth')

    # __getitem__ 메서드 테스트
    data_dict = dataset[0]  # 첫 번째 데이터 가져오기

    # lowdim_dict와 rgb_dict의 각 텐서 차원 출력
    lowdim_dict = data_dict['lowdim']
    rgb_dict = data_dict['rgb']

    print("Lowdim dict dimensions:")
    for key, value in lowdim_dict.items():
        print(f"{key}: {value.shape}")

    print("\nRGB dict dimensions:")
    for key, value in rgb_dict.items():
        if value is not None:
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: None")



if __name__ == "__main__":
    main()
