import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import tqdm
from scaler_M import Scaler
import h5py


class MambaSequenceDatasetNaive(Dataset):
    """
    Naive Action Chunking 버전 Dataset.

    기존 M_dataset.py와의 차이:
      - 기존: 매 프레임마다 (o_t → [a_t..a_{t+K-1}]) 학습
              => 추론에서 매 스텝 쿼리하면 일치, naive chunk 실행이면 불일치

      - 이 버전: query_frequency = K 간격으로만 샘플링
              (o_0 → [a_0..a_{K-1}], o_K → [a_K..a_{2K-1}], ...)
              => 추론에서 K 스텝마다 쿼리하는 naive 방식과 완전히 일치

    chunk_size: 한 번에 Mamba에 공급할 query 횟수 (L).
                실제 프레임 span = L * future_steps.
    """

    def __init__(self, root_dir: str, mode: str = "train", use_pose10d: bool = True,
                 resize_hw=(640, 480), selected_cameras: List[str] = None,
                 scaler: Optional[Scaler] = None,
                 future_steps: int = 16,
                 chunk_size: int = 8):
        super().__init__()
        assert mode in ["train", "test"]
        self.dataset_dir = os.path.join(root_dir, mode)
        self.use_pose10d = use_pose10d
        self.resize_hw = resize_hw
        self.future_steps = future_steps   # K: action chunk 크기 = query 주기
        self.chunk_size = chunk_size       # L: Mamba 시퀀스 길이 (query 횟수 단위)
        self.selected_cameras = selected_cameras or ['top']

        # 궤적 로드
        self.records = []
        self.lengths = []
        for item in sorted(os.listdir(self.dataset_dir)):
            record_path = os.path.join(self.dataset_dir, item)
            self.records.append(record_path)
            with h5py.File(record_path, 'r') as root:
                self.lengths.append(root['/observations/qpos'].shape[0])

        # chunk 인덱스 빌드: (traj_idx, q_start, q_end) - query step 단위
        #   query step i  =>  실제 프레임 i * K
        #   유효 query 수 = traj_len // K  (마지막 K개 액션이 온전히 존재하는 범위)
        self.chunks = []
        for traj_idx, traj_len in enumerate(self.lengths):
            num_valid_queries = traj_len // future_steps
            if num_valid_queries == 0:
                continue
            for q_start in range(0, num_valid_queries, chunk_size):
                q_end = min(q_start + chunk_size, num_valid_queries)
                self.chunks.append((traj_idx, q_start, q_end))

        self.lowdim_shapes = {
            'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
            'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
            'gripper_pos': 1, 'gripper_pos2': 1,
            'agl_1_act': (future_steps, 1), 'agl_2_act': (future_steps, 1),
            'agl_3_act': (future_steps, 1), 'agl_4_act': (future_steps, 1),
            'agl_5_act': (future_steps, 1), 'agl_6_act': (future_steps, 1),
            'agl2_1_act': (future_steps, 1), 'agl2_2_act': (future_steps, 1),
            'agl2_3_act': (future_steps, 1), 'agl2_4_act': (future_steps, 1),
            'agl2_5_act': (future_steps, 1), 'agl2_6_act': (future_steps, 1),
            'gripper_act': (future_steps, 1), 'gripper_act2': (future_steps, 1)
        }

        self.scaler = scaler
        if self.scaler is None and mode == "train":
            self.scaler = Scaler(lowdim_dict=self.lowdim_shapes)
        self.fitting = False

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, q_start, q_end = self.chunks[chunk_idx]
        record_path = self.records[traj_idx]
        traj_len = self.lengths[traj_idx]
        K = self.future_steps
        L = q_end - q_start  # 실제 query 횟수 (Mamba 시퀀스 길이)

        with h5py.File(record_path, 'r') as root:
            qpos = root['/observations/qpos'][()]   # [T, 14]
            act  = root['/action'][()]              # [T, 14]

        # query step → 실제 프레임 인덱스: [0, K, 2K, ...]
        obs_frame_indices = np.arange(q_start, q_end) * K   # [L]

        # 관측: query 프레임만 샘플링 [L, 14]
        q = qpos[obs_frame_indices]

        # 액션: 각 query 프레임부터 K개 [L, K]
        action_offsets = (
            obs_frame_indices[:, None]              # [L, 1]
            + np.arange(K)[None, :]                 # [1, K]
        )  # [L, K]
        action_offsets = np.clip(action_offsets, 0, traj_len - 1)

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

        # action tensors: [L, K, 1]
        agl_1_act    = ft(act[action_offsets, 0:1])
        agl_2_act    = ft(act[action_offsets, 1:2])
        agl_3_act    = ft(act[action_offsets, 2:3])
        agl_4_act    = ft(act[action_offsets, 3:4])
        agl_5_act    = ft(act[action_offsets, 4:5])
        agl_6_act    = ft(act[action_offsets, 5:6])
        gripper_act  = ft(act[action_offsets, 6:7])
        agl2_1_act   = ft(act[action_offsets, 7:8])
        agl2_2_act   = ft(act[action_offsets, 8:9])
        agl2_3_act   = ft(act[action_offsets, 9:10])
        agl2_4_act   = ft(act[action_offsets, 10:11])
        agl2_5_act   = ft(act[action_offsets, 11:12])
        agl2_6_act   = ft(act[action_offsets, 12:13])
        gripper_act2 = ft(act[action_offsets, 13:14])

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

        rgb_dict = {}
        if not getattr(self, 'fitting', False):
            for cam in self.selected_cameras:
                with h5py.File(record_path, 'r') as root:
                    # K 간격의 비연속 프레임 읽기 (fancy indexing)
                    imgs = root[f'observations/images/{cam}'][obs_frame_indices]  # [L, H, W, C]
                imgs = imgs.astype(np.float32) / 255.0
                imgs = np.transpose(imgs, (0, 3, 1, 2))  # [L, C, H, W]
                rgb_dict[cam] = torch.from_numpy(imgs)

        return {
            'lowdim': lowdim_dict,
            'rgb': rgb_dict,
            'traj_idx': traj_idx,
            'is_first_chunk': int(q_start == 0)
        }
