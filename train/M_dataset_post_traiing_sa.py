import os
from typing import Dict, List

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SAPostTraiingDataset(Dataset):
    """
    (s, a, z) post-training dataset.
    - s: current state qpos[t]
    - a: future action chunk action[t:t+future_steps] (training target)
    - prev_action: action[t-1] (optional policy input for SAZ mode)
    - z input: image history I[t-history_steps+1:t] with repeat-first-frame padding
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        history_steps: int = 16,
        future_steps: int = 16,
        selected_cameras: List[str] = None,
        resize_hw=(640, 480),
    ):
        super().__init__()
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.dataset_dir = os.path.join(root_dir, mode)
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.selected_cameras = selected_cameras or ["top"]
        self.resize_hw = resize_hw

        self.records = []
        self.lengths = []
        for item in sorted(os.listdir(self.dataset_dir)):
            record_path = os.path.join(self.dataset_dir, item)
            if not os.path.isfile(record_path):
                continue
            with h5py.File(record_path, "r") as root:
                seq_len = root["/observations/qpos"].shape[0]
                if seq_len <= 0:
                    continue
            self.records.append(record_path)
            self.lengths.append(seq_len)

        self.cum_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _global_to_local(self, idx: int):
        traj_idx = int(np.searchsorted(self.cum_lengths, idx, side="right") - 1)
        frame_idx = int(idx - self.cum_lengths[traj_idx])
        return traj_idx, frame_idx

    def _to_chw_float(self, img_hwc: np.ndarray) -> np.ndarray:
        if self.resize_hw is not None:
            w, h = self.resize_hw
            if img_hwc.shape[1] != w or img_hwc.shape[0] != h:
                img_hwc = cv2.resize(img_hwc, (w, h), interpolation=cv2.INTER_AREA)
        img = img_hwc.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, frame_idx = self._global_to_local(idx)
        record_path = self.records[traj_idx]

        with h5py.File(record_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
            seq_len = qpos.shape[0]

            state_t = qpos[frame_idx]
            if frame_idx > 0:
                prev_action = action[frame_idx - 1]
                prev_action_valid = 1.0
            else:
                prev_action = np.zeros_like(action[0], dtype=np.float32)
                prev_action_valid = 0.0

            action_indices = np.clip(
                np.arange(frame_idx, frame_idx + self.future_steps),
                0,
                seq_len - 1,
            )
            action_chunk = action[action_indices]

            valid_len = min(frame_idx + 1, self.history_steps)
            history_indices = []
            start_idx = frame_idx - self.history_steps + 1
            for offset in range(self.history_steps):
                src_idx = start_idx + offset
                if src_idx < 0:
                    src_idx = 0
                history_indices.append(src_idx)

            rgb_hist = {}
            for cam in self.selected_cameras:
                cam_ds = root[f"observations/images/{cam}"]
                frames = [self._to_chw_float(cam_ds[src_idx]) for src_idx in history_indices]
                rgb_hist[cam] = torch.from_numpy(np.stack(frames, axis=0)).float()

        history_mask = torch.zeros(self.history_steps, dtype=torch.float32)
        history_mask[-valid_len:] = 1.0
        valid_ratio = torch.tensor(float(valid_len) / float(self.history_steps), dtype=torch.float32)

        return {
            "state": torch.from_numpy(state_t).float(),
            "prev_action": torch.from_numpy(prev_action).float(),
            "prev_action_valid": torch.tensor(prev_action_valid, dtype=torch.float32),
            "action": torch.from_numpy(action_chunk).float(),
            "rgb_hist": rgb_hist,
            "traj_idx": torch.tensor(traj_idx, dtype=torch.long),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "valid_len": torch.tensor(valid_len, dtype=torch.long),
            "valid_ratio": valid_ratio,
            "history_mask": history_mask,
        }
