"""
inference_M_naive.py - Naive Action Chunking 추론 모델

기존 inference_M.py와의 차이:
  - get_action()이 action buffer를 관리
  - K 스텝마다 한 번만 policy.step() 호출 (Mamba hidden state 업데이트)
  - 그 사이 K-1 스텝은 버퍼에서 액션을 꺼내 실행
  => 학습(train_naive.py)에서 K 간격 샘플링과 완전히 일치
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np

from train.scaler_M import Scaler
from train.mamba_policy import MambaPolicy, MambaConfig
import train.mamba_policy as _mamba_policy_module
import train.scaler_M as _scaler_module
sys.modules.setdefault('mamba_policy', _mamba_policy_module)
sys.modules.setdefault('scaler_M', _scaler_module)


class MyInferenceModelNaive(nn.Module):
    """
    Naive chunking 추론 모델.
    get_action()을 매 스텝 호출하면 내부적으로:
      - buf_idx < K: 버퍼에서 액션 반환 (policy 미호출)
      - buf_idx == K: policy.step() 호출 → 새 K개 액션 버퍼에 저장
    """

    def __init__(self, checkpoint_path: str, scaler_path: str,
                 config: MambaConfig, lowdim_dict: dict,
                 future_steps: int = 16):
        super().__init__()
        self.future_steps = future_steps  # K

        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            sum_camera_feats=config.sum_camera_feats,
            num_blocks=config.num_blocks,
            mamba_cfg={
                'd_state': config.d_state,
                'd_conv': config.d_conv,
                'expand': config.expand,
                'headdim': config.headdim,
                'activation': config.activation,
                'use_mem_eff_path': config.use_mem_eff_path,
            }
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = Scaler(lowdim_dict=lowdim_dict)
        self.scaler.load(scaler_path)
        self.policy.to(self.device)

        print(f"[MyInferenceModelNaive] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt['state_dict']
        filtered = {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
        self.policy.load_state_dict(filtered, strict=True)
        print("[MyInferenceModelNaive] Weights loaded.")

        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)

        # action buffer: [K, 14] normalized
        self._action_buffer = None  # [K, 14]
        self._buf_idx = future_steps  # 처음에 즉시 쿼리하도록 K로 초기화

        self.cuda()
        self.eval()

    def reset(self):
        """에피소드 시작 시 hidden state와 action buffer 초기화."""
        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        self._action_buffer = None
        self._buf_idx = self.future_steps
        print("[MyInferenceModelNaive] Reset.")

    def get_action(self, lowdim_norm: torch.Tensor, rgb: dict) -> torch.Tensor:
        """
        매 스텝 호출. naive chunking으로 K 스텝마다 policy 쿼리.

        Args:
            lowdim_norm: [1, 14] 정규화된 저차원 상태
            rgb: {cam: [1, C, H, W]}

        Returns:
            action: [14] denormalized 액션 (numpy가 아닌 tensor)
        """
        with torch.no_grad():
            if self._buf_idx >= self.future_steps:
                # K 스텝이 지났거나 처음 → Mamba 쿼리
                pred_action, self.hiddens = self.policy.step(lowdim_norm, rgb, self.hiddens)
                # pred_action: [1, K, 14] (normalized)
                self._action_buffer = pred_action.squeeze(0)  # [K, 14]
                self._buf_idx = 0

            action_norm = self._action_buffer[self._buf_idx]   # [14]
            self._buf_idx += 1

        return action_norm  # normalized, denormalize는 호출부에서

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [..., 14] normalized
        returns: [..., 14] denormalized
        """
        arm1_dict = {
            'agl_1_act': actions[..., 0:1], 'agl_2_act': actions[..., 1:2],
            'agl_3_act': actions[..., 2:3], 'agl_4_act': actions[..., 3:4],
            'agl_5_act': actions[..., 4:5], 'agl_6_act': actions[..., 5:6],
            'gripper_act': actions[..., 6:7]
        }
        arm2_dict = {
            'agl2_1_act': actions[..., 7:8], 'agl2_2_act': actions[..., 8:9],
            'agl2_3_act': actions[..., 9:10], 'agl2_4_act': actions[..., 10:11],
            'agl2_5_act': actions[..., 11:12], 'agl2_6_act': actions[..., 12:13],
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
