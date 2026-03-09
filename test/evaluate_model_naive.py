"""
evaluate_model_naive.py - Naive Action Chunking 평가

기존 evaluate_model_transfer.py와의 차이:
  - MyInferenceModelNaive 사용
  - query_frequency = future_steps (K 스텝마다 쿼리)
  - temporal ensembling 없음 (버퍼에서 직접 실행)
  - 학습(train_naive.py)과 동일한 K 간격 패턴 → train-test 일치
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference_M_naive import MyInferenceModelNaive
import numpy as np
import torch
from einops import rearrange
from sim_env import make_sim_env, BOX_POSE
from visualize_episodes import save_videos
from train.mamba_policy import MambaConfig
from train.scaler_M import Scaler


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    return np.concatenate([cube_position, np.array([1, 0, 0, 0])])


def get_image(ts, camera_names):
    curr_images = {}
    for cam_name in camera_names:
        if cam_name in ts.observation['images']:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images[cam_name] = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_images


# ── 설정 ──────────────────────────────────────────────────────────────────────
scaler_path  = 'scaler_params.pth'
checkpoint   = '/home/jeonchanwook/MTIL.main/logs/naive/transfer20/lightning_logs/version_0/checkpoints/last.ckpt'
results_dir  = 'video_naive'
FUTURE_STEPS = 16   # K: train_naive.py와 반드시 동일하게
camera_names = ['top']

config = MambaConfig()
config.camera_names = camera_names
config.embed_dim    = 2048
config.lowdim_dim   = 14
config.d_model      = 2048
config.action_dim   = 14
config.sum_camera_feats = False
config.num_blocks   = 4

lowdim_dict = {
    'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
    'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
    'gripper_pos': 1, 'gripper_pos2': 1,
    'agl_1_act': (FUTURE_STEPS, 1), 'agl_2_act': (FUTURE_STEPS, 1),
    'agl_3_act': (FUTURE_STEPS, 1), 'agl_4_act': (FUTURE_STEPS, 1),
    'agl_5_act': (FUTURE_STEPS, 1), 'agl_6_act': (FUTURE_STEPS, 1),
    'agl2_1_act': (FUTURE_STEPS, 1), 'agl2_2_act': (FUTURE_STEPS, 1),
    'agl2_3_act': (FUTURE_STEPS, 1), 'agl2_4_act': (FUTURE_STEPS, 1),
    'agl2_5_act': (FUTURE_STEPS, 1), 'agl2_6_act': (FUTURE_STEPS, 1),
    'gripper_act': (FUTURE_STEPS, 1), 'gripper_act2': (FUTURE_STEPS, 1)
}

max_timesteps = 400
num_rollouts  = 50
DT = 0.02

os.makedirs(results_dir, exist_ok=True)

# ── 모델 초기화 ────────────────────────────────────────────────────────────────
infer_model = MyInferenceModelNaive(
    checkpoint_path=checkpoint,
    scaler_path=scaler_path,
    config=config,
    lowdim_dict=lowdim_dict,
    future_steps=FUTURE_STEPS,
).to('cuda')

# 웜업
print("Warmup...")
rgb_dummy  = {cam: torch.zeros(1, 3, 480, 640).cuda() for cam in camera_names}
lowdim_dummy = torch.zeros(1, 14).cuda()
infer_model.get_action(lowdim_dummy, rgb_dummy)
infer_model.reset()
print("Warmup done.")

# ── 환경 ──────────────────────────────────────────────────────────────────────
task_name = 'sim_transfer_cube'
env = make_sim_env(task_name)
env_max_reward = env.task.max_reward

episode_returns  = []
highest_rewards  = []

# ── 평가 루프 ──────────────────────────────────────────────────────────────────
for rollout_id in range(num_rollouts):
    BOX_POSE[0] = sample_box_pose()
    ts = env.reset()
    infer_model.reset()

    image_list = []
    qpos_list  = []
    target_qpos_list = []
    rewards    = []

    with torch.inference_mode():
        for t in range(max_timesteps):
            obs = ts.observation
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})

            qpos_numpy = np.array(obs['qpos'])
            dev = infer_model.device

            # 저차원 상태 정규화
            agl_1  = torch.from_numpy(qpos_numpy[0:1]).to(dev)
            agl_2  = torch.from_numpy(qpos_numpy[1:2]).to(dev)
            agl_3  = torch.from_numpy(qpos_numpy[2:3]).to(dev)
            agl_4  = torch.from_numpy(qpos_numpy[3:4]).to(dev)
            agl_5  = torch.from_numpy(qpos_numpy[4:5]).to(dev)
            agl_6  = torch.from_numpy(qpos_numpy[5:6]).to(dev)
            grip1  = torch.from_numpy(qpos_numpy[6:7]).to(dev)
            agl2_1 = torch.from_numpy(qpos_numpy[7:8]).to(dev)
            agl2_2 = torch.from_numpy(qpos_numpy[8:9]).to(dev)
            agl2_3 = torch.from_numpy(qpos_numpy[9:10]).to(dev)
            agl2_4 = torch.from_numpy(qpos_numpy[10:11]).to(dev)
            agl2_5 = torch.from_numpy(qpos_numpy[11:12]).to(dev)
            agl2_6 = torch.from_numpy(qpos_numpy[12:13]).to(dev)
            grip2  = torch.from_numpy(qpos_numpy[13:14]).to(dev)

            norm1 = infer_model.scaler.normalize(
                {'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4,
                 'agl_5': agl_5, 'agl_6': agl_6, 'gripper_pos': grip1})
            norm2 = infer_model.scaler.normalize(
                {'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4,
                 'agl2_5': agl2_5, 'agl2_6': agl2_6, 'gripper_pos2': grip2})

            lowdim_norm = torch.cat([
                norm1['agl_1'], norm1['agl_2'], norm1['agl_3'], norm1['agl_4'],
                norm1['agl_5'], norm1['agl_6'], norm1['gripper_pos'],
                norm2['agl2_1'], norm2['agl2_2'], norm2['agl2_3'], norm2['agl2_4'],
                norm2['agl2_5'], norm2['agl2_6'], norm2['gripper_pos2']
            ], dim=0).unsqueeze(0).float()  # [1, 14]

            curr_image = get_image(ts, camera_names)

            # naive chunking: get_action이 K 스텝마다 내부적으로 policy 쿼리
            action_norm = infer_model.get_action(lowdim_norm, curr_image)  # [14]
            action_denorm = infer_model.denormalize(action_norm.unsqueeze(0))  # [1, 14]
            target_qpos = action_denorm.squeeze(0).cpu().numpy()

            ts = env.step(target_qpos)
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            rewards.append(ts.reward)

    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards is not None])
    episode_returns.append(episode_return)
    episode_highest_reward = np.max(rewards)
    highest_rewards.append(episode_highest_reward)

    print(f'Rollout {rollout_id}: return={episode_return:.2f}, '
          f'highest_reward={episode_highest_reward}, '
          f'success={episode_highest_reward == env_max_reward}')

    save_videos(image_list, DT, video_path=os.path.join(results_dir, f'video{rollout_id}.mp4'))

# ── 최종 통계 ──────────────────────────────────────────────────────────────────
success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
avg_return   = np.mean(episode_returns)
summary_str  = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
for r in range(env_max_reward + 1):
    cnt  = (np.array(highest_rewards) >= r).sum()
    rate = cnt / num_rollouts
    summary_str += f'Reward >= {r}: {cnt}/{num_rollouts} = {rate * 100:.1f}%\n'

print(summary_str)
