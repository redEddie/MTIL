import os
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))
import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from test.sim_env import make_sim_env, BOX_POSE
from test.visualize_episodes import save_videos

from mamba_policy import MambaConfig
from post_training_sa import LitPostTraiingSA
from scaler_M import Scaler
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def get_image(ts, camera_names):
    curr_images = {}  
    for cam_name in camera_names:
        if cam_name in ts.observation['images']:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images[cam_name] = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0) # [1, 1, C, H, W] for sequence format
        else:
            print(f"Warning: Camera '{cam_name}' not found in images.")
    return curr_images

# --- Configuration paths ---
scaler_path = '../train/scaler_params.pth'  # Needs to exist
# TODO: Update this to point to your actual post_training_sa .ckpt file
checkpoint = '../lightning_logs/post_traiing_sa_saz/version_8/checkpoints/last.ckpt' 
# TODO: Update this to point to your JEPA pre-trained checkpoint that the policy model requires
jepa_ckpt_path = '../train/lightning_logs/jepa_pretrain_final/version_2/checkpoints/epoch=2-step=2616.ckpt'
results_dir = 'jepa.video'  
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Initialize configuration
config = MambaConfig()
config.camera_names = ['top']
config.img_size = (640, 480)
config.num_blocks = 4
config.max_t = 10 

policy_input_mode = "saz" # Make sure this matches how you trained it
future_steps = 16
action_dim = 14
state_dim = 14

lowdim_dict = {
    'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
    'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
    'gripper_pos': 1,
    'gripper_pos2': 1,
    'agl_1_act': (16, 1), 'agl_2_act': (16, 1), 'agl_3_act': (16, 1),
    'agl_4_act': (16, 1), 'agl_5_act': (16, 1), 'agl_6_act': (16, 1),
    'agl2_1_act': (16, 1), 'agl2_2_act': (16, 1), 'agl2_3_act': (16, 1),
    'agl2_4_act': (16, 1), 'agl2_5_act': (16, 1), 'agl2_6_act': (16, 1),
    'gripper_act': (16, 1), 'gripper_act2': (16, 1)
}

scaler = Scaler(lowdim_dict=lowdim_dict)
scaler.load(scaler_path)
print("Scaler loaded.")

# Load JEPA Policy Model
try:
    infer_model = LitPostTraiingSA.load_from_checkpoint(
        checkpoint,
        config=config,
        jepa_ckpt_path=jepa_ckpt_path,
        future_steps=future_steps,
        action_dim=action_dim,
        state_dim=state_dim,
        policy_input_mode=policy_input_mode
    )
    infer_model = infer_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    infer_model.eval()
    
    # Move scaler to the same device as the model
    scaler.to(infer_model.device)
    
    print("JEPA Inference Model Loaded.")
except Exception as e:
    print(f"Error loading model check paths: {e}")
    sys.exit(1)

task_name = 'sim_transfer_cube'
env = make_sim_env(task_name)
env_max_reward = env.task.max_reward
query_frequency = 1  
max_timesteps = 400  
num_queries = future_steps  
num_rollouts = 50  
episode_returns = []  
highest_rewards = []  
DT = 0.02

# We need a context history of images for JEPA (max_t=10)
history_frames = config.max_t

with torch.inference_mode():
    for rollout_id in range(num_rollouts):
        BOX_POSE[0] = sample_box_pose()
        all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim]).cuda()
        ts = env.reset()
        
        image_list = []  
        qpos_list = []
        target_qpos_list = []
        rewards = []
        inference_times = []
        temporal_agg = True  

        # Image history buffer for JEPA
        rgb_buffer = []
        frame_skip = 10 

        for t in tqdm(range(max_timesteps), desc=f"Rollout {rollout_id}", leave=False):
            obs = ts.observation
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
                
            qpos_numpy = np.array(obs['qpos'])
            
            # --- Format State Features ---
            agl_1 = torch.from_numpy(qpos_numpy[0:1]).to(infer_model.device)
            agl_2 = torch.from_numpy(qpos_numpy[1:2]).to(infer_model.device)
            agl_3 = torch.from_numpy(qpos_numpy[2:3]).to(infer_model.device)
            agl_4 = torch.from_numpy(qpos_numpy[3:4]).to(infer_model.device)
            agl_5 = torch.from_numpy(qpos_numpy[4:5]).to(infer_model.device)
            agl_6 = torch.from_numpy(qpos_numpy[5:6]).to(infer_model.device)
            gripper_pos = torch.from_numpy(qpos_numpy[6:7]).to(infer_model.device)
            agl2_1 = torch.from_numpy(qpos_numpy[7:8]).to(infer_model.device)
            agl2_2 = torch.from_numpy(qpos_numpy[8:9]).to(infer_model.device)
            agl2_3 = torch.from_numpy(qpos_numpy[9:10]).to(infer_model.device)
            agl2_4 = torch.from_numpy(qpos_numpy[10:11]).to(infer_model.device)
            agl2_5 = torch.from_numpy(qpos_numpy[11:12]).to(infer_model.device)
            agl2_6 = torch.from_numpy(qpos_numpy[12:13]).to(infer_model.device)
            gripper_pos2 = torch.from_numpy(qpos_numpy[13:14]).to(infer_model.device)

            lowdim_arm1_norm = scaler.normalize(
                {'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4,
                 'agl_5': agl_5, 'agl_6': agl_6, 'gripper_pos': gripper_pos})
            lowdim_arm1_norm = torch.cat([lowdim_arm1_norm['agl_1'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_2'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_3'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_4'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_5'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_6'].unsqueeze(1),
                                          lowdim_arm1_norm['gripper_pos'].unsqueeze(1)], dim=1) 
                                          
            lowdim_arm2_norm = scaler.normalize(
                {'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4,
                 'agl2_5': agl2_5, 'agl2_6': agl2_6, 'gripper_pos2': gripper_pos2})
            lowdim_arm2_norm = torch.cat([lowdim_arm2_norm['agl2_1'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_2'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_3'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_4'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_5'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_6'].unsqueeze(1),
                                          lowdim_arm2_norm['gripper_pos2'].unsqueeze(1)], dim=1) 

            # Current state shape: [1, 14]
            state = torch.cat([lowdim_arm1_norm, lowdim_arm2_norm], dim=1).float()
            
            # --- Format Image Features ---
            curr_image_dict = get_image(ts, config.camera_names)
            curr_img_tensor = curr_image_dict['top'] # [1, 1, 3, 480, 640]
            
            rgb_buffer.append(curr_img_tensor)
            
            # Extract history frames using frame_skip
            padded_buffer = []
            for f in range(history_frames - 1, -1, -1): # From oldest to newest
                idx = len(rgb_buffer) - 1 - (f * frame_skip)
                if idx >= 0:
                    padded_buffer.append(rgb_buffer[idx])
                else:
                    padded_buffer.append(rgb_buffer[0]) # Pad early frames with the first frame
                    
            # Combine history: [1, history_frames, 3, 480, 640]
            rgb_hist = {'top': torch.cat(padded_buffer, dim=1)}

            # --- Inference ---
            if t % query_frequency == 0:
                inf_start = time.time()
                
                # Encode images with JEPA
                z = infer_model.jepa.encode_z(rgb_hist) # [1, 1024]
                
                # Run SAMambaPolicyHead
                if infer_model.policy_input_mode == "sz":
                    head_input = torch.cat([state, z], dim=-1).float() 
                else: 
                     # For 'saz' mode, we need previous action. 
                     prev_a = torch.zeros(1, action_dim).to(state.device) if t == 0 else raw_action_t
                     head_input = torch.cat([state, prev_a, z], dim=-1).float()

                pred_action_norm = infer_model.head(head_input) # [1, 16, 14]
                
                # Denormalize output
                denorm_dict = {}
                for idx, k in enumerate(['agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act', 'gripper_act',
                                         'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act', 'gripper_act2']):
                    denorm_dict[k] = pred_action_norm[:, :, idx:idx+1]
                
                pred_denorm = scaler.denormalize(denorm_dict)
                a_hat = torch.cat([pred_denorm[k] for k in ['agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act', 'gripper_act',
                                         'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act', 'gripper_act2']], dim=-1)
                
                all_actions = a_hat
                inference_times.append(time.time() - inf_start)
                
            if temporal_agg:
                all_time_actions[[t], t:t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]

            raw_action_t = raw_action # Save for next step in 'saz' mode
            target_qpos = raw_action.squeeze(0).cpu().numpy()
            ts = env.step(target_qpos)
            
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            rewards.append(ts.reward)
            
        plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards is not None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        avg_inf_ms = np.mean(inference_times) * 1000
        fps = 1.0 / np.mean(inference_times)
        
        print(f'Rollout {rollout_id}\t{episode_return=}\t{episode_highest_reward=}\tSuccess: {episode_highest_reward == env_max_reward}')
        print(f'Avg Inference Time: {avg_inf_ms:.2f} ms, FPS: {fps:.2f}')
        save_videos(image_list, DT, video_path=os.path.join(results_dir, f'video{rollout_id}.mp4'))

success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
avg_return = np.mean(episode_returns)
summary_str = f'\nOverall Success rate: {success_rate * 100}%\nAverage return: {avg_return}\n'
print(summary_str)
