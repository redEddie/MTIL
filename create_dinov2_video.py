import torch
import numpy as np
import cv2
from tqdm import tqdm
from train.mamba_policy import FrozenDinov2
from train.M_dataset import MambaSequenceDataset
import os

def create_dinov2_video():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 모델 로드 (-4 레이어)
    backbone = FrozenDinov2(layer_index=-4).to(device)
    backbone.eval()

    # 2. 데이터셋 로드
    root_path = "/home/jeonchanwook/MTIL/transfer.100"
    dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=['top'])
    
    # 첫 번째 궤적 전체 프레임 인덱스
    first_traj_len = dataset.lengths[0]
    indices = range(first_traj_len)

    all_features = []
    original_images = []

    print(f"Extracting features for {first_traj_len} frames...")
    
    with torch.no_grad():
        for i in tqdm(indices):
            data = dataset[i]
            img_tensor = data['rgb']['top'].unsqueeze(0).to(device)
            # 원본 이미지 저장 (0~255 uint8 BGR for OpenCV)
            img_np = (data['rgb']['top'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            original_images.append(img_bgr)

            # DINOv2 특징 추출 후 즉시 CPU로 이동하여 GPU 메모리 해제
            feat = backbone(img_tensor).cpu()
            all_features.append(feat)

    # 3. 전체 궤적에 대한 PCA (CPU에서 수행하여 GPU OOM 방지)
    B, C, H, W = all_features[0].shape
    all_feats_tensor = torch.cat(all_features, dim=0) # [L, 1024, H, W]
    feats_flattened = all_feats_tensor.permute(0, 2, 3, 1).reshape(-1, C) # [L*H*W, 1024]
    
    feats_mean = feats_flattened.mean(dim=0, keepdim=True)
    feats_centered = feats_flattened - feats_mean
    
    print("Running SVD on CPU for consistent color mapping (to avoid GPU OOM)...")
    # torch.pca_lowrank는 CPU에서도 잘 동작합니다.
    U, S, V = torch.pca_lowrank(feats_centered, q=3)
    pca_features = torch.matmul(feats_centered, V[:, :3])
    
    pca_min = pca_features.min(dim=0)[0]
    pca_max = pca_features.max(dim=0)[0]
    pca_features = (pca_features - pca_min) / (pca_max - pca_min + 1e-6)
    
    pca_images = pca_features.reshape(len(indices), H, W, 3).numpy()

    # 4. 동영상 저장 (OpenCV)
    os.makedirs("video", exist_ok=True)
    video_path = "video/dinov2_trajectory.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 480))

    print(f"Saving video to {video_path}...")
    for i in tqdm(range(len(indices)), desc="Writing video"):
        # PCA 특징 지도를 640x480으로 확대
        vis_feat = (pca_images[i] * 255).astype(np.uint8)
        vis_feat_bgr = cv2.cvtColor(vis_feat, cv2.COLOR_RGB2BGR)
        vis_feat_resized = cv2.resize(vis_feat_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
        
        # 원본과 특징 지도를 좌우로 결합
        combined_frame = np.hstack((original_images[i], vis_feat_resized))
        out.write(combined_frame)

    out.release()
    print(f"Done! Video saved at {video_path}")

if __name__ == "__main__":
    create_dinov2_video()
