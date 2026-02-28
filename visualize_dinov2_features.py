import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from train.mamba_policy import FrozenDinov2
from train.M_dataset import MambaSequenceDataset
import os

def visualize_trajectory_dinov2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 모델 로드 (-4 레이어)
    backbone = FrozenDinov2(layer_index=-4).to(device)
    backbone.eval()

    # 2. 데이터셋 로드 (첫 번째 궤적)
    root_path = "/home/jeonchanwook/MTIL/transfer.100"
    dataset = MambaSequenceDataset(root_dir=root_path, mode="train", selected_cameras=['top'])
    
    # 궤적 길이 확인 (기존 MambaSequenceDataset의 속성 활용)
    first_traj_len = dataset.lengths[0]
    indices = np.linspace(0, first_traj_len - 1, 10, dtype=int) # 궤적에서 10장 균등 샘플링

    all_features = []
    original_images = []

    print(f"Processing {len(indices)} frames from the first trajectory...")
    
    with torch.no_grad():
        for i in indices:
            data = dataset[i]
            img_tensor = data['rgb']['top'].unsqueeze(0).to(device) # [1, 3, 480, 640]
            original_images.append(data['rgb']['top'].permute(1, 2, 0).numpy())

            # DINOv2 특징 추출 [1, 1024, H_patch, W_patch]
            feat = backbone(img_tensor)
            all_features.append(feat)

    # 3. PyTorch SVD를 이용한 PCA (3개 성분 추출)
    # [N, 1024, H, W] -> [N*H*W, 1024]
    B, C, H, W = all_features[0].shape
    all_feats_tensor = torch.cat(all_features, dim=0) # [N, 1024, H, W]
    feats_flattened = all_feats_tensor.permute(0, 2, 3, 1).reshape(-1, C) # [N*H*W, 1024]
    
    # 중앙값 제거 (Centering)
    feats_mean = feats_flattened.mean(dim=0, keepdim=True)
    feats_centered = feats_flattened - feats_mean
    
    print("Running SVD for PCA on GPU...")
    # SVD를 통한 주성분 3개 추출
    U, S, V = torch.pca_lowrank(feats_centered, q=3)
    pca_features = torch.matmul(feats_centered, V[:, :3]) # [N*H*W, 3]
    
    # 정규화 (0~1)
    pca_min = pca_features.min(dim=0)[0]
    pca_max = pca_features.max(dim=0)[0]
    pca_features = (pca_features - pca_min) / (pca_max - pca_min + 1e-6)
    
    # [N, H, W, 3] 형태로 복원
    pca_images = pca_features.reshape(len(indices), H, W, 3).cpu().numpy()

    # 4. 시각화 결과 저장
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    
    for i in range(num_samples):
        # 원본 이미지
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title(f"Original Frame {indices[i]}")
        axes[i, 0].axis('off')
        
        # DINOv2 PCA 특징 지도 (업샘플링)
        vis_feat = cv2.resize(pca_images[i], (640, 480), interpolation=cv2.INTER_NEAREST)
        axes[i, 1].imshow(vis_feat)
        axes[i, 1].set_title(f"DINOv2 Feature PCA (-4 layer)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    os.makedirs("video", exist_ok=True)
    save_path = "video/dinov2_vis.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_trajectory_dinov2()
