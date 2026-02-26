import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
try:
    from train.mamba_policy import MambaPolicy, MambaConfig
except ImportError:
    from mamba_policy import MambaPolicy, MambaConfig

class MambaJEPA(nn.Module):
    def __init__(self, config: MambaConfig, base_momentum=0.9, final_momentum=0.999):
        super().__init__()
        # 1. Context Model (Optimized)
        self.context_model = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            num_blocks=config.num_blocks,
            img_size=config.img_size
        )
        
        # 2. Target Model (EMA Updated, No Gradient)
        self.target_model = deepcopy(self.context_model)
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        # 3. Predictor (MLP)
        # Input: [Hidden_State(t), delta_t]
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model * 2), 
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )

        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.current_momentum = base_momentum

    def update_ema(self, step, max_steps):
        self.current_momentum = self.base_momentum + \
            (self.final_momentum - self.base_momentum) * min(1.0, step / max_steps)
            
        with torch.no_grad():
            for c_param, t_param in zip(self.context_model.parameters(), self.target_model.parameters()):
                t_param.data.mul_(self.current_momentum).add_(c_param.data, alpha=1 - self.current_momentum)

    def forward(self, images, lowdim, num_context_points=10, prediction_horizon=16):
        """
        images: dict of {cam_name: [B, L, 3, H, W]}
        lowdim: [B, L, D]
        num_context_points: 한 궤적 내에서 학습할 'Context 종료 지점'의 개수
        prediction_horizon: 각 지점에서 미래로 몇 스텝을 예측할지
        """
        B, L, _ = lowdim.shape
        device = lowdim.device
        
        # 1. Target Encoding (전체 시퀀스의 표현형을 먼저 추출)
        with torch.no_grad():
            target_repr_list = []
            h_target = self.target_model.init_hidden_states(B, device)
            for t in range(L):
                imgs_t = {cam: images[cam][:, t] for cam in self.target_model.camera_names}
                _, h_target, hidden = self.target_model.step(lowdim[:, t], imgs_t, h_target, return_repr=True)
                target_repr_list.append(hidden.unsqueeze(1)) 
            target_repr = torch.cat(target_repr_list, dim=1) # [B, L, d_model]

        # 2. Context Encoding (전체 시퀀스를 훑으며 각 시점의 Hidden State 추출)
        context_repr_list = []
        pred_actions_list = []
        h_context = self.context_model.init_hidden_states(B, device)
        for t in range(L):
            imgs_t = {cam: images[cam][:, t] for cam in self.context_model.camera_names}
            pred_act, h_context, hidden = self.context_model.step(lowdim[:, t], imgs_t, h_context, return_repr=True)
            context_repr_list.append(hidden.unsqueeze(1))
            pred_actions_list.append(pred_act.unsqueeze(1))
        
        context_reprs = torch.cat(context_repr_list, dim=1) # [B, L, d_model]
        pred_actions = torch.cat(pred_actions_list, dim=1) # [B, L, future_steps, action_dim]

        # 3. Predictor Training (여러 시점을 Context로 삼아 학습)
        # 너무 촘촘하면 계산량이 많으므로 num_context_points 만큼 무작위 지점을 선택
        # 단, 미래 데이터가 남아있는 지점들 중에서 선택 (1 ~ L - 2)
        if L < 5: # 궤적이 너무 짧으면 처리 불가
            return torch.tensor(0.0, device=device, requires_grad=True), pred_actions

        loss = 0
        actual_points = 0
        
        # Context 종료 지점 샘플링
        indices = torch.randperm(L - 1)[:num_context_points].tolist()
        
        for t_ctx in indices:
            ctx_v = context_reprs[:, t_ctx] # t_ctx 시점까지의 누적 정보
            
            # 해당 시점으로부터 미래 prediction_horizon 만큼 예측
            max_h = min(prediction_horizon, L - 1 - t_ctx)
            if max_h <= 0: continue
            
            for h in range(1, max_h + 1):
                t_future = t_ctx + h
                delta_t = torch.tensor([[float(h)]], device=device).expand(B, 1)
                
                pred_input = torch.cat([ctx_v, delta_t], dim=-1)
                predicted_repr = self.predictor(pred_input)
                actual_target_v = target_repr[:, t_future]
                
                loss += F.mse_loss(predicted_repr, actual_target_v)
                actual_points += 1
        
        if actual_points > 0:
            return loss / actual_points, pred_actions
        else:
            return torch.tensor(0.0, device=device, requires_grad=True), pred_actions
