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
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model * 2), 
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )

        # EMA Settings
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.current_momentum = base_momentum

    def update_ema(self, step, max_steps):
        """
        Adjust EMA momentum based on training progress.
        """
        self.current_momentum = self.base_momentum + \
            (self.final_momentum - self.base_momentum) * min(1.0, step / max_steps)
            
        with torch.no_grad():
            for c_param, t_param in zip(self.context_model.parameters(), self.target_model.parameters()):
                t_param.data.mul_(self.current_momentum).add_(c_param.data, alpha=1 - self.current_momentum)

    def forward(self, images, lowdim, mask_ratio=0.6):
        """
        images: dict of {cam_name: [B, L, 3, H, W]}
        lowdim: [B, L, D]
        """
        B, L, _ = lowdim.shape
        device = lowdim.device
        
        # 1. Split sequence (Context vs Target)
        context_len = max(1, int(L * (1 - mask_ratio)))
        
        # 2. Target Encoding (Full sequence)
        with torch.no_grad():
            target_repr_list = []
            h_target = self.target_model.init_hidden_states(B, device)
            for t in range(L):
                imgs_t = {cam: images[cam][:, t] for cam in self.target_model.camera_names}
                _, h_target, hidden = self.target_model.step(lowdim[:, t], imgs_t, h_target, return_repr=True)
                target_repr_list.append(hidden.unsqueeze(1)) 
            
            target_repr = torch.cat(target_repr_list, dim=1) 

        # 3. Context Encoding (Only context part)
        context_repr_list = []
        h_context = self.context_model.init_hidden_states(B, device)
        for t in range(context_len):
            imgs_t = {cam: images[cam][:, t] for cam in self.context_model.camera_names}
            _, h_context, hidden = self.context_model.step(lowdim[:, t], imgs_t, h_context, return_repr=True)
            context_repr_list.append(hidden.unsqueeze(1))
        
        last_context_repr = context_repr_list[-1].squeeze(1) 

        # 4. Predictor: Predict future Target representations
        loss = 0
        num_targets = L - context_len
        if num_targets > 0:
            for t_future in range(context_len, L):
                delta_t = torch.tensor([[float(t_future - context_len + 1)]], device=device).expand(B, 1)
                pred_input = torch.cat([last_context_repr, delta_t], dim=-1)
                predicted_repr = self.predictor(pred_input)
                actual_repr = target_repr[:, t_future]
                loss += F.mse_loss(predicted_repr, actual_repr)
            
            return loss / num_targets
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
