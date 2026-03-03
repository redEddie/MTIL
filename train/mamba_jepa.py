import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange, repeat
import numpy as np
from mamba_policy import Mamba2, Block, FrozenDinov2, MambaConfig

class MambaJEPA(nn.Module):
    def __init__(self, config: MambaConfig, mask_ratio=0.6, base_momentum=0.996):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.mask_ratio = mask_ratio
        
        # 1. 시각 백본 (DINOv2)
        self.backbone = FrozenDinov2(layer_index=-4)
        dinov2_dim = 1024 
        
        # 2. 패치 토큰 매핑 (Pre-alignment)
        self.patch_proj = nn.Sequential(
            nn.Linear(dinov2_dim, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # 3. 시공간 임베딩 (Space: 34x45 Sin/Cos, Time: config.max_t Learnable)
        self.num_h, self.num_w = 34, 45 
        self.max_t = config.max_t
        
        pos_embed = self._get_2d_sincos_pos_embed(self.d_model, (self.num_h, self.num_w))
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=True)
        
        self.time_embed = nn.Parameter(torch.zeros(1, self.max_t, 1, self.d_model))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        
        # 4. Encoder & Predictor 블록
        def mlp_fn(dim):
            hidden_dim = 4 * dim
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

        self.context_encoder = nn.ModuleList([
            Block(dim=self.d_model, mixer_cls=lambda d: Mamba2(d, d_state=64, expand=2), 
                  mlp_cls=mlp_fn, norm_cls=nn.LayerNorm)
            for _ in range(config.num_blocks)
        ])
        
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = nn.ModuleList([
            Block(dim=self.d_model, mixer_cls=lambda d: Mamba2(d, d_state=64, expand=2), 
                  mlp_cls=mlp_fn, norm_cls=nn.LayerNorm)
            for _ in range(2) 
        ])
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.base_momentum = base_momentum
        self.current_momentum = base_momentum

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
        
        emb_h = self._get_1d_sincos_pos_embed_from_values(embed_dim // 2, grid[1].flatten())
        emb_w = self._get_1d_sincos_pos_embed_from_values(embed_dim // 2, grid[0].flatten())
        return np.concatenate([emb_h, emb_w], axis=1)

    def _get_1d_sincos_pos_embed_from_values(self, embed_dim, pos):
        omega = 1. / 10000**(np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.))
        out = np.einsum('m,d->md', pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    def update_ema(self, step, max_steps):
        self.current_momentum = 1.0 - (1.0 - self.base_momentum) * (torch.cos(torch.tensor(step * 3.14159 / max_steps)) + 1) / 2
        with torch.no_grad():
            for c_param, t_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                t_param.data.mul_(self.current_momentum).add_(c_param.data, alpha=1 - self.current_momentum)

    @torch.no_grad()
    def encode_z(self, images):
        # images['top']: [B, T, 3, H, W]
        B, T, C, H, W = images['top'].shape
        device = images['top'].device

        flat_images = images['top'].flatten(0, 1) # [B*T, C, H, W]
        frame_patches = self.backbone(flat_images)
        h_p, w_p = frame_patches.shape[-2], frame_patches.shape[-1]
        N = h_p * w_p
        
        # [B*T, 1024, h_p, w_p] -> [B, T, N, 1024]
        patches = rearrange(frame_patches, '(b t) c h w -> b t (h w) c', b=B, t=T)
        tokens = self.patch_proj(patches)

        curr_pos_embed = self.pos_embed
        if curr_pos_embed.shape[1] != N:
            curr_pos_embed = rearrange(curr_pos_embed, '1 (h w) d -> 1 d h w', h=self.num_h, w=self.num_w)
            curr_pos_embed = F.interpolate(curr_pos_embed, size=(h_p, w_p), mode='bicubic', align_corners=False)
            curr_pos_embed = rearrange(curr_pos_embed, '1 d h w -> 1 (h w) d')

        time_embed = self.time_embed[:, :T]
        tokens = tokens + curr_pos_embed.unsqueeze(1).to(device) + time_embed.to(device)
        context_tokens = rearrange(tokens, 'b t n d -> b (t n) d')

        context_latents = context_tokens
        for blk in self.context_encoder:
            context_latents, _ = blk(context_latents)

        context_latents = rearrange(context_latents, 'b (t n) d -> b t n d', t=T)
        z = context_latents.mean(dim=(1, 2))
        return z

    def forward(self, images):
        B, T, C, H, W = images['top'].shape
        device = images['top'].device
        
        # 1. 시각 특징 추출 (Batching 처리하여 GPU 병렬 효율 극대화)
        with torch.no_grad():
            flat_images = images['top'].flatten(0, 1) # [B*T, C, H, W]
            frame_patches = self.backbone(flat_images)
            h_p, w_p = frame_patches.shape[-2], frame_patches.shape[-1]
            N = h_p * w_p
            
            # [B*T, 1024, h_p, w_p] -> [B, T, N, 1024]
            patches = rearrange(frame_patches, '(b t) c h w -> b t (h w) c', b=B, t=T)
        
        tokens = self.patch_proj(patches) # [B, T, N, D]
        
        # 2. 동적 위치 임베딩 적용
        curr_pos_embed = self.pos_embed 
        if curr_pos_embed.shape[1] != N:
            curr_pos_embed = rearrange(curr_pos_embed, '1 (h w) d -> 1 d h w', h=self.num_h, w=self.num_w)
            curr_pos_embed = F.interpolate(curr_pos_embed, size=(h_p, w_p), mode='bicubic', align_corners=False)
            curr_pos_embed = rearrange(curr_pos_embed, '1 d h w -> 1 (h w) d')
            
        time_embed = self.time_embed[:, :T]
        tokens = tokens + curr_pos_embed.unsqueeze(1).to(device) + time_embed.to(device)
        
        # 3. Tube Masking
        num_masked = int(N * self.mask_ratio)
        rand_indices = torch.rand(B, N, device=device).argsort(dim=-1)
        mask_idx = rand_indices[:, :num_masked]     
        context_idx = rand_indices[:, num_masked:]  
        
        # 4. Target Encoding (EMA)
        with torch.no_grad():
            target_input = rearrange(tokens, 'b t n d -> b (t n) d')
            target_tokens = target_input
            for blk in self.target_encoder:
                target_tokens, _ = blk(target_tokens)
            
            target_tokens = rearrange(target_tokens, 'b (t n) d -> b t n d', t=T)
            target_latents = torch.gather(target_tokens, 2, mask_idx.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, self.d_model))

        # 5. Context Encoding
        context_tokens = torch.gather(tokens, 2, context_idx.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, self.d_model))
        context_tokens = rearrange(context_tokens, 'b t n d -> b (t n) d')
        
        context_latents = context_tokens
        for blk in self.context_encoder:
            context_latents, _ = blk(context_latents)
            
        # 6. Predictor (All-frame Prediction)
        space_pos = curr_pos_embed.unsqueeze(1).expand(B, T, -1, -1).to(device)
        time_pos = time_embed.expand(B, -1, N, -1).to(device)
        full_pos = space_pos + time_pos
        
        target_pos = torch.gather(full_pos, 2, mask_idx.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, self.d_model))
        target_pos = rearrange(target_pos, 'b t n d -> b (t n) d')
        
        mask_tokens = self.mask_token.expand(B, target_pos.shape[1], -1) + target_pos
        
        pred_input = torch.cat([context_latents, mask_tokens], dim=1)
        pred_output = pred_input
        for blk in self.predictor:
            pred_output, _ = blk(pred_output)
            
        predicted_latents = pred_output[:, context_latents.shape[1]:]
        predicted_latents = rearrange(predicted_latents, 'b (t n) d -> b t n d', t=T)
        
        # 7. Loss
        loss = F.l1_loss(predicted_latents, target_latents, reduction='mean')
        return loss
