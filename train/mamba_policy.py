import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

from einops import rearrange, repeat
from mamba_ssm.modules.block import Block
from torchvision.transforms import functional as TF

# 0) MambaConfig
#########################################
class MambaConfig:
    def __init__(self):
        # 1) Mamba2
        self.d_model = 2048
        self.d_state = 512
        self.d_conv = 4
        self.expand = 2
        self.headdim = 128
        self.ngroups = 1
        self.A_init_range = (1 , 16)
        self.dt_min=0.001
        self.dt_max=0.02
        self.dt_init_floor=1e-4
        self.dt_limit=(0.0, float("inf"))
        self.learnable_init_states=False
        self.activation="swish"
        self.mamba_bias=False
        self.mamba_conv_bias=True
        self.chunk_size=256
        self.use_mem_eff_path=True

        # 2) Policy (多相机+lowdim)
        self.camera_names = ['top'] # or whichever,'angle', etc
        self.pretrained_backbone=True
        self.freeze_backbone=True
        self.embed_dim=2048       # each camera output
        self.lowdim_dim=14       # state_dim
        self.action_dim=14
        self.sum_camera_feats=False
        self.num_blocks=4
        self.img_size=(640,480)

#########################################
# 1)  Mamba2
#########################################

class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=256,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.02,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class CrossCameraAttention(nn.Module):
    def __init__(self, d_model=2048, num_heads=16):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + attn_output)


class CrossModalAttention(nn.Module):
    def __init__(self, d_model=1024, num_heads=8, lowdim_dim=14):
        super().__init__()
        # 层次化投影 + 非线性增强
        self.proj_lowdim = nn.Sequential(
            nn.Linear(14, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.Dropout(0.2),
            nn.Linear(512, d_model)  # d_model=1536或2048
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        key = self.proj_lowdim(key)  # [B, 1, 14] → [B, 1, 1536(2048)]
        value = self.proj_lowdim(value)
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + attn_output)


# 动态适配DINOv2特征提取器
class FrozenDinov2(nn.Module):
    def __init__(self, patch_size=14, layer_index=-4):  # 取倒数第4层特征
        super().__init__()
        # 加载完整预训练模型
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.patch_size = patch_size
        self.layer_index = layer_index  # 控制特征提取层

        # 冻结所有参数
        for param in self.dino.parameters():
            param.requires_grad_(False)
        # 注册前向钩子获取中间层输出
        self.feature_hook = self._register_hook()

    def _register_hook(self):
        # 获取完整的三维输出张量
        def hook(module, input, output):
            self.intermediate_output = output
        handle = self.dino.blocks[self.layer_index].register_forward_hook(hook)
        return handle

    # adaptive_resize
    def adaptive_resize(self, img, min_patches=8):
        B, C, H, W = img.shape
        min_size = self.patch_size * min_patches
        scale = max(min_size / min(H, W), 1.0)  # 确保缩放比例≥1
        new_H = max(round(H * scale), min_size)
        new_W = max(round(W * scale), min_size)
        # 强制对齐到patch_size的整数倍
        new_H = ((new_H + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_W = ((new_W + self.patch_size - 1) // self.patch_size) * self.patch_size
        return TF.resize(img, (new_H, new_W), antialias=True)

    def forward(self, x):
        # 动态调整尺寸（保证最小分辨率）
        x = self.adaptive_resize(x)
        B, C, H, W = x.shape

        # 前向传播获取中间层特征
        _ = self.dino(x)  # 触发前向钩子

        # 获取中间层特征（shape: [B, num_patches, dim]）
        features = self.intermediate_output

        # 重组为2D特征图
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        features = features[:, 1:, :]  # 去除CLS token
        features = features.permute(0, 2, 1).view(B, -1, H_patch, W_patch)
        return features  # [B, dim, H_patch, W_patch]

#########################################
# 2) MambaPolicy (多相机 + lowdim -> Mamba2 -> action)
#########################################
import torch
# import torch.nn as nn
from torchvision import models

class MambaPolicy(nn.Module):
    """
    多相机 + lowdim -> backbone -> concat/sum -> in_proj -> Block (with Mamba2) -> action
    """
    def __init__(
        self,
        camera_names,
        img_size=(640, 480),
        embed_dim=2048,
        lowdim_dim=14,
        d_model=2048,
        action_dim=14,
        num_blocks=4,  # 支持多个 Block
        sum_camera_feats=False,  # sum or concat
        block_cfg=None,  # Block 的配置
        mamba_cfg=None,  # Mamba2 的配置
        future_steps=16,  # 预测未来16步，可调
    ):
        super().__init__()
        self.camera_names = camera_names
        self.future_steps = future_steps
        self.img_size = img_size
        self.lowdim_dim = lowdim_dim
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.action_dim = action_dim
        self.sum_camera_feats = sum_camera_feats
        dinov2_dim = 1024  # dinov2_vitl14的dim=1024
        if mamba_cfg is None or not isinstance(mamba_cfg, MambaConfig):
            mamba_cfg = MambaConfig()
        self.mamba_cfg = mamba_cfg
        # 初始化DINOv2特征提取器
        self.shared_backbone = FrozenDinov2(layer_index=-4)

        # 添加空间压缩层（保持合理分辨率）
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(dinov2_dim, 512, 3, padding=1), #[B, 512, 45, 34]
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), #[B, 256, 45, 34]
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # 输出 [B,128,22,17]
            nn.Flatten(1), # [B, 128*(H*W)],输入为（640，480）时为（B, 128*23*18=52992）
            nn.Linear(128*23*18, self.embed_dim),  # [B, embed_dim]
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10)
        )
        self.spatial_adpater_low = nn.Sequential(
            nn.Conv2d(dinov2_dim, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Flatten(1),  # [B, 128*(H*W)],输入为（128，128）时为（B, 128*8*8=8192）
            nn.Linear(128*8*8, self.embed_dim),  # [B, embed_dim],输入为（640，480）
        )
        # 输入特征的拼接和投影
        # self.in_dim = embed_dim + lowdim_dim
        self.cross_attn = CrossModalAttention(d_model)
        self.in_dim = embed_dim * len(self.camera_names)
        # 跨相机交叉注意力模块
        self.num_cameras = len(camera_names)
        if self.num_cameras > 1:
            self.cross_cam_attn = CrossCameraAttention(d_model=self.in_dim)
        self.in_proj = nn.Linear(self.in_dim, d_model)

        # Block 配置
        if block_cfg is None:
            block_cfg = {}

        # Mamba2 配置
        def mixer_fn(dim):
            return Mamba2(
                d_model=dim,
                d_state=self.mamba_cfg.d_state,
                d_conv=self.mamba_cfg.d_conv,
                expand=self.mamba_cfg.expand,
                headdim=self.mamba_cfg.headdim,
                ngroups=self.mamba_cfg.ngroups,
                A_init_range=self.mamba_cfg.A_init_range,
                dt_min=self.mamba_cfg.dt_min,
                dt_max=self.mamba_cfg.dt_max,
                dt_init_floor=self.mamba_cfg.dt_init_floor,
                dt_limit=self.mamba_cfg.dt_limit,
                chunk_size=self.mamba_cfg.chunk_size,
                use_mem_eff_path=self.mamba_cfg.use_mem_eff_path,
            )

        def mlp_fn(dim):
            hidden_dim = 4 * dim
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
        # 构建多个 Block
        self.blocks = nn.ModuleList([
            Block(
                dim=self.d_model,
                mixer_cls=mixer_fn,
                mlp_cls=mlp_fn,
                norm_cls=nn.LayerNorm,
                fused_add_norm=block_cfg.get("fused_add_norm", False),
                residual_in_fp32=block_cfg.get("residual_in_fp32", False),
            )
            for _ in range(num_blocks)
        ])

        self.flat_action_dim = action_dim * future_steps
        self.out_proj = nn.Linear(d_model, self.flat_action_dim)


    def init_hidden_states(self, batch_size, device=None):
        """
        For single-step inference: gather each block's Mamba2 => allocate_inference_cache
        """
        if device is None:
            device = next(self.parameters()).device
        hidden_list = []
        for blk in self.blocks:
            if hasattr(blk.mixer, "allocate_inference_cache"):
                conv_st, ssm_st = blk.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=None)
            else:
                conv_st, ssm_st = None, None
            hidden_list.append((conv_st, ssm_st))
        return hidden_list

    def step(self, lowdim_t, images_t, hidden_states):
        """
        单帧前向:
          lowdim_t: [B, lowdim_dim]
          images_t: dict of [B, 3, H, W] => 单帧输入
          hidden_states: list of (conv_st, ssm_st) per block，每个 Block 的隐藏状态
        返回:
          pred_action: [B, action_dim]
          new_hidden_states: List[Tensor]
        """
        B, _ = lowdim_t.shape
        device = lowdim_t.device

        # 1. 多相机特征提取

        feats_all = []
        for cam in self.camera_names:
            img = images_t[cam]
            raw_feat = self.shared_backbone(img)  # [B, 1024, H_patch, W_patch]
            # 空间压缩与通道调整
            if self.img_size == (640, 480):
                feat = self.spatial_adapter(raw_feat)  # [B, 128*11*8]
            elif self.img_size == (128, 128):
                feat = self.spatial_adapter_low(raw_feat) # [B, 128*8*8]
            feats_all.append(feat)

        cam_feats = torch.cat(feats_all, dim=1)
        # 跨相机注意力
        if self.num_cameras > 1:
            cam_feats = self.cross_cam_attn(cam_feats.unsqueeze(1),
                                            cam_feats.unsqueeze(1), cam_feats.unsqueeze(1)).squeeze(1)

        # 2. 特征融合与投影
        lowdim_feat = lowdim_t.unsqueeze(1)  # [B, 1, 14]
        # 投影到d_model并交叉注意力
        cam_feats_proj = self.in_proj(cam_feats)  # [B, d_model]
        fused_feat = self.cross_attn(
            query=cam_feats_proj.unsqueeze(1),
            key=lowdim_feat,
            value=lowdim_feat
        ).squeeze(1)
        x_t = fused_feat # [B, d_model]

        # 3) 经过 blocks (单步)
        residual = None
        new_states = []
        hidden = x_t
        for i, blk in enumerate(self.blocks):
            conv_st, ssm_st = hidden_states[i] if i < len(hidden_states) else (None, None)
            if residual is None:
                residual = hidden
            else:
                residual = residual + hidden

            hidden_ln = blk.norm(residual.to(dtype=blk.norm.weight.dtype))

            # => step
            # we need: out, new_conv, new_ssm = blk.mixer.step(...)
            if hasattr(blk.mixer, "step"):
                y_t, new_conv_st, new_ssm_st = blk.mixer.step(hidden_ln.unsqueeze(1), conv_st, ssm_st)
                y_t = y_t.squeeze(1)   # => (B, d_model)
            else:
                # fallback
                y_t = blk.mixer(hidden_ln.unsqueeze(1))
                y_t = y_t.squeeze(1)
                new_conv_st, new_ssm_st = conv_st, ssm_st

            hidden_out = y_t + residual

            # mlp
            if blk.mlp is not None:
                r2 = blk.norm2(hidden_out.to(dtype=blk.norm2.weight.dtype))
                hidden_out = blk.mlp(r2) + hidden_out

            new_states.append((new_conv_st, new_ssm_st))
            hidden = hidden_out
            residual = hidden_out

            # d) out => action
        action_flat = self.out_proj(hidden)# => [B, 16×14=224]
        action_t = action_flat.view(-1, self.future_steps, 14)  # => [B,16,14]
        return action_t, new_states

    def _mamba2_block_forward(self, mixer, hidden_ln, conv_state, ssm_state):
        """
        Mamba2 블록 하나를 알고리즘적으로 forward.
        conv_state와 ssm_state를 받아 청크 경계를 가로질러 기억을 유지한다.

        Args:
            mixer     : Mamba2 인스턴스
            hidden_ln : [B, L, d_model]  (pre-norm 이후 입력)
            conv_state: [B, conv_dim, d_conv] or None  (이전 청크의 conv 슬라이딩 윈도우)
            ssm_state : [B, nheads, headdim, d_state] or None  (이전 청크의 SSM 상태)
        Returns:
            out           : [B, L, d_model]
            new_conv_state: [B, conv_dim, d_conv]
            new_ssm_state : [B, nheads, headdim, d_state]
        """
        B, L, _ = hidden_ln.shape
        dtype = hidden_ln.dtype

        # ── 1. in_proj ────────────────────────────────────────────────────────
        zxbcdt = mixer.in_proj(hidden_ln)   # [B, L, d_in_proj]
        d_mlp = (zxbcdt.shape[-1]
                 - 2 * mixer.d_ssm
                 - 2 * mixer.ngroups * mixer.d_state
                 - mixer.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, mixer.d_ssm,
             mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state,
             mixer.nheads],
            dim=-1
        )
        # d_mlp=0 이 일반적이므로 z0, x0 는 비어있음

        # ── 2. causal conv1d (conv_state로 청크 경계 연속 처리) ───────────────
        conv_dim = mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state
        d_conv   = mixer.d_conv
        xBC_t    = xBC.transpose(1, 2)   # [B, conv_dim, L]

        if conv_state is not None:
            # 이전 청크의 마지막 d_conv-1 값을 앞에 이어붙임
            prefix     = conv_state[:, :, -(d_conv - 1):]          # [B, conv_dim, d_conv-1]
            xBC_padded = torch.cat([prefix, xBC_t], dim=-1)        # [B, conv_dim, d_conv-1+L]
        else:
            xBC_padded = F.pad(xBC_t, (d_conv - 1, 0))             # zero pad (첫 청크)

        # padding 없이 conv1d 적용 (이미 수동으로 padding했으므로)
        xBC_conv = F.conv1d(xBC_padded, mixer.conv1d.weight,
                            mixer.conv1d.bias, groups=conv_dim)     # [B, conv_dim, L]
        xBC_act  = F.silu(xBC_conv).transpose(1, 2)                # [B, L, conv_dim]

        # 다음 청크를 위한 새 conv_state: 현재 청크 raw xBC의 마지막 d_conv 값
        prev = conv_state if conv_state is not None \
               else xBC_t.new_zeros(B, conv_dim, 0)
        new_conv_state = torch.cat([prev, xBC_t], dim=-1)[:, :, -d_conv:]  # [B, conv_dim, d_conv]

        # ── 3. xBC_act 분리 ───────────────────────────────────────────────────
        x_ssm, B_ssm, C_ssm = torch.split(
            xBC_act,
            [mixer.d_ssm,
             mixer.ngroups * mixer.d_state,
             mixer.ngroups * mixer.d_state],
            dim=-1
        )

        # ── 4. dt 이산화 ──────────────────────────────────────────────────────
        dt_disc = F.softplus(dt + mixer.dt_bias.to(dtype=dt.dtype))  # [B, L, nheads]
        A       = -torch.exp(mixer.A_log.float())                     # [nheads]
        x_ssm_r = rearrange(x_ssm, "b l (h p) -> b l h p", p=mixer.headdim)  # [B, L, nheads, headdim]

        # ── 5. SSM sequential loop (알고리즘 검증 버전) ───────────────────────
        if ssm_state is not None:
            h = ssm_state.float().clone()
        else:
            h = torch.zeros(B, mixer.nheads, mixer.headdim, mixer.d_state,
                            device=hidden_ln.device, dtype=torch.float32)

        ys = []
        for t in range(L):
            dt_t = dt_disc[:, t, :].float()          # [B, nheads]
            x_t  = x_ssm_r[:, t, :, :].float()      # [B, nheads, headdim]
            B_t  = B_ssm[:, t, :].float()            # [B, ngroups*d_state]
            C_t  = C_ssm[:, t, :].float()            # [B, ngroups*d_state]

            dA  = torch.exp(dt_t * A)                # [B, nheads]
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt_t, B_t, x_t)  # [B, nheads, headdim, d_state]
            h   = h * rearrange(dA, "b h -> b h 1 1") + dBx

            y_t = torch.einsum("bhpn,bn->bhp", h, C_t)
            y_t = y_t + rearrange(mixer.D.float(), "h -> h 1") * x_t
            ys.append(rearrange(y_t, "b h p -> b (h p)").to(dtype))

        y             = torch.stack(ys, dim=1)   # [B, L, d_ssm]
        new_ssm_state = h.to(dtype)

        # ── 6. RMSNorm + gate z ───────────────────────────────────────────────
        if mixer.rmsnorm:
            y = mixer.norm(y, z)
        else:
            y = y * F.silu(z)

        # ── 7. out_proj ───────────────────────────────────────────────────────
        out = mixer.out_proj(y)   # [B, L, d_model]

        return out, new_conv_state, new_ssm_state

    def forward_seq(self, lowdim_seq, images_seq, hidden_states=None):
        """
        시퀀스 forward (학습용).
          lowdim_seq    : [B, L, lowdim_dim]
          images_seq    : {cam: [B, L, C, H, W]}
          hidden_states : list of (conv_st, ssm_st) per block, or None
        반환:
          actions       : [B, L, future_steps, action_dim]
          hidden_states : 청크 마지막 프레임의 hidden state (다음 청크로 전달)

        DINOv2 : [B*L] 병렬 처리 (속도)
        Mamba  : conv_state·ssm_state를 전달하며 순차 처리 (기억 유지)
        """
        B, L, _ = lowdim_seq.shape

        # ── 1. DINOv2 병렬 특징 추출 [B*L] ───────────────────────────────────
        feats_all = []
        for cam in self.camera_names:
            img      = images_seq[cam]                           # [B, L, C, H, W]
            img_flat = img.reshape(B * L, *img.shape[2:])        # [B*L, C, H, W]
            raw_feat = self.shared_backbone(img_flat)             # [B*L, 1024, H_p, W_p]
            if self.img_size == (640, 480):
                feat = self.spatial_adapter(raw_feat)
            elif self.img_size == (128, 128):
                feat = self.spatial_adpater_low(raw_feat)
            feats_all.append(feat)

        cam_feats = torch.cat(feats_all, dim=1)                  # [B*L, embed_dim*num_cams]

        if self.num_cameras > 1:
            cam_feats = self.cross_cam_attn(
                cam_feats.unsqueeze(1), cam_feats.unsqueeze(1), cam_feats.unsqueeze(1)
            ).squeeze(1)

        # ── 2. in_proj + cross-modal attention [B*L] ─────────────────────────
        cam_feats_proj = self.in_proj(cam_feats)                  # [B*L, d_model]
        lowdim_flat    = lowdim_seq.reshape(B * L, 1, self.lowdim_dim)

        fused = self.cross_attn(
            query=cam_feats_proj.unsqueeze(1),
            key=lowdim_flat,
            value=lowdim_flat
        ).squeeze(1)                                              # [B*L, d_model]

        x = fused.reshape(B, L, self.d_model)                    # [B, L, d_model]

        # ── 3. Mamba 블록: hidden_state 전달하며 처리 ─────────────────────────
        if hidden_states is None:
            hidden_states = self.init_hidden_states(batch_size=B, device=x.device)

        new_hidden_states = []
        residual = None
        hidden   = x

        for i, blk in enumerate(self.blocks):
            conv_st, ssm_st = hidden_states[i]

            if residual is None:
                residual = hidden
            else:
                residual = residual + hidden

            hidden_ln = blk.norm(residual.to(dtype=blk.norm.weight.dtype))

            # conv_state·ssm_state를 받아 알고리즘적으로 forward
            y, new_conv_st, new_ssm_st = self._mamba2_block_forward(
                blk.mixer, hidden_ln, conv_st, ssm_st
            )

            hidden_out = y + residual

            if blk.mlp is not None:
                r2         = blk.norm2(hidden_out.to(dtype=blk.norm2.weight.dtype))
                hidden_out = blk.mlp(r2) + hidden_out

            new_hidden_states.append((new_conv_st, new_ssm_st))
            hidden   = hidden_out
            residual = hidden_out

        # ── 4. 출력 투영 ──────────────────────────────────────────────────────
        action_flat = self.out_proj(hidden)
        actions     = action_flat.view(B, L, self.future_steps, self.action_dim)
        return actions, new_hidden_states

# forward一般不使用
    def forward(self, lowdim, images):
        """
        lowdim: [B, L, lowdim_dim]
        images: dict of [B, L, 3, H, W]
        返回: [B, L, action_dim]
        """
        device = lowdim.device
        B, L, _ = lowdim.shape

        # 1. 多相机特征提取
        feats_all = []
        for cam in self.camera_names:
            if cam not in images:
                feats_all.append(torch.zeros(B, L, self.embed_dim, device=device))
                continue
            x = images[cam]
            x = x.view(B * L, *x.shape[2:])
            net = self.backbones[cam]
            feats = net(x)  # [B*L,512,15,20]
            projected_feat = self.feature_extractors(feats)
            projected_feat = projected_feat.view(B, L, -1)
            feats_all.append(projected_feat)

        if self.sum_camera_feats:
            cam_feats = torch.stack(feats_all, dim=0).sum(dim=0)
        else:
            cam_feats = torch.cat(feats_all, dim=2)

        # 2. 特征拼接与投影
        x = torch.cat([cam_feats, lowdim], dim=2)
        x = self.in_proj(x)

        # 3. 逐层通过 Block
        residual = None
        for blk in self.blocks:
            x, residual = blk(x, residual)

        # 4. 输出动作
        actions = self.out_proj(x)
        actions = actions.view(B, L, self.future_steps, self.action_dim)
        return actions


