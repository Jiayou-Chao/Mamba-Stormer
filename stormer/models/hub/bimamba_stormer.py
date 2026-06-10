import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, trunc_normal_

from .mamba_utils import build_mamba_module
from .stormer import FinalLayer, Stormer, TimestepEmbedder, modulate
from .weather_embedding import WeatherEmbedding


class BiMambaAttention(nn.Module):
    def __init__(self, hidden_size, d_state=16, d_conv=4, expand=1, **mamba_kwargs):
        super().__init__()
        self.forward_mamba = build_mamba_module(
            hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
        self.backward_mamba = build_mamba_module(
            hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
        # Add normalization after each direction to stabilize bidirectional fusion
        self.norm_f = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_b = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        # Forward direction
        forward = self.norm_f(self.forward_mamba(x))
        
        # Backward direction: flip sequence, process, flip back
        # Ensure contiguous after flip for CUDA kernels
        x_flipped = torch.flip(x, dims=[1]).contiguous()
        backward_out = self.backward_mamba(x_flipped)
        backward = torch.flip(backward_out, dims=[1]).contiguous()
        backward = self.norm_b(backward)
        
        return 0.5 * (forward + backward)


class BiMambaBlock(nn.Module):
    """
    A Stormer block that keeps the Transformer residual skeleton while
    replacing attention with bidirectional Mamba.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        d_state=16,
        d_conv=4,
        expand=1,
        **mamba_kwargs,
    ):
        super().__init__()
        del num_heads  # Kept for interface parity with Stormer blocks.

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = BiMambaAttention(
            hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **mamba_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class BiMambaStormer(Stormer):
    def __init__(
        self,
        in_img_size,
        variables,
        patch_size=4,
        hidden_size=1024,
        depth=14,
        num_heads=16,
        mlp_ratio=4.0,
        d_state=32,
        d_conv=4,
        expand=2,
        **kwargs,
    ):
        del kwargs
        nn.Module.__init__(self)

        self.original_in_img_size = in_img_size

        new_in_img_size = list(in_img_size)
        for i in range(2):
            if new_in_img_size[i] % patch_size != 0:
                pad_size = patch_size - (new_in_img_size[i] % patch_size)
                new_in_img_size[i] += pad_size
        self.in_img_size = tuple(new_in_img_size)

        self.variables = variables
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.embedding = WeatherEmbedding(
            variables=variables,
            img_size=self.in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                BiMambaBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(depth)
            ]
        )
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def forward(self, x, variables, time_interval):
        # Pad input if necessary to match self.in_img_size
        pad_h = self.in_img_size[0] - self.original_in_img_size[0]
        pad_w = self.in_img_size[1] - self.original_in_img_size[1]
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
        
        x = self.embedding(x, variables) # B, L, D
        x = self.embed_norm_layer(x)

        # Calculate grid dimensions for alternating scans.
        wp = self.in_img_size[0] // self.patch_size
        hp = self.in_img_size[1] // self.patch_size

        time_interval_emb = self.t_embedder(time_interval)
        
        for i, block in enumerate(self.blocks):
            if i % 2 == 1:
                # Vertical scan: transpose (B, Wp*Hp, D) -> (B, Wp, Hp, D) -> (B, Hp, Wp, D) -> (B, Hp*Wp, D)
                B, L, D = x.shape
                x = x.reshape(B, wp, hp, D).transpose(1, 2).reshape(B, hp * wp, D)
                x = block(x, time_interval_emb)
                # Transpose back to original orientation
                x = x.reshape(B, hp, wp, D).transpose(1, 2).reshape(B, wp * hp, D)
            else:
                # Horizontal scan (Standard Row-major)
                x = block(x, time_interval_emb)
        
        x = self.head(x, time_interval_emb)
        x = self.unpatchify(x)
        
        # Crop output back to original size
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :self.original_in_img_size[0], :self.original_in_img_size[1]]
            
        return x

    def initialize_weights(self):
        # Apply basic initialization but skip modules that are part of Mamba
        # Mamba internal layers have their own specific initializations.
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "mamba" in name.lower():
                    continue
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)
