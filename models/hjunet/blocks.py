import math

import torch
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_, to_2tuple
from torch import nn


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ShiftMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, drop=0.0, shift_size=5):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.pad(x, (self.pad,) * 4, "constant", 0)
        chunks = torch.chunk(x, self.shift_size, 1)
        shifted = [torch.roll(chunk, shift, 2) for chunk, shift in zip(chunks, range(-self.pad, self.pad + 1))]
        x = torch.cat(shifted, 1)
        x = torch.narrow(x, 2, self.pad, H)
        x = torch.narrow(x, 3, self.pad, W)
        x = x.reshape(B, C, H * W).contiguous().transpose(1, 2)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.pad(x, (self.pad,) * 4, "constant", 0)
        chunks = torch.chunk(x, self.shift_size, 1)
        shifted = [torch.roll(chunk, shift, 3) for chunk, shift in zip(chunks, range(-self.pad, self.pad + 1))]
        x = torch.cat(shifted, 1)
        x = torch.narrow(x, 2, self.pad, H)
        x = torch.narrow(x, 3, self.pad, W)
        x = x.reshape(B, C, H * W).contiguous().transpose(1, 2)
        x = self.fc2(x)
        return self.drop(x)


class ShiftedBlock(nn.Module):
    def __init__(self, dim, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = ShiftMLP(dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        return x + self.drop_path(self.mlp(self.norm(x), H, W))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


__all__ = ["DWConv", "ShiftMLP", "ShiftedBlock", "OverlapPatchEmbed"]
