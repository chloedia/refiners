from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import pad


class PatchEncoder(fl.Chain):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 96,
        patch_size: int = 4,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_bias = use_bias
        super().__init__(
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size),
                use_bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
            fl.Permute(0, 2, 3, 1),
            fl.LayerNorm(out_channels, device=device, dtype=dtype),
            fl.Permute(0, 3, 2, 1),
        )


def window_partition(x: Tensor, window_size: int):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5)
    windows = windows.reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class RelativePositionAttention(fl.ContextModule):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        spatial_size: tuple[int, int],
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.spatial_size = spatial_size
        self.qkv = fl.Linear(embedding_dim, embedding_dim * 3, bias=True, device=device, dtype=dtype)
        self.proj = fl.Linear(embedding_dim, embedding_dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * spatial_size[0] - 1) * (2 * spatial_size[1] - 1), self.num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.spatial_size[0])
        coords_w = torch.arange(self.spatial_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.spatial_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.spatial_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.spatial_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)

    def forward(self, x: Tensor) -> Tensor:
        B_, N, C = x.shape
        x = self.qkv(x)
        x = x.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        query, key, value = x.unbind(dim=0)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        relative_position_bias = relative_position_bias.reshape(
            self.spatial_size[0] * self.spatial_size[1], self.spatial_size[0] * self.spatial_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww

        attention = (query * self.head_dim**-0.5) @ key.transpose(dim0=-2, dim1=-1)
        attention = attention + relative_position_bias.unsqueeze(0)

        context = self.use_context(context_name="SwinT")
        mask = context["attn_mask"]

        if mask:
            nW = mask.shape[0]
            attention = attention.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, N, N)
            attention = attention.softmax()
        else:
            attention = attention.softmax(dim=-1)

        attention = (attention @ value).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(attention)
        return attention


class FeedForward(fl.Chain):
    """Multi-Layer Perceptron"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        super().__init__(
            fl.Linear(self.in_features, self.hidden_features, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Linear(self.hidden_features, self.out_features, device=device, dtype=dtype),
        )


class CyclicShift(fl.ContextModule):
    def __init__(self, shift_size: int, window_size: int) -> None:
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        self.shape = None

    def forward(self, x: Tensor) -> Tensor:
        context = self.use_context(context_name="SwinT")
        mask_matrix = context["attn_mask"]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        B, _, _, C = x_windows.shape
        x_windows = x_windows.reshape(B, self.window_size * self.window_size, C)
        context.update({"attn_mask": attn_mask})
        # View as (nW*B, window_size*window_size, C)

        return x_windows


class ReverseCyclicShift(fl.ContextModule):
    def __init__(
        self,
        shift_size: int,
        window_size: int,
        pad: tuple[int, int],
        input_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        self.pad_r, self.pad_b = pad
        self.H, self.W = input_size

    def forward(self, attn_windows: Tensor) -> Tensor:
        B, _, C = attn_windows.shape
        context = self.use_context("SwinT")
        Hp, Wp = context["dim_p"]
        pad_r, pad_b = context["pad"]
        # merge windows
        attn_windows = attn_windows.reshape(B, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, : self.H, : self.W, :]

        x = x.reshape(-1, self.H * self.W, C)
        return x


class Softmax(fl.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=self.dim)


class SwinTransformerBlock(fl.Chain):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        H: int,
        W: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias

        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.H = H
        self.W = W

        self.Hp = 0
        self.Wp = 0

        self.pad_r, self.pad_b = 0, 0

        assert 0 <= self.shift_size < self.window_size, "shift_size must in [0 ; window_size]"

        super().__init__(
            fl.Residual(
                fl.LayerNorm(self.dim),
                fl.Reshape(self.H, self.W, self.dim),
                Pad(window_size=self.window_size, W=self.W, H=self.H),
                CyclicShift(shift_size=self.shift_size, window_size=self.window_size),
                RelativePositionAttention(
                    embedding_dim=self.dim,
                    num_heads=self.num_heads,
                    spatial_size=(self.window_size, self.window_size),
                    device=device,
                    dtype=dtype,
                ),
                Softmax(dim=-1),
                ReverseCyclicShift(
                    shift_size=self.shift_size,
                    window_size=self.window_size,
                    pad=(self.pad_r, self.pad_b),
                    input_size=(self.H, self.W),
                ),  # type:ignore
                fl.LayerNorm(self.dim),
                FeedForward(self.dim, self.mlp_hidden_dim, device=device, dtype=dtype),
                fl.Identity(),
            ),
        )


class Pad(fl.ContextModule):
    def __init__(self, window_size: int, W: int, H: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.W = W
        self.H = H

    def forward(self, x: Tensor) -> Tensor:
        pad_l = pad_t = 0
        pad_r = (self.window_size - self.W % self.window_size) % self.window_size
        pad_b = (self.window_size - self.H % self.window_size) % self.window_size

        x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # type: ignore

        _, Hp, Wp, _ = x.shape

        context = self.use_context("SwinT")
        context["dim_p"] = (Hp, Wp)
        context["pad"] = (pad_r, pad_b)
        return x


class PatchMerging(fl.Chain):
    def __init__(
        self,
        dim: int,
        H: int,
        W: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.dim = dim
        self.H = H
        self.W = W

        super().__init__(
            fl.Lambda(func=self.concat_patch),
            fl.LayerNorm(self.dim * 4),
            fl.Linear(self.dim * 4, self.dim * 2, bias=False, device=device, dtype=dtype),
        )

    def concat_patch(self, x: Tensor):
        B, L, C = x.shape
        assert L == self.H * self.W, "input feature has wrong size"

        x = x.view(B, self.H, self.W, C)
        pad_input = (self.H % 2 == 1) or (self.W % 2 == 1)
        if pad_input:
            x = pad(x, (0, 0, 0, self.W % 2, 0, self.H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        return x


class InitAttentionMask(fl.ContextModule):
    def __init__(self, H: int, W: int, shift_size: int, window_size: int) -> None:
        super().__init__()
        self.Hp = H
        self.Wp = W
        self.shift_size = shift_size
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        img_mask = torch.zeros((1, self.Hp, self.Wp, 1))  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        context = self.use_context(context_name="SwinT")
        context.update({"attn_mask": attn_mask})

        return x


class BasicLayer(fl.Chain):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        downsample: type[PatchMerging] | None,
        norm_feature: int | None,
        H: int,
        W: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.mlp_ratio = mlp_ratio
        self.norm_feature = norm_feature
        self.H = H
        self.W = W
        self.Hp = int(np.ceil(self.H / self.window_size)) * self.window_size
        self.Wp = int(np.ceil(self.W / self.window_size)) * self.window_size

        super().__init__(
            InitAttentionMask(self.Hp, self.Wp, self.shift_size, self.window_size),
            *[
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    H=self.H,
                    W=self.W,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    device=device,
                    dtype=dtype,
                )
                for i in range(self.depth)
            ],
            fl.SetContext(context="SwinT", key="x"),
            fl.Chain(
                fl.LayerNorm(self.norm_feature),
                fl.Reshape(self.H, self.W, self.norm_feature),
                fl.Permute(0, 3, 1, 2),
                fl.Lambda(func=self.update_outs),
            )
            if (self.norm_feature)
            else fl.Identity(),
            fl.UseContext(context="SwinT", key="x"),
            downsample(self.dim, self.H, self.W) if downsample else fl.Identity(),
        )

    def update_outs(self, x: Tensor):
        context = self.use_context(context_name="SwinT")
        outs = context["outs"]
        outs.append(x)
        context.update({"outs": outs})
        return x


class SwinTransformer(fl.Chain):
    def __init__(
        self,
        pretrain_img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices

        self.H, self.W = pretrain_img_size // patch_size, pretrain_img_size // patch_size

        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]

        super().__init__(
            PatchEncoder(
                in_channels=in_chans, patch_size=patch_size, out_channels=embed_dim, device=device, dtype=dtype
            ),
            fl.Flatten(2),
            fl.Transpose(1, 2),
            *[
                fl.Chain(
                    BasicLayer(
                        dim=self.num_features[i_layer],
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                        norm_feature=self.num_features[i_layer] if i_layer in self.out_indices else None,
                        H=int(np.ceil(self.H / (2 ** (i_layer)))),
                        W=int(np.ceil(self.W / (2 ** (i_layer)))),
                        device=device,
                        dtype=dtype,
                    ),
                )
                for i_layer in range(self.num_layers)
            ],
            fl.UseContext(context="SwinT", key="outs"),
        )

    def init_context(self):  # type:ignore
        return {"SwinT": {"attn_mask": None, "dim_p": None, "pad": None, "outs": []}}  # type:ignore


class SwinTransformerH(SwinTransformer):
    def __init__(
        self,
        pretrain_img_size: int = 224,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        super().__init__(
            pretrain_img_size=pretrain_img_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            device=device,
            dtype=dtype,
        )
