# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    Hiera 是一种层次化的Transformer架构，适用于图像处理任务。该模型通过多尺度块（MultiScaleBlock）和窗口化位置嵌入（Windowed Positional Embedding）实现高效的信息交互和特征提取。
    """

    def __init__(
        self,
        embed_dim: int = 96,  # 初始嵌入维度
        num_heads: int = 1,  # 初始注意力头数
        drop_path_rate: float = 0.0,  # 随机深度丢弃率
        q_pool: int = 3,  # q_pool 阶段的数量
        q_stride: Tuple[int, int] = (2, 2),  # 阶段之间的下采样步幅
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # 每个阶段的块数
        dim_mul: float = 2.0,  # 阶段切换时嵌入维度的倍增因子
        head_mul: float = 2.0,  # 阶段切换时注意力头数的倍增因子
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),  # 非全局注意力时的窗口大小
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),  # 每个阶段的窗口大小
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),  # 具有全局注意力的块索引
        return_interm_layers=True,  # 是否返回每个阶段的特征
    ):
        super().__init__()  # 调用父类的初始化方法

        assert len(stages) == len(window_spec), "stages 和 window_spec 的长度必须相同"  # 确保阶段数与窗口规格数相同
        self.window_spec = window_spec  # 保存窗口规格

        depth = sum(stages)  # 总块数
        self.q_stride = q_stride  # 保存下采样步幅
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]  # 每个阶段的结束块索引
        assert 0 <= q_pool <= len(self.stage_ends[:-1]), "q_pool 必须在0和阶段数之间"  # 确保 q_pool 在有效范围内
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]  # q_pool 阶段对应的块索引
        self.return_interm_layers = return_interm_layers  # 是否返回中间层的特征

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,  # 初始化 PatchEmbed 模块，负责将输入图像分割为patch并嵌入
        )
        # 哪些块具有全局注意力
        self.global_att_blocks = global_att_blocks  # 保存具有全局注意力的块索引

        # 窗口化位置嵌入（参考：https://arxiv.org/abs/2311.05613）
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size  # 保存窗口位置嵌入的空间尺寸
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)  # 初始化位置嵌入参数
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])  # 初始化窗口位置嵌入参数
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # 随机深度丢弃率的衰减规则

        cur_stage = 1  # 当前阶段索引
        self.blocks = nn.ModuleList()  # 初始化块的模块列表

        for i in range(depth):  # 遍历所有块
            dim_out = embed_dim  # 初始化输出嵌入维度
            # 延迟一个块，因此下一阶段的第一个块使用前一阶段的初始窗口大小
            window_size = self.window_spec[cur_stage - 1]  # 获取当前阶段的窗口大小

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size  # 如果当前块需要全局注意力，则窗口大小设为0

            if i - 1 in self.stage_ends:  # 如果前一个块是阶段结束块
                dim_out = int(embed_dim * dim_mul)  # 增加嵌入维度
                num_heads = int(num_heads * head_mul)  # 增加注意力头数
                cur_stage += 1  # 切换到下一个阶段

            block = MultiScaleBlock(
                dim=embed_dim,  # 当前嵌入维度
                dim_out=dim_out,  # 输出嵌入维度
                num_heads=num_heads,  # 注意力头数
                drop_path=dpr[i],  # 随机深度丢弃率
                q_stride=self.q_stride if i in self.q_pool_blocks else None,  # 如果是 q_pool 块，则应用下采样步幅
                window_size=window_size,  # 窗口大小
            )

            embed_dim = dim_out  # 更新嵌入维度为输出维度
            self.blocks.append(block)  # 添加块到模块列表

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )  # 根据是否返回中间层，保存每个阶段的通道数

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """
        获取位置嵌入，根据输入的高度和宽度进行插值和拼接。

        Args:
            hw (Tuple[int, int]): 输入的高度和宽度。

        Returns:
            torch.Tensor: 调整后的位置嵌入，形状为 [1, C, H, W]。
        """
        h, w = hw  # 解包高度和宽度
        window_embed = self.pos_embed_window  # 获取窗口位置嵌入
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")  # 对位置嵌入进行插值以匹配输入尺寸
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )  # 将窗口位置嵌入重复并加到插值后的位置嵌入
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # 调整维度顺序为 [B, H, W, C]
        return pos_embed  # 返回调整后的位置嵌入

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]。

        Returns:
            List[torch.Tensor]: 各阶段输出的特征图列表。
        """
        x = self.patch_embed(x)  # 通过 PatchEmbed 模块，输出形状为 [B, H, W, C]
        # x: (B, H, W, C)

        # 添加位置嵌入
        x = x + self._get_pos_embed(x.shape[1:3])  # 将位置嵌入添加到输入特征

        outputs = []  # 初始化输出列表
        for i, blk in enumerate(self.blocks):  # 遍历所有块
            x = blk(x)  # 通过当前块进行前向传播
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):  # 如果当前块是最后一个阶段的块，或是阶段结束且需要返回中间层
                feats = x.permute(0, 3, 1, 2)  # 调整特征图的维度顺序为 [B, C, H, W]
                outputs.append(feats)  # 添加到输出列表

        return outputs  # 返回所有阶段的输出特征图
