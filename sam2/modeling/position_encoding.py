# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    这是位置嵌入的更标准版本，与 "Attention is All You Need" 论文中使用的类似，扩展到适用于图像。
    使用正弦和余弦函数为每个位置生成唯一的嵌入，以捕捉空间信息。
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,  # 用于缩放的位置温度参数
        normalize: bool = True,  # 是否对位置进行归一化
        scale: Optional[float] = None,  # 归一化时的缩放因子
    ):
        super().__init__()  # 调用父类的初始化方法
        assert num_pos_feats % 2 == 0, "Expecting even model width"  # 确保位置特征数为偶数
        self.num_pos_feats = num_pos_feats // 2  # 每个轴的特征数
        self.temperature = temperature  # 位置温度参数
        self.normalize = normalize  # 是否归一化
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")  # 如果提供了缩放因子，normalize 必须为 True
        if scale is None:
            scale = 2 * math.pi  # 默认缩放因子为 2π
        self.scale = scale  # 保存缩放因子

        self.cache = {}  # 用于缓存位置嵌入

    def _encode_xy(self, x, y):
        # 期望位置已被归一化
        assert len(x) == len(y) and x.ndim == y.ndim == 1  # 确保 x 和 y 的长度和维度相同
        x_embed = x * self.scale  # 缩放 x 坐标
        y_embed = y * self.scale  # 缩放 y 坐标

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # 创建维度索引
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 计算频率

        pos_x = x_embed[:, None] / dim_t  # 计算 x 的位置嵌入
        pos_y = y_embed[:, None] / dim_t  # 计算 y 的位置嵌入
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2  # 使用正弦和余弦函数
        ).flatten(1)  # 展平
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2  # 使用正弦和余弦函数
        ).flatten(1)  # 展平
        return pos_x, pos_y  # 返回 x 和 y 的位置嵌入

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)  # 获取 x 和 y 的位置嵌入
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)  # 将 y、x、h、w 拼接
        return pos  # 返回位置嵌入

    encode = encode_boxes  # 向后兼容，使用 encode 作为 encode_boxes 的别名

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape  # 获取形状
        assert bx == by and nx == ny and bx == bl and nx == nl  # 确保形状匹配
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())  # 扁平化并编码 x 和 y
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)  # 重塑形状
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)  # 拼接 y、x 和标签
        return pos  # 返回位置嵌入

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])  # 创建缓存键，基于高度和宽度
        if cache_key in self.cache:  # 如果缓存中存在
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)  # 返回缓存的嵌入，复制 batch 维度
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])  # 创建 y 坐标网格
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)  # 创建 x 坐标网格
        )

        if self.normalize:  # 如果需要归一化
            eps = 1e-6  # 防止除以零
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 归一化 y
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 归一化 x

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # 创建维度索引
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # 计算频率

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算 x 的位置嵌入
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算 y 的位置嵌入
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4  # 使用正弦和余弦函数
        ).flatten(3)  # 展平
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4  # 使用正弦和余弦函数
        ).flatten(3)  # 展平
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 拼接 y 和 x 的位置嵌入，并调整维度顺序
        self.cache[cache_key] = pos[0]  # 缓存第一个样本的嵌入
        return pos  # 返回位置嵌入



class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率进行位置编码。
    这种方法通过随机生成的频率矩阵，为每个位置生成独特的嵌入，增强模型对空间关系的捕捉能力。
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()  # 调用父类的初始化方法
        if scale is None or scale <= 0.0:
            scale = 1.0  # 默认缩放因子为1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),  # 注册随机生成的高斯频率矩阵为缓冲区
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """对归一化到 [0,1] 的点进行位置编码。"""
        # 假设 coords 在 [0, 1]^2 的方形区域内，形状为 d_1 x ... x d_n x 2
        coords = 2 * coords - 1  # 将坐标从 [0,1] 线性映射到 [-1,1]
        coords = coords @ self.positional_encoding_gaussian_matrix  # 矩阵乘法，生成频率编码
        coords = 2 * np.pi * coords  # 缩放频率编码
        # 输出形状为 d_1 x ... x d_n x C
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # 拼接正弦和余弦值

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """为指定大小的网格生成位置编码。"""
        h, w = size  # 获取高度和宽度
        device: Any = self.positional_encoding_gaussian_matrix.device  # 获取设备
        grid = torch.ones((h, w), device=device, dtype=torch.float32)  # 创建全1的网格
        y_embed = grid.cumsum(dim=0) - 0.5  # 计算 y 坐标的累积和并偏移
        x_embed = grid.cumsum(dim=1) - 0.5  # 计算 x 坐标的累积和并偏移
        y_embed = y_embed / h  # 归一化 y 坐标到 [0,1]
        x_embed = x_embed / w  # 归一化 x 坐标到 [0,1]

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # 堆叠 x 和 y 坐标并进行位置编码
        return pe.permute(2, 0, 1)  # 调整维度顺序为 [C, H, W]

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """对未归一化到 [0,1] 的点进行位置编码。"""
        coords = coords_input.clone()  # 克隆输入坐标
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # 归一化 x 坐标到 [0,1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # 归一化 y 坐标到 [0,1]
        return self._pe_encoding(coords.to(torch.float))  # 对归一化后的坐标进行位置编码 # B x N x C


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    """
    初始化 x 和 y 坐标。

    Args:
        end_x (int): x 轴的结束位置。
        end_y (int): y 轴的结束位置。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: x 和 y 坐标张量。
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)  # 创建从0到end_x*end_y-1的序列
    t_x = (t % end_x).float()  # 计算 x 坐标
    t_y = torch.div(t, end_x, rounding_mode="floor").float()  # 计算 y 坐标
    return t_x, t_y  # 返回 x 和 y 坐标


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    计算轴向旋转复数（axial rotary complex numbers）。

    Args:
        dim (int): 位置特征维度。
        end_x (int): x 轴的结束位置。
        end_y (int): y 轴的结束位置。
        theta (float, optional): 温度参数。默认值为10000.0。

    Returns:
        torch.Tensor: 旋转复数矩阵。
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # 计算 x 频率
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # 计算 y 频率

    t_x, t_y = init_t_xy(end_x, end_y)  # 初始化 x 和 y 坐标
    freqs_x = torch.outer(t_x, freqs_x)  # 计算 x 的频率编码
    freqs_y = torch.outer(t_y, freqs_y)  # 计算 y 的频率编码
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)  # 生成 x 轴的旋转复数
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)  # 生成 y 轴的旋转复数
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)  # 拼接 x 和 y 的旋转复数


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    重新调整频率复数张量的形状以便广播。

    Args:
        freqs_cis (torch.Tensor): 频率复数张量。
        x (torch.Tensor): 输入张量，用于确定广播的形状。

    Returns:
        torch.Tensor: 重塑后的频率复数张量。
    """
    ndim = x.ndim  # 获取输入张量的维度
    assert 0 <= 1 < ndim  # 确保至少有两个维度
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])  # 确保频率复数的形状匹配
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]  # 构建新的形状列表
    return freqs_cis.view(*shape)  # 重新调整形状

def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    应用旋转位置编码到查询和键。

    Args:
        xq (torch.Tensor): 查询张量，形状 [..., C]。
        xk (torch.Tensor): 键张量，形状 [..., C]。
        freqs_cis (torch.Tensor): 旋转复数张量，形状适合 xq 和 xk。
        repeat_freqs_k (bool, optional): 是否重复频率以匹配键的序列长度。默认值为 False。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 应用旋转编码后的查询和键张量。
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 将查询张量转换为复数形式
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )  # 将键张量转换为复数形式，如果键的长度为0则为 None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # 调整频率复数的形状以便广播
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # 应用旋转编码并展平
    if xk_ is None:
        # 如果没有键进行旋转（由于 dropout）
        return xq_out.type_as(xq).to(xq.device), xk  # 仅返回查询
    # 如果需要重复频率以匹配键的序列长度
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]  # 计算重复因子
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)  # 重复频率复数
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # 应用旋转编码到键并展平
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)  # 返回编码后的查询和键
