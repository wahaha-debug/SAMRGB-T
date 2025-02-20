# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d


class MaskDownSampler(nn.Module):
    """
    逐步下采样掩码，按照总步幅（total_stride）进行，每次下采样步幅为 stride。
    注意，LayerNorm 是按每个 *token* 应用的，就像在 ViT 中一样。

    每次下采样（步幅为 stride**2）后，通道容量按相同的因子增加。
    最后，线性投影到 embed_dim 通道。
    """

    def __init__(
        self,
        embed_dim=256,  # 嵌入维度
        kernel_size=4,  # 卷积核大小
        stride=4,  # 卷积步幅
        padding=0,  # 卷积填充
        total_stride=16,  # 总步幅
        activation=nn.GELU,  # 激活函数
    ):
        super().__init__()  # 调用父类的初始化方法
        num_layers = int(math.log2(total_stride) // math.log2(stride))  # 计算下采样层数
        assert stride**num_layers == total_stride, "stride 的幂必须等于 total_stride"  # 确保 stride 的幂等于 total_stride
        self.encoder = nn.Sequential()  # 初始化编码器为顺序容器
        mask_in_chans, mask_out_chans = 1, 1  # 初始化输入和输出通道数为1
        for _ in range(num_layers):  # 遍历下采样层数
            mask_out_chans = mask_in_chans * (stride**2)  # 每次下采样后通道数增加 stride^2 倍
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )  # 添加卷积层
            self.encoder.append(LayerNorm2d(mask_out_chans))  # 添加 LayerNorm 层
            self.encoder.append(activation())  # 添加激活函数
            mask_in_chans = mask_out_chans  # 更新输入通道数为当前输出通道数

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))  # 添加最后的 1x1 卷积层，将通道数投影到 embed_dim

    def forward(self, x):
        return self.encoder(x)  # 通过编码器进行前向传播并返回结果


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. 有两个等效的实现：
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; 全部在 (N, C, H, W) 中
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    我们使用 (2)，因为我们发现它在 PyTorch 中稍微快一些

    Args:
        dim (int): 输入通道数。
        drop_path (float): 随机深度率。默认值：0.0
        layer_scale_init_value (float): Layer Scale 的初始化值。默认值：1e-6。
    """

    def __init__(
        self,
        dim,
        kernel_size=7,  # 卷积核大小
        padding=3,  # 卷积填充
        drop_path=0.0,  # 随机深度率
        layer_scale_init_value=1e-6,  # Layer Scale 初始化值
        use_dwconv=True,  # 是否使用深度可分离卷积
    ):
        super().__init__()  # 调用父类的初始化方法
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,  # 如果使用深度可分离卷积，则 groups=dim
        )  # 深度可分离卷积
        self.norm = LayerNorm2d(dim, eps=1e-6)  # LayerNorm 层，按通道进行规范化
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # 点卷积/1x1 卷积，用线性层实现
        self.act = nn.GELU()  # GELU 激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 点卷积/1x1 卷积，用线性层实现
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  # Layer Scale 参数，如果初始化值大于0，则创建可训练参数
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # 随机深度层，如果 drop_path > 0，则使用 DropPath，否则使用恒等映射

    def forward(self, x):
        input = x  # 保存输入用于残差连接
        x = self.dwconv(x)  # 通过深度可分离卷积
        x = self.norm(x)  # 通过 LayerNorm
        x = x.permute(0, 2, 3, 1)  # 将张量从 [N, C, H, W] 转换为 [N, H, W, C]
        x = self.pwconv1(x)  # 通过第一个线性层
        x = self.act(x)  # 应用 GELU 激活函数
        x = self.pwconv2(x)  # 通过第二个线性层
        if self.gamma is not None:
            x = self.gamma * x  # 如果有 gamma 参数，则缩放输出
        x = x.permute(0, 3, 1, 2)  # 将张量从 [N, H, W, C] 转换回 [N, C, H, W]

        x = input + self.drop_path(x)  # 残差连接并应用 DropPath
        return x  # 返回输出


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()  # 调用父类的初始化方法
        self.proj = nn.Identity()  # 初始化投影为恒等映射
        self.layers = get_clones(layer, num_layers)  # 克隆多个层
        if input_projection:
            assert dim is not None, "dim 必须在 input_projection 为 True 时提供"  # 如果需要输入投影，确保 dim 已提供
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)  # 使用 1x1 卷积进行输入投影

    def forward(self, x):
        # 通常 x: (N, C, H, W)
        x = self.proj(x)  # 应用输入投影（如果有的话）
        for layer in self.layers:
            x = layer(x)  # 逐层应用
        return x  # 返回融合后的输出


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,  # 输出维度
        mask_downsampler,  # 掩码下采样模块
        fuser,  # 融合模块
        position_encoding,  # 位置编码模块
        in_dim=256,  # 输入维度（像素特征的输入维度）
    ):
        super().__init__()  # 调用父类的初始化方法

        self.mask_downsampler = mask_downsampler  # 保存掩码下采样模块

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)  # 像素特征的 1x1 卷积投影
        self.fuser = fuser  # 保存融合模块
        self.position_encoding = position_encoding  # 保存位置编码模块
        self.out_proj = nn.Identity()  # 初始化输出投影为恒等映射
        if out_dim != in_dim:  # 如果输出维度不同于输入维度
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)  # 使用 1x1 卷积进行输出投影

    def forward(
        self,
        pix_feat: torch.Tensor,  # 像素特征，形状 [B, C, H, W]
        masks: torch.Tensor,  # 掩码，形状 [B, 1, H, W]
        skip_mask_sigmoid: bool = False,  # 是否跳过掩码的 sigmoid 处理
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## 处理掩码
        # sigmoid，使得与布尔值的真实掩码的领域偏移较小
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)  # 应用 sigmoid 函数
        masks = self.mask_downsampler(masks)  # 通过掩码下采样模块

        ## 融合像素特征和下采样后的掩码
        # 以防视觉特征在 CPU 上，将其转换到 masks 的设备
        pix_feat = pix_feat.to(masks.device)  # 将像素特征移动到 masks 的设备上

        x = self.pix_feat_proj(pix_feat)  # 通过像素特征的 1x1 卷积投影
        x = x + masks  # 将掩码与像素特征相加
        x = self.fuser(x)  # 通过融合模块
        x = self.out_proj(x)  # 通过输出投影模块

        pos = self.position_encoding(x).to(x.dtype)  # 通过位置编码模块并转换数据类型

        return {"vision_features": x, "vision_pos_enc": [pos]}  # 返回视觉特征和位置编码