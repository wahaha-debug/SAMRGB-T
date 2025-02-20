# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk  # 主干网络（通常是一个深度学习模型，用于提取基础特征）
        self.neck = neck  # 颈部模块（通常用于进一步处理主干网络的输出特征）
        self.scalp = scalp  # 表示是否丢弃最低分辨率的特征，默认为0（不丢弃）
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        # 检查 trunk 和 neck 的通道尺寸是否匹配，否则抛出错误

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))  # 样本通过 trunk（主干）提取初步特征，再传入 neck 进一步处理，返回特征和位置信息
        #torch.Size([8, 256, 160, 160])
        #torch.Size([8, 256, 80, 80])
        #torch.Size([8, 256, 40, 40])
        #torch.Size([8, 256, 20, 20])

        # print(features)
        # print(pos)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]
            # 如果 scalp > 0，则丢弃最低分辨率的特征和对应的位置编码，保留较高分辨率的特征

        src = features[-1]  # 获取最后一个特征（通常是分辨率最高的特征）
        output = {
            "vision_features": src,  # 提取的视觉特征
            "vision_pos_enc": pos,  # 对应的位置信息编码
            "backbone_fpn": features,  # 主干网络生成的特征金字塔
        }
        return output  # 返回一个包含多个键值的字典



class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        # 创建一个与卷积层数量相同的列表，用于存储输出和位置编码
        out = [None] * len(self.convs)  # 存储输出的特征
        pos = [None] * len(self.convs)  # 存储每个特征图的位置编码

        # 确保输入的特征图数量与卷积层数量一致
        assert len(xs) == len(self.convs)

        prev_features = None  # 初始化前一个特征图为None

        # 从高分辨率到低分辨率顺序进行前向传播
        n = len(self.convs) - 1  # 获取卷积层的索引上限
        for i in range(n, -1, -1):  # 从高分辨率到低分辨率进行循环
            x = xs[i]  # 获取输入的第i个特征图（从低到高分辨率）

            # 使用卷积层处理当前输入特征图，得到侧支特征（lateral features）
            lateral_features = self.convs[n - i](x)

            # 如果当前层属于FPN的顶层，并且上一个特征图不为None，则进行自上而下的特征融合
            if i in self.fpn_top_down_levels and prev_features is not None:
                # 通过上采样（插值）将上一个特征图的分辨率提高一倍
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),  # 将prev_features转换为浮动类型
                    scale_factor=2.0,  # 通过2倍的缩放因子来上采样
                    mode=self.fpn_interp_model,  # 上采样使用的插值模型（例如nearest或bilinear）
                    align_corners=None if self.fpn_interp_model == "nearest" else False,  # 确定是否对齐角点
                    antialias=False,  # 是否启用抗锯齿
                )

                # 将侧支特征与上采样后的特征进行相加，融合不同层的特征
                prev_features = lateral_features + top_down_features

                # 如果融合方式是平均（"avg"），则将结果除以2
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                # 如果当前层不是FPN顶层或没有prev_features，直接使用侧支特征
                prev_features = lateral_features

            # 输出当前层的特征图
            x_out = prev_features
            out[i] = x_out  # 将当前层的输出特征图保存到out列表中

            # 计算位置编码并保存到pos列表
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        # 返回所有层的特征图和位置编码
        return out, pos
