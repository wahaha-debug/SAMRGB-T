# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

# from .sam2_utils import LayerNorm2d, MLP
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        根据图像和提示嵌入预测掩码，使用 Transformer 架构。

        参数:
          transformer_dim (int): Transformer 的通道维度
          transformer (nn.Module): 用于预测掩码的 Transformer 模块
          num_multimask_outputs (int): 当存在多个掩码输出时的掩码数量
          activation (nn.Module): 用于掩码上采样的激活函数
          iou_head_depth (int): 用于预测掩码质量的 MLP 深度
          iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏层维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim  # Transformer 特征维度
        self.transformer = transformer  # Transformer 模块

        self.num_multimask_outputs = num_multimask_outputs  # 多掩码输出的数量

        self.iou_token = nn.Embedding(1, transformer_dim)  # 用于预测交并比（IoU）的嵌入 token
        self.num_mask_tokens = num_multimask_outputs + 1  # 掩码 token 的总数（包括单掩码 token 和多掩码 token）
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)  # 掩码 token 的嵌入

        self.pred_obj_scores = pred_obj_scores  # 是否启用物体得分预测
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)  # 物体得分 token 嵌入
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr  # 是否为对象指针启用多掩码 token

        #################？？？？？？？？？？？？？？？？冻结？？？？？？？？？？？？？？？？？？？？？？############################
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),  # 转置卷积用于特征上采样
            LayerNorm2d(transformer_dim // 4),  # 归一化
            activation(),  # 激活函数
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),  # 再次上采样
            activation(),  # 激活函数
        )

        self.use_high_res_features = use_high_res_features  # 是否使用高分辨率特征
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )  # 高分辨率特征的第 0 层卷积
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )  # 高分辨率特征的第 1 层卷积
        # 训练  超网络 MLP，用于生成每个掩码的特定输出
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)     # 每个掩码 token 都有一个对应的 MLP，输入维度和隐藏维度为 transformer_dim，输出维度为 transformer_dim // 8
                for i in range(self.num_mask_tokens)
            ]
        )

        # 冻结
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        # 用于预测每个掩码的质量（交并比 IoU）的 MLP，输入维度为 transformer_dim
        # 隐藏层维度为 iou_head_hidden_dim，输出维度为 num_mask_tokens，深度为 iou_head_depth
        # 可选使用 Sigmoid 激活函数

        # 冻结
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
        # 如果启用了物体得分预测，定义一个线性层或 MLP 用于输出物体得分

        # 动态多掩码稳定性策略
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability  # 是否启用基于稳定性的动态多掩码
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta  # 动态稳定性策略的 Delta 值
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh  # 动态稳定性策略的阈值

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据图像和提示嵌入预测掩码。

        参数:
          image_embeddings (torch.Tensor): 从图像编码器提取的嵌入
          image_pe (torch.Tensor): 与图像嵌入形状一致的位置编码
          sparse_prompt_embeddings (torch.Tensor): 点和框的稀疏提示嵌入
          dense_prompt_embeddings (torch.Tensor): 掩码输入的稠密提示嵌入
          multimask_output (bool): 是否返回多个掩码
          repeat_image (bool): 图像是否需要重复处理
          high_res_features (List[torch.Tensor] 或 None): 可选的高分辨率特征

        返回:
          torch.Tensor: 批处理的预测掩码
          torch.Tensor: 批处理的掩码质量预测
          torch.Tensor: 用于掩码输出的 SAM token
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )
        # 调用 `predict_masks` 方法，根据输入嵌入生成预测掩码（masks）、交并比预测（iou_pred）、掩码token输出（mask_tokens_out）和物体分数logits（object_score_logits）

        # 根据是否启用多掩码输出，选择正确的掩码或调整掩码
        if multimask_output:
            masks = masks[:, 1:, :, :]  # 如果多掩码输出，则从第二个掩码开始保留掩码
            iou_pred = iou_pred[:, 1:]  # 只保留对应的交并比预测
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
            # 如果启用了基于稳定性的动态多掩码策略（且模型处于测试模式），则调用对应方法处理掩码和交并比
        else:
            masks = masks[:, 0:1, :, :]  # 默认情况下，仅保留第一个掩码
            iou_pred = iou_pred[:, 0:1]  # 只保留第一个掩码的交并比预测

        # 处理 SAM token 输出
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # 如果是多掩码输出且启用了对象指针多掩码token，则返回对应token
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]  # 始终返回单个掩码的token（包括测试时的多掩码模式）
            # 即使在多点击跟踪时训练模型，过去的token仍然被认为是单掩码token

        # 返回处理后的输出
        return masks, iou_pred, sam_tokens_out, object_score_logits
        # 返回预测掩码、掩码质量预测（iou_pred）、SAM token输出和物体分数logits

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测掩码。更多细节参考 `forward` 方法。"""

        # 拼接输出 tokens
        s = 0  # 起始索引
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,  # 物体分数 token
                    self.iou_token.weight,  # IoU token
                    self.mask_tokens.weight,  # 掩码 tokens
                ],
                dim=0,
            )
            s = 1  # 如果启用了物体分数预测，IoU token 在第一个位置
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )  # 如果未启用物体分数预测，直接使用 IoU 和掩码 tokens

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        # 扩展输出 tokens，使其与批次大小一致
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # 将输出 tokens 和稀疏提示嵌入拼接

        # 扩展每张图像的特征，使其与每个掩码的 token 匹配
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        # 将稠密提示嵌入加到图像嵌入上

        assert (
                image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # 扩展图像的位置编码，使其匹配批次大小

        b, c, h, w = src.shape  # 获取图像嵌入的形状
        # 冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结
        # 通过 Transformer 处理特征
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]  # 提取 IoU token 输出
        mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]  # 提取掩码 tokens 输出

        # 冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结冻结
        # 上采样掩码嵌入并使用掩码 tokens 预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)  # 恢复特征的空间维度
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)  # 通过转置卷积上采样特征
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling  # 高分辨率特征的处理
            feat_s0, feat_s1 = high_res_features  # 获取高分辨率特征
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))  # 合并高分辨率特征和原始特征
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)  # 进一步处理
        #
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
        #     ),  # 转置卷积用于特征上采样
        #     LayerNorm2d(transformer_dim // 4),  # 归一化
        #     activation(),  # 激活函数
        #     nn.ConvTranspose2d(
        #         transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
        #     ),  # 再次上采样
        #     activation(),  # 激活函数
        # )

        # 使用超网络 MLP 生成掩码   训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )  # 对每个掩码 token 通过 MLP 生成超网络输入
        hyper_in = torch.stack(hyper_in_list, dim=1)  # 将所有超网络输入堆叠
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # 通过矩阵乘法生成最终的掩码，恢复为图像的空间维度

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)  # 使用 IoU token 输出预测掩码质量
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
            # 如果启用了物体分数预测，使用对应的输出 token 预测物体分数
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
            # 如果未启用物体分数预测，默认分数为 10.0（表示高置信度）

        return masks, iou_pred, mask_tokens_out, object_score_logits
        # 返回预测的掩码、掩码质量预测、掩码 tokens 和物体分数

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
