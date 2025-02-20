# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional  # 从 typing 模块导入 Optional

import torch  # 导入 PyTorch
from torch import nn, Tensor  # 从 torch 模块导入 nn 和 Tensor

from sam2.modeling.sam.transformer import RoPEAttention  # 从 sam2.modeling.sam.transformer 模块导入 RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones  # 从 sam2.modeling.sam2_utils 模块导入 get_activation_fn 和 get_clones


class MemoryAttentionLayer(nn.Module):
    """
    MemoryAttentionLayer 是一个Transformer层，包含自注意力、交叉注意力和前馈神经网络（MLP）模块。
    它用于在当前目标（tgt）和记忆（memory）之间进行信息交互。
    """

    def __init__(
        self,
        activation: str,  # 激活函数的名称
        cross_attention: nn.Module,  # 交叉注意力模块
        d_model: int,  # 模型的嵌入维度
        dim_feedforward: int,  # 前馈网络的隐藏层维度
        dropout: float,  # dropout 概率
        pos_enc_at_attn: bool,  # 是否在自注意力中添加位置编码
        pos_enc_at_cross_attn_keys: bool,  # 是否在交叉注意力的键中添加位置编码
        pos_enc_at_cross_attn_queries: bool,  # 是否在交叉注意力的查询中添加位置编码
        self_attention: nn.Module,  # 自注意力模块
    ):
        super().__init__()  # 调用父类的初始化方法
        self.d_model = d_model  # 保存嵌入维度
        self.dim_feedforward = dim_feedforward  # 保存前馈网络的隐藏层维度
        self.dropout_value = dropout  # 保存 dropout 概率
        self.self_attn = self_attention  # 自注意力模块
        self.cross_attn_image = cross_attention  # 交叉注意力模块

        # 实现前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 前馈网络的第一层线性变换
        self.dropout = nn.Dropout(dropout)  # dropout 层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 前馈网络的第二层线性变换

        self.norm1 = nn.LayerNorm(d_model)  # 第一个 LayerNorm，用于自注意力后
        self.norm2 = nn.LayerNorm(d_model)  # 第二个 LayerNorm，用于交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)  # 第三个 LayerNorm，用于前馈网络后
        self.norm4 = nn.LayerNorm(d_model)  # 第四个 LayerNorm，用于反向交叉注意力后
        self.dropout1 = nn.Dropout(dropout)  # 第一个 dropout 层
        self.dropout2 = nn.Dropout(dropout)  # 第二个 dropout 层
        self.dropout3 = nn.Dropout(dropout)  # 第三个 dropout 层

        self.activation_str = activation  # 保存激活函数的名称
        self.activation = get_activation_fn(activation)  # 获取激活函数实例

        # 决定在哪里添加位置编码
        self.pos_enc_at_attn = pos_enc_at_attn  # 是否在自注意力中添加位置编码
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys  # 是否在交叉注意力的键中添加位置编码
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries  # 是否在交叉注意力的查询中添加位置编码

    def _forward_sa(self, tgt, query_pos):
        # 自注意力块
        if self.skip_first_layer_pe:
            tgt2 = self.self_attn(q=tgt, k=tgt, v=tgt)  # 如果跳过第一层PE，直接进行自注意力
        else:
            q = tgt + query_pos  # 添加查询位置编码
            attn_out = self.self_attn(q=q, k=q, v=tgt)  # 计算自注意力输出
            tgt2 = tgt + attn_out  # 更新目标
        tgt = self.norm1(tgt2)  # 应用第一个 LayerNorm
        tgt = self.dropout1(tgt)  # 应用第一个 dropout
        return tgt  # 返回更新后的目标

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)  # 确保 cross_attn_image 是 RoPEAttention 的实例
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}  # 设置额外的参数

        # 交叉注意力块，tokens 关注图像嵌入
        q = tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt  # 根据配置添加查询位置编码
        k = memory + pos if self.pos_enc_at_cross_attn_keys else memory  # 根据配置添加键位置编码
        attn_out = self.cross_attn_image(q=q, k=k, v=memory, **kwds)  # 计算交叉注意力输出
        tgt2 = tgt + attn_out  # 更新目标
        tgt2 = self.norm2(tgt2)  # 应用第二个 LayerNorm
        tgt2 = self.dropout2(tgt2)  # 应用第二个 dropout
        return tgt2  # 返回更新后的目标

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        # 自注意力
        tgt = self._forward_sa(tgt, query_pos)  # 调用自注意力函数
        # 交叉注意力
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)  # 调用交叉注意力函数
        # 前馈网络
        tgt2 = self.norm3(tgt)  # 应用第三个 LayerNorm
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))  # 通过前馈网络
        tgt = tgt + self.dropout3(tgt2)  # 更新目标并应用第三个 dropout
        return tgt  # 返回最终的目标



class MemoryAttention(nn.Module):
    """
    MemoryAttention 类用于堆叠多个 MemoryAttentionLayer，以实现深层次的注意力机制。
    它可以选择是否在输入时添加位置编码，并支持批次优先的输入格式。
    """

    def __init__(
        self,
        d_model: int,  # 模型的嵌入维度
        pos_enc_at_input: bool,  # 是否在输入时添加位置编码
        layer: nn.Module,  # MemoryAttentionLayer 实例
        num_layers: int,  # 堆叠的层数
        batch_first: bool = True,  # 输入是否以 batch 为第一维
    ):
        super().__init__()  # 调用父类的初始化方法
        self.d_model = d_model  # 保存嵌入维度
        self.layers = get_clones(layer, num_layers)  # 克隆多个 MemoryAttentionLayer
        self.num_layers = num_layers  # 保存层数
        self.norm = nn.LayerNorm(d_model)  # 最后一层的 LayerNorm
        self.pos_enc_at_input = pos_enc_at_input  # 是否在输入时添加位置编码
        self.batch_first = batch_first  # 输入是否以 batch 为第一维

    def forward(
        self,
        curr: torch.Tensor,  # 当前目标的输入（自注意力）
        memory: torch.Tensor,  # 记忆的输入（交叉注意力）
        curr_pos: Optional[Tensor] = None,  # 当前目标的位置信息编码
        memory_pos: Optional[Tensor] = None,  # 记忆的位置信息编码
        num_obj_ptr_tokens: int = 0,  # 对象指针 token 的数量
    ):
        if isinstance(curr, list):  # 如果当前目标是列表
            assert isinstance(curr_pos, list)  # 确保位置编码也是列表
            assert len(curr) == len(curr_pos) == 1  # 确保列表长度为1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )  # 解包列表中的第一个元素

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"  # 确保当前目标和记忆的 batch size 相同

        output = curr  # 初始化输出为当前目标
        if self.pos_enc_at_input and curr_pos is not None:  # 如果配置为在输入时添加位置编码
            output = output + 0.1 * curr_pos  # 添加位置编码（缩放因子为0.1）

        if self.batch_first:  # 如果输入以 batch 为第一维
            # 转换为序列优先
            output = output.transpose(0, 1)  # 转置为 [N, B, C]
            curr_pos = curr_pos.transpose(0, 1)  # 转置位置编码为 [N, B, C]
            memory = memory.transpose(0, 1)  # 转置记忆为 [N, B, C]
            memory_pos = memory_pos.transpose(0, 1)  # 转置记忆位置编码为 [N, B, C]

        for layer in self.layers:  # 遍历所有 MemoryAttentionLayer
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):  # 如果交叉注意力是 RoPEAttention 实例
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}  # 设置额外参数

            output = layer(
                tgt=output,  # 目标输入
                memory=memory,  # 记忆输入
                pos=memory_pos,  # 记忆位置编码
                query_pos=curr_pos,  # 查询位置编码
                **kwds,  # 额外参数
            )  # 通过 MemoryAttentionLayer 进行前向传播
        normed_output = self.norm(output)  # 应用最后的 LayerNorm

        if self.batch_first:  # 如果输入以 batch 为第一维
            # 转换回 batch 优先
            normed_output = normed_output.transpose(0, 1)  # 转置回 [B, N, C]
            curr_pos = curr_pos.transpose(0, 1)  # 转置位置编码回 [B, N, C]

        return normed_output  # 返回规范化后的输出

