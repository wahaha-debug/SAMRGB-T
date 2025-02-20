import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic
# from .modeling.sam2_base_mem import SAM2Base
from sam2.modeling.sam2_base import SAM2Base

import torch.nn.init as init

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
        self.num_layers = num_layers  # 定义网络的层数
        h = [hidden_dim] * (num_layers - 1)  # 隐藏层的维度
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )  # 构建MLP的线性层
        self.sigmoid_output = sigmoid_output  # 是否在输出层使用Sigmoid激活
        self.act = activation()  # 使用用户指定的激活函数

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)  # 对每一层使用激活函数，最后一层不激活
        if self.sigmoid_output:
            x = F.sigmoid(x)  # 如果需要，应用Sigmoid激活函数
        return x  # 返回MLP的输出
    
class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_q2: nn.Module = None,  # Second LoRA module
            linear_b_q2: nn.Module = None,  # Second LoRA module
            linear_a_v2: nn.Module = None,  # Second LoRA module
            linear_b_v2: nn.Module = None,  # Second LoRA module

            linear_a_q3: nn.Module = None,  # Second LoRA module
            linear_b_q3: nn.Module = None,  # Second LoRA module
            linear_a_v3: nn.Module = None,  # Second LoRA module
            linear_b_v3: nn.Module = None,  # Second LoRA module

    ):
        super().__init__()
        self.qkv = qkv  # 输入的qkv线性层
        self.linear_a_q = linear_a_q  # 用于Q的低秩适配A
        self.linear_b_q = linear_b_q  # 用于Q的低秩适配B
        self.linear_a_v = linear_a_v  # 用于V的低秩适配A
        self.linear_b_v = linear_b_v  # 用于V的低秩适配B

        self.linear_a_q2 = linear_a_q2
        self.linear_b_q2 = linear_b_q2
        self.linear_a_v2 = linear_a_v2
        self.linear_b_v2 = linear_b_v2

        self.linear_a_q3 = linear_a_q3
        self.linear_b_q3 = linear_b_q3
        self.linear_a_v3 = linear_a_v3
        self.linear_b_v3 = linear_b_v3

        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # 原始QKV计算
        new_q = self.linear_b_q(self.linear_a_q(x))  # 计算低秩适配后的Q
        new_v = self.linear_b_v(self.linear_a_v(x))  # 计算低秩适配后的V
        qkv[:, :, :, : self.dim] += new_q  # 将低秩Q加到原始Q部分
        qkv[:, :, :, -self.dim:] += new_v  # 将低秩V加到原始V部分
        
        # Apply the second LoRA module if they exist
        if self.linear_a_q2 and self.linear_b_q2:
            new_q2 = self.linear_b_q2(self.linear_a_q2(x))
            qkv[:, :, :, :self.dim] += new_q2
            # print("aaaaaa",qkv[:, :, :, :self.dim].shape)

        if self.linear_a_v2 and self.linear_b_v2:
            new_v2 = self.linear_b_v2(self.linear_a_v2(x))
            qkv[:, :, :, -self.dim:] += new_v2
            # print("bbbbbbb",qkv[:, :, :, :self.dim].shape)

        if self.linear_a_q3 and self.linear_b_q3:
            new_q3 = self.linear_b_q3(self.linear_a_q3(x))
            qkv[:, :, :, :self.dim] += new_q3
            # print("aaaaaa",qkv[:, :, :, :self.dim].shape)

        if self.linear_a_v3 and self.linear_b_v3:
            new_v3 = self.linear_b_v3(self.linear_a_v3(x))
            qkv[:, :, :, -self.dim:] += new_v3
            # print("ccccccc",qkv[:, :, :, :self.dim].shape)

        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.
    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0  # 确保 LoRA 的秩参数 r 为正数
        if lora_layer:
            self.lora_layer = lora_layer  # 使用指定的层
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks))) # 默认将 LoRA 应用于所有的 trunk blocks        # Only apply lora to the image encoder by default

        self.w_As = []  # 第一组 LoRA 层的权重 W_A
        self.w_Bs = []  # 第一组 LoRA 层的权重 W_B

        self.w_As2 = []  # 第二组 LoRA 层的权重 W_A
        self.w_Bs2 = []  # 第二组 LoRA 层的权重 W_B

        self.w_As3 = []  # 第二组 LoRA 层的权重 W_A
        self.w_Bs3 = []  # 第二组 LoRA 层的权重 W_B

        # Freeze original SAM model parameters
        # 冻结原始模型的所有参数
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA to specified layers
        # 遍历图像编码器的每一层
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue  # 如果该层不在指定的 lora_layer 中，跳过
            w_qkv_linear = blk.attn.qkv  # 提取该层的 qkv 线性层
            self.dim = w_qkv_linear.in_features  # 获取线性层的输入维度

            # First LoRA module
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            # Second LoRA module
            w_a_linear_q2 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q2 = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v2 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v2 = nn.Linear(r, self.dim, bias=False)
            self.w_As2.append(w_a_linear_q2)
            self.w_Bs2.append(w_b_linear_q2)
            self.w_As2.append(w_a_linear_v2)
            self.w_Bs2.append(w_b_linear_v2)

            # third LoRA module
            w_a_linear_q3 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q3 = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v3 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v3 = nn.Linear(r, self.dim, bias=False)
            self.w_As2.append(w_a_linear_q3)
            self.w_Bs2.append(w_b_linear_q3)
            self.w_As2.append(w_a_linear_v3)
            self.w_Bs2.append(w_b_linear_v3)

            #使用 _LoRA_qkv 替换原始 qkv 线性层，使其包含 LoRA 的权重更新逻辑。  _LoRA_qkv -> nn.Module -> object
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                w_a_linear_q2,
                w_b_linear_q2,
                w_a_linear_v2,
                w_b_linear_v2,
                w_a_linear_q3,
                w_b_linear_q3,
                w_a_linear_v3,
                w_b_linear_v3
            )
        self.reset_parameters() #调用 reset_parameters 方法，初始化 LoRA 模块的权重（如使用 Kaiming 初始化）。
        self.sam = sam_model

        transformer_dim = self.sam.sam_mask_decoder.transformer_dim
        self.mlp_src = MLP(input_dim=transformer_dim, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        self.mlp_feat_s0 = MLP(input_dim=transformer_dim // 8, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        self.mlp_feat_s1 = MLP(input_dim=transformer_dim // 4, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        self.linear_fuse = nn.Conv2d(transformer_dim // 8 * 3, transformer_dim // 8, kernel_size=1)

        # for param in sam_model.image_encoder.parameters():
        #     param.requires_grad = False

    def save_lora_parameters(self, filename: str) -> None:

        # 确保文件名以 `.pt` 或 `.pth` 结尾，表示支持的保存格式
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # 获取 LoRA 模块的层数（注意：self.w_As 包含多组权重，这里计算层数）
        num_layer = len(self.w_As)  # 实际上它是 LoRA 的一半层数

        # 将第一组 LoRA 模块的权重保存为字典
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        # 用于存储其他模块的参数
        prompt_encoder_tensors = {}  # 存储 prompt encoder 的参数
        mask_decoder_tensors = {}  # 存储 mask decoder 的参数

        # 获取模型的 state_dict（支持 DataParallel 和 DistributedDataParallel）
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam,
                                                                     torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()  # 如果是并行模型，获取实际的模块
        else:
            state_dict = self.sam.state_dict()  # 普通模型直接获取 state_dict

        # 遍历所有参数，将属于 prompt_encoder 和 mask_decoder 的参数分类存储
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value  # 保存 prompt encoder 参数
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value  # 保存 mask decoder 参数

        # 合并所有参数，包括 LoRA 参数和其他模块参数
        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}

        # 使用 safetensors 库保存参数到指定文件
        save_file(merged_dict, filename)

    def reset_parameters(self):
        # 初始化第一组 LoRA 模块的 W_A 参数
        for w_a in self.w_As:
            init.kaiming_uniform_(w_a.weight, a=math.sqrt(5)) # 使用 Kaiming 初始化方法为 W_A 设置初始值，适合 ReLU 激活函数
        # 初始化第一组 LoRA 模块的 W_B 参数
        for w_b in self.w_Bs:
            init.zeros_(w_b.weight) # 将 W_B 的初始值设置为全零，确保初始状态下 LoRA 不影响原模型的输出
        # 初始化第二组 LoRA 模块的 W_A 参数
        for w_a2 in self.w_As2:
            init.kaiming_uniform_(w_a2.weight, a=math.sqrt(5)) # 同样使用 Kaiming 初始化方法
        # 初始化第二组 LoRA 模块的 W_B 参数
        for w_b2 in self.w_Bs2:
            init.zeros_(w_b2.weight)  # 将第二组 W_B 参数初始化为零
    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

    def forward(self, batched_input, multimask_output,text):
        # 对输入的 batched_input 进行处理：
        # 首先去掉维度为1的批次维度（squeeze(0)），然后堆叠为一个统一的张量
        batched_input = torch.stack([x.squeeze(0) for x in batched_input], dim=0).squeeze(0)  # 处理输入的维度，确保适合后续操作
        m, b, _, h, w = batched_input.shape  # 提取输入张量的形状信息：m 是序列数，b 是批量大小，h 和 w 是图像高度和宽度
        batched_input = batched_input.reshape(2 * b, 3, h, w)  # 将输入调整为适合模型的形状，通常是对批次维度的展开

        # 获取图像的嵌入特征
        image_embedding = self.sam.forward_image(batched_input)  # 使用 sam 的 forward_image 方法提取视觉特征

        text_features = self.get_text_feature(text.expand(b, -1)).to(batched_input.dtype)


        # 通过 SAM 解码器生成分割掩码和其他特征
        multi_mask_output = self.sam._forward_sam_heads(
            image_embedding['vision_features'],  # 输入视觉特征
            high_res_features=image_embedding['backbone_fpn'][:2],  # 高分辨率特征，从 FPN 中提取
            multimask_output=multimask_output  # 控制是否生成多掩码输出
        )

        # 提取 multi_mask_output 的第二个元素作为目标输出
        m_output = multi_mask_output[1]  # 通常是分割掩码或其他感兴趣的特征
        _, fc, fh, fw = m_output.size()  # 获取输出的形状信息
        m_output = m_output.reshape(m, b, fc, fh, fw)  # 恢复原始批次维度的形状
        m_output = torch.mean(m_output, dim=0)  # 对维度 m 进行平均，可能是为了合并不同序列的结果

        return m_output  # 返回最终的模型输出
