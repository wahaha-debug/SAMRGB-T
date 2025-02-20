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

        if self.linear_a_v2 and self.linear_b_v2:
            new_v2 = self.linear_b_v2(self.linear_a_v2(x))
            qkv[:, :, :, -self.dim:] += new_v2
        
        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

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
                w_b_linear_v2
            )
        self.reset_parameters() #调用 reset_parameters 方法，初始化 LoRA 模块的权重（如使用 Kaiming 初始化）。
        self.sam = sam_model

        # 获取SAM掩码解码器中的Transformer特征维度大小，用于后续MLP和卷积层的输入维度
        transformer_dim = self.sam.sam_mask_decoder.transformer_dim  #表示 sam_mask_decoder 的 Transformer 的维度大小。
        # 定义MLP模块，用于处理来自Transformer的原始特征
        # 输入维度为transformer_dim，隐藏层和输出维度为transformer_dim // 8，共3层
        self.mlp_src = MLP(input_dim=transformer_dim, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        # 定义MLP模块，用于处理第0层（低级特征）的特征
        self.mlp_feat_s0 = MLP(input_dim=transformer_dim // 8, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        # 定义MLP模块，用于处理第1层（高级特征）的特征
        self.mlp_feat_s1 = MLP(input_dim=transformer_dim // 4, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        # 定义1x1卷积层，用于融合来自mlp_src、mlp_feat_s0和mlp_feat_s1的特征
        # 输入通道为transformer_dim // 8 * 3（3个MLP输出拼接），输出通道为transformer_dim // 8
        # 核大小为1x1，用于特征维度的映射和融合，不改变空间分辨率
        self.linear_fuse = nn.Conv2d(transformer_dim // 8 * 3, transformer_dim // 8, kernel_size=1)

        # for param in sam_model.image_encoder.parameters():
        #     param.requires_grad = False

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

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

    def forward(self, batched_input, multimask_output):
        # 对输入的 batched_input 进行处理：
        # 首先去掉维度为1的批次维度（squeeze(0)），然后堆叠为一个统一的张量
        batched_input = torch.stack([x.squeeze(0) for x in batched_input], dim=0).squeeze(0)  # 处理输入的维度，确保适合后续操作
        # print(batched_input.size())
        m, b, _, h, w = batched_input.shape  # 提取输入张量的形状信息：m 是序列数，b 是批量大小，h 和 w 是图像高度和宽度
        batched_input = batched_input.reshape(2 * b, 3, h, w)  # 将输入调整为适合模型的形状，通常是对批次维度的展开

        # 获取图像的嵌入特征
        image_embedding = self.sam.forward_image(batched_input)  # 使用 sam 的 forward_image 方法提取视觉特征
        # "vision_features": src,  # 提取的视觉特征   #8，256，40，40
        # "vision_pos_enc": pos,  # 对应的位置信息编码
        # "backbone_fpn": features,  # 主干网络生成的特征金字塔

        high_res_0 = image_embedding["backbone_fpn"][0]     # 获取图像编码器FPN（Feature Pyramid Network）中第一层的高分辨率特征图
        # torch.Size([8, 32, 160, 160])
        high_res_1 = image_embedding["backbone_fpn"][1] # 获取图像编码器FPN中第二层的高分辨率特征图，
        # torch.Size([8, 32, 80, 80])
        feat = image_embedding['vision_features']  # 获取图像的深层视觉特征，形状为
        # torch.Size([8, 32, 40, 40])

        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt

        # 假设 feat 的形状是 (batch_size, channels, height, width)
        feat_0 = feat[0, 6, :, :].cpu()  # 选择第一张图像的第一个通道
        feat_1 = feat[0, 7, :, :].cpu()  # 选择第一张图像的第一个通道
        high_res_0_0 = high_res_0[0, 6, :, :].cpu()  # 选择第一张图像的第一个通道
        high_res_0_1 = high_res_0[0, 7, :, :].cpu()  # 选择第一张图像的第一个通道
        high_res_1_0 = high_res_1[0, 6, :, :].cpu()  # 选择第一张图像的第一个通道
        high_res_1_1 = high_res_1[0, 7, :, :].cpu()  # 选择第一张图像的第一个通道
        to_pil = transforms.ToPILImage()
        # 使用 ToPILImage 进行转换.
        # 定义转换，将 Tensor 转换为 PIL 图片
        feat_0 = to_pil(feat_0)
        feat_1 = to_pil(feat_1)
        image0_0 = to_pil(high_res_0_0)
        image0_1 = to_pil(high_res_0_1)
        image1_0= to_pil(high_res_1_0)
        image1_1 = to_pil(high_res_1_1)
        mode = 'GnBu'
        #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', '
        # BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
        # 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Grays_r',
        # 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
        # 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
        # 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r',
        # 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd',
        # 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', '
        # RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds',
        # 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
        # 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
        # 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
        # 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r',
        # 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
        # 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
        # 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
        # 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
        # 'gist_grey_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
        # 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
        # 'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2',
        # 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r',
        # 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
        # 'jet_r', 'magma', 'magma_r', 'managua', 'managua_r',
        # 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r',
        # 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
        # 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
        # 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
        # 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
        # 'twilight_shifted', 'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis', 'viridis_r',
        # 'winter', 'winter_r'
        # 显示图片
        plt.imshow(feat_0, cmap=mode)  # 这里使用 'gray' 来显示单通道图像
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        plt.imshow(feat_1,  cmap=mode)  # 这里使用 'gray' 来显示单通道图像
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        plt.imshow(image0_0,  cmap=mode)  # 这里使用 'gray' 来显示单通道图像
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        plt.imshow(image0_1,  cmap=mode)  # 这里使用 'gray' 来显示单通道图像
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        plt.imshow(image1_0,  cmap=mode)  # 这里使用 'gray' 来显示单通道图像
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        plt.imshow(image1_1,  cmap=mode) # 这里使用 'gray' 来显示单通道图像
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        exit()




        # 通过 SAM 解码器生成分割掩码和其他特征
        multi_mask_output = self.sam._forward_sam_heads(
            image_embedding['vision_features'],  # 输入视觉特征
            high_res_features=image_embedding['backbone_fpn'][:2],  # 高分辨率特征，从 FPN 中提取 feat_s0, feat_s1 = high_res_features  # 获取高分辨率特征
            multimask_output=multimask_output  # 控制是否生成多掩码输出
        )

        # 提取 multi_mask_output 的第二个元素作为目标输出
        m_output = multi_mask_output[1]  # 通常是分割掩码或其他感兴趣的特征
        _, fc, fh, fw = m_output.size()  # 获取输出的形状信息
        m_output = m_output.reshape(m, b, fc, fh, fw)  # 恢复原始批次维度的形状
        m_output = torch.mean(m_output, dim=0)  # 对维度 m 进行平均，可能是为了合并不同序列的结果

        return m_output  # 返回最终的模型输出
