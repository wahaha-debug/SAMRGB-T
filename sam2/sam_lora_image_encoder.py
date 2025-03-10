import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic
from sam2.modeling.sam2_base import SAM2Base
import torch.nn.init as init

class MLP_my(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU, sigmoid_output=False):
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


# class LightweightAttentionFusion(nn.Module):
#     def __init__(self, input_dim):
#         super(LightweightAttentionFusion, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 8, kernel_size=1),
#             nn.BatchNorm2d(input_dim // 8),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(input_dim // 8, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         attention_map = self.attention(x)
#         out = self.gamma * attention_map * x + x
#         return out


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv  # 输入的qkv线性层
        self.linear_a_q = linear_a_q  # 用于Q的低秩适配A
        self.linear_b_q = linear_b_q  # 用于Q的低秩适配B
        self.linear_a_v = linear_a_v  # 用于V的低秩适配A
        self.linear_b_v = linear_b_v  # 用于V的低秩适配B
        self.dim = qkv.in_features  # QKV输入的特征维度
        self.w_identity = torch.eye(qkv.in_features)  # 初始化单位矩阵

    def forward(self, x):
        qkv = self.qkv(x)  # 原始QKV计算
        new_q = self.linear_b_q(self.linear_a_q(x))  # 计算低秩适配后的Q
        new_v = self.linear_b_v(self.linear_a_v(x))  # 计算低秩适配后的V
        qkv[:, :, :, : self.dim] += new_q  # 将低秩Q加到原始Q部分
        qkv[:, :, :, -self.dim:] += new_v  # 将低秩V加到原始V部分
        return qkv  # 返回调整后的QKV结果


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

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.trunk.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

        transformer_dim = self.sam.sam_mask_decoder.transformer_dim
        self.mlp_src = MLP_my(input_dim=transformer_dim, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        self.mlp_feat_s0 = MLP_my(input_dim=transformer_dim // 8, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        self.mlp_feat_s1 = MLP_my(input_dim=transformer_dim // 4, hidden_dim=transformer_dim // 8, output_dim=transformer_dim // 8, num_layers=3)
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = nn.Conv2d(transformer_dim // 8 * 3, transformer_dim // 8, kernel_size=1)

        self.linear_pred = nn.Conv2d(transformer_dim // 8, 25, kernel_size=1)
        # self.att_fuse = LightweightAttentionFusion(transformer_dim // 8 * 3)

        # Used for deeplab fuse
        self.conv_1x1_feat = nn.Conv2d(256, 48, kernel_size=1)  # Reduce feat channels to 48
        self.conv_3x3_fusion = nn.Sequential(
            nn.Conv2d(48 + 32 + 64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(256, 9, kernel_size=1)  # Assuming 25 is the number of output classes

        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize weights for MLPs
        for m in [self.mlp_src, self.mlp_feat_s0, self.mlp_feat_s1]:
            for layer in m.layers:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)

        # Initialize weights for Conv2d layer
        init.xavier_uniform_(self.linear_fuse.weight)
        if self.linear_fuse.bias is not None:
            init.constant_(self.linear_fuse.bias, 0)

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""
        加载LoRA和全连接层的参数。仅支持.pt或.pth文件格式。
        如果未安装safetensor库，请运行 `pip install safetensors` 进行安装。
        """

        assert filename.endswith(".pt") or filename.endswith('.pth') # 确保输入的文件名以 .pt 或 .pth 结尾，这两种是常见的模型保存格式。

        state_dict = torch.load(filename)# 使用PyTorch的 `torch.load` 方法加载保存的权重字典。

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}" # 按照保存的键名格式获取对应的权重键，例如 "w_a_000"。
            saved_tensor = state_dict[saved_key] # 从保存的字典中获取对应的权重张量。
            w_A_linear.weight = Parameter(saved_tensor) # 将加载的权重赋值到当前模型的LoRA参数 `w_As` 中。

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}" # 获取对应的权重键名，例如 "w_b_000"。
            saved_tensor = state_dict[saved_key] # 从字典中获取保存的权重张量。
            w_B_linear.weight = Parameter(saved_tensor) # 将加载的权重赋值到当前模型的LoRA参数 `w_Bs` 中。

        sam_dict = self.sam.state_dict()   # 获取SAM模型的当前状态字典。
        sam_keys = sam_dict.keys()  # 获取SAM模型所有权重的键名。

        # 加载Prompt Encoder的参数
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]  # 筛选出所有与Prompt Encoder相关的权重键名。
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]   # 从加载的状态字典中提取Prompt Encoder对应的权重值。
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}  # 构造一个新的状态字典，键值对对应Prompt Encoder的权重。
        sam_dict.update(prompt_encoder_new_state_dict)   # 更新SAM模型的状态字典，加载Prompt Encoder的参数。

        # 加载Mask Decoder的参数
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]  # 筛选出所有与Mask Decoder相关的权重键名。
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys] # 从加载的状态字典中提取Mask Decoder对应的权重值。
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}    # 构造一个新的状态字典，键值对对应Mask Decoder的权重。
        sam_dict.update(mask_decoder_new_state_dict) # 更新SAM模型的状态字典，加载Mask Decoder的参数。

        self.sam.load_state_dict(sam_dict)
        # 将更新后的权重加载到SAM模型中。

    def reset_parameters(self) -> None:
        # 初始化模型中的LoRA参数（w_As 和 w_Bs），确保在训练开始前权重处于合理的状态。
        '''
        功能：
        重新初始化模型中用于低秩适配的参数 w_As 和 w_Bs。
        w_As 是用于生成低秩特征的权重，w_Bs 是用于将低秩特征映射回原始维度的权重。
        Kaiming初始化： 对 w_As 使用 Kaiming 初始化，这种方法适用于具有非线性激活函数（如ReLU）的深度网络。
        它可以根据层的输入大小动态分配权重范围，减少梯度消失或爆炸的可能性。
        零初始化：  对 w_Bs 的权重初始化为零，这意味着初始阶段模型完全依赖于主模型的预训练权重，而非低秩适配层。
        作用： 确保模型在训练前权重处于良好的初始化状态，以提高训练效果并减少不稳定性。
        '''
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5)) # 对每个低秩适配A（w_A）的权重进行Kaiming初始化。
            # Kaiming初始化有助于提高深层网络的收敛速度，特别是对于ReLU激活函数。
            # 参数 `a=math.sqrt(5)` 是用于计算权重初始化的增益因子（gain）。

        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            # 对每个低秩适配B（w_B）的权重初始化为零。
            # 将低秩适配B初始化为零有助于让模型在初始阶段主要依赖原始权重。

    def seg_fuse(self, src, feat_s0, feat_s1):
        '''
        功能：
        对输入的源特征（src）和其他两组特征（feat_s0 和 feat_s1）进行融合，生成分割输出。
        主要步骤：
        展平与变换：将输入特征展平为二维格式，通过MLP变换后恢复原状。
        上采样对齐：将所有特征上采样到统一的目标分辨率（1024x1024）。
        特征融合：通过通道拼接和1x1卷积对所有特征进行融合处理。
        输出生成：通过卷积生成嵌入结果，用于分割任务。
        输出：
        返回一个形状为 [batch_size, num_classes, 1024, 1024] 的张量，用于分割类别预测。
        '''
        # 定义分割任务的特征融合方法，输入包括源特征 `src` 和两个额外特征 `feat_s0` 和 `feat_s1`。

        b, c, _, _ = src.shape # 获取 `src` 特征的形状，`b` 表示批量大小，`c` 表示通道数。
        src_flat = src.view(b, c, -1).transpose(1, 2) # 将 `src` 特征展平为二维（b, c, h*w），然后转置为（b, h*w, c）。
        feat_s0_flat = feat_s0.view(b, feat_s0.size(1), -1).transpose(1, 2) # 同样对 `feat_s0` 进行展平和转置操作，变为（b, h*w, c）。
        feat_s1_flat = feat_s1.view(b, feat_s1.size(1), -1).transpose(1, 2) # 同样对 `feat_s1` 进行展平和转置操作，变为（b, h*w, c）。

        src_transformed = self.mlp_src(src_flat) # 使用 `mlp_src` 对展平后的 `src` 特征进行MLP变换。
        src_transformed = src_transformed.transpose(1, 2).view(b, -1, src.shape[2], src.shape[3])
        # 将变换后的特征转置回原始格式，并恢复为二维空间形状。
        src_transformed = F.interpolate(src_transformed, size=[1024, 1024], mode='bilinear', align_corners=False)# 将特征上采样到固定的分辨率（1024x1024），使用双线性插值。

        feat_s0_transformed = self.mlp_feat_s0(feat_s0_flat)# 对展平后的 `feat_s0` 特征进行MLP变换。
        feat_s0_transformed = feat_s0_transformed.transpose(1, 2).view(b, -1, feat_s0.size(2), feat_s0.size(3))# 将变换后的 `feat_s0` 特征转置回原始格式，并恢复为二维空间形状。
        feat_s0_transformed = F.interpolate(feat_s0_transformed, size=[1024, 1024], mode='bilinear',
                                            align_corners=False)# 对 `feat_s0` 特征上采样到固定分辨率（1024x1024）。

        feat_s1_transformed = self.mlp_feat_s1(feat_s1_flat)# 对展平后的 `feat_s1` 特征进行MLP变换。
        feat_s1_transformed = feat_s1_transformed.transpose(1, 2).view(b, -1, feat_s1.size(2), feat_s1.size(3))# 将变换后的 `feat_s1` 特征转置回原始格式，并恢复为二维空间形状。
        feat_s1_transformed = F.interpolate(feat_s1_transformed, size=[1024, 1024], mode='bilinear',
                                            align_corners=False)# 对 `feat_s1` 特征上采样到固定分辨率（1024x1024）。

        combined_features = torch.cat([src_transformed, feat_s0_transformed, feat_s1_transformed], dim=1)# 将 `src_transformed`、`feat_s0_transformed` 和 `feat_s1_transformed` 在通道维度（dim=1）进行拼接。
        combined_features = self.dropout(combined_features)# 对拼接后的特征进行Dropout操作，随机置零一部分特征，防止过拟合。
        upscaled_embedding = self.linear_fuse(combined_features)# 使用1x1卷积（`linear_fuse`）对拼接的特征进行融合。
        upscaled_embedding = self.linear_pred(upscaled_embedding) # 使用1x1卷积（`linear_pred`）生成最终的分割预测结果，通常为类别数通道。

        return upscaled_embedding # 返回融合后的特征嵌入，用于分割任务的输出。

    def fpn_fuse(self, feat, high_res_0, high_res_1):
        '''
       功能：
        实现特征金字塔网络（FPN）的特征融合，结合不同分辨率的特征图。
        融合后的特征用于分割任务的后续处理。
        主要步骤：
        降维处理：对 high_res_0 和 high_res_1 应用1x1卷积，统一通道数为256。
        分辨率对齐：通过插值将特征图对齐到相同分辨率，便于逐像素操作。
        特征融合：逐像素相加融合不同层次的特征图。
        深层特征增强：融合高层次和低层次特征，进一步提取有用信息。
        输出对齐：将最终的融合特征调整到目标分辨率。
        输出：
        返回融合后的特征图，形状为 [batch_size, 256, 640, 640]，用于后续的分割任务。
        '''
        device = feat.device  # 获取输入特征所在的设备（GPU或CPU）。

        high_res_0_conv = nn.Conv2d(32, 256, kernel_size=1).to(device)(high_res_0)
        # 对高分辨率特征0 (`high_res_0`) 应用1x1卷积，将通道数从32扩展到256。
        high_res_1_conv = nn.Conv2d(64, 256, kernel_size=1).to(device)(high_res_1)
        # 对高分辨率特征1 (`high_res_1`) 应用1x1卷积，将通道数从64扩展到256。

        high_res_1_up = F.interpolate(high_res_1_conv, size=high_res_0.shape[2:], mode='bilinear', align_corners=False)
        # 将特征1的分辨率上采样到与特征0相同的空间尺寸。
        # 使用双线性插值法进行上采样，确保大小对齐以进行加法融合。
        fpn_merge_1 = high_res_0_conv + high_res_1_up
        # 将处理后的特征0和上采样后的特征1进行逐像素相加，完成初步特征融合。
        fpn_merge_1_up = F.interpolate(fpn_merge_1, size=feat.shape[2:], mode='bilinear', align_corners=False)
        # 将融合后的特征再次上采样到深层特征 `feat` 的空间尺寸。
        final_merge = feat + fpn_merge_1_up
        # 将深层特征 `feat` 和上采样后的融合特征相加，完成特征的最终融合。
        final_output = nn.Conv2d(256, 256, kernel_size=3, padding=1).to(device)(final_merge)
        # 对最终融合的特征应用3x3卷积，进一步处理特征，提取上下文信息。
        # 卷积输出的通道数仍然是256。
        final_output = F.interpolate(final_output, size=(640, 640), mode='bilinear', align_corners=False)
        # 最后将输出特征图上采样到固定的目标分辨率（640x640）。
        # 上采样确保输出的大小与分割任务的要求一致。

        return final_output
        # 返回最终融合后的特征图。

    def deeplab_fuse(self, feat, high_res_0, high_res_1):
        # 定义DeepLab特征融合方法，输入为深层特征 `feat` 和两层高分辨率特征 `high_res_0` 和 `high_res_1`。

        feat = self.conv_1x1_feat(feat)
        # 对输入的深层特征 `feat` 进行1x1卷积，减少其通道数。
        # 目的是降低特征维度，减少计算复杂度，增强与其他特征的融合效果。
        feat_up = F.interpolate(feat, size=high_res_1.shape[2:], mode='bilinear', align_corners=False)
        # 将 `feat` 上采样到与 `high_res_1` 相同的分辨率。
        # 使用双线性插值（bilinear interpolation）方法进行上采样。
        high_res_0_up = F.interpolate(high_res_0, size=high_res_1.shape[2:], mode='bilinear', align_corners=False)
        # 将高分辨率特征 `high_res_0` 上采样到与 `high_res_1` 相同的分辨率。
        # 同样使用双线性插值方法。
        combined_features = torch.cat([feat_up, high_res_0_up, high_res_1], dim=1)
        # 将上采样后的 `feat_up`、`high_res_0_up` 和原始 `high_res_1` 在通道维度（dim=1）进行拼接。
        # 拼接的特征整合了深层特征和高分辨率特征。
        fused_features = self.conv_3x3_fusion(combined_features)
        # 对拼接后的特征应用一系列3x3卷积进行特征融合。
        # 该操作能够捕获局部上下文信息，并生成融合后的特征图。
        output = self.final_conv(fused_features)
        # 对融合后的特征图应用1x1卷积，生成最终的分割结果。
        # 这里的通道数通常对应于分割任务中的类别数。
        output = F.interpolate(output, size=(640, 640), mode='bilinear', align_corners=False)
        # 将输出特征图上采样到最终的目标分辨率（640x640）。
        # 双线性插值方法确保输出的分辨率与期望分辨率一致。
        return output
        # 返回最终的分割结果，包含目标类别的预

    def forward(self, batched_input, multimask_output):
        # print(batched_input.shape)
        b, _, h, w = batched_input[0].shape
        # print("sample", b, _, h, w )
        sample = torch.stack([batched_input[0], batched_input[1]], dim=0)
        # print("sample", sample.shape)  #torch.Size([2, 4, 3, 640, 640])
        sample = sample.permute(1, 0, 2, 3, 4)
        # print("sample", sample.shape)  #torch.Size([4, 2, 3, 640, 640])
        batched_input = sample.reshape(2 * b, 3, h, w)
        # print("batched_input", batched_input.shape)  #torch.Size([8, 3, 640, 640])
        image_embedding = self.sam.forward_image(batched_input)     # 使用SAM模型的前向方法对输入图像进行编码，返回图像的嵌入特征，包括多层特征。

        high_res_0 = image_embedding["backbone_fpn"][0]     # 获取图像编码器FPN（Feature Pyramid Network）中第一层的高分辨率特征图
        # torch.Size([8, 32, 160, 160])
        high_res_1 = image_embedding["backbone_fpn"][1] # 获取图像编码器FPN中第二层的高分辨率特征图，
        # torch.Size([8, 32, 80, 80])
        feat = image_embedding['vision_features']  # 获取图像的深层视觉特征，形状为
        # torch.Size([8, 32, 40, 40])
        output = self.seg_fuse(feat, high_res_0, high_res_1) # 使用DeepLab融合方法对 `feat`、`high_res_0` 和 `high_res_1` 进行特征融合，生成输出特征图。
        print(output.shape)

        # # 使用SAM模型的多掩码头（_forward_sam_heads）生成多掩码输出。
        # # 输入包括视觉特征（`vision_features`）和高分辨率特征（FPN的前两层）。
        multi_mask_output = self.sam._forward_sam_heads(image_embedding['vision_features'],
                                                        high_res_features=image_embedding['backbone_fpn'][:2],
                                                        multimask_output=multimask_output)  # , image_size

        m_output = multi_mask_output[1]
        # print(  m_output.shape )  #torch.Size([8, 9, 640, 640])
        b, fc, fh, fw = m_output.size()  #8, 9, 640, 640
        B = int(b)//2
        fc = int(fc)
        fh = int(fh)
        fw = int(fw)
        # print(b, fc, fh, fw )
        # m_output = m_output.reshape(m,b,fc,fh,fw)
        m_output = m_output.reshape(2, B, 9, fh, fw)
        m_output = torch.mean(m_output, dim=0)

        return m_output,output
        # return multi_mask_output, output
        # 返回两个结果：
        # 1. `multi_mask_output`：来自SAM模型的多掩码预测结果。
        # 2. `output`：通过DeepLab融合生成的特征输出结果。
