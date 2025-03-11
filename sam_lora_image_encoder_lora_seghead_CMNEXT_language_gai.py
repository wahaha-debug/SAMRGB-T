import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.codegen.ast import continue_
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from icecream import ic
# from .modeling.sam2_base_mem import SAM2Base
from sam2.modeling.sam2_base import SAM2Base
import torch.nn.init as init
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#+++++++fusion model++++++++++++++++++++++++++++++
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from semseg.models.modules.ffm import FeatureRectifyModule as FRM
#+++++++language++++++++++++++++++++++++++++++
import clip
#+++++++++++++++++++++++++++++++++++++

class PrototypeSegmentation:
    def __init__(self, num_classes, feature_dim):
        self.num_classes = num_classes # 模型需要分割的类别数目。
        self.feature_dim = feature_dim # 每个类别的特征维度。通常是特征图的通道数。
        # Initialize global prototypes for each class
        self.global_prototypes = torch.zeros((num_classes, feature_dim), requires_grad=True).to( 'cuda')  # Set requires_grad=True   #这是一个大小为 (num_classes, feature_dim) 的张量，用来存储每个类别的全局原型。
        # 每个类别的原型是该类别所有样本的特征均值。在训练过程中，原型会不断更新。
        # 初始化时，这个张量被填充为零，并且启用了 requires_grad=True，这意味着它将在反向传播时参与梯度更新。

    def update_global_prototypes(self, current_prototypes): # 来自当前批次的数据（即批次原型）。
        #该方法使用了一个平滑更新策略，将当前批次的原型与已有的全局原型进行加权平均：0.9 是全局原型的权重，0.1 是当前批次原型的权重。
        # Update global prototypes with current prototypes
        self.global_prototypes.data = 0.9 * self.global_prototypes.data + 0.1 * current_prototypes.data
        # Using `.data` avoids creating new computation graphs for the updated values.
        #来避免修改计算图，从而防止影响梯度计算。

    def compute_loss(self, features, labels):
        # Calculate per-class prototypes from current batch
        #通过 calculate_batch_prototypes 方法计算出的当前批次每个类别的原型。
        batch_prototypes = self.calculate_batch_prototypes(features, labels)

        # Update global prototypes  使用当前批次的原型来更新全局原型。
        self.update_global_prototypes(batch_prototypes)

        # Compute the prototype matching loss
        #通过 prototype_loss 方法计算的原型匹配损失。通常使用均方误差损失（MSE Loss）来度量当前批次的原型与全局原型之间的差异。
        prototype_loss = self.prototype_loss(batch_prototypes)

        # Optionally, you can also compute segmentation loss here (e.g., CrossEntropyLoss)
        # segmentation_loss = F.cross_entropy(logits, labels)

        # Combine the losses
        # 最后，将计算出的损失（目前只有 prototype_loss）作为最终的总损失返回。
        total_loss = prototype_loss
        return batch_prototypes, total_loss

    def calculate_batch_prototypes(self, features, labels):
        # Initialize prototypes
        # 初始化为一个大小为 (num_classes, feature_dim) 的零张量，用来存储当前批次每个类别的原型。
        batch_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=features.device) #9 9
        # 用来存储每个类别的像素点数量。
        # 得到batch视觉特征
        count = torch.zeros(self.num_classes, device=features.device) # 9

        # Ensure labels are on the same device as features
        # 将 labels 转换到与 features 相同的设备上，避免跨设备计算错误。
        labels = labels.to(features.device)  #4 640 640

        labels = labels.unsqueeze(1) #4 1 640 640
        # Resize labels to match feature map size
        # 使用 F.interpolate 将标签图像缩放到与特征图相同的尺寸。
        labels_resized = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long().squeeze(1)  #4 640 640

        # Flatten features and resized labels
        # 扁平化特征图和标签，使得每个像素的特征和标签能够一一对应。
        b, c, h, w = features.size() #将特征图 (b,c,h,w) 和标签 (b,h,w) 展平为像素级的一维序列，使得每个像素的特征向量和标签一一对应。
        features = features.permute(0, 2, 3, 1).reshape(-1, c) #  (1638400, 9) 的特征矩阵
        labels_resized = labels_resized.view(-1) #  (1638400) 的标签向量。
        # 对于每个类别（通过 for i in range(self.num_classes) 遍历），计算属于该类别的像素点的特征均值，并更新 batch_prototypes。
        for i in range(self.num_classes):  #循 环遍历所有的类别，self.num_classes 是类别的总数。
            mask = (labels_resized == i)  # 创布尔掩码标记属于类别i的像素
            if mask.sum() > 0:
                # # 所有属于当前类别的特征向量的均值。features[mask] 选取所有属于类别 i 的特征向量，
                # .mean(dim=0) 计算这些向量在每个维度的平均值，结果就是类别 i 的原型向量，存储在 batch_prototypes 数组的第 i 个位置。
                # batch_prototypes[i] = features[mask].mean(dim=0)  #
                # count[i] = mask.sum()  # 录属于类别 i 的特征向量的数量，存储在 count 数组的第 i 个位置。

                # 获取类别i的所有特征向量
                class_features = features[mask]  # (n_pixels, c) 1042319 筛选出所有属于类别 i 的特征向量。
                # 若强行使用GAP，需将特征视为1x1空间维，此处并不合理，仅为示例
                # 此处GAP等价于.mean(dim=0)，因为特征已被展平
                gap_features = class_features.view(1, c, -1, 1)   # 假设空间维度为(n_pixels, 1)  1，9，1042319，1
                gap = torch.nn.AdaptiveAvgPool2d((1, 1))
                batch_prototypes[i] = gap(gap_features).squeeze()  #9，9
                count[i] = mask.sum()

        # Avoid division by zero
        # 最后，返回每个类别的原型（使用 count 来避免除零错误）。
        count = count.clamp(min=1)  # 确保count最小值1，避免除以0
        return batch_prototypes / count.unsqueeze(1)

    def prototype_loss(self, batch_prototypes):
        # Calculate the loss between batch prototypes and global prototypes
        # 当前批次的原型。
        # 计算当前批次的原型与全局原型之间的均方误差损失（MSE Loss）。这种损失度量了当前批次的原型与全局原型之间的差异，目标是使得每个类别的原型在训练过程中逐渐接近其全局原型。
        loss = F.mse_loss(batch_prototypes, self.global_prototypes)
        return loss

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

    def __init__(self, model_clip,sam_model: SAM2Base, r: int, lora_layer=None):
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
        #trunk: sam2.modeling.backbones.hieradet.Hiera
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
        # print(f"transformer_dim (in __init__): {transformer_dim}")

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
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = nn.Conv2d(transformer_dim // 8 * 3, transformer_dim // 8, kernel_size=1)
        self.linear_pred = nn.Conv2d(transformer_dim // 8, 9, kernel_size=1)

        # for param in sam_model.image_encoder.parameters():
        #     param.requires_grad = False

        # Used for deeplab fuse
        self.conv_1x1_feat = nn.Conv2d(256, 48, kernel_size=1)  # Reduce feat channels to 48
        self.conv_3x3_fusion = nn.Sequential(
            nn.Conv2d(48 + 32 + 64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(256, 9, kernel_size=1)  # Assuming 25 is the number of output classes
        self.final_conv_add = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,9,kernel_size=1),
        )

        num_heads = [1, 2, 4]
        self.FRMs = nn.ModuleList([
            FRM(dim=32, reduction=1),
            FRM(dim=64, reduction=1),
            FRM(dim=256, reduction=1)])
        self.FFMs = nn.ModuleList([
            FFM(dim=32, reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
            FFM(dim=64, reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
            FFM(dim=256, reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d)])

        #+++++++++++++++++++language+++++++++++++++++++++language++++++++++++++++++++++++
        self.model_clip = model_clip
        self.class_names =  ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']

        self.prototype_segmentation = PrototypeSegmentation(9, 9)

        # +++++++++++++++++++language+++++++++++++++++++++language++++++++++++++++++++++++


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
        #src：torch.Size([8, 256, 40, 40])
        #feat_s0：torch.Size([8, 32, 160, 160])
        #feat_s1：torch.Size([8, 64, 80, 80])
        # 定义分割任务的特征融合方法，输入包括源特征 `src` 和两个额外特征 `feat_s0` 和 `feat_s1`。
        b, c, _, _ = src.shape # 获取 `src` 特征的形状，`b` 表示批量大小，`c` 表示通道数。
        src_flat = src.view(b, c, -1).transpose(1, 2) # 将 `src` 特征展平为二维（b, c, h*w），然后转置为（b, h*w, c）。
        feat_s0_flat = feat_s0.view(b, feat_s0.size(1), -1).transpose(1, 2) # 同样对 `feat_s0` 进行展平和转置操作，变为（b, h*w, c）。
        feat_s1_flat = feat_s1.view(b, feat_s1.size(1), -1).transpose(1, 2) # 同样对 `feat_s1` 进行展平和转置操作，变为（b, h*w, c）。

        src_transformed = self.mlp_src(src_flat) # 使用 `mlp_src` 对展平后的 `src` 特征进行MLP变换。 #8，1600，32
        src_transformed = src_transformed.transpose(1, 2).view(b, -1, src.shape[2], src.shape[3]) #8，32，40，40
        # 将变换后的特征转置回原始格式，并恢复为二维空间形状。
        src_transformed = F.interpolate(src_transformed, size=[640, 640], mode='bilinear', align_corners=False)# 将特征上采样到固定的分辨率（1024x1024），使用双线性插值。

        feat_s0_transformed = self.mlp_feat_s0(feat_s0_flat)# 对展平后的 `feat_s0` 特征进行MLP变换。 # 8，25600，32
        feat_s0_transformed = feat_s0_transformed.transpose(1, 2).view(b, -1, feat_s0.size(2), feat_s0.size(3))# 将变换后的 `feat_s0` 特征转置回原始格式，并恢复为二维空间形状。 8，32，160，160
        feat_s0_transformed = F.interpolate(feat_s0_transformed, size=[640, 640], mode='bilinear',
                                            align_corners=False)# 对 `feat_s0` 特征上采样到固定分辨率（1024x1024）。

        feat_s1_transformed = self.mlp_feat_s1(feat_s1_flat)# 对展平后的 `feat_s1` 特征进行MLP变换。 #8，6400，32
        feat_s1_transformed = feat_s1_transformed.transpose(1, 2).view(b, -1, feat_s1.size(2), feat_s1.size(3))# 将变换后的 `feat_s1` 特征转置回原始格式，并恢复为二维空间形状。 #
        feat_s1_transformed = F.interpolate(feat_s1_transformed, size=[640, 640], mode='bilinear',
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
        # torch.Size([8, 32, 160, 160]) 对高分辨率特征0 (`high_res_0`) 应用1x1卷积，将通道数从32扩展到256。
        high_res_1_conv = nn.Conv2d(64, 256, kernel_size=1).to(device)(high_res_1)
        # torch.Size([8, 32, 80, 80]) 对高分辨率特征1 (`high_res_1`) 应用1x1卷积，将通道数从64扩展到256。

        feat_up = F.interpolate(feat, size=high_res_1_conv.shape[2:], mode='bilinear', align_corners=False)
        # torch.Size([8, 32, 160, 160]) 将特征1的分辨率上采样到与特征0相同的空间尺寸。
        # 使用双线性插值法进行上采样，确保大小对齐以进行加法融合。
        fpn_merge_1 = feat_up + high_res_1_conv
        # torch.Size([8, 32, 160, 160]) 将处理后的特征0和上采样后的特征1进行逐像素相加，完成初步特征融合。
        fpn_merge_1_up = F.interpolate(fpn_merge_1, size=high_res_0_conv.shape[2:], mode='bilinear', align_corners=False)
        # torch.Size([8, 32, 40, 40]) 将融合后的特征再次上采样到深层特征 `feat` 的空间尺寸。
        final_merge = high_res_0_conv + fpn_merge_1_up
        # torch.Size([8, 32, 40, 40]) 将深层特征 `feat` 和上采样后的融合特征相加，完成特征的最终融合。
        final_output = nn.Conv2d(256, 9, kernel_size=3, padding=1).to(device)(final_merge)
        # 对最终融合的特征应用3x3卷积，进一步处理特征，提取上下文信息。
        # 卷积输出的通道数仍然是256。
        final_output = F.interpolate(final_output, size=(640, 640), mode='bilinear', align_corners=False)
        # 最后将输出特征图上采样到固定的目标分辨率（640x640）。
        # 上采样确保输出的大小与分割任务的要求一致。

        return final_output
        # 返回最终融合后的特征图。
    def fpn_fuseadd(self, feat, high_res_0, high_res_1):
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

        high_res_1_up = F.interpolate(high_res_1_conv, size=high_res_0.shape[2:], mode='bilinear',
                                      align_corners=False)
        # 将特征1的分辨率上采样到与特征0相同的空间尺寸。
        # 使用双线性插值法进行上采样，确保大小对齐以进行加法融合。
        fpn_merge_1 = high_res_0_conv + high_res_1_up
        # 将处理后的特征0和上采样后的特征1进行逐像素相加，完成初步特征融合。
        fpn_merge_1_up = F.interpolate(fpn_merge_1, size=feat.shape[2:], mode='bilinear', align_corners=False)
        # 将融合后的特征再次上采样到深层特征 `feat` 的空间尺寸。
        final_merge = feat + fpn_merge_1_up
        # 将深层特征 `feat` 和上采样后的融合特征相加，完成特征的最终融合。
        final_output = nn.Conv2d(256, 128, kernel_size=3, padding=1).to(device)(final_merge)
        final_output = nn.ReLU(inplace=True)(final_output)  # 对 final_output 张量应用 ReLU 激活函数，所有小于零的值会被设置为零。
        final_output = nn.Conv2d(128, 9, kernel_size=1).to(device)(final_output)
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

    def process_tensor(self,feature, m, b):
        _, fc, fh, fw = feature.size()
        feature = feature.reshape(m, b, fc, fh, fw)
        cam, f = feature[0].squeeze(-1), feature[0].squeeze(-1)
        return cam, f

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature
#====================================================================================================================================


    def forward(self, batched_input,label, multimask_output):
        # 对输入的 batched_input 进行处理：
        # 首先去掉维度为1的批次维度（squeeze(0)），然后堆叠为一个统一的张量
        batched_input = torch.stack([x.squeeze(0) for x in batched_input], dim=0).squeeze(0)  # 处理输入的维度，确保适合后续操作
        # print(batched_input.size())
        m, b, _, h, w = batched_input.shape  # 提取输入张量的形状信息：m 是序列数，b 是批量大小，h 和 w 是图像高度和宽度
        batched_input = batched_input.reshape(2 * b, 3, h, w)  # 将输入调整为适合模型的形状，通常是对批次维度的展开

        # 获取图像的嵌入特征
        image_embedding = self.sam.forward_image(batched_input)  # 使用 sam 的 forward_image 方法提取视觉特征
        # "vision_features": src,  # 提取的视觉特征
        # "vision_pos_enc": pos,  # 对应的位置信息编码
        # "backbone_fpn": features,  # 主干网络生成的特征金字塔
        # 分辨率较高，通常表示较细粒度的特征，应该是 细粒度特征金字塔 (FFP)。
        high_res_0 = image_embedding["backbone_fpn"][0]     # 获取图像编码器FPN（Feature Pyramid Network）中第一层的高分辨率特征图 FO FFP
        # torch.Size([8, 32, 160, 160])  细粒度特征金字塔 (FFP)
        # 分辨率较低，通常表示中等分辨率的特征，应该是 中等分辨率特征金字塔。
        high_res_1 = image_embedding["backbone_fpn"][1]    # 获取图像编码器FPN中第二层的高分辨率特征图， F1 IFP
        # torch.Size([8, 64, 80, 80])
        feat = image_embedding['vision_features']  # 获取图像的深层视觉特征，形状为
        # torch.Size([8, 256, 40, 40])

        # ++++fusion model++++++++fusion model+++++++++fusion model+++++++++++++++
        # x0_cam, x0_f = self.process_tensor(high_res_0, m, b)
        # x0_cam, x0_f = self.FRMs[0](x0_cam, x0_f)
        # x0_fused = self.FFMs[0](x0_cam, x0_f) #torch.Size([4, 32, 160, 160])
        #
        # x1_cam, x1_f = self.process_tensor(high_res_1, m, b)
        # x1_cam, x1_f = self.FRMs[1](x1_cam, x1_f)
        # x1_fused = self.FFMs[1](x1_cam, x1_f)   #torch.Size([4, 64, 80, 80])
        #
        # x_cam, x_f = self.process_tensor(feat, m, b)
        # x_cam, x_f = self.FRMs[2](x_cam, x_f)
        # x_fused = self.FFMs[2](x_cam, x_f) #torch.Size([4, 256, 40, 40])

        output = self.fpn_fuse(feat, high_res_0, high_res_1)
        _, fc, fh, fw = output.size()  # 获取输出的形状信息
        output = output.reshape(m, b, fc, fh, fw)  # 恢复原始批次维度的形状
        output = torch.mean(output, dim=0)  # 对维度 m 进行平均，可能是为了合并不同序列的结果
        # print(output.shape)
        #++++++++language+++++++++++++++language+++++++++++++++++++++language++++++++++++++++++++++++
        if self.training:
            # Tokenize 文本标签
            text_labels = clip.tokenize(self.class_names).to("cuda") #9 77
            # 使用 CLIP 模型编码文本
            feature = self.get_text_feature(text_labels).to(output.dtype) #9 512
            # 计算文本特征的相似度矩阵
            text_sim_matrix = F.cosine_similarity(feature.unsqueeze(0), feature.unsqueeze(1), dim=-1)  # 9x9
            # 获取批次原型和计算原型损失
            batch_prototypes, prototype_loss = self.prototype_segmentation.compute_loss(output, label)  # 9x9
            # 计算视觉特征的相似度矩阵
            # visual_sim_matrix = F.cosine_similarity(batch_prototypes.unsqueeze(0), batch_prototypes.unsqueeze(1),dim=-1)

            # 对文本和视觉相似度矩阵进行softmax归一化
            text_sim_matrix = F.softmax(text_sim_matrix, dim=-1)
            visual_sim_matrix = torch.nn.functional.log_softmax(batch_prototypes, dim=-1)
            # visual_sim_matrix = F.softmax(batch_prototypes, dim=-1)
            # 计算KL损失
            # kl_loss = F.kl_div(visual_sim_matrix, text_sim_matrix, reduction='batchmean')
            kl_loss = F.kl_div(visual_sim_matrix, text_sim_matrix, reduction='none')
            # kl_loss = F.kl_div(visual_sim_matrix, text_sim_matrix, reduction='mean')
            # kl_loss = F.kl_div(visual_sim_matrix, text_sim_matrix, reduction='sum')
        else:
            kl_loss = torch.tensor(0.0).to(output.device)  # 在验证模式下，不计算KL损失
            prototype_loss = torch.tensor(0.0).to(output.device)  # 在验证模式下，不计算KL损失
            # 在评估模式下，我们不进行语言相关操作，只用视觉特征进行分割
        # +++++++++language++++++++++++++language+++++++++++++++++++++language++++++++++++++++++++++++
        # 通过 SAM 解码器生成分割掩码和其他特征
        multi_mask_output = self.sam._forward_sam_heads(
            image_embedding['vision_features'],  # 输入视觉特征
            high_res_features=image_embedding['backbone_fpn'][:2],  # 高分辨率特征，从 FPN 中提取 feat_s0, feat_s1 = high_res_features  # 获取高分辨率特征
            multimask_output=multimask_output  # 控制是否生成多掩码输出
        )

        # 提取 multi_mask_output 的第二个元素作为目标输出
        m_output = multi_mask_output[1]  # 通常是分割掩码或其他感兴趣的特征
        _, fc, fh, fw = output.size()  # 获取输出的形状信息
        m_output = m_output.reshape(m, b, fc, fh, fw)  # 恢复原始批次维度的形状
        m_output = torch.mean(m_output, dim=0)  # 对维度 m 进行平均，可能是为了合并不同序列的结果
        # print(m_output.shape)
        # print(kl_loss)
        # print(prototype_loss)

        return m_output, output, prototype_loss, kl_loss# 返回最终的模型输出
