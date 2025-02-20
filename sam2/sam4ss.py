import torch.nn as nn
from .modeling.sam2_base import SAM2Base

class sam4ss(nn.Module):
    #sam_model: 期望传入一个 SAM2Base 类型的模型实例。这意味着 sam4ss 类是基于已有的 SAM2 模型进行封装或扩展。
    def __init__(self, sam_model: SAM2Base):
        super(sam4ss, self).__init__()

        self.sam = sam_model

    def forward(self, batched_input, multimask_output=True):

            image_embedding = self.sam.forward_image(batched_input)#调用 SAM2 模型的 forward_image 方法，处理输入图像，生成特征嵌入（image_embedding）
            #返回的 image_embedding 预期是一个字典，包含关键的特征图，如 'vision_features' 和 'backbone_fpn'
            print('image_embedding',len(image_embedding) )

            # "vision_features": src,
            # "vision_pos_enc": pos,
            # "backbone_fpn": features,
            #
            # print('image_embedding',image_embedding )
            output  = self.sam._forward_sam_heads(image_embedding['vision_features'], high_res_features=image_embedding['backbone_fpn'][:2], multimask_output=multimask_output)

            return  output  #调用 SAM2 模型的内部方法 _forward_sam_heads，传入处理后的特征图。

            # image_embedding['vision_features']: 视觉特征图，作为解码器的输入。
            # high_res_features = image_embedding['backbone_fpn'][:2]: 高分辨率特征图，可能用于细节恢复或多尺度融合。
            # multimask_output = multimask_output: 控制是否输出多个掩码。