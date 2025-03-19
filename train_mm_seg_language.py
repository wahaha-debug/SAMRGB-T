import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.datasets import *
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from tools.val_mm import evaluate

from sam2.build_sam import build_sam2
# from sam2.sam_lora_image_encoder_lora import LoRA_Sam
# from sam2.sam_lora_image_encoder_lora_seghead_CMNEXT_language_gai import LoRA_Sam
from sam2.sam_lora_fusion_feat_language_2seghead import LoRA_Sam
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import random
import clip

def main(cfg, gpu, save_dir):
    start = time.time()  # 记录训练开始的时间
    best_mIoU = 0.0  # 初始化最佳的Mean Intersection over Union (mIoU)为0
    best_epoch = 0  # 初始化最佳训练轮次为0
    num_workers = 8  # 设置加载数据时的工作进程数为8
    device = torch.device(cfg['DEVICE'])  # 根据配置文件中的DEVICE参数选择训练设备（CPU或GPU）
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']  # 从配置文件中加载训练和验证的相关参数
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']  # 从配置文件中加载数据集和模型的配置
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']  # 加载损失函数、优化器和调度器配置
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']  # 获取训练的总轮次和学习率
    resume_path = cfg['MODEL']['RESUME']  # 获取预训练模型的路径
    # gpus = int(os.environ['WORLD_SIZE'])  # 获取GPU数量，通常用于分布式训练
    gpus = int(os.environ.get('WORLD_SIZE', 1))  # 如果没有设置WORLD_SIZE，则默认使用1个GPU

    # 数据预处理：训练时的数据增强，忽略某些标签
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    # 数据预处理：验证时的图像增强
    valtransform = get_train_augmentation(eval_cfg['IMAGE_SIZE'])

    # 初始化训练数据集，加载数据并应用训练时的增强方式
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    # 初始化验证数据集，加载数据并应用验证时的增强方式
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = trainset.CLASSES  # 获取训练数据集的类名（即类别标签）

    # 初始化模型，配置模型的骨干网络和输出类别数
    # model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    model_clip, _ = clip.load("ViT-B/32")

    # checkpoint = "../checkpoints/sam2_hiera_base_plus.pt"
    checkpoint = model_cfg['PRETRAINED']
    model_cf = model_cfg['MODEL_CONFIG']

    # 加载基础模型：
    sam2 = build_sam2(model_cf, checkpoint)

    model = LoRA_Sam(model_clip,sam2, r=train_cfg['RANK'])
    # print(model)
    for param in model.sam.obj_ptr_proj.parameters():  # 冻结 `obj_ptr_proj` 模块的所有参数。
        param.requires_grad = False  # 这意味着这些参数在训练过程中不会被更新，常用于参数已经预训练完成或固定不变的情况。
    for param in model.sam.sam_mask_decoder.iou_prediction_head.parameters():  # 冻结 `iou_prediction_head` 模块的所有参数
        param.requires_grad = False  # `iou_prediction_head` 可能是分割任务中用于预测目标的IoU得分的部分，这里选择不训练该部分。
    for param in model.sam.sam_mask_decoder.pred_obj_score_head.parameters():  # 冻结 `pred_obj_score_head` 模块的所有参数。
        param.requires_grad = False  # `pred_obj_score_head` 可能是预测目标分数（如目标置信度）的模块，这里也不参与训练。
    for param in model.sam.memory_attention.parameters():  # 解冻 `memory_attention` 模块的参数。
        param.requires_grad = True  # 允许该模块在训练过程中更新权重，通常表示希望微调这一部分以适应新任务。
    for param in model.sam.memory_encoder.parameters():  # 解冻 `memory_encoder` 模块的参数。
        param.requires_grad = True  # 允许该部分的参数在训练中更新，用于学习新的任务特征。
    for param in model.sam.sam_prompt_encoder.parameters():  # 冻结 `sam_prompt_encoder` 模块的参数。
        param.requires_grad = False  # `sam_prompt_encoder` 可能是模型中用于处理输入提示（prompt）的模块。

    for param in model.model_clip.parameters():
        param.requires_grad = False

    def count_parameters(model):
        total_parameters = 0
        trainable_parameters = 0
        for name, param in model.named_parameters():
            total_parameters += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        return total_parameters, trainable_parameters

    total_parameters, trainable_parameters = count_parameters(model)
    print('Total number of parameters: %d' % total_parameters)
    print('Total number of trainable parameters: %d' % trainable_parameters)
    print('Percentage of trainable parameters: %.2f%%' % (trainable_parameters / total_parameters * 100))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    resume_checkpoint = None  # 初始化恢复检查点为None
    if os.path.isfile(resume_path):  # 如果指定的恢复路径存在
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))  # 加载预训练模型
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])  # 加载模型权重
        # print(msg)  # 打印加载的模型信息
        logger.info(msg)  # 将加载的信息记录到日志
    # else:
        # model.init_pretrained(model_cfg['PRETRAINED'])  # 如果没有恢复检查点，初始化预训练模型

    model = model.to(device)  # 将模型移到指定的设备（GPU/CPU）上

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus  # 计算每个 epoch 的迭代次数
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)  # 获取损失函数（如交叉熵、Dice损失等），使用指定的名称和忽略标签
    start_epoch = 0  # 初始化训练的起始 epoch
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr,
                              optim_cfg['WEIGHT_DECAY'])  # 获取优化器（如 SGD、Adam 等），并根据配置文件设置学习率和权重衰减
    # 获取学习率调度器（如多项式学习率衰减），并计算训练过程中的学习率调整
    scheduler = get_scheduler(
        sched_cfg['NAME'], optimizer, int((epochs + 1) * iters_per_epoch),
        sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO']
    )

    if train_cfg['DDP']:  # 如果启用了分布式训练（DDP：Distributed Data Parallel）
        # 使用 DistributedSampler 来分配训练数据给每个进程
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None  # 验证集不需要分布式采样
        # 使用 DDP 包装模型，指定 GPU 设备
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)  # 如果没有启用 DDP，则使用普通的随机采样
        sampler_val = None  # 验证集不使用采样器（可以根据需要修改）

    # 如果有恢复检查点（resume_checkpoint），加载先前的训练状态
    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1  # 恢复训练的 epoch
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])  # 恢复优化器的状态
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])  # 恢复学习率调度器的状态
        loss = resume_checkpoint['loss']  # 恢复损失值（用于监控训练进展）
        best_mIoU = resume_checkpoint['best_miou']  # 恢复最好的 mIoU 值

    # 创建训练数据加载器，指定批量大小、工作线程数等配置
    trainloader = DataLoader(
        trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers,
        drop_last=True, pin_memory=False, sampler=sampler
    )
    # 创建验证数据加载器
    valloader = DataLoader(
        valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers,
        pin_memory=False, sampler=sampler_val
    )

    scaler = GradScaler(enabled=train_cfg['AMP'])  # 如果启用了自动混合精度（AMP），则使用 GradScaler 来动态缩放梯度
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        # 如果是分布式训练的主进程（rank 0），或者没有启用 DDP
        writer = SummaryWriter(str(save_dir))  # 创建 TensorBoard 写入器，保存训练日志
        # logger.info('================== model complexity =====================')  # 记录模型的复杂度（如FLOPs）
        # cal_flops(model, dataset_cfg['MODALS'], logger)  # 计算模型的计算量（FLOPs），根据使用的模态
        logger.info('================== model structure =====================')  # 记录模型的结构
        logger.info(model)  # 打印模型的详细结构
        logger.info('================== training config =====================')  # 记录训练配置
        logger.info(cfg)  # 打印整个训练配置（包括数据集、优化器等参数）

    for epoch in range(start_epoch, epochs):
        model.train()  # 将模型设置为训练模式，这会启用如 Dropout 和 BatchNorm 等层的训练行为

        if train_cfg['DDP']:
            sampler.set_epoch(epoch)  # 如果启用了分布式数据并行（DDP），则在每个 epoch 设置随机数种子，以确保数据打乱顺序一致

        train_loss = 0.0  # 初始化当前 epoch 的累计训练损失
        lr = scheduler.get_lr()  # 获取当前学习率，可能是一个列表，因为可能有多个参数组
        lr = sum(lr) / len(lr)  # 计算所有参数组的平均学习率（如果有多个参数组）

        # 使用 tqdm 库创建进度条，显示当前的 epoch、迭代次数、学习率和损失
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch + 1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        # 遍历训练数据加载器，获取每个批次的样本数据（sample）和标签（lbl）
        for iter, (sample, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)  # 清除优化器中的梯度，避免累积旧的梯度

            # 将输入样本和标签移动到指定设备（通常是 GPU）
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)
            # print(lbl.shape)
            # print(sample[1].shape)
            # print(sample[0].shape) #torch.Size([4, 3, 640, 640])
            # 使用自动混合精度（AMP）进行前向传播和损失计算
            with autocast(enabled=train_cfg['AMP']):
                m_output ,output, prototype_loss, kl_loss = model(sample, lbl,multimask_output = True)  # SAMed
                # feat = torch.mean(m_feat[0],dim=0).unsqueeze(0)

                loss_sam = loss_fn(m_output, lbl)  # 计算损失函数
                loss_aux = loss_fn(output, lbl)  # 计算损失函数

                # print(loss_cfg['WEIGHT_aux'])
                # print(loss_cfg['WEIGHT_prototype'])
                # print(loss_cfg['WEIGHT_language'])
                loss_aux = loss_cfg['WEIGHT_aux'] * loss_aux
                prototype_loss = loss_cfg['WEIGHT_prototype'] * prototype_loss.mean()
                kl_loss = loss_cfg['WEIGHT_language'] * kl_loss.mean()
                loss = loss_sam + loss_aux + prototype_loss + kl_loss
                # print(loss)
                loss = loss.mean()  # 或者 loss.sum()

            # 使用 GradScaler 缩放损失，以防止在混合精度训练时出现梯度下溢
            scaler.scale(loss).backward()  # 反向传播，计算梯度
            scaler.step(optimizer)  # 使用缩放后的梯度更新优化器的参数
            scaler.update()  # 更新 GradScaler 的缩放因子

            scheduler.step()  # 更新学习率调度器，以根据预设的策略调整学习率
            torch.cuda.synchronize()  # 等待所有 CUDA 操作完成，确保同步

            # 获取当前的学习率，并计算其平均值
            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8  # 限制学习率的最小值，以防过小

            #=================================================
            loss_aux += loss_aux.item()  # 累加当前批次的损失
            loss_sam += loss_sam.item()  # 累加当前批次的损失
            kl_loss += kl_loss.item()  # 累加当前批次的损失
            prototype_loss += prototype_loss.item()  # 累加当前批次的损失
            train_loss += loss.item()  # 累加当前批次的损失
            #=================================================


            # 更新进度条的描述信息，显示当前 epoch、迭代次数、学习率和当前的平均损失
            pbar.set_description(f"Epoch: [{epoch + 1}/{epochs}] Iter: [{iter + 1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter + 1):.8f}")

        # 平均化训练损失
        train_loss /= iter + 1
        loss_aux /= iter + 1
        loss_sam /= iter + 1
        kl_loss /= iter + 1
        prototype_loss /= iter + 1
        # print("#=================================================#=================================================oss")
        # print("#=================================================#=================================================oss")
        # print("#=================================================#=================================================oss")
        # print("#=================================================#=================================================oss")
        # print("kl_loss", kl_loss)
        # print("prototype_loss", prototype_loss)
        # # print("loss_aux", loss_aux)
        # # print("loss_sam", loss_sam)
        # print("train_loss", train_loss)

        # 如果启用了分布式训练（DDP）且当前是主进程（rank == 0），或者没有启用DDP，则记录训练损失
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/loss_aux', loss_aux, epoch)
            writer.add_scalar('train/loss_sam', loss_sam, epoch)
            writer.add_scalar('train/kl_loss', kl_loss, epoch)
            writer.add_scalar('train/prototype_loss', prototype_loss, epoch)

        # 清理CUDA缓存
        torch.cuda.empty_cache()

        # 每隔指定的评估周期（EVAL_INTERVAL）进行验证，或者如果已经到达总的epoch数，则进行验证
        if ((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch + 1) > train_cfg['EVAL_START']) or (epoch + 1) == epochs:
            # 如果启用了分布式训练（DDP）且当前是主进程（rank == 0），或者没有启用DDP，则进行验证
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                # 计算验证集上的准确率、平均准确率、IoU等指标
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                # 记录验证集的mIoU值
                writer.add_scalar('val/mIoU', miou, epoch)

                # 如果当前mIoU比历史最佳mIoU更高，则保存当前模型
                if miou > best_mIoU:
                    # 删除之前保存的最佳模型文件（如果存在）
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)

                    # 更新最佳mIoU和最佳epoch
                    best_mIoU = miou
                    best_epoch = epoch + 1

                    # 保存当前最佳模型权重文件
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"

                    # 保存模型状态字典
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)

                    # 保存包含更多信息的checkpoint
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg[
                                    'DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)

                    # 输出当前模型在验证集上的各项指标
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))

                # 输出当前epoch的mIoU和最佳mIoU
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

        # 在所有训练完成后关闭writer对象以便保存Tensorboard日志
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.close()

        # 关闭进度条
        pbar.close()

        # 输出训练总时长
        end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/mfnet_rgbt.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    BATCH_SIZE = cfg['TRAIN']['BATCH_SIZE']
    RANK = cfg['TRAIN']['RANK']
    LR = cfg['OPTIMIZER']['LR']
    exp_name = '_'.join(
        [cfg['DATASET']['NAME'],model, modals, str(BATCH_SIZE), str(RANK), str(LR)])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()