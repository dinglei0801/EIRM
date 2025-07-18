import argparse
import datetime
import os.path as osp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from sklearn.metrics import f1_score, recall_score
import numpy as np
from torch.utils.data import DataLoader

from datasets.mini_imagenet_cub import MiniImageNet
from torch.cuda.amp import autocast, GradScaler  # 添加这一行
from torch.amp import autocast, GradScaler
from datasets.agriculture import AgricultureImageNet, SSLAgricultureImageNet
from datasets.cub import CubImageNet, SSLCubImageNet
from datasets.chinese_medicine import CMedicineImageNet, SSLCMedicineImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.distill import DistillKL, HintLoss, ContrastiveDistillKL, FocalDistillKL
from models.resnet import resnet12, Decoder
from utils import set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, seed_torch, \
    compute_confidence_interval
#Trial 11 finished with value: 0.6932666817754501 and parameters: {'lr': 0.003444642074134048, 'wd': 0.0007451617972708867, 'temperature': 5, 'kd_coef': 0.30000701726508594, 'ssl_coef': 0.3755153058344556, 'min_temperature': 2.1843488527517314}. Best is trial 11 with value: 0.6932666817754501.
class EnhancedIRMLoss(nn.Module):
    def __init__(self, penalty_weight=1.0, num_envs=7):
        super(EnhancedIRMLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.num_envs = num_envs
        self.env_weights = nn.Parameter(torch.ones(num_envs))
        
    def forward(self, logits, labels, environments):
        loss = 0
        penalty = 0
        batch_size = len(labels)
        env_weights = F.softmax(self.env_weights, dim=0)
        
        for env_idx in range(self.num_envs):
            env_mask = (environments == env_idx)
            if not env_mask.any():
                continue
                
            env_logits = logits[env_mask]
            env_labels = labels[env_mask]
            
            # 计算环境特定损失
            env_loss = F.cross_entropy(env_logits, env_labels, reduction='none')
            
            # 梯度归一化
            scale = torch.tensor(1.).cuda().requires_grad_()
            env_loss_scaled = (env_loss * scale).mean()
            grad = torch.autograd.grad(env_loss_scaled, [scale], create_graph=True)[0]
            grad_norm = torch.norm(grad)
            
            # 添加权重和归一化的梯度惩罚
            penalty += env_weights[env_idx] * (grad_norm - 1).pow(2)
            loss += env_weights[env_idx] * env_loss.mean()
            
        # 添加一个正则化项来平衡环境权重
        entropy_reg = -(env_weights * torch.log(env_weights + 1e-8)).sum()
        
        return loss + self.penalty_weight * penalty - 0.1 * entropy_reg

class DynamicIRMWeightScheduler:
    def __init__(self, 
                 initial_weight=0.2,
                 min_weight=0.05,
                 max_weight=0.3,
                 warmup_epochs=20,
                 performance_threshold=0.6):
        """
        动态IRM权重调度器
        
        Args:
            initial_weight (float): 初始IRM权重
            min_weight (float): 最小IRM权重
            max_weight (float): 最大IRM权重
            warmup_epochs (int): 预热轮次
            performance_threshold (float): 性能差距阈值
        """
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs
        self.performance_threshold = performance_threshold
        self.prev_loss = None
        self.loss_ma = None
        self.ma_beta = 0.9  # 移动平均系数
        
    def calculate_weight(self, epoch, max_epoch, teacher_acc, student_acc, irm_loss):
        """
        计算当前epoch的IRM权重
        """
        # 1. 基于训练进度的调整
        progress = epoch / max_epoch
        if epoch < self.warmup_epochs:
            # 预热阶段：逐渐增加权重
            progress_factor = epoch / self.warmup_epochs
        else:
            # 正常训练阶段：余弦退火
            progress_factor = 0.5 * (1 + math.cos(math.pi * (progress - self.warmup_epochs/max_epoch)/(1 - self.warmup_epochs/max_epoch)))
            
        # 2. 基于教师-学生性能差距的调整
        performance_gap = teacher_acc - student_acc
        if performance_gap > self.performance_threshold:
            # 差距大时减小IRM权重，让模型更专注于模仿教师
            gap_factor = 0.7
        else:
            # 差距小时增加IRM权重，加强鲁棒性
            gap_factor = 1.0 + (self.performance_threshold - performance_gap)
            
        # 3. 基于IRM损失变化趋势的调整
        if self.loss_ma is None:
            self.loss_ma = irm_loss
        else:
            self.loss_ma = self.ma_beta * self.loss_ma + (1 - self.ma_beta) * irm_loss
            
        if self.prev_loss is not None:
            loss_change = (irm_loss - self.prev_loss) / self.prev_loss
            # 损失下降时逐渐增加权重，上升时快速减小权重
            loss_factor = 1.0 - math.tanh(max(0, loss_change) * 2)
        else:
            loss_factor = 1.0
            
        self.prev_loss = irm_loss
        
        # 4. 计算最终权重
        weight = self.initial_weight * progress_factor * gap_factor * loss_factor
        
        # 5. 确保权重在合理范围内
        weight = max(self.min_weight, min(self.max_weight, weight))
        
        return weight

def generate_environments(data, args):
    """生成更丰富的不变环境"""
    envs = []
    # 原始数据
    envs.append(data)
    
    # 基础变换环境
    envs.append(torch.flip(data, dims=[3]))  # 水平翻转
    envs.append(torch.flip(data, dims=[2]))  # 垂直翻转
    
    # 颜色变换环境
    brightness = torch.clamp(data * (0.8 + 0.4 * torch.rand_like(data)), 0, 1)
    contrast = torch.clamp((data - 0.5) * (0.8 + 0.4 * torch.rand_like(data)) + 0.5, 0, 1)
    envs.append(brightness)
    envs.append(contrast)
    
    # 噪声环境
    noise = torch.clamp(data + 0.1 * torch.randn_like(data), 0, 1)
    envs.append(noise)
    
    # 局部遮挡环境
    mask = torch.ones_like(data)
    mask_size = args.size // 4
    x = random.randint(0, args.size - mask_size)
    y = random.randint(0, args.size - mask_size)
    mask[:, :, x:x+mask_size, y:y+mask_size] = 0
    masked = data * mask
    envs.append(masked)
    
    return torch.cat(envs, dim=0)

class AdaptiveWeightLoss(nn.Module):
    def __init__(self, num_losses):
        super(AdaptiveWeightLoss, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        weights = F.softplus(self.weights)
        normalized_weights = weights / (weights.sum() + 1e-8)
        return sum(w * l for w, l in zip(normalized_weights, losses))

class DualResNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(DualResNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.relation_module = RelationModule(input_size, hidden_size)
        self.adaptive_weight = AdaptiveWeightLoss(3)  # 3 losses: cls, ssl, recon

    def forward(self, x):
        encoder_output = self.encoder(x)

        if isinstance(encoder_output, tuple):
            features = encoder_output[0]  # 使用第一个返回值作为特征
        else:
            features = encoder_output

        reconstructed = self.decoder(features)
        # return features, self.relation_module(features), reconstructed
        return features, features, reconstructed  # 返回 features 两次，模拟 proto 和 enhanced features


def generate_dual_samples(data_shot):
    # 生成对偶样本的逻辑
    x_90 = data_shot.transpose(2, 3).flip(2)
    x_180 = data_shot.flip(2).flip(3)
    x_270 = data_shot.flip(2).transpose(2, 3)
    return torch.cat((data_shot, x_90, x_180, x_270), 0)  # 返回原始样本和对偶样本


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# 问题出在加载教师模型权重时，模型的状态字典中的键与加载的权重文件中的健不匹配。具体来说，加载的权重前面有一个encoder的前缀，而目标模型的键没有这个前缀
def load_weights(model, weights_path, prefix=''):
    state_dict = torch.load(weights_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            if new_k in model.state_dict():
                new_state_dict[new_k] = v

    missing_keys = set(model.state_dict().keys()) - set(new_state_dict.keys())
    unexpected_keys = set(new_state_dict.keys()) - set(model.state_dict().keys())

    if missing_keys:
        print(f"Warning: Missing keys in loaded weights: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in loaded weights: {unexpected_keys}")

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights into model from {weights_path}")


def get_dataset(args):
    if args.dataset == 'mini':
        trainset = MiniImageNet('train', args.size)
        valset = MiniImageNet('test', args.size)
        print("=> MiniImageNet...")
    elif args.dataset == 'insect':
        trainset = SSLInsectImageNet('train', args)
        valset = InsectImageNet('test', args.size)
        print("=> Insection...")
    elif args.dataset == 'chinese_medicine':
        trainset = SSLCMedicineImageNet('train', args)
        valset = CMedicineImageNet('test', args.size)
        print("=> Chinere Medicine...")
    elif args.dataset == 'agriculture':
        trainset = SSLAgricultureImageNet('train', args)
        valset = AgricultureImageNet('test', args.size)
        print("=> Agriculture...")
    elif args.dataset == 'cub':
        trainset = SSLAgricultureImageNet('train', args)
        valset = AgricultureImageNet('test', args.size)
        print("=> Cub...")
    elif args.dataset == 'tiered':
        trainset = TieredImageNet('train', args.size)
        valset = TieredImageNet('test', args.size)
        print("=> TieredImageNet...")
    elif args.dataset == 'cifarfs':
        trainset = CIFAR_FS('train', args.size)
        valset = CIFAR_FS('test', args.size)
        print("=> CIFAR FS...")
    else:
        print("Invalid dataset...")
        exit()

    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                      args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader

def calculate_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == labels).float().mean().item()
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    return accuracy, f1, recall
def training(args):
    ensure_path(args.save_path)
    # 获取数据
    train_loader, val_loader = get_dataset(args)

    # 初始化模型
    if args.model == 'convnet':
        encoder = Convnet().cuda()
        print("=> Convnet architecture...")
    else:
        if args.dataset in ['mini', 'tiered', 'insect', 'agriculture', 'chinese_medicine', 'cub']:
            encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
        else:
            encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
        print("=> Resnet architecture...")
    # 加载教师模型权重
    decoder = Decoder(640, 3, (args.size, args.size)).cuda()
    teacher = DualResNet(encoder, decoder).cuda()
    teacher.load_state_dict(torch.load(osp.join(args.stage2_path, 'max-acc.pth')))
    print("=> Teacher loaded with stage 2 knowledge...")
    teacher.eval()
    print("=> Teacher loaded with pretrain knowledge...")
    if args.kd_mode != 0:
        # produce a student model with the same structure as teacher model without knowldege
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
        if args.stage1_path:
            load_weights(model, osp.join(args.stage1_path, 'max-acc.pth'))
            print("=> Student loaded with pretrain knowledge...")

    if args.kd_mode == 0:
        # intilialize student with same knowledge as teacher
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
        print("=> Student obtain teacher's knowledge...")

    if args.kd_type == 'kd':
        criterion_kd = DistillKL(args.temperature).cuda()
    elif args.kd_type == 'focal':
        criterion_kd = FocalDistillKL(args.temperature).cuda()
    elif args.kd_type == 'contrastive':
        criterion_kd = ContrastiveDistillKL(args.temperature).cuda()
    elif args.kd_type == 'dual':
        criterion_kd = DualContrastiveDistillKL(args.temperature).cuda()
    else:
        criterion_kd = HintLoss(args.temperature).cuda()

    optimizer = torch.optim.Adam([{'params': model.parameters()},{'params': teacher.adaptive_weight.parameters()}], lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['val_f1'] = []
    trlog['val_recall'] = []
    timer = Timer()
    best_epoch = 0
    cmi = [0.0, 0.0]
    initial_temperature = args.temperature
    min_temperature = 1.0
    # 初始化IRM权重调度器
    irm_scheduler = DynamicIRMWeightScheduler(
        initial_weight=args.irm_coef,
        min_weight=args.irm_min_weight,
        max_weight=args.irm_max_weight,
        warmup_epochs=args.irm_warmup,
        performance_threshold=0.6
    )
    for epoch in range(1, args.max_epoch + 1):
        #tl, ta = train(args, teacher, model, train_loader, optimizer, criterion_kd)
        #vl, va, ci_low, ci_high = validate(args, model, val_loader)
        # 在这里添加学习率调整的代码
        # 更细致的学习率调整策略
        if epoch > 107:
            base_lr = args.lr*0.005
            if epoch <= 130:
                new_lr = base_lr * 0.3  # 先降到0.3
            elif epoch <= 160:
                new_lr = base_lr * 0.1  # 再降到0.1
            else:
                new_lr = base_lr * 0.05  # 最后降到0.05
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
                
        kd_coef = args.kd_coef * (1 - epoch / args.max_epoch) 
        # 动态计算IRM权重
        #current_irm_coef = irm_scheduler.calculate_weight(
            #epoch=epoch,
            #max_epoch=args.max_epoch,
            #teacher_acc=teacher_acc if 'teacher_acc' in locals() else 0.9,
            #student_acc=student_acc if 'student_acc' in locals() else 0.2,
            #irm_loss=trlog['train_loss'][-1] if trlog['train_loss'] else 14.0
        #)
        
        # 使用动态权重进行训练
        tl, ta, teacher_acc, student_acc = train(args, teacher, model, train_loader, optimizer, 
                                               criterion_kd, kd_coef)
        
        # 打印当前的IRM权重
        #print(f"Current IRM weight: {current_irm_coef:.4f}")

        new_temperature = adjust_temperature(epoch, args.max_epoch, initial_temperature, min_temperature, teacher_acc,student_acc)
        criterion_kd.set_temperature(new_temperature)

        vl, va, vf, vr, acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std = validate(args, model, val_loader)
        lr_scheduler.step()
        #lr_scheduler.step()

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
            best_epoch = epoch
            cmi = [acc_mean, acc_std]

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        trlog['val_f1'].append(vf)
        trlog['val_recall'].append(vr)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')
        ot, ots = timer.measure()
        tt, _ = timer.measure(epoch / args.max_epoch)
        current_temperature = criterion_kd.get_temperature()
        print(f"Epoch {epoch}, Current Temperature: {current_temperature}")
        print(f"Epoch {epoch}, Teacher Acc: {teacher_acc:.4f}, Student Acc: {student_acc:.4f}")
        print(
            'Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f}±{:.4f} - F1={:.4f}±{:.4f} - Recall={:.4f}±{:.4f} - max acc={:.4f} - ETA:{}/{}'.format(
                epoch, args.max_epoch, tl, ta, vl, acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std, trlog['max_acc'], ots, timer.tts(tt - ot)))

        if epoch == args.max_epoch:
            print("Best Epoch is {} with acc={:.2f}±{:.2f}%...".format(best_epoch, cmi[0], cmi[1]))
            print("---------------------------------------------------")


def ssl_loss(args, model, data_shot):
    # s1 s2 q1 q2 q1 q2
    x_90 = data_shot.transpose(2, 3).flip(2)
    x_180 = data_shot.flip(2).flip(3)
    x_270 = data_shot.flip(2).transpose(2, 3)
    data_query = torch.cat((x_90, x_180, x_270), 0)

    proto, _, _ = model(data_shot)
    proto = proto.reshape(1, args.shot * args.train_way, -1).mean(dim=0)

    label = torch.arange(args.train_way * args.shot).repeat(args.pre_query)
    label = label.type(torch.cuda.LongTensor)
    query, _, _ = model(data_query)

    logits = euclidean_metric(query, proto)
    loss = F.cross_entropy(logits, label)

    return loss
def contrastive_loss(features, labels, temperature=0.1):
    """
    计算对比损失
    :param features: 样本特征，形状为 [batch_size, feature_dim]
    :param labels: 样本标签，形状为 [batch_size]
    :param temperature: 温度参数
    :return: 对比损失
    """
    features = F.normalize(features, dim=1)  # 归一化特征
    similarity_matrix = torch.matmul(features, features.T)  # 计算相似度矩阵
    similarity_matrix = similarity_matrix / temperature  # 缩放相似度

    # 构建正样本和负样本掩码
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # 正样本掩码
    neg_mask = 1 - mask  # 负样本掩码
    neg_mask.fill_diagonal_(0)  # 排除自身

    # 计算对比损失
    exp_sim = torch.exp(similarity_matrix)
    pos_loss = -torch.log(exp_sim * mask / (exp_sim * mask).sum(dim=1, keepdim=True)).sum(dim=1).mean()
    neg_loss = -torch.log(exp_sim * neg_mask / (exp_sim * neg_mask).sum(dim=1, keepdim=True)).sum(dim=1).mean()
    contrastive_loss = pos_loss + neg_loss

    return contrastive_loss

def train(args, teacher, model, train_loader, optimizer, criterion_kd,kd_coef):
    teacher.eval()
    model.train()
    tl = Averager()
    ta = Averager()
    teacher_acc_avg = Averager()
    student_acc_avg = Averager()
    scaler = GradScaler()
    NUM_ENVS = 7  # 环境数量
    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]
        # 记录原始查询集大小
        original_query_size = data_query.size(0)
        
        with torch.no_grad():
            tproto, _, _ = teacher(data_shot)
            tproto = tproto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            tquery, _, _ = teacher(data_query)
            tlogits = euclidean_metric(tquery, tproto)

        with autocast(device_type='cuda', dtype=torch.float16):
            # 创建IRM损失实例
            irm_criterion = EnhancedIRMLoss(penalty_weight=args.irm_penalty, num_envs=NUM_ENVS).cuda()
            proto, _, _ = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            # 生成原始标签
            label = torch.arange(args.train_way).repeat(args.train_query)
            label = label.type(torch.cuda.LongTensor)

            # 生成环境数据
            data_query_envs = generate_environments(data_query, args)
            # 为每个环境复制标签
            label_envs = label.repeat(NUM_ENVS)
            
            # 对环境数据进行前向传播
            query_envs, _, _ = model(data_query_envs)
            logits_envs = euclidean_metric(query_envs, proto)
            
            # 生成环境标识
            environments = torch.arange(NUM_ENVS, device=data_query.device).repeat_interleave(original_query_size)

            # 计算IRM损失
            irm_loss = irm_criterion(logits_envs, label_envs, environments)
            
            # 使用原始数据计算其他损失
            query, _, _ = model(data_query)
            logits = euclidean_metric(query, proto)
            
            teacher_acc = count_acc(tlogits, label)
            teacher_acc_avg.add(teacher_acc)
            student_acc = count_acc(logits, label)
            student_acc_avg.add(student_acc)
            
            clsloss = F.cross_entropy(logits, label)
            kdloss = criterion_kd(logits, tlogits)
            loss_ss = ssl_loss(args, model, data_shot)

            # 组合所有损失
            losses = [clsloss, kdloss, loss_ss]
            base_loss = teacher.adaptive_weight(losses)
            
            # 动态调整IRM权重
            irm_weight = args.irm_coef
            loss = base_loss + irm_weight * irm_loss

            # 监控损失权重（每100个batch打印一次）
            if i % 100 == 0:
                weights = F.softplus(teacher.adaptive_weight.weights)
                normalized_weights = weights / (weights.sum() + 1e-8)
                print("\nCurrent adaptive weights:")
                print(f"Classification: {normalized_weights[0]:.4f}")
                print(f"Knowledge Distillation: {normalized_weights[1]:.4f}")
                print(f"Self-supervised: {normalized_weights[2]:.4f}")
                print(f"IRM weight: {irm_weight:.4f}")
                
                print("\nCurrent loss values:")
                print(f"Classification: {clsloss:.4f}")
                print(f"Knowledge Distillation: {kdloss:.4f}")
                print(f"Self-supervised: {loss_ss:.4f}")
                print(f"IRM: {irm_loss:.4f}")
                print(f"Total loss: {loss:.4f}")

        acc = count_acc(logits, label)
        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return tl.item(), ta.item(), teacher_acc_avg.item(), student_acc_avg.item()

def validate(args, model, val_loader):
    model.eval()
    vl = Averager()
    va = Averager()
    vf = Averager()
    vr = Averager()
    acc_list = []
    f1_list = []
    recall_list = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            with autocast(device_type='cuda', dtype=torch.float16):
                proto, _, _ = model(data_shot)
                proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

                label = torch.arange(args.test_way).repeat(args.test_query)
                label = label.type(torch.cuda.LongTensor)
                query, _, _ = model(data_query)
                logits = euclidean_metric(query, proto)
                loss = F.cross_entropy(logits, label)
                acc,f1, recall = calculate_metrics(logits, label)

            vl.add(loss.item())
            va.add(acc)
            vf.add(f1)
            vr.add(recall)
            acc_list.append(acc * 100)
            f1_list.append(f1 * 100)
            recall_list.append(recall * 100)

            proto = None;
            logits = None;
            loss = None
    acc_mean, acc_std = compute_confidence_interval(acc_list)
    f1_mean, f1_std = compute_confidence_interval(f1_list)
    recall_mean, recall_std = compute_confidence_interval(recall_list)
    return vl.item(), va.item(), vf.item(), vr.item(), acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std


#def adjust_temperature(epoch, max_epoch, initial_T, min_T, teacher_acc, student_acc):
    # 基于epoch的线性降低
    #base_T = max(min_T, initial_T * (1 - epoch / max_epoch))
    # 根据teacher和student准确率差异调整
    #acc_diff = max(0, teacher_acc - student_acc)
    #adjust_factor = 1 + 2*acc_diff
    #return base_T * adjust_factor
# 1. 改进温度调整策略
def adjust_temperature(epoch, max_epoch, initial_T, min_T, teacher_acc, student_acc):
    # 使用余弦退火来调整基础温度
    progress = epoch / max_epoch
    cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
    base_T = min_T + (initial_T - min_T) * cos_decay
    
    # 根据teacher和student的准确率差异动态调整
    acc_diff = max(0, teacher_acc - student_acc)
    # 使用sigmoid函数使调整更平滑
    adjust_factor = 1 + 2 / (1 + math.exp(-5 * acc_diff))
    
    return base_T * adjust_factor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--pre-query', type=int,
                        default=3)  # for self-supervised process: the number of query image generated based on support image
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0010216960494210821)
    parser.add_argument('--wd', type=float, default=0.003323038317277643)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet', choices=['convnet', 'resnet'])
    parser.add_argument('--dataset', type=str, default='cifarfs',
                        choices=['mini', 'tiered', 'cifarfs', 'insect', 'agriculture', 'chinese_medicine',
                                 'cub'])
    parser.add_argument('--ssl-coef', type=float, default=0.45404962580855557, help='The beta coefficient for self-supervised loss')
    # self-distillation stage parameter
    parser.add_argument('--temperature', type=int, default=7)
    parser.add_argument('--kd-coef', type=float, default=0.4681582350243203, help="The gamma coefficient for distillation loss")
    # 0: copy teacher and only KD       1: common KD
    parser.add_argument('--kd-mode', type=int, default=1, choices=[0, 1])
    parser.add_argument('--kd-type', type=str, default='focal', choices=['kd', 'hint', 'focal', 'dual', 'contrastive'])
    parser.add_argument('--stage1-path', default='')
    parser.add_argument('--stage2-path', default='')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--irm-penalty', type=float, default=0.2,help='Penalty weight for IRM loss')
    parser.add_argument('--irm-coef', type=float, default=0.3,help='Coefficient for IRM loss in total loss')
    parser.add_argument('--irm-min-weight', type=float, default=0.05,
                        help='Minimum IRM weight')
    parser.add_argument('--irm-max-weight', type=float, default=0.3,
                        help='Maximum IRM weight')
    parser.add_argument('--irm-warmup', type=int, default=30,
                        help='Number of epochs for IRM warmup')
    parser.add_argument('--irm-threshold', type=float, default=0.6,
                        help='Performance gap threshold for IRM adjustment')
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    # fix seed
    seed_torch(1)
    set_gpu(args.gpu)

    if args.dataset in ['mini', 'tiered', 'insect', 'agriculture', 'chinese_medicine', 'cub']:
        args.size = 84
    elif args.dataset in ['cifarfs']:
        args.size = 32
        args.worker = 0
    else:
        args.size = 28

    training(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)

