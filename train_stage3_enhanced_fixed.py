import argparse
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import logging
import sys
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import resnet12, Decoder
from utils import set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, seed_torch, compute_confidence_interval


def setup_logging(save_path):
    """设置详细的日志记录"""
    log_file = osp.join(save_path, 'training.log')
    
    # 创建日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 写入文件
            logging.StreamHandler(sys.stdout)          # 输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("Training Log Started")
    logger.info("="*80)
    
    return logger


class LossTracker:
    """损失追踪器"""
    def __init__(self, logger):
        self.logger = logger
        self.losses = {
            'cls': [],
            'kd': [],
            'proto': [],
            'ssl': [],
            'irm': [],
            'align': [],
            'total': []
        }
        self.epoch_losses = {}
    
    def log_batch_loss(self, epoch, batch_idx, losses_dict, acc):
        """记录批次损失"""
        if batch_idx % 50 == 0:  # 每50个batch记录一次
            loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses_dict.items() if v > 0])
            self.logger.info(f"Epoch {epoch}, Batch {batch_idx}: {loss_str}, Acc={acc:.4f}")
    
    def log_epoch_loss(self, epoch, losses_dict, train_acc, val_acc):
        """记录轮次损失"""
        self.epoch_losses[epoch] = losses_dict.copy()
        
        loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses_dict.items() if v > 0])
        self.logger.info(f"Epoch {epoch} Summary: {loss_str}")
        self.logger.info(f"Epoch {epoch} Accuracy: Train={train_acc:.4f}, Val={val_acc:.4f}")
    
    def analyze_loss_trends(self, epoch):
        """分析损失趋势"""
        if epoch >= 10:  # 至少10个epoch后开始分析
            recent_epochs = list(range(max(1, epoch-9), epoch+1))
            
            for loss_type in ['cls', 'kd', 'ssl']:
                if loss_type in self.epoch_losses.get(epoch, {}):
                    values = [self.epoch_losses[e].get(loss_type, 0) for e in recent_epochs if e in self.epoch_losses]
                    if len(values) >= 5:
                        trend = "increasing" if values[-1] > values[0] else "decreasing"
                        self.logger.info(f"Loss trend analysis - {loss_type}: {trend} (latest: {values[-1]:.4f})")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        num_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))


class AdvancedDataAugmentation:
    """高级数据增强策略"""
    def __init__(self, args, logger):
        self.size = args.size
        self.cutmix_prob = 0.3
        self.mixup_alpha = 0.2
        self.logger = logger
        
    def cutmix(self, data, targets):
        """CutMix augmentation"""
        if random.random() > self.cutmix_prob:
            return data, targets
            
        batch_size = data.size(0)
        indices = torch.randperm(batch_size).cuda()
        
        # Generate random bounding box
        lam = np.random.beta(1.0, 1.0)
        W, H = data.size(2), data.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        self.logger.debug(f"CutMix applied with lambda: {lam:.3f}")
        return data, (targets, targets[indices], lam)
    
    def mixup(self, data, targets):
        """MixUp augmentation"""
        if random.random() > 0.5:
            return data, targets
            
        batch_size = data.size(0)
        indices = torch.randperm(batch_size).cuda()
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_data = lam * data + (1 - lam) * data[indices]
        
        self.logger.debug(f"MixUp applied with lambda: {lam:.3f}")
        return mixed_data, (targets, targets[indices], lam)


class EnhancedIRMLoss(nn.Module):
    """增强的IRM损失"""
    def __init__(self, penalty_weight=0.1, num_envs=4):
        super(EnhancedIRMLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.num_envs = num_envs
        
    def forward(self, logits, labels, environments=None):
        """计算增强的IRM损失"""
        base_loss = F.cross_entropy(logits, labels)
        
        if environments is not None:
            # 计算每个环境的损失
            unique_envs = torch.unique(environments)
            grad_norms = []
            
            for env in unique_envs:
                env_mask = (environments == env)
                if env_mask.sum() > 0:
                    env_logits = logits[env_mask]
                    env_labels = labels[env_mask]
                    env_loss = F.cross_entropy(env_logits, env_labels)
                    
                    # 计算梯度范数
                    grad = torch.autograd.grad(
                        env_loss, env_logits,
                        retain_graph=True,
                        create_graph=True,
                        only_inputs=True
                    )[0]
                    grad_norms.append(torch.norm(grad, p=2))
            
            if grad_norms:
                penalty = torch.var(torch.stack(grad_norms))
                return base_loss + self.penalty_weight * penalty
        
        return base_loss


class EnhancedEnvironmentGenerator:
    """增强的环境生成器"""
    def __init__(self, args, logger):
        self.size = args.size
        self.logger = logger
        
    def generate(self, data, num_envs=4):
        """生成4个多样化环境"""
        envs = []
        
        # 环境1: 原始数据
        envs.append(data)
        
        # 环境2: 水平翻转
        envs.append(torch.flip(data.clone(), dims=[3]))
        
        # 环境3: 颜色扰动
        brightness = 0.15 * torch.randn(data.size(0), 1, 1, 1).cuda()
        envs.append(torch.clamp(data.clone() + brightness, 0, 1))
        
        # 环境4: 高斯噪声
        noise = 0.05 * torch.randn_like(data)
        envs.append(torch.clamp(data.clone() + noise, 0, 1))
        
        result = torch.cat(envs[:num_envs], dim=0)
        self.logger.debug(f"Generated {num_envs} environments, output shape: {result.shape}")
        return result


class PrototypicalLoss(nn.Module):
    """原型损失函数"""
    def __init__(self, temperature=1.0):
        super(PrototypicalLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, query_features, support_features, support_labels, query_labels):
        """计算原型损失"""
        # 计算原型
        n_way = len(torch.unique(support_labels))
        n_shot = len(support_labels) // n_way
        
        prototypes = support_features.view(n_shot, n_way, -1).mean(0)
        
        # 计算距离
        distances = euclidean_metric(query_features, prototypes)
        logits = -distances / self.temperature
        
        return F.cross_entropy(logits, query_labels)


class AdaptiveWeightLoss(nn.Module):
    """自适应权重损失 - 与Stage2兼容"""
    def __init__(self, num_losses):
        super(AdaptiveWeightLoss, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        weights = F.softplus(self.weights)
        return sum(w * l for w, l in zip(weights, losses))


class DualResNet(nn.Module):
    """教师模型结构 - 与Stage2完全兼容"""
    def __init__(self, encoder, decoder):
        super(DualResNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adaptive_weight = AdaptiveWeightLoss(3)  # 3 losses: cls, ssl, recon

    def forward(self, x):
        encoder_output = self.encoder(x)
        
        if isinstance(encoder_output, tuple):
            features = encoder_output[0]  # 使用第一个返回值作为特征
        else:
            features = encoder_output

        reconstructed = self.decoder(features)
        return features, features, reconstructed  # 返回 features 两次，模拟 proto 和 enhanced features


class AdvancedSSL:
    """高级自监督学习"""
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
    def rotation_ssl(self, model, data):
        """旋转自监督学习"""
        batch_size = data.size(0)
        
        # 生成旋转数据
        rotated_data = []
        rotation_labels = []
        
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = data
            elif angle == 90:
                rotated = data.transpose(2, 3).flip(2)
            elif angle == 180:
                rotated = data.flip(2).flip(3)
            else:  # 270
                rotated = data.flip(2).transpose(2, 3)
            
            rotated_data.append(rotated)
            rotation_labels.extend([angle // 90] * batch_size)
        
        rotated_data = torch.cat(rotated_data, dim=0)
        rotation_labels = torch.tensor(rotation_labels).cuda()
        
        # 特征提取
        features = model(rotated_data)
        if isinstance(features, tuple):
            features = features[0]
        
        # 旋转分类头
        rotation_logits = F.linear(features, torch.randn(4, features.size(-1)).cuda())
        
        loss = F.cross_entropy(rotation_logits, rotation_labels)
        self.logger.debug(f"Rotation SSL loss: {loss.item():.4f}")
        return loss
    
    def contrastive_ssl(self, model, data_shot, data_query):
        """对比自监督学习"""
        # 特征提取
        shot_features = model(data_shot)
        query_features = model(data_query)
        
        if isinstance(shot_features, tuple):
            shot_features = shot_features[0]
        if isinstance(query_features, tuple):
            query_features = query_features[0]
        
        # 正负样本对比
        similarity_matrix = torch.mm(F.normalize(query_features), F.normalize(shot_features).t())
        
        # 对比损失
        temperature = 0.1
        loss = -torch.log(torch.exp(similarity_matrix / temperature).sum(dim=1)).mean()
        
        self.logger.debug(f"Contrastive SSL loss: {loss.item():.4f}")
        return loss


def ssl_loss(args, model, data_shot, logger):
    """自监督学习损失 - 与Stage2兼容"""
    # 生成旋转变换
    x_90 = data_shot.transpose(2, 3).flip(2)
    x_180 = data_shot.flip(2).flip(3)
    x_270 = data_shot.flip(2).transpose(2, 3)
    data_query = torch.cat((x_90, x_180, x_270), 0)

    def get_features(x):
        output = model(x)
        if isinstance(output, tuple):
            return output[0]  # 假设第一个元素是我们需要的特征
        return output

    proto = get_features(data_shot)
    proto = proto.reshape(1, args.train_way * args.shot, -1).mean(dim=0)
    query = get_features(data_query)

    label = torch.arange(args.train_way * args.shot).repeat(3)
    label = label.type(torch.cuda.LongTensor)

    logits = euclidean_metric(query, proto)
    loss = F.cross_entropy(logits, label)
    
    logger.debug(f"Traditional SSL loss: {loss.item():.4f}")
    return loss


def advanced_ssl_loss(args, model, data_shot, data_query, logger):
    """高级自监督学习损失"""
    ssl_module = AdvancedSSL(args, logger)
    
    # 旋转SSL
    rotation_loss = ssl_module.rotation_ssl(model, data_shot)
    
    # 对比SSL
    contrastive_loss = ssl_module.contrastive_ssl(model, data_shot, data_query)
    
    # 传统SSL
    traditional_ssl = ssl_loss(args, model, data_shot, logger)
    
    total_ssl = traditional_ssl + 0.3 * rotation_loss + 0.2 * contrastive_loss
    logger.debug(f"Advanced SSL total: {total_ssl.item():.4f}")
    return total_ssl


def calculate_metrics(logits, labels):
    """计算评估指标"""
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == labels).float().mean().item()
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    return accuracy, f1, recall


def get_dataset(args, logger):
    """获取数据集"""
    if args.dataset == 'mini':
        trainset = MiniImageNet('train', args.size)
        valset = MiniImageNet('test', args.size)
        logger.info("=> MiniImageNet loaded")
    elif args.dataset == 'tiered':
        trainset = TieredImageNet('train', args.size)
        valset = TieredImageNet('test', args.size)
        logger.info("=> TieredImageNet loaded")
    elif args.dataset == 'cifarfs':
        trainset = CIFAR_FS('train', args.size)
        valset = CIFAR_FS('test', args.size)
        logger.info("=> CIFAR FS loaded")
    else:
        logger.error("Invalid dataset specified")
        exit()

    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                      args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)
    
    logger.info(f"Dataset loaded - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


def train_enhanced(args, teacher, model, train_loader, optimizer, epoch, logger, loss_tracker):
    """增强训练函数"""
    logger.info(f"Starting training epoch {epoch}")
    teacher.eval()
    model.train()
    
    # 初始化组件
    env_generator = EnhancedEnvironmentGenerator(args, logger)
    irm_criterion = EnhancedIRMLoss(penalty_weight=args.irm_penalty, num_envs=4).cuda()
    focal_loss = FocalLoss(alpha=1, gamma=2).cuda()
    label_smoothing = LabelSmoothingLoss(smoothing=0.1).cuda()
    prototypical_loss = PrototypicalLoss(temperature=1.0).cuda()
    augmentation = AdvancedDataAugmentation(args, logger)
    
    tl = Averager()
    ta = Averager()
    scaler = GradScaler()
    
    epoch_losses = {
        'cls': Averager(),
        'kd': Averager(),
        'proto': Averager(),
        'ssl': Averager(),
        'irm': Averager(),
        'align': Averager(),
        'total': Averager()
    }
    
    for i, batch in enumerate(train_loader, 1):
        try:
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]
            
            logger.debug(f"Batch {i}: data_shot shape: {data_shot.shape}, data_query shape: {data_query.shape}")
            
            # 数据增强
            aug_info = None
            if epoch > 20:
                if random.random() < 0.3:
                    data_query, aug_info = augmentation.cutmix(data_query, 
                                                              torch.arange(args.train_way).repeat(args.train_query).cuda())
                elif random.random() < 0.3:
                    data_query, aug_info = augmentation.mixup(data_query, 
                                                             torch.arange(args.train_way).repeat(args.train_query).cuda())
            
            with torch.no_grad():
                # 教师模型推理
                t_features, t_proto, _ = teacher(data_shot)
                t_proto = t_proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
                t_query_features, t_query, _ = teacher(data_query)
                t_logits = euclidean_metric(t_query, t_proto)
                
                logger.debug(f"Teacher features - shot: {t_features.shape}, query: {t_query_features.shape}")
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # 学生模型特征提取
                s_features_shot = model(data_shot)
                if isinstance(s_features_shot, tuple):
                    s_features_shot = s_features_shot[0]
                proto = s_features_shot.reshape(args.shot, args.train_way, -1).mean(dim=0)
                
                # 生成标签
                label = torch.arange(args.train_way).repeat(args.train_query).cuda()
                
                # 基础推理
                s_features_query = model(data_query)
                if isinstance(s_features_query, tuple):
                    s_features_query = s_features_query[0]
                logits = euclidean_metric(s_features_query, proto)
                
                logger.debug(f"Student features - shot: {s_features_shot.shape}, query: {s_features_query.shape}")
                
                # 1. 主要分类损失
                if epoch > 50:
                    cls_loss = focal_loss(logits, label)
                    logger.debug(f"Using Focal Loss: {cls_loss.item():.4f}")
                else:
                    cls_loss = label_smoothing(logits, label)
                    logger.debug(f"Using Label Smoothing: {cls_loss.item():.4f}")
                
                # 2. 蒸馏损失
                temperature = max(1.5, 4.0 - epoch * 0.015)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(t_logits / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # 3. 原型损失
                proto_loss = prototypical_loss(s_features_query, s_features_shot, 
                                             torch.arange(args.train_way).repeat(args.shot).cuda(), 
                                             label)
                
                # 4. 高级SSL损失
                ssl_loss_value = advanced_ssl_loss(args, model, data_shot, data_query, logger)
                
                # 5. IRM损失
                irm_loss = 0
                if epoch > 80:
                    try:
                        data_query_envs = env_generator.generate(data_query, num_envs=4)
                        label_envs = label.repeat(4)
                        
                        query_envs = model(data_query_envs)
                        if isinstance(query_envs, tuple):
                            query_envs = query_envs[0]
                        
                        proto_expanded = proto.unsqueeze(0).expand(4, -1, -1).reshape(-1, proto.size(-1))
                        logits_envs = euclidean_metric(query_envs, proto_expanded)
                        
                        environments = torch.arange(4).repeat_interleave(data_query.size(0)).cuda()
                        irm_loss = irm_criterion(logits_envs, label_envs, environments)
                        
                        logger.debug(f"IRM loss computed: {irm_loss.item():.4f}")
                    except Exception as e:
                        #logger.warning(f"IRM loss computation failed: {e}")
                        irm_loss = 0
                
                # 6. 特征对齐损失
                feature_align_loss = 0
                if epoch > 30:
                    feature_align_loss = F.mse_loss(s_features_query, t_query_features) * 0.1
                
                # 动态权重
                progress = epoch / args.max_epoch
                cls_weight = 1.0
                kd_weight = args.kd_coef * (1 - progress * 0.2)
                proto_weight = 0.3 * min(1.0, progress * 2)
                ssl_weight = args.ssl_coef * max(0.2, 1 - progress * 0.4)
                irm_weight = args.irm_coef * min(1.0, max(0, (epoch - 80) / 30)) if epoch > 80 else 0
                
                # 总损失
                loss = (cls_weight * cls_loss + 
                       kd_weight * kd_loss + 
                       proto_weight * proto_loss +
                       ssl_weight * ssl_loss_value +
                       irm_weight * irm_loss +
                       feature_align_loss)
                
                # 记录各项损失
                losses_dict = {
                    'cls': cls_loss.item(),
                    'kd': kd_loss.item(),
                    'proto': proto_loss.item(),
                    'ssl': ssl_loss_value.item(),
                    'irm': irm_loss.item() if isinstance(irm_loss, torch.Tensor) else irm_loss,
                    'align': feature_align_loss.item() if isinstance(feature_align_loss, torch.Tensor) else feature_align_loss,
                    'total': loss.item()
                }
                
                # 更新损失追踪器
                for k, v in losses_dict.items():
                    if k in epoch_losses:
                        epoch_losses[k].add(v)
                
                # 记录批次损失
                acc = count_acc(logits, label)
                loss_tracker.log_batch_loss(epoch, i, losses_dict, acc)
            
            tl.add(loss.item())
            ta.add(acc)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            logger.debug(f"Gradient norm: {grad_norm:.4f}")
            
            scaler.step(optimizer)
            scaler.update()
            
            # 内存清理
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in batch {i}: {e}")
            continue
    
    # 记录轮次损失
    epoch_losses_dict = {k: v.item() for k, v in epoch_losses.items()}
    loss_tracker.log_epoch_loss(epoch, epoch_losses_dict, ta.item(), 0)  # val_acc will be filled later
    
    logger.info(f"Epoch {epoch} training completed - Loss: {tl.item():.4f}, Acc: {ta.item():.4f}")
    return tl.item(), ta.item()


def validate(args, model, val_loader, logger):
    """验证函数"""
    logger.info("Starting validation")
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
            try:
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]

                # 多次推理平均（TTA）
                logits_list = []
                for tta_idx in range(3):
                    # 轻微数据增强
                    if random.random() < 0.5:
                        data_query_aug = torch.flip(data_query, dims=[3])
                    else:
                        data_query_aug = data_query
                    
                    # 特征提取
                    features_shot = model(data_shot)
                    if isinstance(features_shot, tuple):
                        features_shot = features_shot[0]
                    proto = features_shot.reshape(args.shot, args.test_way, -1).mean(dim=0)

                    features_query = model(data_query_aug)
                    if isinstance(features_query, tuple):
                        features_query = features_query[0]
                    
                    logits = euclidean_metric(features_query, proto)
                    logits_list.append(logits)
                
                # 平均预测
                logits = torch.stack(logits_list).mean(0)
                label = torch.arange(args.test_way).repeat(args.test_query).cuda()
                
                loss = F.cross_entropy(logits, label)
                acc, f1, recall = calculate_metrics(logits, label)

                vl.add(loss.item())
                va.add(acc)
                vf.add(f1)
                vr.add(recall)
                acc_list.append(acc * 100)
                f1_list.append(f1 * 100)
                recall_list.append(recall * 100)
                
                if i % 500 == 0:
                    logger.debug(f"Validation batch {i}: Loss={loss.item():.4f}, Acc={acc:.4f}")
                    
            except Exception as e:
                logger.error(f"Error in validation batch {i}: {e}")
                continue

    acc_mean, acc_std = compute_confidence_interval(acc_list)
    f1_mean, f1_std = compute_confidence_interval(f1_list)
    recall_mean, recall_std = compute_confidence_interval(recall_list)
    
    logger.info(f"Validation completed - Acc: {acc_mean:.2f}±{acc_std:.2f}%, F1: {f1_mean:.2f}±{f1_std:.2f}%")
    return vl.item(), va.item(), vf.item(), vr.item(), acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std


def check_model_compatibility(teacher, model, args, logger):
    """检查模型兼容性"""
    logger.info("Checking model compatibility...")
    
    try:
        # 测试数据
        test_data = torch.randn(args.shot * args.train_way, 3, args.size, args.size).cuda()
        
        # 测试教师模型
        with torch.no_grad():
            t_out = teacher(test_data)
            logger.info(f"Teacher output shapes: {[x.shape if torch.is_tensor(x) else type(x) for x in t_out]}")
        
        # 测试学生模型
        with torch.no_grad():
            s_out = model(test_data)
            if isinstance(s_out, tuple):
                s_out = s_out[0]
            logger.info(f"Student output shape: {s_out.shape}")
        
        logger.info("Model compatibility check passed")
        return True
        
    except Exception as e:
        logger.error(f"Model compatibility check failed: {e}")
        return False


def save_checkpoint(model, optimizer, epoch, best_acc, save_path, logger):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    checkpoint_path = osp.join(save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def training(args):
    """主训练函数"""
    ensure_path(args.save_path)
    
    # 设置日志
    logger = setup_logging(args.save_path)
    loss_tracker = LossTracker(logger)
    
    # 记录训练配置
    logger.info("Training Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # 获取数据
    logger.info("Loading datasets...")
    train_loader, val_loader = get_dataset(args, logger)

    # 初始化教师模型
    logger.info("Initializing teacher model...")
    if args.dataset in ['mini', 'tiered']:
        encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
    else:
        encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
    
    decoder = Decoder(640, 3, (args.size, args.size)).cuda()
    teacher = DualResNet(encoder, decoder).cuda()
    
    # 加载教师模型权重
    teacher_path = osp.join(args.stage2_path, 'max-acc.pth')
    if osp.exists(teacher_path):
        try:
            teacher.load_state_dict(torch.load(teacher_path, weights_only=False))
            logger.info("Teacher model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            return
    else:
        logger.error(f"Teacher model not found at {teacher_path}")
        return
    teacher.eval()
    
    # 初始化学生模型
    logger.info("Initializing student model...")
    if args.dataset in ['mini', 'tiered']:
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
    else:
        model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
    
    # 加载预训练权重
    if args.stage1_path:
        stage1_path = osp.join(args.stage1_path, 'max-acc.pth')
        if osp.exists(stage1_path):
            try:
                checkpoint = torch.load(stage1_path, weights_only=False)
                model.load_state_dict(checkpoint, strict=False)
                logger.info("Stage1 weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load stage1 weights: {e}")
                logger.info("Using random initialization")
        else:
            logger.warning(f"Stage1 weights not found at {stage1_path}")
    
    # 检查模型兼容性
    if not check_model_compatibility(teacher, model, args, logger):
        logger.error("Model compatibility check failed, exiting...")
        return
    
    # 优化器设置
    logger.info("Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd}
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=args.lr * 0.001
    )
    
    def save_model(name):
        model_path = osp.join(args.save_path, name + '.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved: {model_path}")

    # 训练日志
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
    patience = 0
    
    logger.info("Starting training loop...")
    
    # 主训练循环
    for epoch in range(1, args.max_epoch + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.max_epoch}")
        logger.info(f"{'='*50}")
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # 训练
        try:
            tl, ta = train_enhanced(args, teacher, model, train_loader, optimizer, epoch, logger, loss_tracker)
        except Exception as e:
            logger.error(f"Training failed in epoch {epoch}: {e}")
            continue
        
        # 验证
        try:
            vl, va, vf, vr, acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std = validate(args, model, val_loader, logger)
        except Exception as e:
            logger.error(f"Validation failed in epoch {epoch}: {e}")
            continue
        
        # 更新损失追踪器的验证准确率
        if epoch in loss_tracker.epoch_losses:
            loss_tracker.epoch_losses[epoch]['val_acc'] = va
        
        # 保存最佳模型
        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
            best_epoch = epoch
            cmi = [acc_mean, acc_std]
            patience = 0
            logger.info(f"New best model saved! Accuracy: {acc_mean:.2f}±{acc_std:.2f}%")
        else:
            patience += 1
            logger.info(f"No improvement. Patience: {patience}")
        
        # 动态学习率调整
        scheduler.step()
        
        # 高级早停策略
        if patience > 100 and epoch > 100:
            logger.info("Early stopping triggered due to no improvement")
            break
        
        if patience == 15 and epoch > 50:
            logger.info("Reducing learning rate due to no improvement")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                logger.info(f"Learning rate reduced to: {param_group['lr']:.6f}")
        
        # 更新训练日志
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        trlog['val_f1'].append(vf)
        trlog['val_recall'].append(vr)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        save_model('epoch-last')

        # 打印详细信息
        ot, ots = timer.measure()
        tt, _ = timer.measure(epoch / args.max_epoch)
        
        logger.info(f"Epoch {epoch} Results:")
        logger.info(f"  Train Loss: {tl:.4f}, Train Acc: {ta:.4f}")
        logger.info(f"  Val Loss: {vl:.4f}, Val Acc: {acc_mean:.2f}±{acc_std:.2f}%")
        logger.info(f"  Val F1: {f1_mean:.2f}±{f1_std:.2f}%, Val Recall: {recall_mean:.2f}±{recall_std:.2f}%")
        logger.info(f"  Best Acc: {trlog['max_acc']:.4f} (Epoch {best_epoch})")
        logger.info(f"  ETA: {ots} / {timer.tts(tt - ot)}")
        
        # 分析损失趋势
        loss_tracker.analyze_loss_trends(epoch)
        
        # 每50轮保存检查点
        if epoch % 50 == 0:
            save_model(f'epoch-{epoch}')
            save_checkpoint(model, optimizer, epoch, trlog['max_acc'], args.save_path, logger)
        
        # 内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")

    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best Epoch: {best_epoch} with Acc: {cmi[0]:.2f}±{cmi[1]:.2f}%")
    logger.info(f"Total epochs: {epoch}")
    logger.info(f"Final patience: {patience}")
    logger.info("="*80)
    
    # 保存最终的损失分析
    logger.info("\nFinal Loss Analysis:")
    for epoch_num, losses in loss_tracker.epoch_losses.items():
        if epoch_num % 20 == 0:  # 每20轮记录一次
            loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses.items() if k != 'val_acc'])
            logger.info(f"Epoch {epoch_num}: {loss_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--max-epoch', type=int, default=250)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    
    # 路径参数
    parser.add_argument('--save-path', default='./save/cifarfs-stage3-enhanced-5s')
    parser.add_argument('--stage1-path', default='./save/cifarfs-stage1')
    parser.add_argument('--stage2-path', default='./save/cifarfs-stage2-5s')
    
    # 硬件参数
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--worker', type=int, default=0)
    
    # 数据参数
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='cifarfs',
                        choices=['mini', 'tiered', 'cifarfs'])
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--train-batch', type=int, default=120)
    parser.add_argument('--test-batch', type=int, default=2000)
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['convnet', 'resnet'])
    
    # 损失权重参数
    parser.add_argument('--kd-coef', type=float, default=0.6,
                        help='Knowledge distillation coefficient')
    parser.add_argument('--ssl-coef', type=float, default=0.5,
                        help='Self-supervised learning coefficient')
    parser.add_argument('--irm-coef', type=float, default=0.3,
                        help='IRM loss coefficient')
    parser.add_argument('--irm-penalty', type=float, default=0.15,
                        help='IRM penalty weight')
    
    # 日志参数
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()

    # 记录开始时间
    start_time = datetime.datetime.now()

    # 设置随机种子
    seed_torch(1)
    set_gpu(args.gpu)

    # 根据数据集调整参数
    if args.dataset in ['mini', 'tiered']:
        args.size = 84
    elif args.dataset == 'cifarfs':
        args.size = 32
        args.worker = 0
    
    # 打印配置信息
    print("="*80)
    print("Enhanced Stage 3 Training with Detailed Logging")
    print(f"Dataset: {args.dataset}")
    print(f"Image size: {args.size}")
    print(f"Shot: {args.shot}, Way: {args.train_way}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"Learning rate: {args.lr}")
    print(f"Train batch: {args.train_batch}")
    print(f"Stage1 path: {args.stage1_path}")
    print(f"Stage2 path: {args.stage2_path}")
    print(f"Save path: {args.save_path}")
    print("="*80)
    
    training(args)

    # 记录结束时间
    end_time = datetime.datetime.now()
    print(f"\nTotal execution time: {end_time - start_time}")
