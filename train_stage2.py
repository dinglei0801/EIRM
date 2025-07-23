import argparse
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import f1_score, recall_score
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # 添加这一行
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets.mini_imagenet_cub import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import resnet12,Decoder
from utils import set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, seed_torch, compute_confidence_interval
from torch.amp import autocast, GradScaler

class AdaptiveWeightLoss(nn.Module):
    def __init__(self, num_losses):
        super(AdaptiveWeightLoss, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        weights = F.softplus(self.weights)
        return sum(w * l for w, l in zip(weights, losses))

class DualResNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(DualResNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adaptive_weight = AdaptiveWeightLoss(3)  # 3 losses: cls, ssl, recon

    def forward(self, x):
        encoder_output = self.encoder(x)
        
        if isinstance(encoder_output, tuple):
            features = encoder_output[0] #使用第一个返回值作为特征
        else:
            features = encoder_output

        reconstructed = self.decoder(features)
        return features, features, reconstructed  # 返回 features 两次，模拟 proto 和 enhanced features

def get_dataset(args):
    if args.dataset == 'mini':
        trainset = MiniImageNet('train', args.size)
        valset = MiniImageNet('test', args.size)
        print("=> MiniImageNet...")
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

    train_loader, val_loader = get_dataset(args)

    if args.model == 'convnet':
        encoder = Convnet().cuda()
        print("=> Convnet architecture...")
    else:
        if args.dataset in ['mini', 'tiered','cub']:
            encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
            print("=> Large block resnet architecture...")
        else:
            encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
            print("=> Small block resnet architecture...")
    decoder = Decoder(640, 3, (args.size, args.size)).cuda()  # 假设特征维度为640，输出通道为3
    model = DualResNet(encoder, decoder).cuda()


    if args.stage1_path:
        pretrained_dict = torch.load(osp.join(args.stage1_path, 'max-acc.pth'))
        model_dict = model.state_dict()

        # 调整键名以匹配当前模型结构
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('relation_module'):
                new_k = 'encoder.' + k
            else:
                new_k = 'encoder.' + k
            new_pretrained_dict[new_k] = v

        # 只更新存在于当前模型中的键
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}

        # 更新模型状态字典
        model_dict.update(new_pretrained_dict)
        model.load_state_dict(model_dict)

        print("=> Pretrained weights loaded successfully")
    if args.dataset in ['mini', 'tiered']:
        output_size = (84, 84)
    elif args.dataset in ['cifarfs']:
        output_size = (32, 32)
    else:
        output_size = (28, 28)

    
    #使用 ID 来区分参数
    model_params = set(id(p) for p in model.parameters())
    adaptive_weight_params = set(id(p) for p in model.adaptive_weight.parameters())

   
    # 创建优化器，使用 ID 来区分参数
    optimizer = torch.optim.Adam([
        {'params': [p for p in model.parameters() if id(p) not in adaptive_weight_params], 'weight_decay': args.wd},
        {'params': model.adaptive_weight.parameters(), 'weight_decay': 0}
    ], lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    #adaptive_weight_loss = AdaptiveWeightLoss(num_losses=3).cuda()  # 3个损失：分类损失、SSL损失、重构损失
    # 添加调试信息
    print(f"Total parameters: {len(model_params)}")
    print(f"Adaptive weight parameters: {len(adaptive_weight_params)}")
    print(f"Parameters with weight decay: {len(model_params - adaptive_weight_params)}")
    print(f"Parameters in optimizer: {sum(len(g['params']) for g in optimizer.param_groups)}")
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

    for epoch in range(1, args.max_epoch + 1):

        tl, ta = train(args, model, train_loader, optimizer)
        lr_scheduler.step()
        vl, va, vf, vr, acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std = validate(args, model, val_loader)

       
        if va > trlog['max_acc']:
            #print(f"New best accuracy: {va}, previous best: {trlog['max_acc']}")
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

        print(
            'Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f}±{:.4f} - F1={:.4f}±{:.4f} - Recall={:.4f}±{:.4f} - max acc={:.4f} - ETA:{}/{}'.format(
                epoch, args.max_epoch, tl, ta, vl, acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std, trlog['max_acc'], ots, timer.tts(tt - ot)))

        if epoch == args.max_epoch:
            print("Best Epoch is {} with acc={:.2f}±{:.2f}%...".format(best_epoch, cmi[0], cmi[1]))
            print("---------------------------------------------------")

def ssl_loss(args, encoder, data_shot):
    # s1 s2 q1 q2 q1 q2
    x_90 = data_shot.transpose(2,3).flip(2)
    x_180 = data_shot.flip(2).flip(3)
    x_270 = data_shot.flip(2).transpose(2,3)
    data_query = torch.cat((x_90, x_180, x_270), 0)

    def get_features(x):
        output = encoder(x)
        if isinstance(output, tuple):
            return output[0]  # 假设第一个元素是我们需要的特征
        return output

    proto = get_features(data_shot)
    proto = proto.reshape(1, args.train_way*args.shot, -1).mean(dim=0)
    query = get_features(data_query)

    label = torch.arange(args.train_way*args.shot).repeat(args.pre_query)
    label = label.type(torch.cuda.LongTensor)

    logits = euclidean_metric(query, proto)
    loss = F.cross_entropy(logits, label)

    return loss

def train(args, model, train_loader, optimizer):
    model.train()
    tl = Averager()
    ta = Averager()
    scaler = GradScaler()
    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]
        #print(f"\nBatch {i}:")
        #print(f"data_shot shape: {data_shot.shape}")
        #print(f"data_query shape: {data_query.shape}")
        with autocast(device_type='cuda', dtype=torch.float16):
            features, proto, reconstructed_shot= model(data_shot)
            #print(f"After first model forward:")
            #print(f"features shape: {features.shape}")
            #print(f"proto shape before reshape: {proto.shape}")
            
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            #print(f"proto shape after reshape: {proto.shape}")
            
            features, query, _ = model(data_query)
            #print(f"query shape: {query.shape}")
            #print(f"proj_query shape: {proj_query.shape}")
            label = torch.arange(args.train_way).repeat(args.train_query).type(torch.cuda.LongTensor)

            logits = euclidean_metric(query, proto)
            loss_cls = F.cross_entropy(logits, label)
            loss_ss = ssl_loss(args, model.encoder, data_shot)
            loss_recon = F.mse_loss(reconstructed_shot, data_shot)
            # 计算对比损失
            #temp = 0.07  # 温度系数
            # 将特征标准化
            #normalized_proto = F.normalize(proto, dim=1)
            #normalized_query = F.normalize(query, dim=1)
            
            # 计算余弦相似度矩阵
            #sim_matrix = torch.mm(normalized_query, normalized_proto.t()) / temp
            # 对比学习标签与原始分类标签相同
            #loss_contrast = F.cross_entropy(sim_matrix, label)
            # 添加正则化项
            epsilon = 1e-8
          
            #组合损失
            losses = [loss_cls + epsilon, loss_ss + epsilon, loss_recon + epsilon]
            losses =[loss_cls,loss_ss,loss_recon]
            loss = model.adaptive_weight(losses)
            acc = count_acc(logits, label)

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        torch.cuda.empty_cache()  # 清理未使用的内存
            
        proto = None; query = None; logits = None; loss = None
    return tl.item(), ta.item()
def validate(args, model, val_loader):
    model.eval()

    vl = Averager()
    va = Averager()
    vf = Averager()
    vr = Averager()
    acc_list = []
    f1_list = []
    recall_list = []

    for i, batch in enumerate(val_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]
        with autocast(device_type='cuda', dtype=torch.float16):
            _, proto, _ = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
            _, query, _ = model(data_query)

            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(query, proto)
            loss = F.cross_entropy(logits, label)
            acc,f1, recall = calculate_metrics(logits, label)
            acc = count_acc(logits, label)

        vl.add(loss.item())
        va.add(acc)
        vf.add(f1)
        vr.add(recall)
        acc_list.append(acc * 100)
        f1_list.append(f1 * 100)
        recall_list.append(recall * 100)
            
        proto = None; query = None; logits = None; loss = None
    acc_mean, acc_std = compute_confidence_interval(acc_list)
    f1_mean, f1_std = compute_confidence_interval(f1_list)
    recall_mean, recall_std = compute_confidence_interval(recall_list)
    return vl.item(), va.item(), vf.item(), vr.item(), acc_mean, acc_std, f1_mean, f1_std, recall_mean, recall_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--pre-query', type=int, default=3)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/cub-stage2-1s')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--lr', type=float, default=0.00020334184370359855)
    parser.add_argument('--wd', type=float, default=0.0008190363126360188)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--model', type=str, default='resnet', choices=['convnet', 'resnet'])
    parser.add_argument('--mode', type=int, default=0, choices=[0,1])
    parser.add_argument('--stage1-path', default='./save/cub-stage1')
    parser.add_argument('--beta', type=float, default=0.055083770829349454)
    parser.add_argument('--dataset', type=str, default='cifarfs', choices=['mini','tiered','cifarfs'])
    parser.add_argument('--gamma', type=float, default=0.4125195004202946, help='Weight for reconstruction loss')
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    # fix seed
    seed_torch(1)
    set_gpu(args.gpu)

    if args.dataset in ['mini', 'tiered']:
        args.size = 84
    elif args.dataset in ['cifarfs']:
        args.size = 32
        args.worker = 0
    else:
        args.size = 28

    training(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)


