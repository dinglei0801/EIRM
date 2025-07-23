#python improved_test.py \
 # --save-path ./save/cifarfs-stage3-enhanced-logged-new \
 # --dataset cifarfs \
 # --shot 1 \
  #--compare \
  #--validation-acc 70.52

import argparse
import datetime
import os.path as osp
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import resnet12
from utils import set_gpu, Averager, count_acc, euclidean_metric, seed_torch, compute_confidence_interval
from sklearn.metrics import f1_score
import numpy as np

def final_evaluate(args):
    if args.dataset == 'mini':
        valset = MiniImageNet('test', args.size)
    elif args.dataset == 'tiered':
        valset = TieredImageNet('test', args.size)
    elif args.dataset == "cifarfs":
        valset = CIFAR_FS('test', args.size)
    else:
        print("Invalid dataset...")
        exit()
        
    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)

    if args.model == 'convnet':
        model = Convnet().cuda()
        print("=> Convnet architecture...")
    else:
        if args.dataset in ['mini', 'tiered']:
            model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
        else:
            model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
        print("=> Resnet architecture...")

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max-acc.pth')))
    print("=> Model loaded...")
    model.eval()

    ave_acc = Averager()
    acc_list = []
    all_preds = []
    all_labels = []
    
    print("Starting enhanced evaluation with TTA...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.test_way * args.shot
            data_shot, data_query = data[:k], data[k:]
            
            # 与验证时保持一致：使用TTA（Test Time Augmentation）
            logits_list = []
            
            for tta_idx in range(args.tta_times):
                # 轻微数据增强（与验证时一致）
                if tta_idx == 0:
                    # 第一次：原始数据
                    data_query_aug = data_query
                elif tta_idx == 1:
                    # 第二次：水平翻转
                    data_query_aug = torch.flip(data_query, dims=[3])
                else:
                    # 第三次：随机选择原始或翻转
                    if random.random() < 0.5:
                        data_query_aug = torch.flip(data_query, dims=[3])
                    else:
                        data_query_aug = data_query
                
                # 支持集特征提取（每次TTA都重新计算以保证一致性）
                support_features = model(data_shot)
                if isinstance(support_features, tuple):
                    support_features = support_features[0]
                
                # 计算原型
                proto = support_features.reshape(args.shot, args.test_way, -1).mean(dim=0)
                
                # 查询集特征提取
                query_features = model(data_query_aug)
                if isinstance(query_features, tuple):
                    query_features = query_features[0]
                
                # 计算logits
                logits = euclidean_metric(query_features, proto)
                logits_list.append(logits)
            
            # 平均多次TTA的结果（与验证时一致）
            final_logits = torch.stack(logits_list).mean(0)
            
            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            acc = count_acc(final_logits, label)
            ave_acc.add(acc)
            acc_list.append(acc * 100)
            
            # 收集预测和标签用于计算F1分数
            _, preds = torch.max(final_logits, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # 进度显示
            if i % 500 == 0:
                print(f"Processed {i}/{len(loader)} batches, Current Acc: {acc:.4f}")

    a, b = compute_confidence_interval(acc_list)
    print("\n" + "="*60)
    print("ENHANCED EVALUATION RESULTS")
    print("="*60)
    print("Final accuracy with 95% interval : {:.2f}±{:.2f}".format(a, b))

    # 计算F1分数
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("F1 Score: {:.4f}".format(f1))
    
    # 与验证集结果对比
    if args.validation_acc:
        improvement = a - args.validation_acc
        print(f"Validation Accuracy: {args.validation_acc:.2f}%")
        print(f"Test vs Validation: {improvement:+.2f}%")
        if improvement > -1.0:
            print("✅ Gap reduced to acceptable range!")
        else:
            print("⚠️  Still significant gap, consider further improvements")
    
    print("="*60)


def final_evaluate_simple(args):
    """原始简单版本（用于对比）"""
    if args.dataset == 'mini':
        valset = MiniImageNet('test', args.size)
    elif args.dataset == 'tiered':
        valset = TieredImageNet('test', args.size)
    elif args.dataset == "cifarfs":
        valset = CIFAR_FS('test', args.size)
    else:
        print("Invalid dataset...")
        exit()
        
    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)

    if args.model == 'convnet':
        model = Convnet().cuda()
    else:
        if args.dataset in ['mini', 'tiered']:
            model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
        else:
            model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()

    model.load_state_dict(torch.load(osp.join(args.save_path, 'max-acc.pth')))
    model.eval()

    ave_acc = Averager()
    acc_list = []
    
    print("Running simple evaluation (original method)...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.test_way * args.shot
            data_shot, data_query = data[:k], data[k:]
            
            # 原始方法：单次推理
            x, _, _ = model(data_shot)
            x = x.reshape(args.shot, args.test_way, -1).mean(dim=0)
            p = x
            
            query, _, _ = model(data_query)
            logits = euclidean_metric(query, p)

            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            acc = count_acc(logits, label)
            ave_acc.add(acc)
            acc_list.append(acc * 100)

    a, b = compute_confidence_interval(acc_list)
    print("Simple evaluation accuracy: {:.2f}±{:.2f}".format(a, b))
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--test-batch', type=int, default=5000)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--model', type=str, default='resnet', choices=['convnet', 'resnet'])
    parser.add_argument('--dataset', type=str, default='cifarfs', choices=['mini','tiered','cifarfs'])
    
    # 新增参数
    parser.add_argument('--tta-times', type=int, default=3, 
                       help='Number of test time augmentation iterations')
    parser.add_argument('--validation-acc', type=float, default=None,
                       help='Validation accuracy for comparison')
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced evaluation with TTA')
    parser.add_argument('--compare', action='store_true',
                       help='Compare both simple and enhanced methods')
    
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

    print("="*60)
    print("Few-Shot Learning Test Evaluation")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Shot: {args.shot}")
    print(f"Model path: {args.save_path}")
    print(f"TTA times: {args.tta_times}")
    print("="*60)

    if args.compare:
        # 对比两种方法
        print("\n1. Running simple evaluation...")
        simple_acc = final_evaluate_simple(args)
        
        print("\n2. Running enhanced evaluation...")
        final_evaluate(args)
        
        print(f"\nComparison Summary:")
        print(f"Simple method: {simple_acc:.2f}%")
        print(f"Expected enhancement: +0.5~1.5%")
        
    elif args.enhanced:
        # 只运行增强版本
        final_evaluate(args)
    else:
        # 只运行简单版本
        final_evaluate_simple(args)

    end_time = datetime.datetime.now()
    print(f"\nTotal executed time: {end_time - start_time}")

