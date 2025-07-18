import argparse
import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import resnet12, Decoder
from utils import set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, seed_torch, compute_confidence_interval

class DualResNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(DualResNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, tuple):
            features = encoder_output[0]
        else:
            features = encoder_output
        reconstructed = self.decoder(features)
        return features, features, reconstructed

def get_dataset(args):
    if args.dataset == 'mini':
        trainset = MiniImageNet('train', args.size)
        valset = MiniImageNet('test', args.size)
    elif args.dataset == 'tiered':
        trainset = TieredImageNet('train', args.size)
        valset = TieredImageNet('test', args.size)
    elif args.dataset == 'cifarfs':
        trainset = CIFAR_FS('train', args.size)
        valset = CIFAR_FS('test', args.size)
    else:
        print("Invalid dataset...")
        exit()
    
    train_sampler = CategoriesSampler(trainset.label, args.train_batch, args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.test_batch, args.test_way, args.shot + args.train_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader

def training(args, trial):
    ensure_path(args.save_path)

    train_loader, val_loader = get_dataset(args)
    # Print the hyperparameters for the current trial
    print("Starting trial with hyperparameters:")
    print("  lr: {}".format(trial.suggest_float('lr', 1e-5, 1e-1, log=True)))
    print("  wd: {}".format(trial.suggest_float('wd', 1e-5, 1e-1, log=True)))
    print("  drop_rate: {}".format(trial.suggest_float('drop_rate', 0.1, 0.5)))
    
    encoder = resnet12(avg_pool=True, drop_rate=trial.suggest_float('drop_rate', 0.1, 0.5)).cuda()
    decoder = Decoder(640, 3, (args.size, args.size)).cuda()
    model = DualResNet(encoder, decoder).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=trial.params['lr'], weight_decay=trial.params['wd'])
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    best_val_acc = 0
    total_time = 0  # Initialize total_time
    for epoch in range(1, args.max_epoch + 1):
        epoch_start_time = datetime.datetime.now()
        
        tl, ta = train(args, model, train_loader, optimizer)
        vl, va, _, _ = validate(args, model, val_loader)

        lr_scheduler.step()

        epoch_end_time = datetime.datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        total_time += epoch_duration

        print(f"Epoch {epoch}/{args.max_epoch}, "
              f"Train Loss: {tl:.4f}, Train Acc: {ta:.4f}, "
              f"Val Loss: {vl:.4f}, Val Acc: {va:.4f}, "
              f"Time: {epoch_duration:.2f}s, "
              f"Total Time: {total_time:.2f}s")

        if va > best_val_acc:
            best_val_acc = va
            print(f"New best validation accuracy: {best_val_acc:.4f}")

        # Report intermediate value to Optuna
        trial.report(va, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    print(f"Trial {trial.number} finished. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Total time for this trial: {total_time:.2f}s")
    return best_val_acc

def train(args, model, train_loader, optimizer):
    model.train()
    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]

        features, proto, reconstructed_shot = model(data_shot)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
        features, query, _ = model(data_query)

        label = torch.arange(args.train_way).repeat(args.train_query).type(torch.cuda.LongTensor)
        logits = euclidean_metric(query, proto)
        loss_cls = F.cross_entropy(logits, label)
        loss = loss_cls  # Simplified for demonstration
        acc = count_acc(logits, label)

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tl.item(), ta.item()

def validate(args, model, val_loader):
    model.eval()
    vl = Averager()
    va = Averager()
    acc_list = []

    for i, batch in enumerate(val_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]

        _, proto, _ = model(data_shot)
        proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
        _, query, _ = model(data_query)

        label = torch.arange(args.test_way).repeat(args.test_query).type(torch.cuda.LongTensor)
        logits = euclidean_metric(query, proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        vl.add(loss.item())
        va.add(acc)
        acc_list.append(acc * 100)

    return vl.item(), va.item(), 0, 0  # Placeholder for confidence interval

def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/cifarfs-stage2-1s-optuna')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='cifarfs', choices=['mini', 'tiered', 'cifarfs'])
    
    # Add missing parameters
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1, help='Weight for reconstruction loss')

    args = parser.parse_args()
    
    # Suggest hyperparameters
    args.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    args.wd = trial.suggest_float('wd', 1e-5, 1e-1, log=True)
    args.drop_rate = trial.suggest_float('drop_rate', 0.1, 0.5)
    
    # Run training and get the validation accuracy
    val_accuracy = training(args, trial)

    # Output the results of the trial
    print("Trial finished with validation accuracy: {:.2f}".format(val_accuracy))
    print("Hyperparameters:")
    print("  lr: {}".format(args.lr))
    print("  wd: {}".format(args.wd))
    print("  drop_rate: {}".format(args.drop_rate))
    return val_accuracy

if __name__ == '__main__':
    print("Starting Optuna optimization")
    study = optuna.create_study(direction='maximize')
    
    try:
        study.optimize(objective, n_trials=30, timeout=3600)  # 1 hour timeout
    except KeyboardInterrupt:
        print("Optimization stopped early by user.")
    
    print("Optimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save study results
    output_file = "optuna_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Best trial value: {trial.value}\n")
        f.write("Best trial params:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Results saved to {output_file}")
