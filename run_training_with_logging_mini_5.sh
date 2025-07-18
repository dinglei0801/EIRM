#!/bin/bash

# ====================================================================
# CIFARFS Enhanced Stage 3 Training with Detailed Logging
# ====================================================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建保存目录
SAVE_DIR="./save/mini-stage3-enhanced-logged"
mkdir -p $SAVE_DIR

# 检查依赖文件
echo "Checking required files..."
if [ ! -f "./save/mini-stage1/max-acc.pth" ]; then
    echo "ERROR: Stage1 model not found at ./save/mini-stage1/max-acc.pth"
    exit 1
fi

if [ ! -f "./save/mini-stage2-1s/max-acc.pth" ]; then
    echo "ERROR: Stage2 model not found at ./save/mini-stage2-1s/max-acc.pth"
    exit 1
fi

echo "All required files found."

# 记录开始时间
start_time=$(date)
echo "Training started at: $start_time"

# ====================================================================
# 1. 快速测试运行（推荐先运行）
# ====================================================================
echo "Running quick test..."
python train_stage3_enhanced_fixed.py \
    --dataset mini \
    --max-epoch 3 \
    --shot 5 \
    --train-way 10 \
    --test-way 5 \
    --train-query 15 \
    --test-query 15 \
    --lr 0.0015 \
    --wd 0.0005 \
    --train-batch 20 \
    --test-batch 100 \
    --kd-coef 0.6 \
    --ssl-coef 0.5 \
    --irm-coef 0.3 \
    --irm-penalty 0.15 \
    --save-path ./save/mini-stage3-test \
    --stage1-path ./save/mini-stage1 \
    --stage2-path ./save/mini-stage2-5s \
    --gpu 0 \
    --worker 0 \
    --log-level INFO

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo "Quick test passed! Starting full training..."
else
    echo "Quick test failed! Please check the logs."
    exit 1
fi

# ====================================================================
# 2. 完整训练
# ====================================================================
echo "Starting full training with detailed logging..."
python train_stage3_enhanced_fixed.py \
    --dataset mini \
    --max-epoch 200 \
    --shot 5 \
    --train-way 10 \
    --test-way 5 \
    --train-query 15 \
    --test-query 15 \
    --lr 0.0015 \
    --wd 0.0005 \
    --train-batch 120 \
    --test-batch 2000 \
    --kd-coef 0.6 \
    --ssl-coef 0.5 \
    --irm-coef 0.3 \
    --irm-penalty 0.15 \
    --save-path $SAVE_DIR \
    --stage1-path ./save/mini-stage1 \
    --stage2-path ./save/mini-stage2-5s \
    --gpu 0 \
    --worker 0 \
    --log-level INFO

# 记录结束时间
end_time=$(date)
echo "Training completed at: $end_time"

# 显示训练结果
echo "="*60
echo "Training Results:"
echo "Log file: $SAVE_DIR/training.log"
echo "Best model: $SAVE_DIR/max-acc.pth"
echo "Training log: $SAVE_DIR/trlog"
echo "="*60

# 分析训练结果
if [ -f "$SAVE_DIR/trlog" ]; then
    python -c "
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

print('Analyzing training results...')
if os.path.exists('$SAVE_DIR/trlog'):
    trlog = torch.load('$SAVE_DIR/trlog')
    
    print(f'Total epochs: {len(trlog[\"train_acc\"])}')
    print(f'Best validation accuracy: {trlog[\"max_acc\"]:.4f}')
    print(f'Final train accuracy: {trlog[\"train_acc\"][-1]:.4f}')
    print(f'Final val accuracy: {trlog[\"val_acc\"][-1]:.4f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(trlog['train_loss'], label='Train Loss')
    plt.plot(trlog['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(trlog['train_acc'], label='Train Acc')
    plt.plot(trlog['val_acc'], label='Val Acc')
    plt.axhline(y=0.77, color='r', linestyle='--', label='Target 77%')
    plt.axhline(y=0.7461, color='g', linestyle='--', label='Baseline 74.61%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(trlog['val_f1'], label='F1 Score')
    plt.plot(trlog['val_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('F1 & Recall Curves')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('$SAVE_DIR/training_analysis.png', dpi=300, bbox_inches='tight')
    print('Training curves saved to $SAVE_DIR/training_analysis.png')
    
    # 性能分析
    if trlog['max_acc'] > 0.70:
        print('🎉 SUCCESS: Target 70% accuracy achieved!')
    elif trlog['max_acc'] > 0.6682:
        improvement = (trlog['max_acc'] - 0.6682) * 100
        print(f'📈 IMPROVEMENT: {improvement:.2f}% improvement over baseline')
    else:
        print('❌ PERFORMANCE: Below baseline, check logs for issues')
else:
    print('Training log not found')
"
else
    echo "Training log not found"
fi