# CIFAR-10 Training Optimization Guide

## 🚀 Quick Start - Optimized Training

### Run the optimized training script:
```bash
python train_optimized.py
```

### Or run directly with optimized parameters:
```bash
python main.py --batch_size 128 --epochs 100 --lr 0.1 --momentum 0.9 --weight_decay 1e-4 --warmup_epochs 5 --scheduler cosine --save_best
```

## 🔧 Key Optimizations Applied

### 1. **Learning Rate Schedule**
- **Cosine Annealing**: Smoothly decreases LR from 0.1 to 1e-6 over 100 epochs
- **Warmup**: Gradually increases LR from 0 to 0.1 over first 5 epochs
- **Why**: Prevents initial instability and provides better convergence

### 2. **Optimizer Improvements**
- **Nesterov Momentum**: `momentum=0.9, nesterov=True`
- **Weight Decay**: `weight_decay=1e-4` for regularization
- **Why**: Better gradient updates and prevents overfitting

### 3. **Loss Function Enhancement**
- **Label Smoothing**: `label_smoothing=0.1`
- **Why**: Reduces overconfidence and improves generalization

### 4. **Batch Size Optimization**
- **Batch Size 128**: Optimal for your GPU memory and training stability
- **Why**: Better gradient estimates, faster training, stable convergence

## 📊 Expected Performance Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Epoch 1 Accuracy | ~36% | ~45-50% |
| Epoch 5 Accuracy | ~60% | ~75-80% |
| Epoch 10 Accuracy | ~65% | ~85-90% |
| Final Accuracy | ~70% | ~92-95% |
| Training Time | ~25 min/epoch | ~15-20 min/epoch |

## 🎯 Training Strategies

### Strategy 1: Quick Test (10 epochs)
```bash
python main.py --batch_size 128 --epochs 10 --lr 0.1 --warmup_epochs 3 --scheduler cosine --save_best
```

### Strategy 2: Full Training (100 epochs)
```bash
python main.py --batch_size 128 --epochs 100 --lr 0.1 --warmup_epochs 5 --scheduler cosine --save_best --snapshot_freq 10
```

### Strategy 3: Resume from Snapshot
```bash
python main.py --resume_from ./snapshots_optimized/cifar_epoch_50.pth --epochs 100 --save_best
```

## 🔍 Monitoring Training

### Key Metrics to Watch:
1. **Learning Rate**: Should start low, peak at 0.1, then decrease
2. **Train/Test Accuracy Gap**: Should be < 5% (indicates good generalization)
3. **Loss Convergence**: Should decrease smoothly without spikes
4. **Best Model Saving**: Will save automatically when test accuracy improves

### Expected Learning Curve:
```
Epoch 1:  Train Acc: ~45% | Test Acc: ~50% | LR: 0.020
Epoch 5:  Train Acc: ~75% | Test Acc: ~80% | LR: 0.100
Epoch 10: Train Acc: ~85% | Test Acc: ~90% | LR: 0.095
Epoch 20: Train Acc: ~90% | Test Acc: ~92% | LR: 0.080
Epoch 50: Train Acc: ~93% | Test Acc: ~94% | LR: 0.040
Epoch 100: Train Acc: ~95% | Test Acc: ~95% | LR: 0.001
```

## 🛠️ Troubleshooting

### If training is still slow:
1. **Reduce batch size**: Try `--batch_size 64` if memory allows
2. **Increase learning rate**: Try `--lr 0.2` for faster initial learning
3. **Reduce warmup**: Try `--warmup_epochs 2`

### If accuracy plateaus:
1. **Increase epochs**: Try `--epochs 150`
2. **Adjust weight decay**: Try `--weight_decay 5e-5` or `--weight_decay 2e-4`
3. **Try different scheduler**: `--scheduler step` with `--step_size 30 --gamma 0.1`

### If overfitting occurs:
1. **Increase weight decay**: Try `--weight_decay 2e-4`
2. **Reduce learning rate**: Try `--lr 0.05`
3. **Add more data augmentation** (modify preprocess.py)

## 📈 Advanced Techniques

### Mixed Precision Training (if supported):
```bash
# Add to your training command
python main.py --batch_size 128 --epochs 100 --lr 0.1 --warmup_epochs 5 --scheduler cosine --save_best --mixed_precision
```

### Learning Rate Finder:
```bash
# Find optimal learning rate
python main.py --batch_size 128 --epochs 1 --lr 0.001 --scheduler onecycle --find_lr
```

## 🎉 Expected Results

With these optimizations, you should see:
- **Faster convergence**: 2-3x faster than before
- **Higher accuracy**: 90%+ test accuracy
- **Stable training**: No loss spikes or instability
- **Better generalization**: Small train/test gap
- **Automatic best model saving**: No manual intervention needed

The training should now be much more efficient and achieve better results with the same number of parameters!

