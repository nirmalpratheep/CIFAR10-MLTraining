# CIFAR-10 Image Classifier: Advanced CNN with Residual Connections

This project implements a sophisticated CNN architecture for CIFAR-10 image classification, featuring depthwise separable convolutions, residual connections, spatial dropout, and advanced training techniques. The model achieves **81.61% test accuracy** on CIFAR-10 dataset.

## 🎯 Project Overview

The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). This project implements an advanced CNN with:

- **Depthwise Separable Convolutions**: Efficient feature extraction with reduced parameters
- **Residual Connections**: Improved gradient flow and training stability
- **Spatial Dropout**: Better regularization for convolutional layers
- **Advanced Augmentation**: Mandatory augmentations including HorizontalFlip, ShiftScaleRotate, and CoarseDropout
- **Cosine Annealing**: Smooth learning rate scheduling without warmup
- **Mixed Precision Training**: AMP support for faster training

## 🏗️ Model Architecture

### **CIFARNet: Advanced CNN with Residual Connections**

**Architecture Details:**
- **Input**: 32×32×3 RGB images
- **Channel Progression**: 3 → 40 → 128 → 240 → 384 → 10
- **Total Parameters**: ~1.2M parameters
- **Final Test Accuracy**: **81.61%**

### **Layer Breakdown:**

| Layer | Type | Input Channels | Output Channels | Kernel Size | Stride | Padding | Special Features |
|-------|------|----------------|-----------------|-------------|--------|---------|------------------|
| **C1** | Conv2D | 3 | 40 | 3×3 | 1 | 1 | BatchNorm + ReLU + Spatial Dropout |
| **C2** | Depthwise Separable | 40 | 128 | 3×3 (depth) + 1×1 (point) | 1 | 1 | **Residual Connection** + BatchNorm + ReLU + Spatial Dropout |
| **C3** | Dilated Depthwise Separable | 128 | 240 | 3×3 (dilation=4) + 1×1 | 1 | 4 | **Residual Connection** + BatchNorm + ReLU + Spatial Dropout |
| **C4** | Depthwise Separable | 240 | 384 | 3×3 (depth) + 1×1 (point) | 2 | 1 | BatchNorm + ReLU + Spatial Dropout |
| **Output** | GAP + FC | 384 | 10 | Global Average Pool + Linear | - | - | Fully Connected Layer |

### **Key Architectural Features:**

- **Depthwise Separable Convolutions**: Efficient feature extraction with reduced computational cost
- **Residual Connections**: Added to C2 and C3 layers for improved gradient flow
- **Spatial Dropout**: Applied after each block for better regularization
- **Dilated Convolutions**: C3 uses dilation=4 for larger receptive field without additional parameters
- **Global Average Pooling**: Reduces overfitting compared to fully connected layers

## 📊 Performance Results

### **Final Training Results (250 Epochs)**
```
🎯 ACHIEVED: 81.61% test accuracy on CIFAR-10
📊 TRAIN ACCURACY: 85.13%
📊 TEST ACCURACY: 81.61%
📊 TOP-3 ACCURACY: 95.81%
📊 TOP-5 ACCURACY: 98.70%
```

### **Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **airplane** | 0.8193 | 0.8340 | 0.8266 | 1000 |
| **automobile** | 0.9135 | 0.8980 | 0.9057 | 1000 |
| **bird** | 0.7371 | 0.7290 | 0.7330 | 1000 |
| **cat** | 0.7395 | 0.6530 | 0.6936 | 1000 |
| **deer** | 0.7787 | 0.7880 | 0.7833 | 1000 |
| **dog** | 0.7734 | 0.7510 | 0.7620 | 1000 |
| **frog** | 0.7771 | 0.8960 | 0.8323 | 1000 |
| **horse** | 0.8503 | 0.8290 | 0.8395 | 1000 |
| **ship** | 0.8796 | 0.9060 | 0.8926 | 1000 |
| **truck** | 0.8895 | 0.8770 | 0.8832 | 1000 |

### **Overall Metrics:**
- **Macro F1-Score**: 0.8152
- **Weighted F1-Score**: 0.8152
- **Macro Precision**: 0.8158
- **Macro Recall**: 0.8161

## 🔬 Training Configuration

### **Training Parameters:**
- **Batch Size**: 512
- **Epochs**: 250
- **Learning Rate**: 0.1 (initial)
- **Momentum**: 0.9
- **Weight Decay**: 5e-4
- **Scheduler**: Cosine Annealing (no warmup)
- **Optimizer**: SGD with Nesterov momentum
- **Mixed Precision**: AMP enabled
- **Gradient Clipping**: 1.0 max norm

### **Data Augmentation (Mandatory):**
- **HorizontalFlip**: 50% probability
- **ShiftScaleRotate**: shift_limit=0.0625, scale_limit=0.1, rotate_limit=15°
- **CoarseDropout**: 1 hole, 16×16 pixels, filled with dataset mean
- **ColorJitter**: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
- **RandomCrop**: 32×32 from 40×40 padded input

### **Key Training Insights:**

#### **Architecture Effectiveness:**
- **Residual Connections**: Significantly improved gradient flow in C2 and C3 layers
- **Depthwise Separable Convolutions**: Efficient feature extraction with reduced parameters
- **Spatial Dropout**: Better regularization compared to standard dropout
- **Dilated Convolutions**: C3 layer's dilation=4 provided larger receptive field

#### **Training Strategy Success:**
- **Cosine Annealing**: Smooth learning rate decay from 0.1 to 1e-6 over 250 epochs
- **No Warmup**: Direct cosine scheduling proved effective
- **Mixed Precision**: AMP provided training speedup without accuracy loss
- **Gradient Clipping**: Prevented gradient explosion during training

## 🚀 Usage Instructions

### **Prerequisites**
```bash
pip install torch torchvision tqdm albumentations torchsummary
```

### **Quick Start**
```bash
# Run complete training with all features
python run_complete_training.py

# Direct training with custom parameters
python main.py --batch_size 512 --epochs 250 --lr 0.1 --weight_decay 5e-4
```

### **Advanced Training Options**
```bash
# Enable mixed precision training
python main.py --amp --batch_size 512 --epochs 250

# Custom gradient clipping
python main.py --max_grad_norm 2.0 --epochs 250

# Enable data caching for faster subsequent runs
python main.py --cache_transforms --cache_dir ./cache --epochs 250

# Custom snapshot frequency
python main.py --snapshot_freq 10 --save_best --epochs 250

# Generate visualizations
python main.py --plot_training --plot_evaluation --plot_freq 50 --epochs 250
```

### **Key Command Line Arguments**
- `--batch_size`: Batch size (default: 512)
- `--epochs`: Number of training epochs (default: 250)
- `--lr`: Initial learning rate (default: 0.1)
- `--weight_decay`: Weight decay for regularization (default: 5e-4)
- `--amp`: Enable mixed precision training
- `--max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `--cache_transforms`: Cache augmented data for faster training
- `--plot_training`: Generate training curves
- `--plot_evaluation`: Generate confusion matrix and metrics
- `--save_best`: Save model only when test accuracy improves

## 📊 Generated Outputs

### **Training Logs**
Complete training logs are saved to `./log/` directory with timestamps:
- `training_complete_YYYYMMDD-HHMMSS.log`

### **Model Snapshots**
Model checkpoints saved to `./snapshots_complete/`:
- Automatic snapshots every 5 epochs
- Best model saving when test accuracy improves
- Complete state preservation (model, optimizer, scheduler, training history)

### **Visualizations**
All plots saved to `./plots_complete/`:
- `training_curves.png` - Loss and accuracy curves
- `confusion_matrix.png` - Classification confusion matrix
- `class_metrics.png` - Per-class performance metrics
- `learning_rate_schedule.png` - LR schedule over time
- `classification_report.txt` - Detailed metrics report

## 🎓 Key Technical Achievements

### **Architecture Innovations:**
1. **Residual Connections**: Successfully integrated in C2 and C3 layers
2. **Depthwise Separable Convolutions**: Efficient feature extraction
3. **Dilated Convolutions**: C3 layer with dilation=4 for larger receptive field
4. **Spatial Dropout**: Better regularization than standard dropout

### **Training Optimizations:**
1. **Cosine Annealing**: Smooth LR decay without warmup complications
2. **Mixed Precision Training**: AMP for faster training
3. **Gradient Clipping**: Stable training with max_norm=1.0
4. **Advanced Augmentation**: Mandatory augmentations for better generalization

### **Performance Highlights:**
- **81.61% test accuracy** on CIFAR-10
- **95.81% top-3 accuracy** showing strong confidence
- **Balanced performance** across all 10 classes
- **Efficient architecture** with ~1.2M parameters

## 🔮 Future Enhancements

- **Attention Mechanisms**: Add spatial/channel attention modules
- **EfficientNet Scaling**: Compound scaling of depth, width, and resolution
- **Knowledge Distillation**: Train smaller student models
- **Advanced Augmentation**: MixUp, CutMix, or AutoAugment
- **Ensemble Methods**: Combine multiple model predictions

## 📝 Conclusion

This CIFAR-10 classifier demonstrates the effectiveness of modern CNN architectures with residual connections, depthwise separable convolutions, and advanced training techniques. The model achieves competitive 81.61% accuracy through:

- **Thoughtful Architecture Design**: Residual connections and efficient convolutions
- **Advanced Training Strategies**: Cosine annealing, mixed precision, and gradient clipping
- **Robust Data Augmentation**: Mandatory augmentations for better generalization
- **Comprehensive Evaluation**: Detailed metrics and visualizations

The project showcases how combining architectural innovations with modern training techniques can achieve strong performance on challenging computer vision tasks like CIFAR-10 classification.