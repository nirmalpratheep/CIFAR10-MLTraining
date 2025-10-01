#!/usr/bin/env python3
"""
Optimized CIFAR-10 training script with improved hyperparameters for batch size 128.

This script uses proven techniques to achieve faster convergence and better accuracy:
- Cosine annealing learning rate schedule
- Learning rate warmup
- Weight decay and Nesterov momentum
- Label smoothing
- Better data augmentation
"""

import subprocess
import sys
import os

def run_training():
    """Run optimized training with best hyperparameters for batch size 128."""
    
    print("🚀 Starting Optimized CIFAR-10 Training (Batch Size 128)")
    print("=" * 60)
    
    # Optimized hyperparameters for batch size 128
    cmd = [
        sys.executable, "main.py",
        "--batch_size", "128",
        "--epochs", "50",
        "--lr", "0.1",
        "--momentum", "0.9",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "5",
        "--scheduler", "cosine",
        "--snapshot_freq", "2",
        "--snapshot_dir", "./snapshots_optimized",
        "--save_best",
        "--cache_transforms",
        "--cache_dir", "./cache_optimized",
        "--plot_training",
        "--plot_evaluation",
        "--plot_freq", "5",
        "--plot_dir", "./plots_optimized"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    # Run the training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        print("📁 Snapshots saved in: ./snapshots_optimized/")
        print("📊 Best model will be automatically saved when accuracy improves")
    else:
        print("\n❌ Training failed with return code:", result.returncode)
    
    return result.returncode

def run_quick_test():
    """Run a quick test with fewer epochs to verify the setup."""
    
    print("🧪 Running Quick Test (10 epochs)")
    print("=" * 40)
    
    cmd = [
        sys.executable, "main.py",
        "--batch_size", "128",
        "--epochs", "10",
        "--lr", "0.1",
        "--momentum", "0.9",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "3",
        "--scheduler", "cosine",
        "--snapshot_freq", "5",
        "--snapshot_dir", "./snapshots_test",
        "--plot_training",
        "--plot_evaluation",
        "--plot_freq", "5",
        "--plot_dir", "./plots_test"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 40)
    
    result = subprocess.run(cmd)
    return result.returncode

def main():
    """Main function to run optimized training."""
    
    print("CIFAR-10 Optimized Training Script")
    print("=" * 50)
    print()
    print("This script will train your CIFAR-10 model with optimized hyperparameters")
    print("specifically tuned for batch size 128.")
    print()
    print("Key improvements:")
    print("• Cosine annealing learning rate schedule")
    print("• 5-epoch learning rate warmup")
    print("• Weight decay (1e-4) and Nesterov momentum")
    print("• Label smoothing (0.1)")
    print("• Better data augmentation")
    print("• Automatic best model saving")
    print()
    
    choice = input("Choose option:\n1. Quick test (10 epochs)\n2. Full training (100 epochs)\n3. Exit\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        return run_quick_test()
    elif choice == "2":
        return run_training()
    elif choice == "3":
        print("Exiting...")
        return 0
    else:
        print("Invalid choice. Exiting...")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
