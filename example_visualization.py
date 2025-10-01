#!/usr/bin/env python3
"""
Example script demonstrating comprehensive visualization and metrics for CIFAR-10 training.

This script shows how to:
1. Train with automatic plotting
2. Generate comprehensive evaluation reports
3. Create confusion matrices and class metrics
4. Visualize training progress
"""

import subprocess
import sys
import os

def run_training_with_plots():
    """Run training with comprehensive visualization."""
    
    print("🎨 CIFAR-10 Training with Comprehensive Visualization")
    print("=" * 60)
    
    # Training command with all visualization features
    cmd = [
        sys.executable, "main.py",
        "--batch_size", "128",
        "--epochs", "20",
        "--lr", "0.1",
        "--momentum", "0.9",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "3",
        "--scheduler", "cosine",
        "--snapshot_freq", "5",
        "--snapshot_dir", "./snapshots_demo",
        "--save_best",
        "--plot_training",
        "--plot_evaluation",
        "--plot_freq", "5",
        "--plot_dir", "./plots_demo"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 60)
    print("This will generate:")
    print("• Training curves (loss, accuracy, learning rate)")
    print("• Confusion matrix")
    print("• Per-class performance metrics")
    print("• Classification report")
    print("• Learning rate schedule")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   • ./snapshots_demo/ - Model snapshots")
        print("   • ./plots_demo/ - All visualizations and reports")
        print("\n📊 Generated files:")
        print("   • training_curves.png - Loss and accuracy curves")
        print("   • confusion_matrix.png - Confusion matrix")
        print("   • class_metrics.png - Per-class precision/recall/F1")
        print("   • learning_rate_schedule.png - LR schedule")
        print("   • classification_report.txt - Detailed metrics")
    else:
        print("\n❌ Training failed with return code:", result.returncode)
    
    return result.returncode

def run_quick_demo():
    """Run a quick 5-epoch demo with visualization."""
    
    print("⚡ Quick Demo (5 epochs) with Visualization")
    print("=" * 50)
    
    cmd = [
        sys.executable, "main.py",
        "--batch_size", "128",
        "--epochs", "5",
        "--lr", "0.1",
        "--momentum", "0.9",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "2",
        "--scheduler", "cosine",
        "--snapshot_freq", "2",
        "--snapshot_dir", "./snapshots_quick",
        "--plot_training",
        "--plot_evaluation",
        "--plot_freq", "2",
        "--plot_dir", "./plots_quick"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Quick demo completed!")
        print("📁 Check ./plots_quick/ for generated visualizations")
    else:
        print("\n❌ Demo failed with return code:", result.returncode)
    
    return result.returncode

def show_usage_examples():
    """Show various usage examples for different scenarios."""
    
    print("📚 Usage Examples for CIFAR-10 Training with Visualization")
    print("=" * 70)
    
    examples = [
        {
            "name": "Basic Training with Plots",
            "command": "python main.py --batch_size 128 --epochs 20 --plot_training --plot_evaluation",
            "description": "Train for 20 epochs with all visualizations"
        },
        {
            "name": "Training with Custom Plot Frequency",
            "command": "python main.py --batch_size 128 --epochs 50 --plot_training --plot_freq 10",
            "description": "Generate plots every 10 epochs"
        },
        {
            "name": "Training with Custom Plot Directory",
            "command": "python main.py --batch_size 128 --epochs 30 --plot_dir ./my_plots --plot_training",
            "description": "Save plots to custom directory"
        },
        {
            "name": "Training without Plots",
            "command": "python main.py --batch_size 128 --epochs 20 --no_plots",
            "description": "Disable all plotting for faster training"
        },
        {
            "name": "Resume Training with Plots",
            "command": "python main.py --resume_from ./snapshots/cifar_epoch_10.pth --plot_training --plot_evaluation",
            "description": "Resume training and generate plots"
        },
        {
            "name": "Evaluation Only (from snapshot)",
            "command": "python main.py --resume_from ./snapshots/cifar_epoch_20.pth --epochs 20 --plot_evaluation --no_plots",
            "description": "Load model and generate evaluation plots only"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
    
    print("\n" + "=" * 70)

def main():
    """Main function to run visualization examples."""
    
    print("CIFAR-10 Visualization Examples")
    print("=" * 40)
    print()
    print("This script demonstrates comprehensive visualization features:")
    print("• Training curves (loss, accuracy, learning rate)")
    print("• Confusion matrix with class names")
    print("• Per-class performance metrics")
    print("• Classification reports")
    print("• Learning rate schedules")
    print()
    
    choice = input("Choose option:\n1. Full training demo (20 epochs)\n2. Quick demo (5 epochs)\n3. Show usage examples\n4. Exit\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        return run_training_with_plots()
    elif choice == "2":
        return run_quick_demo()
    elif choice == "3":
        show_usage_examples()
        return 0
    elif choice == "4":
        print("Exiting...")
        return 0
    else:
        print("Invalid choice. Exiting...")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

