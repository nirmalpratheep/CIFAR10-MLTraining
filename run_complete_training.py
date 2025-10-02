#!/usr/bin/env python3
"""
Complete CIFAR-10 training script with all features enabled.

This script demonstrates:
- Optimized training with batch size 128
- Model snapshots and resuming
- Comprehensive visualization and metrics
- Confusion matrix and classification reports
- Training curves and learning rate visualization
"""

import subprocess
import sys
import os
import time
from datetime import datetime


def _ensure_log_dir() -> str:
    """Ensure the log directory exists and return its path."""
    log_dir = os.path.join(".", "log")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _timestamp() -> str:
    """Return a compact timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _tee_run_command_and_log(cmd, log_file_path: str) -> int:
    """Run a command, tee stdout/stderr to console and a log file, and return exit code."""
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        print ("Command: " + " ".join(cmd) + "\n")
        log_file.write("Command: " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
        return process.returncode

def run_complete_training():
    """Run complete training with all features enabled."""
    
    print("🚀 Complete CIFAR-10 Training with All Features")
    print("=" * 60)
    print()
    print("Features enabled:")
    print("✅ Optimized hyperparameters for batch size 128")
    print("✅ Model snapshots every 5 epochs")
    print("✅ Best model saving")
    print("✅ Comprehensive visualization")
    print("✅ Confusion matrix and metrics")
    print("✅ Training curves and learning rate plots")
    print("✅ Classification reports")
    print()
    
    # Complete training command with all features
    cmd = [
        sys.executable, "main.py",
        "--batch_size", "512",
        "--epochs", "180",
        "--lr", "0.1",
        "--momentum", "0.9",
        "--weight_decay", "5e-4",
        "--warmup_epochs", "5",
        "--scheduler", "cosine",
        "--snapshot_freq", "5",
        "--snapshot_dir", "./snapshots_complete",
        "--save_best",
        "--plot_training",
        "--plot_evaluation",
        "--plot_freq", "180",
        "--plot_dir", "./plots_complete",
        "--cache_transforms",
        "--cache_dir", "./cache_complete",
        "--num_workers", "2",
        "--max_grad_norm", "1.0"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 60)
    print (cmd)
    # Prepare logging
    log_dir = _ensure_log_dir()
    log_path = os.path.join(log_dir, f"training_complete_{_timestamp()}.log")
    print(f"📝 Writing full training log to: {log_path}")
    
    start_time = time.time()
    return_code = _tee_run_command_and_log(cmd, log_path)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    if return_code == 0:
        print(f"\n✅ Training completed successfully in {training_time/60:.1f} minutes!")
        print()
        print("📁 Generated outputs:")
        print("   • ./snapshots_complete/ - Model snapshots")
        print("   • ./plots_complete/ - All visualizations")
        print("   • ./cache_complete/ - Cached data transforms")
        print()
        print("📊 Generated visualizations:")
        print("   • training_curves.png - Loss and accuracy curves")
        print("   • confusion_matrix.png - Classification confusion matrix")
        print("   • class_metrics.png - Per-class performance metrics")
        print("   • learning_rate_schedule.png - LR schedule over time")
        print("   • classification_report.txt - Detailed metrics report")
        print()
        print("🎯 Next steps:")
        print("   • Check ./plots_complete/ for all visualizations")
        print("   • Use snapshots to resume training if needed")
        print("   • Analyze classification_report.txt for detailed metrics")
    else:
        print(f"\n❌ Training failed with return code: {return_code}")
        print("Check the error messages above for troubleshooting.")
    
    return return_code

def run_quick_demo():
    """Run a quick 10-epoch demo with all features."""
    
    print("⚡ Quick Demo (10 epochs) with All Features")
    print("=" * 50)
    
    cmd = [
        sys.executable, "main.py",
        "--batch_size", "128",
        "--epochs", "10",
        "--lr", "0.1",
        "--momentum", "0.9",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "3",
        "--scheduler", "cosine",
        "--snapshot_freq", "3",
        "--snapshot_dir", "./snapshots_demo",
        "--save_best",
        "--plot_training",
        "--plot_evaluation",
        "--plot_freq", "3",
        "--plot_dir", "./plots_demo"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # Prepare logging for demo
    log_dir = _ensure_log_dir()
    log_path = os.path.join(log_dir, f"training_demo_{_timestamp()}.log")
    print(f"📝 Writing full demo log to: {log_path}")
    
    return_code = _tee_run_command_and_log(cmd, log_path)
    
    if return_code == 0:
        print("\n✅ Quick demo completed!")
        print("📁 Check ./plots_demo/ for generated visualizations")
    else:
        print("\n❌ Demo failed with return code:", return_code)
    
    return return_code

def show_feature_summary():
    """Show a summary of all available features."""
    
    print("🎯 CIFAR-10 Training Features Summary")
    print("=" * 50)
    print()
    
    features = {
        "Training Optimizations": [
            "Cosine annealing learning rate schedule",
            "Learning rate warmup (5 epochs)",
            "Weight decay (1e-4) and Nesterov momentum",
            "Label smoothing (0.1)",
            "Optimized for batch size 128"
        ],
        "Model Management": [
            "Automatic model snapshots",
            "Best model saving",
            "Resume training from snapshots",
            "Complete state preservation"
        ],
        "Visualization": [
            "Training curves (loss, accuracy, LR)",
            "Confusion matrix with class names",
            "Per-class performance metrics",
            "Learning rate schedule plots",
            "Classification reports"
        ],
        "Metrics": [
            "Accuracy and Top-K accuracy",
            "Precision, Recall, F1-score",
            "Macro and weighted averages",
            "Per-class detailed metrics",
            "Support and confusion analysis"
        ]
    }
    
    for category, items in features.items():
        print(f"📊 {category}:")
        for item in items:
            print(f"   • {item}")
        print()
    
    print("🚀 Quick Start Commands:")
    print("   python run_complete_training.py")
    print("   python main.py --batch_size 128 --epochs 20 --plot_training --plot_evaluation")
    print("   python example_visualization.py")

def main():
    """Main function to run complete training."""
    
    print("CIFAR-10 Complete Training System")
    print("=" * 40)
    print()
    print("This system provides comprehensive CIFAR-10 training with:")
    print("• Optimized hyperparameters")
    print("• Model snapshots and resuming")
    print("• Complete visualization suite")
    print("• Detailed metrics and analysis")
    print()
    
    run_complete_training()
    return 0
    
if __name__ == "__main__":
    main()
    

