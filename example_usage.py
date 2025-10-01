#!/usr/bin/env python3
"""
Example usage of the CIFAR-10 training script with snapshot functionality.

This script demonstrates how to:
1. Train a model with automatic snapshots
2. Resume training from a snapshot
3. Use different snapshot strategies
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and print the output."""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("-" * 50)
    return result.returncode == 0

def main():
    print("CIFAR-10 Training with Snapshot Examples")
    print("=" * 50)
    
    # Example 1: Basic training with snapshots every 3 epochs
    print("\n1. Basic training with snapshots every 3 epochs:")
    cmd1 = [
        sys.executable, "main.py",
        "--epochs", "10",
        "--snapshot_freq", "3",
        "--snapshot_dir", "./snapshots_example"
    ]
    run_command(cmd1)
    
    # Example 2: Training with best model saving
    print("\n2. Training with best model saving (only save when accuracy improves):")
    cmd2 = [
        sys.executable, "main.py",
        "--epochs", "10",
        "--save_best",
        "--snapshot_dir", "./snapshots_best"
    ]
    run_command(cmd2)
    
    # Example 3: Resume training from a snapshot
    print("\n3. Resume training from a snapshot:")
    # First, check if we have any snapshots
    snapshot_dir = "./snapshots_example"
    if os.path.exists(snapshot_dir):
        snapshots = [f for f in os.listdir(snapshot_dir) if f.endswith('.pth')]
        if snapshots:
            latest_snapshot = os.path.join(snapshot_dir, snapshots[-1])
            print(f"Resuming from: {latest_snapshot}")
            cmd3 = [
                sys.executable, "main.py",
                "--epochs", "15",
                "--resume_from", latest_snapshot,
                "--snapshot_freq", "2"
            ]
            run_command(cmd3)
        else:
            print("No snapshots found to resume from.")
    else:
        print("Snapshot directory not found.")
    
    print("\nExample usage completed!")

if __name__ == "__main__":
    main()

