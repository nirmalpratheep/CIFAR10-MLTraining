import argparse
import importlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime

from preprocess import get_data_loaders
from train import train_epoch, evaluate
from torch.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
from torchsummary import summary
from visualization import (
    create_training_summary, create_evaluation_summary, 
    evaluate_model_comprehensive, TrainingVisualizer
)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensures reproducibility for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name: str, device: torch.device):
    module = importlib.import_module("model_cifar")
    return module.build_model(device)


def save_snapshot(model, optimizer, scheduler, epoch, train_losses, train_acc, test_losses, test_acc, 
                 snapshot_dir: str, model_name: str):
    """Save model snapshot with training state."""
    os.makedirs(snapshot_dir, exist_ok=True)
    
    snapshot = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'train_acc': train_acc,
        'test_losses': test_losses,
        'test_acc': test_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    snapshot_path = os.path.join(snapshot_dir, f"{model_name}_epoch_{epoch}.pth")
    torch.save(snapshot, snapshot_path)
    print(f"Snapshot saved: {snapshot_path}")
    return snapshot_path


def load_snapshot(snapshot_path: str, model, optimizer, scheduler, device):
    """Load model snapshot and return training state."""
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    
    print(f"Loading snapshot: {snapshot_path}")
    snapshot = torch.load(snapshot_path, map_location=device)
    
    model.load_state_dict(snapshot['model_state_dict'])
    optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    scheduler.load_state_dict(snapshot['scheduler_state_dict'])
    
    epoch = snapshot['epoch']
    train_losses = snapshot.get('train_losses', [])
    train_acc = snapshot.get('train_acc', [])
    test_losses = snapshot.get('test_losses', [])
    test_acc = snapshot.get('test_acc', [])
    
    print(f"Resumed from epoch {epoch}")
    return epoch, train_losses, train_acc, test_losses, test_acc


def get_lr_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """Calculate learning rate warmup factor."""
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def apply_warmup_lr(optimizer, base_lr: float, warmup_factor: float):
    """Apply warmup learning rate to optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * warmup_factor


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training")
    parser.add_argument("--model", type=str, default="cifar", help="Model to use: cifar|model1|model2|model3")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs (disabled)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "onecycle"], help="Learning rate scheduler")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_transforms", action="store_true", help="Cache transformed samples to disk")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to store cached samples")
    parser.add_argument("--cache_namespace", type=str, default=None, help="Namespace for cache (change if transforms change)")
    parser.add_argument("--no_cuda", action="store_true")
    
    # Snapshot-related arguments
    parser.add_argument("--snapshot_dir", type=str, default="./snapshots", help="Directory to save model snapshots")
    parser.add_argument("--snapshot_freq", type=int, default=5, help="Save snapshot every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to snapshot file to resume training from")
    parser.add_argument("--save_best", action="store_true", help="Save snapshot only when test accuracy improves")
    
    # Visualization arguments
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Directory to save plots and visualizations")
    parser.add_argument("--plot_training", action="store_true", help="Generate training curves and plots")
    parser.add_argument("--plot_evaluation", action="store_true", help="Generate evaluation plots (confusion matrix, metrics)")
    parser.add_argument("--plot_freq", type=int, default=10, help="Generate plots every N epochs")
    parser.add_argument("--no_plots", action="store_true", help="Disable all plotting")
    
    # Training features
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (AMP)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()
    set_seed(42)
    device = get_device(prefer_cuda=not args.no_cuda)

    # Optimize data loading based on device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    pin_memory = use_cuda and args.cache_transforms  # Only use pin_memory with caching for stability
    num_workers = min(args.num_workers, 2) if not use_cuda else args.num_workers
    
    print(f"Data loading: num_workers={num_workers}, pin_memory={pin_memory}, use_cuda={use_cuda}")
    print(f"Caching: {args.cache_transforms} (dir: {args.cache_dir})")
    
    print("Loading datasets...")
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_train=True,
        model_name=args.model,
        cache_transforms=args.cache_transforms,
        cache_dir=args.cache_dir,
        cache_namespace=args.cache_namespace,
    )
    
    # Test data loading
    print("Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"Data loading successful! Batch shape: {test_batch[0].shape}, labels: {test_batch[1].shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return 1

    model = build_model(args.model, device)
    print(f"Device: {device}")
    print(f"Model loaded, starting training...")
    # CIFAR-10 input shape
    summary(model, input_size=(3, 32, 32))
    
    # Improved optimizer with weight decay
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay, nesterov=True)
    
    # Improved scheduler setup
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                              steps_per_epoch=len(train_loader),
                              epochs=args.epochs, pct_start=0.3)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing

    # Initialize training state
    start_epoch = 1
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    learning_rates = []
    best_test_acc = 0.0

    # Resume from snapshot if specified
    if args.resume_from:
        start_epoch, train_losses, train_acc, test_losses, test_acc = load_snapshot(
            args.resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        best_test_acc = max(test_acc) if test_acc else 0.0
        print(f"Resuming training from epoch {start_epoch}")

    scaler = GradScaler('cuda', enabled=args.amp)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # No warmup - use scheduler directly
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        print("Starting training...")
        tr_loss, tr_acc = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            scaler=scaler if args.amp else None,
            use_amp=args.amp,
            max_grad_norm=args.max_grad_norm,
        )
        print("Starting evaluation...")
        te_loss, te_acc = evaluate(model, device, test_loader, criterion, use_amp=args.amp)
        
        # Step scheduler every epoch
        if args.scheduler == "onecycle":
            scheduler.step()  # OneCycleLR steps per batch
        else:
            scheduler.step()  # Other schedulers step per epoch

        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        test_losses.append(te_loss)
        test_acc.append(te_acc)

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save snapshot based on frequency or best accuracy
        should_save = False
        if args.save_best and te_acc > best_test_acc:
            best_test_acc = te_acc
            should_save = True
            print(f"New best test accuracy: {te_acc:.2f}%")
        elif not args.save_best and epoch % args.snapshot_freq == 0:
            should_save = True

        if should_save:
            save_snapshot(
                model, optimizer, scheduler, epoch, train_losses, train_acc, 
                test_losses, test_acc, args.snapshot_dir, args.model
            )
        
        # Generate plots if requested
        if not args.no_plots and (args.plot_training or args.plot_evaluation):
            if epoch % args.plot_freq == 0 or epoch == args.epochs:
                print(f"\n📊 Generating plots for epoch {epoch}...")
                
                # Create plots directory
                os.makedirs(args.plot_dir, exist_ok=True)
                
                # Generate training curves
                if args.plot_training:
                    create_training_summary(
                        train_losses, train_acc, test_losses, test_acc, 
                        learning_rates, args.plot_dir
                    )
                
                # Generate evaluation plots
                if args.plot_evaluation:
                    create_evaluation_summary(
                        model, device, test_loader, criterion, args.plot_dir
                    )
                
                print("✅ Plots generated successfully!")
    
    # Final comprehensive evaluation and plotting
    if not args.no_plots:
        print("\n🎯 Generating final comprehensive evaluation...")
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # Final training summary
        if args.plot_training:
            create_training_summary(
                train_losses, train_acc, test_losses, test_acc, 
                learning_rates, args.plot_dir
            )
        
        # Final evaluation summary
        if args.plot_evaluation:
            metrics, eval_results = create_evaluation_summary(
                model, device, test_loader, criterion, args.plot_dir
            )
            
            print(f"\n📈 Final Performance Summary:")
            print(f"   • Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"   • Top-3 Accuracy: {metrics.get('top_3_accuracy', 'N/A'):.4f}")
            print(f"   • Top-5 Accuracy: {metrics.get('top_5_accuracy', 'N/A'):.4f}")
            print(f"   • Macro F1-Score: {metrics['f1_macro']:.4f}")
            print(f"   • Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        
        print(f"\n📁 All plots and reports saved in: {args.plot_dir}")
        print("🎉 Training completed with comprehensive analysis!")


if __name__ == "__main__":
    main()
