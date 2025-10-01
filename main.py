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
from torch.optim.lr_scheduler import OneCycleLR
from torchsummary import summary

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


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training")
    parser.add_argument("--model", type=str, default="cifar", help="Model to use: cifar|model1|model2|model3")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.1)
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
    
    args = parser.parse_args()
    set_seed(42)
    device = get_device(prefer_cuda=not args.no_cuda)

    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=2,
        pin_memory=True,
        shuffle_train=True,
        model_name=args.model,
        cache_transforms=args.cache_transforms,
        cache_dir=args.cache_dir,
        cache_namespace=args.cache_namespace,
    )

    model = build_model(args.model, device)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    # CIFAR-10 input shape
    summary(model, input_size=(3, 32, 32))
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = OneCycleLR(optimizer, max_lr=0.1,
                            steps_per_epoch=len(train_loader),
                            epochs=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Initialize training state
    start_epoch = 1
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    best_test_acc = 0.0

    # Resume from snapshot if specified
    if args.resume_from:
        start_epoch, train_losses, train_acc, test_losses, test_acc = load_snapshot(
            args.resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        best_test_acc = max(test_acc) if test_acc else 0.0
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}")
        tr_loss, tr_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, device, test_loader, criterion)
        scheduler.step(te_loss)   # pass validation loss

        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        test_losses.append(te_loss)
        test_acc.append(te_acc)

        print(
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%"
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


if __name__ == "__main__":
    main()
