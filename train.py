import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


def train_epoch(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
):
    """Train the model for one epoch with optional AMP and gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, device, test_loader, criterion, use_amp: bool = False):
    """Evaluate the model on test set (optionally with AMP)."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_val = criterion(output, target).item()
            test_loss += loss_val
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        # Intentionally no per-batch logging; epoch summary is printed in main.py

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc

