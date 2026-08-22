import os
import traceback
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from augmentation import augment, mixup_data, mixup_criterion
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, N_CLASSES
from data_utils import load_features
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from smartpool import best_device, move_optimizer_to


@dataclass
class TrainingResult:
    fold_idx: int
    model_name: str
    val_accuracy: float


@dataclass
class ProgressInfo:
    model_name: str
    fold_idx: int
    epoch: int
    batch: int
    total_batches: int
    device: str
    avg_loss: float
    val_accuracy: float
    pid: int


@dataclass
class ErrorInfo:
    exception: BaseException
    traceback: str


class _AugmentedDataset(TensorDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    def __getitem__(self, idx):
        feat, target = super().__getitem__(idx)
        feat_np = feat.squeeze(0).numpy()
        feat_np = augment(feat_np)
        return torch.from_numpy(feat_np).unsqueeze(0), target


def train_single_fold(fold_idx, model_class, train_data, train_targets, val_data, val_targets, progress_queue, user_device=None):
    try:
        return _train_single_fold(fold_idx, model_class, train_data, train_targets, val_data, val_targets, progress_queue, user_device)
    except Exception as e:
        error_info = ErrorInfo(e, traceback.format_exc())
        progress_queue.put(error_info)
        raise e


def _train_single_fold(fold_idx, model_class, train_data, train_targets, val_data, val_targets, progress_queue, user_device):
    model = model_class(n_classes=N_CLASSES)
    if user_device is None:
        device = torch.device(best_device().torch_device)
    else:
        device = torch.device(user_device)

    train_dataset = _AugmentedDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == 'cuda'))
    num_batches = len(train_loader)

    old_device = device
    model.to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    initial_progress = ProgressInfo(
        model_name=model_class.__name__,
        fold_idx=fold_idx,
        epoch=1,
        batch=0,
        total_batches=num_batches,
        device=device,
        avg_loss=0.0,
        val_accuracy=0.0,
        pid=os.getpid()
    )
    progress_queue.put(initial_progress)

    last_val_accuracy = 0.0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if user_device is None:
                device = torch.device(best_device().torch_device)

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if device != old_device:
                model.to(device, non_blocking=True)
                move_optimizer_to(optimizer, device)
                old_device = device

            data, target_a, target_b, lam = mixup_data(data, target, alpha=0.2)

            optimizer.zero_grad()
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_info = ProgressInfo(
                model_name=model_class.__name__,
                fold_idx=fold_idx,
                epoch=epoch + 1,
                batch=batch_idx + 1,
                total_batches=num_batches,
                device=device,
                avg_loss=epoch_loss / (batch_idx + 1),
                val_accuracy=last_val_accuracy,
                pid=os.getpid()
            )
            progress_queue.put(progress_info)

        model.eval()
        correct = 0
        total = 0
        val_accuracy = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        val_accuracy = correct / total
        last_val_accuracy = val_accuracy
        scheduler.step()
        model.train()

        final_progress = ProgressInfo(
            model_name=model_class.__name__,
            fold_idx=fold_idx,
            epoch=epoch + 1,
            batch=num_batches,
            total_batches=num_batches,
            device=device,
            avg_loss=epoch_loss / num_batches,
            val_accuracy=val_accuracy,
            pid=os.getpid()
        )
        progress_queue.put(final_progress)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    val_accuracy = correct / total

    return TrainingResult(fold_idx, model_class.__name__, val_accuracy)