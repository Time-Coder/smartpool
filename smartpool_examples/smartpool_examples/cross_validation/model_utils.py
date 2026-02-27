import torch
import torch.nn as nn
import torch.optim as optim

import traceback
from dataclasses import dataclass

from .config import LEARNING_RATE, EPOCHS
from .data_utils import create_data_loaders

from smartpool import move_optimizer_to


@dataclass
class TrainingResult:
    fold_idx:int
    model_name:str
    val_accuracy:float


@dataclass
class ProgressInfo:
    model_name:str
    fold_idx:int
    epoch:int
    batch:int
    total_batches:int
    device:str
    avg_loss:float
    val_accuracy:float


@dataclass
class ErrorInfo:
    exception:BaseException
    traceback:str

def train_single_fold(fold_idx, model_class, train_indices, val_indices, dataset, progress_queue, device=None):
    try:
        return _train_single_fold(fold_idx, model_class, train_indices, val_indices, dataset, progress_queue, device)
    except BaseException as e:
        error_info = ErrorInfo(e, traceback.format_exc())
        progress_queue.put(error_info)
        raise e

def _train_single_fold(fold_idx, model_class, train_indices, val_indices, dataset, progress_queue, device):
    train_loader, val_loader = create_data_loaders(dataset, train_indices, val_indices)
    num_batches = len(train_loader)
    model = model_class()
    
    if hasattr(train_single_fold, "device"):
        device = train_single_fold.device()

    old_device = device
    model.to(device, non_blocking=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    initial_progress = ProgressInfo(
        model_name=model_class.__name__,
        fold_idx=fold_idx,
        epoch=1,
        batch=0,
        total_batches=num_batches,
        device=device,
        avg_loss=0.0,
        val_accuracy=0.0
    )
    progress_queue.put(initial_progress)

    last_val_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if hasattr(train_single_fold, "device"):
                device = train_single_fold.device()

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
                    
            if device != old_device:
                model.to(device, non_blocking=True)
                move_optimizer_to(optimizer, device)
                old_device = device
                    
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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
                val_accuracy=last_val_accuracy
            )
            progress_queue.put(progress_info)
        
        model.eval()
        correct = 0
        total = 0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                if hasattr(train_single_fold, "device"):
                    device = train_single_fold.device()

                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                        
                if device != old_device:
                    model.to(device, non_blocking=True)
                    move_optimizer_to(optimizer, device)
                    old_device = device
                        
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_accuracy = correct / total
        last_val_accuracy = val_accuracy
        model.train()
        
        final_progress = ProgressInfo(
            model_name=model_class.__name__,
            fold_idx=fold_idx,
            epoch=epoch + 1,
            batch=num_batches,
            total_batches=num_batches,
            device=device,
            avg_loss=epoch_loss / num_batches,
            val_accuracy=val_accuracy
        )
        progress_queue.put(final_progress)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            if hasattr(train_single_fold, "device"):
                device = train_single_fold.device()

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if device != old_device:
                model.to(device, non_blocking=True)
                move_optimizer_to(optimizer, device)
                old_device = device

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    val_accuracy = correct / total
    
    return TrainingResult(fold_idx, model_class.__name__, val_accuracy)
