import os
import tarfile
import urllib.request

import torch
from config import BATCH_SIZE, DATA_ROOT
from rich.progress import (
    BarColumn, DownloadColumn, Progress, TextColumn,
    TimeRemainingColumn, TransferSpeedColumn,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download_with_progress(url, dest, description):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(description, total=None)

        def reporthook(block_count, block_size, total_size):
            if total_size > 0 and progress.tasks[task_id].total is None:
                progress.update(task_id, total=total_size)
            downloaded = block_count * block_size
            progress.update(task_id, completed=min(downloaded, total_size) if total_size > 0 else downloaded)

        urllib.request.urlretrieve(url, dest, reporthook)


def _base_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


def prepare_data():
    if not os.path.exists(os.path.join(DATA_ROOT, 'cifar-10-batches-py')):
        os.makedirs(DATA_ROOT, exist_ok=True)
        tar_dest = os.path.join(DATA_ROOT, 'cifar-10-python.tar.gz')
        if not os.path.exists(tar_dest):
            _download_with_progress(CIFAR10_URL, tar_dest, "Downloading CIFAR-10")
        with tarfile.open(tar_dest, 'r:gz') as tar:
            tar.extractall(DATA_ROOT)

    dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=False,
        transform=_base_transform()
    )
    if not isinstance(dataset.data, torch.Tensor):
        dataset.data = torch.from_numpy(dataset.data)
    if not isinstance(dataset.targets, torch.Tensor):
        dataset.targets = torch.tensor(dataset.targets)
    dataset.data.share_memory_()
    dataset.targets.share_memory_()
    return dataset


class _FoldDataset(Dataset):
    def __init__(self, data, targets, indices, transform):
        self._data = data
        self._targets = targets
        self._indices = indices
        self._transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample_idx = int(self._indices[idx])
        img = self._data[sample_idx].numpy()
        target = int(self._targets[sample_idx])
        if self._transform is not None:
            img = self._transform(img)
        return img, target


def create_data_loaders(dataset, train_indices, val_indices, train_augment=True):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]) if train_augment else _base_transform()

    train_dataset = _FoldDataset(dataset.data, dataset.targets, train_indices, train_transform)
    val_dataset = _FoldDataset(dataset.data, dataset.targets, val_indices, _base_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader