import os
import urllib.request

from .config import BATCH_SIZE, DATA_ROOT
from rich.progress import (
    BarColumn, DownloadColumn, Progress, TextColumn,
    TimeRemainingColumn, TransferSpeedColumn,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_URLS = [
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
]


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


def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_raw = os.path.join(DATA_ROOT, 'MNIST', 'raw')
    required_files = [f for _, f in MNIST_URLS]
    
    # 检查 raw 目录下是否已有全部 4 个 gz 文件
    all_exist = all(os.path.exists(os.path.join(mnist_raw, f)) for f in required_files)

    if not all_exist:
        os.makedirs(mnist_raw, exist_ok=True)
        for url, filename in MNIST_URLS:
            dest = os.path.join(mnist_raw, filename)
            if not os.path.exists(dest):
                _download_with_progress(url, dest, f"Downloading {filename}")

    dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform
    )
    dataset.data.share_memory_()
    dataset.targets.share_memory_()
    return dataset


def create_data_loaders(dataset, train_indices, val_indices):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader
