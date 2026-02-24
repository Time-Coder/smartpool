import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DATA_ROOT, BATCH_SIZE


def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_exists = (
        os.path.exists(os.path.join(DATA_ROOT, 'MNIST', 'raw')) and
        os.path.exists(os.path.join(DATA_ROOT, 'MNIST', 'processed'))
    )
    
    return datasets.MNIST(
        root=DATA_ROOT, 
        train=True, 
        download=not mnist_exists, 
        transform=transform
    )


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