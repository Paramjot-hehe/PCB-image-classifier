import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_pcb_loaders(data_dir, batch_size=16, train_split=0.8):
    # 1. THE DATA TRANSFORMS
    # Training needs 'Augmentation' to prevent the model from memorizing.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images
        transforms.RandomVerticalFlip(p=0.5),   # PCBs can be viewed from any side
        transforms.RandomRotation(15),          # Slight rotations help with real-world drift
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Lighting differences
        transforms.ToTensor(),                  # Convert to PyTorch Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet Stats
    ])

    # Validation/Testing should NOT be augmented, only resized and normalized.
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. LOAD THE RAW DATA
    # ImageFolder automatically uses folder names as labels (e.g., 'Short' = 0, 'Spur' = 1)
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # 3. SPLIT INTO TRAIN AND VALIDATION (Reproducible)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # manual_seed(42) ensures the 'random' split is the same every time you run it
    train_ds, val_ds = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Attach the transforms to the specific splits
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    # 4. CREATE THE DATALOADERS
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes