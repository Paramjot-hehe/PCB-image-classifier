from dataset import get_pcb_loaders

# Set your data directory path
data_path = r'D:\pcb_ai_project\train\train' 

train_loader, val_loader, classes = get_pcb_loaders(data_path)

print(f"Success! Found {len(classes)} classes: {classes}")
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}") # Should be [16, 3, 224, 224]
print(f"Labels in this batch: {labels}")