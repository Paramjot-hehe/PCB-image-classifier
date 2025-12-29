import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the logic we wrote in previous steps
from dataset import get_pcb_loaders
from model import get_pcb_model

# --- CONFIGURATION ---
DATA_DIR = r'D:\pcb_ai_project\train_600x_jitter'  # Path to your new high-quality data
BATCH_SIZE = 32                 # RTX 5050 can handle 32 easily with 600px patches
EPOCHS = 15
LEARNING_RATE = 0.0001          # Slow and steady for fine-tuning

def train_engine():
    # 1. Load Data
    train_loader, val_loader, class_names = get_pcb_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_pcb_model(len(class_names)).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print(f"Starting Training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

    # 5. FINAL EVALUATION (The Metrics Part)
    print("\nCalculating Final Metrics...")
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # 6. OUTPUT RESULTS
    print("\n" + "="*30)
    print("   PCB DEFECT CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save the weights
    torch.save(model.state_dict(), 'pcb_model_final.pth')
    print("\nModel saved as pcb_model_final.pth")

if __name__ == "__main__":
    train_engine()