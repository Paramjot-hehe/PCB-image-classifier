import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import get_pcb_loaders
from model import get_pcb_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def plot_loss_surface():
    device = "cuda"
    # 1. Load your trained model and data
    # Note: Use a small batch for speed during visualization
    train_loader, _, class_names = get_pcb_loaders(r'D:\pcb_ai_project\train_600x_jitter', batch_size=32)
    model = get_pcb_model(len(class_names))
    model.load_state_dict(torch.load('pcb_model_final.pth'))
    model.to(device).eval()

    # 2. Pick two random directions (X and Y axes) to explore the 'valley'
    params = [p for p in model.fc.parameters() if p.requires_grad]
    dir_x = [torch.randn_like(p) for p in params]
    dir_y = [torch.randn_like(p) for p in params]

    # 3. Create a grid around your final weights
    steps = 30
    r = 1  # Range: how far to 'walk' away from the optimum
    x_coords = np.linspace(-r, r, steps)
    y_coords = np.linspace(-r, r, steps)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.zeros((steps, steps))

    # 4. Calculate 'Potential Energy' (Loss) at each point
    criterion = nn.CrossEntropyLoss()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    print("Mapping the potential surface (RTX 5050 at work)...")
    orig_weights = [p.data.clone() for p in params]

    for i in range(steps):
        for j in range(steps):
            for p, w, dx, dy in zip(params, orig_weights, dir_x, dir_y):
                p.data = w + X[i,j]*dx + Y[i,j]*dy
            
            with torch.no_grad():
                Z[i,j] = criterion(model(images), labels).item()

    # 5. Render the 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')
    
    ax.set_title('PCB Model: 3D Potential Surface (Loss Landscape)')
    ax.set_zlabel('Loss (Potential)')
    plt.colorbar(surf, ax=ax, shrink=0.5)
    plt.show()

if __name__ == "__main__":
    plot_loss_surface()