import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# 1. SETUP & DIRECTORIES
RESULTS_DIR = "results"
SUBDIRS = ["validation", "test", "qualitative"]
for d in SUBDIRS:
    os.makedirs(os.path.join(RESULTS_DIR, d), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_pcb_project():
    # 2. LOAD CLASSES & DATA
    # Assuming your data is in 'curated_patches' or a similar ImageFolder structure
    data_dir = r'D:\pcb_ai_project\train_600x_jitter' 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    
    # Splitting exactly as we did in training for consistency
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3. LOAD TRAINED MODEL
    # Import your get_pcb_model from your original script
    from model import get_pcb_model 
    model = get_pcb_model(num_classes)
    model.load_state_dict(torch.load('finetunning_approach/pcb_model_final.pth')) # Your fine-tuned weights
    model.to(device).eval()

    def get_predictions(loader):
        all_preds = []
        all_labels = []
        all_probs = []
        all_inputs = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Predicting"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                # Store a few images for qualitative analysis
                if len(all_inputs) < 100:
                    all_inputs.extend(images.cpu())
                    
        return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_inputs

    # 4. RUN EVALUATION
    for name, loader in [("validation", val_loader), ("test", test_loader)]:
        print(f"\n--- Evaluating {name.upper()} SET ---")
        y_true, y_pred, y_probs, sample_imgs = get_predictions(loader)
        
        # A. Classification Report
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f"{RESULTS_DIR}/{name}/classification_report.csv")
        
        # B. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{RESULTS_DIR}/{name}/confusion_matrix_raw.png", dpi=300)
        plt.close()

        # C. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
            plt.plot(recall, precision, label=f'{classes[i]}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {name}')
        plt.legend()
        plt.savefig(f"{RESULTS_DIR}/{name}/pr_curves.png", dpi=300)
        plt.close()

    # 5. QUALITATIVE ANALYSIS (Visual Grids)
    # 5. QUALITATIVE ANALYSIS (Visual Grids)
    print("\nGenerating Qualitative Grids...")
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    def save_grid(indices, all_imgs, labels, preds, probs, type_name):
        if len(indices) == 0:
            print(f"Skipping {type_name} grid: No samples found.")
            return
            
        num_samples = min(len(indices), 12)
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            idx = indices[i]
            plt.subplot(3, 4, i+1)
            
            # Undo normalization for display
            img = inv_normalize(all_imgs[idx]).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            color = 'green' if labels[idx] == preds[idx] else 'red'
            
            title_text = (f"True: {classes[labels[idx]]}\n"
                          f"Pred: {classes[preds[idx]]}\n"
                          f"Conf: {probs[idx][preds[idx]]:.2f}")
            
            plt.title(title_text, color=color, fontsize=10)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/qualitative/{type_name}_examples.png")
        plt.close()

    # Identify indices for Correct and Error samples
    correct_indices = np.where(y_true == y_pred)[0]
    error_indices = np.where(y_true != y_pred)[0]

    # Save the grids
    save_grid(correct_indices, sample_imgs, y_true, y_pred, y_probs, "correct")
    save_grid(error_indices, sample_imgs, y_true, y_pred, y_probs, "misclassified")
    # Correct samples
    correct_idx = np.where(y_true == y_pred)[0][:12]
    save_grid([sample_imgs[i] for i in correct_idx], y_true[correct_idx], y_pred[correct_idx], y_probs[correct_idx], "correct")
    
    # Error samples
    error_idx = np.where(y_true != y_pred)[0][:12]
    if len(error_idx) > 0:
        save_grid([sample_imgs[i] for i in error_idx], y_true[error_idx], y_pred[error_idx], y_probs[error_idx], "misclassified")

    print(f"\nALL EVALUATION COMPLETE. Files saved in /{RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_pcb_project()