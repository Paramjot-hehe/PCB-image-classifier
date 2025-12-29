import torch
import cv2
import numpy as np
from model import get_pcb_model
from torchvision import transforms
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def start_ui_inspector(image_path):
    device = "cuda"
    classes = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    
    # 1. Load Model
    model = get_pcb_model(len(classes))
    model.load_state_dict(torch.load('pcb_model_final.pth'))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Image
    original_img = cv2.imread(image_path)
    if original_img is None: return
    h, w, _ = original_img.shape
    
    print("Scanning board... please wait.")
    detections = []
    patch_size = 600
    stride = 300 # Overlap helps ensure we don't miss anything

    # 3. Scanning (Hidden from user)
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch_cv = original_img[y:y+patch_size, x:x+patch_size]
            patch_rgb = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB)
            patch_pil = Image.fromarray(patch_rgb)
            
            input_tensor = transform(patch_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            if conf.item() > 0.96: # High confidence threshold
                detections.append({
                    'box': (x, y, x+patch_size, y+patch_size),
                    'label': classes[pred.item()],
                    'conf': conf.item(),
                    'patch': patch_cv.copy()
                })

    # 4. UI Loop
    if not detections:
        print("No defects found.")
        return

    current_idx = 0
    while True:
        det = detections[current_idx]
        
        # Create a display for the specific patch
        display_patch = det['patch']
        text = f"Defect {current_idx + 1}/{len(detections)}: {det['label']} ({det['conf']*100:.1f}%)"
        
        # Draw info bar at the top
        header = np.zeros((70, display_patch.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Combine header and patch
        ui_frame = np.vstack((header, display_patch))
        
        # Add navigation hints at the bottom
        cv2.putText(ui_frame, "[N] Next  [P] Prev  [Q] Quit", (10, ui_frame.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("PCB AI Inspector", ui_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            current_idx = (current_idx + 1) % len(detections)
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(detections)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
if __name__ == "__main__":
    start_ui_inspector(r"D:\pcb_ai_project\train\train\Spur\08_spur_08.jpg")