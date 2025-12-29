import torch
import cv2
import numpy as np
from model import get_pcb_model
from torchvision import transforms
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output): self.activations = output
    def save_gradients(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_grads[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        return heatmap / (np.max(heatmap) + 1e-8)

def explore_board_with_ai(image_path):
    device = "cuda"
    classes = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    
    # Load Model
    model = get_pcb_model(len(classes))
    model.load_state_dict(torch.load(r'D:\pcb_ai_project\finetunning_approach\pcb_model_final.pth'))
    model.to(device).eval()
    cam = GradCAM(model, model.layer4[-1])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_img = cv2.imread(image_path)
    h, w, _ = full_img.shape
    x, y, size = 0, 0, 400  # Starting position and window size

    print("Controls: WASD to move, Q to quit.")

    while True:
        # 1. Extract and Prep Patch
        patch = full_img[y:y+size, x:x+size]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        input_tensor = transform(Image.fromarray(patch_rgb)).unsqueeze(0).to(device)

        # 2. Get AI Logic
        output = model(input_tensor)
        conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
        heatmap = cam.generate(input_tensor, pred.item())

        # 3. Create Heatmap Overlay
        heatmap_resized = cv2.resize(heatmap, (size, size))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        result_patch = cv2.addWeighted(patch, 0.6, heatmap_color, 0.4, 0)

        # 4. Create UI: Full Board (Left) and AI Zoom (Right)
        # Scale the big board down so it fits on screen
        scale = 800 / w
        board_view = cv2.resize(full_img, (800, int(h * scale)))
        # Draw the "Scanning Box" on the board view
        cv2.rectangle(board_view, (int(x*scale), int(y*scale)), 
                      (int((x+size)*scale), int((y+size)*scale)), (0, 255, 0), 2)

        # Stack views side-by-side
        # Ensure result_patch is same height as board_view for display
        zoom_view = cv2.resize(result_patch, (board_view.shape[0], board_view.shape[0]))
        
        info_text = f"AI Thinks: {classes[pred.item()]} ({conf.item()*100:.1f}%)"
        cv2.putText(zoom_view, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        combined_ui = np.hstack((board_view, zoom_view))
        cv2.imshow("PCB AI Explorer: Move with WASD", combined_ui)

        # 5. Handle Keyboard
        key = cv2.waitKey(1) & 0xFF
        step = 50
        if key == ord('w') and y > 0: y -= step
        elif key == ord('s') and y < h - size: y += step
        elif key == ord('a') and x > 0: x -= step
        elif key == ord('d') and x < w - size: x += step
        elif key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test on one of your 600x600 patches
    explore_board_with_ai(r"D:\pcb_ai_project\train\train\Short\05_short_05.jpg")