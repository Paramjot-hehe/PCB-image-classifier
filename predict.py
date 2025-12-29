import torch
from PIL import Image
from torchvision import transforms
from model import get_pcb_model

def predict_pcb_defect(img_path):
    device = "cuda"
    classes = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    
    # 1. Load Model
    model = get_pcb_model(len(classes))
    model.load_state_dict(torch.load('pcb_model_final.pth'))
    model.to(device).eval()

    # 2. Prepare Image (Must match the 224x224 training size)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

    print(f"Result: {classes[predicted.item()]} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    # Put a path to a single 600x600 patch here to test
    predict_pcb_defect(r"C:\Users\poram\OneDrive\Pictures\Screenshots\test_img_pcb.png")