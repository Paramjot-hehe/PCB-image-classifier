import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random  # Added for random jitter

# Configuration
INPUT_DIR = r"D:\pcb_ai_project\train\train"
OUTPUT_DIR = r"D:\pcb_ai_project\train_600x_jitter"
CROP_SIZE = 600
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 800

# Augmentation Settings
NUM_SNAPS = 3        # How many images to take per click
JITTER_RANGE = 270    # Max pixels to shift left/right/up/down

class PCBCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"PCB Defect Cropper (Auto-Jitter x{NUM_SNAPS})")
        
        self.image_paths = []
        self.current_idx = 0
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.category = ""
        self.filename = ""

        self.load_image_list()

        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<Button-1>", self.on_click)
        root.bind("<space>", self.next_image)
        root.bind("<Escape>", self.close_app)
        root.bind("<Right>", self.next_image)

        if self.image_paths:
            self.load_current_image()
        else:
            messagebox.showerror("Error", f"No images found in {INPUT_DIR}")
            root.destroy()

    def load_image_list(self):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for root_dir, dirs, files in os.walk(INPUT_DIR):
            for file in files:
                if any(file.lower().endswith(ext.replace("*", "")) for ext in extensions):
                    self.image_paths.append(os.path.join(root_dir, file))
        print(f"Found {len(self.image_paths)} images.")

    def load_current_image(self):
        if self.current_idx >= len(self.image_paths):
            self.status_label.config(text="End of dataset.")
            return

        full_path = self.image_paths[self.current_idx]
        rel_path = os.path.relpath(full_path, INPUT_DIR)
        self.category = os.path.dirname(rel_path)
        self.filename = os.path.basename(full_path)

        self.original_image = cv2.imread(full_path)
        if self.original_image is None:
            self.next_image()
            return

        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape

        scale_w = DISPLAY_MAX_WIDTH / w
        scale_h = DISPLAY_MAX_HEIGHT / h
        self.scale_factor = min(scale_w, scale_h, 1.0)

        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)

        resized_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.pil_image = Image.fromarray(resized_image)
        self.display_image = ImageTk.PhotoImage(self.pil_image)

        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)
        
        self.root.geometry(f"{new_w}x{new_h + 30}")
        self.status_label.config(text=f"Image {self.current_idx + 1}/{len(self.image_paths)}: {self.category}/{self.filename}")

    def on_click(self, event):
        if self.original_image is None: return

        # 1. Map Display Click to Original Coordinates
        click_x = int(event.x / self.scale_factor)
        click_y = int(event.y / self.scale_factor)
        
        img_h, img_w, _ = self.original_image.shape
        save_dir = os.path.join(OUTPUT_DIR, self.category)
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(self.filename)[0]

        # Loop to create multiple jittered crops
        for i in range(NUM_SNAPS):
            # 2. Calculate Random Jitter
            # First snap is exactly where you clicked, others are random
            if i == 0:
                offset_x, offset_y = 0, 0
            else:
                offset_x = random.randint(-JITTER_RANGE, JITTER_RANGE)
                offset_y = random.randint(-JITTER_RANGE, JITTER_RANGE)

            # Center of the new crop
            center_x = click_x + offset_x
            center_y = click_y + offset_y

            # 3. Calculate Bounds
            half_crop = CROP_SIZE // 2
            x_start = center_x - half_crop
            y_start = center_y - half_crop
            
            # 4. Edge Clamping (Ensure box stays inside image)
            if x_start < 0: x_start = 0
            if x_start + CROP_SIZE > img_w: x_start = img_w - CROP_SIZE
            
            if y_start < 0: y_start = 0
            if y_start + CROP_SIZE > img_h: y_start = img_h - CROP_SIZE

            x_end = x_start + CROP_SIZE
            y_end = y_start + CROP_SIZE

            # 5. Crop and Save
            crop = self.original_image[y_start:y_end, x_start:x_end]
            
            # Save with unique name including coordinate and jitter index
            save_name = f"{base_name}_x{x_start}_y{y_start}_j{i}.jpg"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, crop)

            # 6. UI Feedback (Draw rectangle)
            disp_x1 = x_start * self.scale_factor
            disp_y1 = y_start * self.scale_factor
            disp_x2 = x_end * self.scale_factor
            disp_y2 = y_end * self.scale_factor

            # Use slightly different colors or thickness to differentiate
            color = "red" if i == 0 else "orange"
            self.canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline=color, width=2)

        print(f"Generated {NUM_SNAPS} crops for {base_name}")
        self.canvas.create_text(click_x * self.scale_factor, (click_y * self.scale_factor) - 10, 
                                text=f"Saved x{NUM_SNAPS}", fill="red", anchor=tk.SW)

    def next_image(self, event=None):
        self.current_idx += 1
        if self.current_idx < len(self.image_paths):
            self.canvas.delete("all")
            self.load_current_image()
        else:
            messagebox.showinfo("Done", "All images processed!")
            self.root.quit()

    def close_app(self, event=None):
        self.root.quit()

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created {INPUT_DIR}.")
    
    root = tk.Tk()
    app = PCBCropperApp(root)
    root.mainloop()