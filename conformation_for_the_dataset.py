import os

# Path to your main 'data' folder
data_dir = r'D:\pcb_ai_project\train_600x_jitter'

def audit_pcb_dataset(directory):
    if not os.path.exists(directory):
        print(f"Error: Folder '{directory}' not found!")
        return

    print(f"{'Category':<20} | {'Image Count':<12}")
    print("-" * 35)
    
    total_images = 0
    categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for category in categories:
        cat_path = os.path.join(directory, category)
        # Filters for common image extensions
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        count = len(images)
        print(f"{category:<20} | {count:<12}")
        total_images += count
        
    print("-" * 35)
    print(f"{'TOTAL':<20} | {total_images:<12}")

if __name__ == "__main__":
    audit_pcb_dataset(data_dir)
# from PIL import Image
# import os

# # Path to one of your image folders
# sample_path = r"C:\Users\poram\Downloads\167242322-0eedf9a4-ab4d-4354-aa7c-4c7138bb0192.png"
# first_image = os.listdir(sample_path)[50]

# with Image.open(os.path.join(sample_path, first_image)) as img:
#     width, height = img.size
#     print(f"Image Resolution: {width}x{height} pixels")