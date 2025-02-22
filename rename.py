import os
import uuid

# Folder 
IMAGE_FOLDER = "data/test"

# Check for folder
if not os.path.exists(IMAGE_FOLDER):
    print("Folder not found!")
    exit()

# Rename images
for img_name in os.listdir(IMAGE_FOLDER):
    old_path = os.path.join(IMAGE_FOLDER, img_name)
    
    # Check the extension
    if os.path.isfile(old_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Generate a unique random filename
        new_name = str(uuid.uuid4()) + os.path.splitext(img_name)[1]  # Preserve extension
        new_path = os.path.join(IMAGE_FOLDER, new_name)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed {img_name} → {new_name}")

print("✅ All images renamed successfully!")
