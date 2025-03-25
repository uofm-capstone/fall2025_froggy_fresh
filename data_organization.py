import os
import random
import shutil
import logging
from pathlib import Path

def create_train_test_split(source_dir="frog_images", split_ratio=0.8):
    """
    Create a train/test split from source_dir/frog and source_dir/not_frog folders.
    Moves images to train/test structure and removes the original empty directories.
    
    Args:
        source_dir: Base directory containing frog and not_frog subdirectories
        split_ratio: Train split ratio (default: 0.8 for 80% training, 20% testing)
    
    Returns:
        Dictionary with paths to created directories, or None if validation fails
    """
    # Define directory paths
    source_frogs_dir = os.path.join(source_dir, "frog")
    source_not_frogs_dir = os.path.join(source_dir, "not_frog")
    
    # Validate source directories
    if not os.path.exists(source_frogs_dir):
        logging.error(f"Source frog directory not found: {source_frogs_dir}")
        return None
        
    if not os.path.exists(source_not_frogs_dir):
        logging.error(f"Source not-frog directory not found: {source_not_frogs_dir}")
        return None
    
    # Create train/test directory structure
    train_dir = os.path.join(source_dir, "train")
    test_dir = os.path.join(source_dir, "test")
    
    train_frogs_dir = os.path.join(train_dir, "frog")
    train_not_frogs_dir = os.path.join(train_dir, "not_frog")
    test_frogs_dir = os.path.join(test_dir, "frog")
    test_not_frogs_dir = os.path.join(test_dir, "not_frog")
    
    # Create directories
    os.makedirs(train_frogs_dir, exist_ok=True)
    os.makedirs(train_not_frogs_dir, exist_ok=True)
    os.makedirs(test_frogs_dir, exist_ok=True)
    os.makedirs(test_not_frogs_dir, exist_ok=True)
    
    # Process frog images
    split_and_move_images(source_frogs_dir, train_frogs_dir, test_frogs_dir, split_ratio)
    
    # Process not-frog images
    split_and_move_images(source_not_frogs_dir, train_not_frogs_dir, test_not_frogs_dir, split_ratio)
    
    # Remove original directories if they're empty
    try:
        # Only remove if empty
        if os.path.exists(source_frogs_dir) and len(os.listdir(source_frogs_dir)) == 0:
            os.rmdir(source_frogs_dir)
            logging.info(f"Removed empty directory: {source_frogs_dir}")
            
        if os.path.exists(source_not_frogs_dir) and len(os.listdir(source_not_frogs_dir)) == 0:
            os.rmdir(source_not_frogs_dir)
            logging.info(f"Removed empty directory: {source_not_frogs_dir}")
    except Exception as e:
        logging.warning(f"Could not remove empty directories: {e}")
    
    # Print statistics
    train_frogs = count_images(train_frogs_dir)
    train_not_frogs = count_images(train_not_frogs_dir)
    test_frogs = count_images(test_frogs_dir)
    test_not_frogs = count_images(test_not_frogs_dir)
    
    print(f"Train/Test Split ({split_ratio:.0%}/{1-split_ratio:.0%}) completed:")
    print(f"Train set: {train_frogs} frogs, {train_not_frogs} not-frogs")
    print(f"Test set: {test_frogs} frogs, {test_not_frogs} not-frogs")
    
    return {
        "base_dir": source_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_frogs_dir": train_frogs_dir,
        "train_not_frogs_dir": train_not_frogs_dir,
        "test_frogs_dir": test_frogs_dir,
        "test_not_frogs_dir": test_not_frogs_dir
    }

def split_and_move_images(source_dir, train_dir, test_dir, split_ratio):
    """Split images from source directory into train and test directories by moving them."""
    image_files = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Shuffle the files for randomization
    random.shuffle(image_files)
    
    # Calculate split point
    split_idx = int(len(image_files) * split_ratio)
    
    # Move train files
    for img_file in image_files[:split_idx]:
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(train_dir, img_file)
        shutil.move(src_path, dst_path)
    
    # Move test files
    for img_file in image_files[split_idx:]:
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(test_dir, img_file)
        shutil.move(src_path, dst_path)

def count_images(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Count image files in a directory."""
    if not os.path.exists(directory):
        return 0
    
    return sum(1 for file in os.listdir(directory) 
              if file.lower().endswith(extensions))

def get_category_images(base_dir, category, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Get list of images in a specific category directory."""
    category_dir = os.path.join(base_dir, category)
    if not os.path.exists(category_dir):
        return []
    
    return [os.path.join(category_dir, f) for f in os.listdir(category_dir)
            if f.lower().endswith(extensions)]

def validate_image_dir(directory):
    """Validate that a directory contains frog and not_frog subdirectories with images."""
    frog_dir = os.path.join(directory, "frog")
    not_frog_dir = os.path.join(directory, "not_frog")
    
    if not os.path.exists(directory):
        return False, f"Directory '{directory}' does not exist"
        
    if not os.path.exists(frog_dir):
        return False, f"Missing 'frog' subdirectory in '{directory}'"
        
    if not os.path.exists(not_frog_dir):
        return False, f"Missing 'not_frog' subdirectory in '{directory}'"
        
    frog_count = count_images(frog_dir)
    not_frog_count = count_images(not_frog_dir)
    
    if frog_count == 0:
        return False, f"No images found in '{frog_dir}'"
        
    if not_frog_count == 0:
        return False, f"No images found in '{not_frog_dir}'"
        
    return True, f"Found {frog_count} frog images and {not_frog_count} not-frog images"