import os
import random
import shutil

def move_files(src_img_dir, src_label_dir, dest_img_dir, dest_label_dir, sample_ratio):
    """
    Randomly selects a percentage of images from src_img_dir and moves them, along with their
    associated label files from src_label_dir, to dest_img_dir and dest_label_dir.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}  # adjust as needed

    # Get a list of image files in the source directory
    images = [
        f for f in os.listdir(src_img_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not images:
        print("No images found in", src_img_dir)
        return

    # Calculate how many files to move; ensure at least 1 file is moved if the ratio > 0 and there are files.
    sample_size = int(len(images) * sample_ratio)
    if sample_size == 0 and sample_ratio > 0:
        sample_size = 1

    # Randomly sample images
    sampled_files = random.sample(images, sample_size)
    print(f"Moving {len(sampled_files)} files from {src_img_dir} to {dest_img_dir}")
    
    for file in sampled_files:
        # Build full paths for image and label files
        src_image = os.path.join(src_img_dir, file)
        dst_image = os.path.join(dest_img_dir, file)
        
        # Move the image file
        shutil.move(src_image, dst_image)
        
        # For the label, we assume it's a .txt file with the same base name.
        base_name, _ = os.path.splitext(file)
        label_filename = base_name + ".txt"
        src_label = os.path.join(src_label_dir, label_filename)
        dst_label = os.path.join(dest_label_dir, label_filename)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"Label file not found for image {file} at {src_label}")

if __name__ == "__main__":
    # Set the base directory for your dataset
    base_dir = os.path.join(os.getcwd(), "datasets", "dataset_webcam")
    
    # Directories for images
    train_images_dir = os.path.join(base_dir, "images", "train")
    val_images_dir = os.path.join(base_dir, "images", "val")
    test_images_dir = os.path.join(base_dir, "images", "test")
    
    # Directories for labels
    train_labels_dir = os.path.join(base_dir, "labels", "train")
    val_labels_dir = os.path.join(base_dir, "labels", "val")
    test_labels_dir = os.path.join(base_dir, "labels", "test")
    
    # Ensure destination directories exist
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    # Move 15% of images/labels to the val folders
    move_files(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, sample_ratio=0.15)
    
    # Move 1% of images/labels to the test folders
    move_files(train_images_dir, train_labels_dir, test_images_dir, test_labels_dir, sample_ratio=0.01)
