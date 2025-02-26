import os
import random
import shutil

from config.config import train_val_test_settings, frame_capture_settings

def move_files(src_img_dir, src_labels_dir, dest_img_dir, dest_labels_dir, sample_size):
    """
    Randomly selects a number of images from src_img_dir and moves them, along with their
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
        src_label = os.path.join(src_labels_dir, label_filename)
        dst_label = os.path.join(dest_labels_dir, label_filename)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"Label file not found for image {file} at {src_label}")

if __name__ == "__main__":
    
    src_imgs_dir = frame_capture_settings.img_dir
    src_labels_dir = frame_capture_settings.edited_label_dir


    dest_train_img_dir = train_val_test_settings.train_images_dir
    dest_train_label_dir = train_val_test_settings.train_labels_dir
    dest_val_img_dir = train_val_test_settings.val_images_dir
    dest_val_label_dir = train_val_test_settings.val_labels_dir
    dest_test_img_dir = train_val_test_settings.test_images_dir
    dest_test_label_dir = train_val_test_settings.test_labels_dir

       
    # Ensure destination directories exist
    os.makedirs(train_val_test_settings.val_images_dir, exist_ok=True)
    os.makedirs(train_val_test_settings.val_labels_dir, exist_ok=True)
    os.makedirs(train_val_test_settings.test_images_dir, exist_ok=True)
    os.makedirs(train_val_test_settings.test_labels_dir, exist_ok=True)
    os.makedirs(train_val_test_settings.train_images_dir, exist_ok=True)
    os.makedirs(train_val_test_settings.train_labels_dir, exist_ok=True)

    
    val_sample_size = int(len(os.listdir(src_imgs_dir)) * train_val_test_settings.sample_ratio_val)
    test_sample_size = int(len(os.listdir(src_imgs_dir)) * train_val_test_settings.sample_ratio_test)
    train_sample_size = len(os.listdir(src_imgs_dir)) - val_sample_size - test_sample_size
    
    
    move_files(src_imgs_dir, src_labels_dir, dest_val_img_dir, dest_val_label_dir, sample_size=val_sample_size)
    move_files(src_imgs_dir, src_labels_dir, dest_test_img_dir, dest_test_label_dir, sample_size=test_sample_size)
    move_files(src_imgs_dir, src_labels_dir, dest_train_img_dir, dest_train_label_dir, sample_size=train_sample_size)
