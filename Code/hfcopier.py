import os
import shutil
import random
from pathlib import Path


def copy_random_samples(base_data_path, output_path, num_samples_per_split=20):
    """
    Copy random samples from each split to a new directory structure
    
    Args:
        base_data_path: Path to the base data directory
        output_path: Path where to save the copied samples
        num_samples_per_split: Number of samples to copy from each split
    """
    
    # Create output directory structure
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    splits = ["train", "test", "val"]
    
    for split in splits:
        print(
            f"Processing {split} split - copying {num_samples_per_split} samples...")
        
        # Source directories
        source_images = Path(base_data_path) / split / "images"
        source_masks = Path(base_data_path) / split / "masks"
        
        # Create output directories
        output_split = output_path / split
        output_split.mkdir(exist_ok=True)
        output_images = output_split / "images"
        output_masks = output_split / "masks"
        output_images.mkdir(exist_ok=True)
        output_masks.mkdir(exist_ok=True)
        
        # Check if source directories exist
        if not source_images.exists():
            print(
                f"Warning: {source_images} does not exist. Skipping {split} split.")
            continue
        if not source_masks.exists():
            print(
                f"Warning: {source_masks} does not exist. Skipping {split} split.")
            continue
        
        # Get all image files (looking for _slice.png files)
        image_files = [f for f in os.listdir(source_images)
                      if f.endswith("_slice.png")]
        
        if len(image_files) < num_samples_per_split:
            print(
                f"Warning: Only {len(image_files)} images available in {split}, copying all of them.")
            selected_files = image_files
        else:
            # Randomly select samples
            selected_files = random.sample(image_files, num_samples_per_split)
        
        # Copy selected files
        copied_count = 0
        for image_filename in selected_files:
            # Get corresponding mask filename
            # Replace _slice.png with _mask.png
            mask_filename = image_filename.replace("_slice.png", "_mask.png")
            
            # Source paths
            src_image = source_images / image_filename
            src_mask = source_masks / mask_filename
            
            # Destination paths
            dst_image = output_images / image_filename
            dst_mask = output_masks / mask_filename
            
            # Copy files if they exist
            if src_image.exists() and src_mask.exists():
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_mask, dst_mask)
                copied_count += 1
            else:
                print(f"  Warning: Missing file pair - {image_filename} or {mask_filename}")
        
        print(f"Completed {split} split: {copied_count} file pairs copied\n")


def main():
    # Configuration
    patch_size = 256
    version = 0
    
    # Source data path - updated to match your structure
    base_data_path = f"Data/Data_{patch_size}_{version}"
    
    # Output path for the copied samples
    output_path = f"Data/hf_Data_{patch_size}_{version}"
    
    # Number of samples to copy from each split
    num_samples_per_split = 20
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    print("Starting random data sampling...")
    print(f"Source: {base_data_path}")
    print(f"Destination: {output_path}")
    print(f"Samples per split: {num_samples_per_split}")
    print("-" * 50)
    
    # Copy the samples
    copy_random_samples(base_data_path, output_path, num_samples_per_split)
    
    print("Data sampling completed!")
    print(f"Check the '{output_path}' folder for your copied samples.")


if __name__ == "__main__":
    main()