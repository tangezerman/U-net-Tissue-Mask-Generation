import os
import csv
import cv2
import numpy as np
import tensorflow as tf
import tools

# Suppress TensorFlow info and warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Define paths and parameters
model_path = "Models\model__256_16_0"
external_dataset_path = "Paths\external_data_test.csv"
export_path = "Export"
patch_size = 256

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Read external dataset paths from CSV
with open(external_dataset_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Extract information from the CSV row
        WSI_path = row['WSI_path']
        # Provide the path to the XML file containing ground truth masks
        real_mask_path = row['real_mask_path']

        # Process the WSI
        slide = tools.open_slide(WSI_path)
        level = 5  # Define the level for processing
        if level > slide.level_count:
            level = slide.level_count - 1
        downscale = slide.level_downsamples[level] // 1
        canvas = slide.level_dimensions[level]
        img_array = tools.generate_img_array(slide, level)
        patch_list = tools.slice(img_array)
        matrix_shape = len(img_array) // patch_size

        # Predict masks for the patches
        predicted_patches = model.predict(patch_list)
        predicted_patches = (predicted_patches > 0.5).astype(np.uint8) * 255
        original_array_shape = predicted_patches.shape
        reshaped_array = predicted_patches.reshape(
            matrix_shape, matrix_shape, original_array_shape[1], original_array_shape[2], original_array_shape[3])
        predicted_mask = reshaped_array.swapaxes(1, 2).reshape(
            (original_array_shape[1]*matrix_shape, original_array_shape[2]*matrix_shape, original_array_shape[3]))
        predicted_mask = cv2.resize(
            predicted_mask, dsize=canvas, interpolation=cv2.INTER_NEAREST)

        # Calculate IOU and Dice coefficient if ground truth mask is available
        try:
            real_mask = tools.generate_mask(real_mask_path, downscale, canvas)
            iou = tools.calculate_iou(real_mask, predicted_mask)
            dice = tools.calculate_dice_coefficient(real_mask, predicted_mask)
            print(f"For {WSI_path}:")
            print(f"IOU: {iou}")
            print(f"Dice: {dice}")
        except Exception as e:
            print(f"For {WSI_path}:")
            print("Error:", e)

        # Write the predicted mask
        output_mask_path = os.path.join(export_path, os.path.splitext(
            os.path.basename(WSI_path))[0] + "_predicted_mask.png")
        cv2.imwrite(output_mask_path, predicted_mask)
