import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tools
import tensorflow as tf
import cv2
import numpy as np

model_path = r"C:\Users\tange\Vs Code\16th\Submission\Models\model__256_16_0"
WSI_path = r"D:\AIN3007_project\train_valid_data\Breast1__he\TCGA-A1-A0SM-01Z-00-DX1.svs"
real_mask_path = r"D:\AIN3007_project\train_valid_data\Breast1__he\TCGA-A1-A0SM-01Z-00-DX1.xml"
export_path = r"C:\Users\tange\Vs Code\16th\Submission\export"

patch_size = 256

slide = tools.open_slide(WSI_path)

level = 5 
if level > slide.level_count : level = slide.level_count-1

downscale = slide.level_downsamples[level]//1
canvas = slide.level_dimensions[level]

img_array = tools.generate_img_array(slide,level)
patch_list = tools.slice(img_array)
matrix_shape = len(img_array) // patch_size

model = tf.keras.models.load_model(model_path)

predicted_patches = model.predict(patch_list)
predicted_patches = (predicted_patches > 0.5).astype(np.uint8) * 255

original_array_shape = predicted_patches.shape
reshaped_array = predicted_patches.reshape(7, 7, original_array_shape[1], original_array_shape[2], original_array_shape[3])
predicted_mask = reshaped_array.swapaxes(1, 2).reshape((original_array_shape[1]*7, original_array_shape[2]*7, original_array_shape[3]))

predicted_mask = cv2.resize(predicted_mask, dsize=canvas, interpolation=cv2.INTER_NEAREST)

try:
    real_mask = tools.generate_mask(real_mask_path,downscale,canvas)
    iou = tools.calculate_iou(real_mask,predicted_mask)
    dice = tools.calculate_dice_coefficient(real_mask,predicted_mask)
    print(f"IOU: {iou}")
    print(f"Dice: {dice}")
except:
    print("No real mask")

cv2.imwrite("slice.png", predicted_mask)