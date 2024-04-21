import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
OPENSLIDE_PATH = r"D:\openslide-win64-20231011"

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def open_slide(wsi_path):
    slide = openslide.OpenSlide(wsi_path)
    return slide


def generate_img_array(slide, level, patch_size=256):
    canvas = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, canvas)
    img_arr = np.array(img)[:, :, :3]
    size = min(canvas)//patch_size*patch_size
    img_arr = cv2.resize(img_arr, dsize=(size, size),
                         interpolation=cv2.INTER_AREA)

    # mask out blacked out areas
    mask = (img_arr < 10).any(axis=2)
    img_arr[mask] = [255, 255, 255]

    return img_arr


def slice(img_array, patch_size=256):

    matrix_size = len(img_array[0]) // patch_size
    patch_list = []
    for row in range(matrix_size):
        for col in range(matrix_size):

            start_row = row * patch_size
            end_row = start_row + patch_size

            start_col = col * patch_size
            end_col = start_col + patch_size

            cropped_img = img_array[start_row:end_row, start_col:end_col]

            cropped_img = cropped_img / 255.0
            patch_list.append(cropped_img)

    return np.array(patch_list)


def generate_mask(xml, downscale, canvas):
    mask = np.zeros(canvas[::-1], dtype=np.uint8)
    tree = ET.parse(xml)
    root = tree.getroot()

    for annotation in root.findall("./Annotations/Annotation"):
        coordinates = []
        for coordinate in annotation.findall(".//Coordinates/Coordinate"):
            x = float(coordinate.attrib["X"].replace(",", ".")) / downscale
            y = float(coordinate.attrib["Y"].replace(",", ".")) / downscale
            coordinates.append((x, y))

        if annotation.attrib.get("PartOfGroup") == "tissue":
            pts = np.array(coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=255)
        elif annotation.attrib.get("PartOfGroup") == "bg":
            pts = np.array(coordinates, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=0)

    return mask


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    if union == 0:
        return 1.0

    dice_coeff = (2.0 * intersection) / union
    return dice_coeff
