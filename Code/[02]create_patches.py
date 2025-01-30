# %% Imports
import csv
import psutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import threading
OPENSLIDE_PATH = r"D:\openslide-win64-20231011"

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


# %% Deffinitions

def get_data_list():
    full_data_lists = []
    for file_name in ["train", "val", "test"]:
        csv_file_path = rf"C:\Users\tange\vscode\U-net-Tissue-Mask-Generation\Paths\split_data_{file_name}.csv"

        data_list = []
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                row = [element for element in row]
                data_list.append(row)

        full_data_lists.append(data_list)
    return full_data_lists


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

    mask = cv2.resize(mask, dsize=(size, size),
                      interpolation=cv2.INTER_NEAREST)

    return mask


def generate_img_array(slide, level, canvas):
    img = slide.read_region((0, 0), level, canvas)
    img_arr = np.array(img)[:, :, :3]

    img_arr = cv2.resize(img_arr, dsize=(size, size),
                         interpolation=cv2.INTER_AREA)

    mask = (img_arr < 5).any(axis=2)
    img_arr[mask] = [255, 255, 255]

    return img_arr


def not_empty(mask):
    total_pixels = mask.size
    non_empty_pixels = np.count_nonzero(mask)

    if non_empty_pixels > 0:
        empty_percentage = (1 - non_empty_pixels / total_pixels) * 100
        return empty_percentage < 99
    else:
        return False


def slice(mask, slice_img, folder):
    row_col = size // patch_size
    for row in range(row_col):
        for col in range(row_col):
            start_row = row * patch_size
            end_row = start_row + patch_size
            start_col = col * patch_size
            end_col = start_col + patch_size

            cropped_img = slice_img[start_row:end_row, start_col:end_col]
            cropped_mask = mask[start_row:end_row, start_col:end_col]

            index_str = f"{index}_{row}_{col}"

            if not_empty(cropped_mask):
                mask_path = os.path.join(
                    main_folder, folder, "masks", f"{index_str}_mask.png")
                img_path = os.path.join(
                    main_folder, folder, "images", f"{index_str}_slice.png")

                cv2.imwrite(mask_path, cropped_mask)
                cv2.imwrite(img_path, cropped_img)


def parse(file_info, folder):
    """
    Parse slide image and generate patches with masks.
    Returns True if successful, False if file is unsupported or missing.

    Parameters:
    file_info (list): Contains path to slide and XML annotation
    folder (str): Output folder name for the patches
    """
    try:
        global size
        path = file_info[0]
        slide = openslide.OpenSlide(path)

        level = 5
        if level > slide.level_count:
            level = slide.level_count-1

        downscale = slide.level_downsamples[level]//1
        canvas = slide.level_dimensions[level]
        size = min(canvas)//patch_size*patch_size
        xml = file_info[1]

        slice_img = generate_img_array(slide, level, canvas)
        mask = generate_mask(xml, downscale, canvas)

        slice(mask, slice_img, folder)
        return True

    except openslide.OpenSlideUnsupportedFormatError:
        print(f"Skipping unsupported file: {path}")
        return False
    except Exception as e:
        print(f"Error processing file {path}: {str(e)}")
        return False
    finally:
        # Ensure slide is properly closed even if an error occurs
        if 'slide' in locals():
            try:
                slide.close()
            except:
                pass

# %% RUN


patch_size = 256
version = 0
limit = 1000

main_folder = rf"Data\Data_{patch_size}_{version}"

os.makedirs(os.path.join(main_folder), exist_ok=True)
os.makedirs(os.path.join(main_folder, "trash"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "trash", "masks"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "trash", "images"), exist_ok=True)

index = 0
process = psutil.Process()

train_files, val_files, test_files = get_data_list()
for split in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:

    os.makedirs(os.path.join(main_folder, split[1], "masks"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, split[1], "images"), exist_ok=True)

    progress_bar = tqdm(total=len(split[0]),
                        desc=f"Processing {split[1]} files")
    skipped_files = 0

    for file in split[0]:
        if index >= limit:
            break

        success = parse(file, split[1])
        if not success:
            skipped_files += 1
        else:
            index += 1

        progress_bar.update(1)

    progress_bar.close()
    print(f"Completed {split[1]} split. Skipped {skipped_files} files.")

    if index >= limit:
        break
# %%
