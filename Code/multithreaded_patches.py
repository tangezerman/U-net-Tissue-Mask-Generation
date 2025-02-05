import csv
import multiprocessing
import psutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

OPENSLIDE_PATH = r"D:\openslide-win64-20231011"

patch_size = 256
version = 0
limit = 1000

main_folder = rf"Data\Data_{patch_size}_{version}"

# Create necessary directories
os.makedirs(os.path.join(main_folder), exist_ok=True)
os.makedirs(os.path.join(main_folder, "trash"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "trash", "masks"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "trash", "images"), exist_ok=True)

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def get_cores() -> int :
    try:

        logical_cores = (multiprocessing.cpu_count())//2
        return logical_cores
    except Exception:
        return 2

    
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

def generate_mask(xml, downscale, canvas, size):
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

    mask = cv2.resize(mask, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
    return mask

def generate_img_array(slide, level, canvas, size):
    img = slide.read_region((0, 0), level, canvas)
    img_arr = np.array(img)[:, :, :3]

    img_arr = cv2.resize(img_arr, dsize=(size, size), interpolation=cv2.INTER_AREA)

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

def slice(mask, slice_img, folder, index, patch_size):
    row_col = mask.shape[0] // patch_size
    results = []  # List to store results for each patch

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
                # Prepare the data to be returned
                mask_path = os.path.join(main_folder, folder, "masks", f"{index_str}_mask.png")
                img_path = os.path.join(main_folder, folder, "images", f"{index_str}_slice.png")
                results.append((cropped_mask, cropped_img, mask_path, img_path))

    return results

def save_image_atomically(image, path, lock):
    with lock:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=os.path.dirname(path)) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, image)
        shutil.move(tmp_path, path)

def parse(file_info, folder, index, patch_size, lock):
    try:
        path = file_info[0]
        slide = openslide.OpenSlide(path)

        level = 5
        if level > slide.level_count:
            level = slide.level_count - 1

        downscale = slide.level_downsamples[level] // 1
        canvas = slide.level_dimensions[level]
        size = min(canvas) // patch_size * patch_size
        xml = file_info[1]

        slice_img = generate_img_array(slide, level, canvas, size)
        mask = generate_mask(xml, downscale, canvas, size)

        # Get the results from slice
        results = slice(mask, slice_img, folder, index, patch_size)

        # Write files using the results
        for cropped_mask, cropped_img, mask_path, img_path in results:
            save_image_atomically(cropped_mask, mask_path, lock)
            save_image_atomically(cropped_img, img_path, lock)

        return True

    except openslide.OpenSlideUnsupportedFormatError:
        return False
    except Exception as e:
        print(f"Error processing file {file_info[0]}: {e}")
        return False
    finally:
        if 'slide' in locals():
            try:
                slide.close()
            except:
                pass

def process_split(split_files, split_name, shared_index, skipped_files, patch_size, lock):
    workers = get_cores()
    os.makedirs(os.path.join(main_folder, split_name, "masks"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, split_name, "images"), exist_ok=True)

    progress_bar = tqdm(total=len(split_files), desc=f"Processing {split_name} files")
    skipped_count = 0

    # Pre-filter files based on limit
    with lock:
        current_index = shared_index.value
        files_to_process = split_files[:limit - current_index]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, file in enumerate(files_to_process):
            futures.append(executor.submit(parse, file, split_name, current_index + i, patch_size, lock))

        for future in as_completed(futures):
            success = future.result()
            if not success:
                skipped_count += 1
            else:
                with lock:
                    shared_index.value += 1
            progress_bar.update(1)

    progress_bar.close()
    print(f"Completed {split_name} split. Skipped {skipped_count} files.")

if __name__ == "__main__":
    process = psutil.Process()

    train_files, val_files, test_files = get_data_list()

    manager = Manager()
    shared_index = manager.Value('i', 0)
    skipped_files = manager.list()
    lock = manager.Lock()  # Create a shared lock using Manager

    process_split(train_files, "train", shared_index, skipped_files, patch_size, lock)
    process_split(val_files, "val", shared_index, skipped_files, patch_size, lock)
    process_split(test_files, "test", shared_index, skipped_files, patch_size, lock)

    if skipped_files:
        print("\n" + "---" * 20)
        print("Skipped files:")
        for file_path, reason in skipped_files:
            print(f"File: {file_path}, Reason: {reason}")
        print("---" * 20)
    else:
        print("\nNo files were skipped.")