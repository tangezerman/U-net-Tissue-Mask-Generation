import os
import random
import csv


def gather_file_info(data_folder, extensions):

    wsi_info_list = []

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                wsi_name, file_extension = os.path.splitext(file)
                if file_extension in extensions:
                    wsi_path = os.path.join(folder_path, file)
                    xml_file = wsi_name + ".xml"
                    xml_path = os.path.join(folder_path, xml_file)
                    if os.path.exists(xml_path):
                        wsi_info_list.append([wsi_path, xml_path, wsi_name, folder_name])

    return wsi_info_list

def split_data(info_list: list):
    split_dict = {"train": [], "val": [], "test": []}

    for folder_name in set([info[3] for info in info_list]):
        folder_files = [info for info in info_list if info[3] == folder_name]
        random.shuffle(folder_files)
        total_files = len(folder_files)

        train_size = int(0.7 * total_files)
        val_size = int(0.2 * total_files)

        train_files = folder_files[:train_size]
        val_files = folder_files[train_size:train_size + val_size]
        test_files = folder_files[train_size + val_size:]

        split_dict["train"].extend(train_files)
        split_dict["val"].extend(val_files)
        split_dict["test"].extend(test_files)

    return split_dict

def create_data_list(file_info_list, base_filename):
    csv_filename = rf"Paths\{base_filename}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["WSI Path", "XML Path", "WSI Name", "Folder Name"])
        csv_writer.writerows(file_info_list)

def create_split_csv(split_dict, base_filename):
    for split_type, split_files in split_dict.items():
        csv_filename = rf"Paths\{base_filename}_{split_type}.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["WSI Path", "XML Path", "WSI Name", "Folder Name"])
            csv_writer.writerows(split_files)

   
if __name__ == "__main__":
    random.seed(41)
    
    data_folder = r"E:\temp_data\AIN3007_project\train_valid_data"
    extensions = [".svs", ".mrxs", ".ndpi", ".tif", ".tiff"]

    file_info_list = gather_file_info(data_folder, extensions)
    split_dict = split_data(file_info_list)

    create_data_list(file_info_list, "full_data")
    create_split_csv(split_dict, "split_data")

