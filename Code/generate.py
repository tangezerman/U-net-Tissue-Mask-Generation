import os
import csv
OPENSLIDE_PATH = r"D:\openslide-win64-20231011"

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def generate_csv(directory_path):
    # Open CSV file in write mode
    with open('wsi_labels.csv', 'w', newline='') as csvfile:
        fieldnames = ['WSI_Path', 'real_mask_patch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Loop through files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".svs") or filename.endswith(".dat"):
                wsi_path = os.path.join(directory_path, filename)
                label_name = "real_mask_" + \
                    os.path.splitext(filename)[0] + \
                    "_patch"  # Generating label name
                writer.writerow(
                    {'WSI_Path': wsi_path, 'real_mask_patch': label_name})


# Directory where WSI files are stored
directory_path = r"D:\AIN3007_project\eval\raw-internal-test-dataset"

# Generate CSV file
generate_csv(directory_path)
