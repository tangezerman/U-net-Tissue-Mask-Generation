import os
import pandas as pd

# Read the CSV file
csv_path = "your_csv_file.csv"  # Update with the path to your CSV file
df = pd.read_csv(csv_path)

# Function to generate XML paths


def generate_xml_path(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    xml_path = os.path.join(os.path.dirname(file_path), file_name + ".xml")
    return xml_path


# Apply the function to generate XML paths
df['real_mask_path'] = df['WSI_path'].apply(generate_xml_path)

# Save the updated DataFrame to a new CSV file
output_csv_path = "output_file.csv"  # Specify the path for the output CSV file
df.to_csv(output_csv_path, index=False)

print("XML paths generated and saved to:", output_csv_path)
