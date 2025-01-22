import os
import json

def merge_ground_truth_with_folder(gt_folder, output_file):
    merged_data = {}

    # Iterate through all folders and JSON files in the ground truth directory
    for root, _, files in os.walk(gt_folder):
        folder_name = os.path.basename(root)
        folder_data = []

        for file in files:
            if file.endswith('.json'):  # Process only JSON files
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        folder_data.extend(data)  # Add data from the file to the folder's list
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON file: {file_path}")
        
        if folder_data:  # Only add non-empty data
            merged_data[folder_name] = folder_data

    # Write the merged data to the output file
    with open(output_file, 'w') as out_f:
        json.dump(merged_data, out_f, indent=4)
    
    print(f"All ground truth JSON files have been merged into {output_file}")

# Example usage
gt_folder_path = "/projects/MAD3D/Zhuoli/MICCAI/VerSe2019/dataset-verse19test/derivatives"
output_file_path = "eval/merged_ground_truth.json"

merge_ground_truth_with_folder(gt_folder_path, output_file_path)

