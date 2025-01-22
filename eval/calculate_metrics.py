import os
import json
import pandas as pd
import numpy as np


def calculate_identification_rate_and_l_error(gt_folder, pred_file, output_csv_path):
    # Read prediction JSON file
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    results = []
    identification_rate_all = 0
    l_error_all = 0
    # Iterate over each subject in the predictions
    for subject_id, subject_predictions in predictions.items():
        # Path to the corresponding ground truth file
        subject_folder = os.path.join(gt_folder, f"{subject_id[4:]}")
        print(subject_folder) 
        if not os.path.exists(subject_folder):
            print(f"Folder for subject {subject_id} not found. Skipping...")
            continue
        # Look for a JSON file in the subject's folder
        gt_file_path = None
        for file_name in os.listdir(subject_folder):
            if file_name.endswith('.json'):
                gt_file_path = os.path.join(subject_folder, file_name)
                break

        if not gt_file_path:
            print(f"No JSON file found for subject {subject_id}. Skipping...")
            continue
        # Read the ground truth file
        with open(gt_file_path, 'r') as gt_file:
            gt_data = json.load(gt_file)

        # Extract ground truth with labels and coordinates
        gt_df = pd.DataFrame([item for item in gt_data if 'label' in item])

        # Extract predictions into a DataFrame
        pred_df = pd.DataFrame(
            [(label, *coords) for label, coords in subject_predictions],
            columns=['label', 'X', 'Y', 'Z']
        )

        # Merge ground truth and predictions based on labels
        merged_df = pd.merge(gt_df, pred_df, on='label', suffixes=('_gt', '_pred'))

        # Calculate identification rate
        identified_labels = merged_df['label'].nunique()
        total_gt_labels = gt_df['label'].nunique()
        identification_rate = identified_labels / total_gt_labels

        # Calculate L-error (mean Euclidean distance)
        distances = np.sqrt(
            ((merged_df['X_gt'] - merged_df['X_pred']) ** 2) +
            ((merged_df['Y_gt'] - merged_df['Y_pred']) ** 2) +
            ((merged_df['Z_gt'] - merged_df['Z_pred']) ** 2)
        )
        mean_l_error = distances.mean()

        # Print results for each subject
        print(f"Subject ID: {subject_id}")
        print(f"  Identification Rate: {identification_rate:.2f}")
        print(f"  L-error (mean distance): {mean_l_error:.2f}\n")
        # Append results to a list
        results.append({
            'Subject ID': subject_id,
            'Identification Rate': identification_rate,
            'Mean L-error': mean_l_error
        })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

# Example usage
gt_folder_path = "/projects/MAD3D/Zhuoli/MICCAI/VerSe2019/dataset-verse19test/derivatives"
prediction_file_path = "/projects/MAD3D/Zhuoli/MICCAI/clean_code/Multi-View-Vertebra-Localization-and-Identification-from-CT-Images/inference_save/fcn_test_voting.json"
output_csv_path = "inference_save/identification_metrics.csv"
calculate_identification_rate_and_l_error(gt_folder_path, prediction_file_path, output_csv_path)

#print(f"Identification Rate: {ident_rate}")
#print(f"L-error (mean distance): {l_error}")
