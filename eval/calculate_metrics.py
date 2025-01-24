import os
import json
import pandas as pd
import numpy as np
import SimpleITK as sitk
from utils.landmark.common import Landmark
from utils.landmark.visualization.landmark_visualization_matplotlib import LandmarkVisualizationMatplotlib

def calculate_identification_rate_and_l_error(gt_folder, pred_file, output_csv_path):
    # Read prediction JSON file
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    vis = LandmarkVisualizationMatplotlib(dim=3,
                                              annotations=dict([(i, f'C{i + 1}') for i in range(7)] +        # 0-6: C1-C7
                                                               [(i, f'T{i - 6}') for i in range(7, 19)] +    # 7-18: T1-12
                                                               [(i, f'L{i - 18}') for i in range(19, 25)] +  # 19-24: L1-6
                                                               [(25, 'T13')]))                               # 25: T13

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
        
        landmarks = [Landmark(coords=coords, is_valid=True, value=label) for label, coords in subject_predictions]
        
        gt_landmarks = [Landmark(coords=[item['X'], item['Y'], item['Z']], is_valid=True, value=item['label']) for item in gt_data if 'label' in item]
        #vis.visualize_landmark_projections(input_image, target_landmarks, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks_gt.png'))
        #vis.visualize_prediction_groundtruth_projections(input_image, curr_landmarks_no_postprocessing, target_landmarks, filename=self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_landmarks.png'))

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
    #results_df = pd.DataFrame(results)
    #results_df.to_csv(output_csv_path, index=False)

        # Path to the main directory containing multiple subfolders.
        base_dir = "/projects/MAD3D/Zhuoli/MICCAI/VerSe2019/dataset-verse19test/rawdata/"
        
        
        # Loop through all subdirectories and files.
        for (i, folder) in enumerate(os.listdir(base_dir)):
            for filename in os.listdir(os.path.join(base_dir, folder)):
                # Check if the file ends with the desired pattern.
                
                if filename.endswith("ct.nii.gz") and folder == subject_id[4:]:
                    file_path = os.path.join(base_dir, folder, filename)
                    print(f"Reading: {file_path}")

                    # Read using SimpleITK
                    ct_image = sitk.ReadImage(file_path)
                    vis.visualize_landmark_projections(ct_image, gt_landmarks, filename="/projects/MAD3D/Zhuoli/MICCAI/VerSe2019/dataset-verse19test/multi-view-predict/" +subject_id+ '_landmarks_gt.png')
                    vis.visualize_landmark_projections(ct_image, landmarks, filename="/projects/MAD3D/Zhuoli/MICCAI/VerSe2019/dataset-verse19test/multi-view-predict/" +subject_id+ '_landmarks_pp.png')
        
# Example usage
gt_folder_path = "/projects/MAD3D/Zhuoli/MICCAI/VerSe2019/dataset-verse19test/derivatives"
prediction_file_path = "/projects/MAD3D/Zhuoli/MICCAI/clean_code/Multi-View-Vertebra-Localization-and-Identification-from-CT-Images/inference_save/fcn_test_voting.json"
output_csv_path = "inference_save/identification_metrics.csv"
calculate_identification_rate_and_l_error(gt_folder_path, prediction_file_path, output_csv_path)

#print(f"Identification Rate: {ident_rate}")
#print(f"L-error (mean distance): {l_error}")
