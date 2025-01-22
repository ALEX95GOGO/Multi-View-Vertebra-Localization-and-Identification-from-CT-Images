import json
import numpy as np

gt_json_file = "/projects/MAD3D/Zhuoli/MICCAI/clean_code/Multi-View-Vertebra-Localization-and-Identification-from-CT-Images/eval/merged_ground_truth.json"
save_json_file = "/projects/MAD3D/Zhuoli/MICCAI/clean_code/Multi-View-Vertebra-Localization-and-Identification-from-CT-Images/inference_save/fcn_test_voting.json"

if __name__ == "__main__":
    with open(gt_json_file) as json_data:
        gt_dic_list = json.load(json_data)
        json_data.close()
    with open(save_json_file) as json_data:
        pred_dic_list = json.load(json_data)
        json_data.close()
    
    distance_error = 0
    count_all, count_right = 0,0
    count = 0
    miss_vert_count = {}
    for k in gt_dic_list.keys():
        pred_centroids = []
        pred_ids = []
        gt_centroids = []
        gt_ids = []
        for i in range(len(gt_dic_list[k])):
            #import pdb; pdb.set_trace()
            #print(gt_dic_list[k][i])
            try:
                gt_ids.append(gt_dic_list[k][i]['label'])
                gt_centroids.append([gt_dic_list[k][i]['X'],gt_dic_list[k][i]['Y'],gt_dic_list[k][i]['Z']])
            except Exception as e:
                print(e)
        if k in pred_dic_list:
            for i in range(len(pred_dic_list[k])):
                pred_ids.append(pred_dic_list[k][i][0])
                pred_centroids.append(pred_dic_list[k][i][1])
    
            pred_len = len(pred_dic_list[k])
            gt_len = len(gt_dic_list[k])
            if gt_len - pred_len > 0:
                miss_vert_count[k] = gt_len - pred_len
            distance_matrix = np.zeros((pred_len, gt_len))
            for i in range(pred_len):
                for j in range(gt_len):
                    distance_matrix[i][j] = np.sqrt(np.sum(np.square(np.array(pred_centroids[i])+1 - np.array(gt_centroids[j]))))
            index = distance_matrix.argmin(axis=1)
            for i in range(len(index)):
                if distance_matrix[i][index[i]] < 20 :
                    distance_error += distance_matrix[i][index[i]]
                    count_all += 1
                    count_right += (pred_ids[i]==gt_ids[index[i]])
                    if pred_ids[i]!=gt_ids[index[i]]:
                        print(f'{k} pred is {pred_ids[i]}, gt is {gt_ids[index[i]]}')
                else:
                    count += 1
                    count_all += 1

    for i,k in enumerate(miss_vert_count.keys()):
        if miss_vert_count[k] > 0:
            print(k,miss_vert_count[k])
    mode = 'test' 
    print((f'[*] {mode.zfill(10)}_voting: id rate is {100*count_right/(count_all):.2f}%, distance error is {distance_error/count_all:.4f} + {abs(sum(miss_vert_count.values())*1000)/(count_all):.4f} = {(distance_error + abs(sum(miss_vert_count.values())*1000))/(count_all):.4f} mm, over dis num: {count}, mis num {sum(miss_vert_count.values())}.'))
    #txt_file.write(f'[*] {mode.zfill(10)}_voting: id rate is {100*count_right/(count_all):.2f}%, distance error is {distance_error/count_all:.4f} + {abs(sum(miss_vert_count.values())*1000)/(count_all):.4f} = {(distance_error + abs(sum(miss_vert_count.values())*1000))/(count_all):.4f} mm, over dis num: {count}, mis num {sum(miss_vert_count.values())}.')
