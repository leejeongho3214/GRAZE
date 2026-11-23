"""Regroup data/hot3d/data/acc_ori.pkl by object -> data/hot3d/pre-process_dataset/acc_ori.pkl."""
import glob
import os
import os.path as osp
import pickle

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
data_root = osp.join(PROJECT_ROOT, "data/hot3d/data")

from collections import defaultdict
import pickle
total_dict = defaultdict(lambda: defaultdict(list))
act_dict = defaultdict(int)

file_name = "acc_ori"

with open(os.path.join(data_root, f"{file_name}.pkl"), "rb") as f:
    data = pickle.load(f)
    
for idx, value in enumerate(data['meta']):
    if len(value) == 0:
        continue
    total_dict[value[0][1]]['obj_pose_rts'].append(data['obj_pose_rts'][idx])
    total_dict[value[0][1]]['cam_poses'].append(data['cam_poses'][idx])
    total_dict[value[0][1]]['hand_pose_manos'].append(data['hand_pose_manos'][idx])
    total_dict[value[0][1]]['action_labels'].append(data['action_labels'][idx])
    total_dict[value[0][1]]['meta'].append(data['meta'][idx][0])
    total_dict[value[0][1]]['gaze'].append(data['gaze'][idx])
    total_dict[value[0][1]]['act_id'].append(data['act_id'][idx])
    
total_dict = dict(total_dict)            

os.makedirs(osp.join(data_root, "../pre-process_dataset"), exist_ok=True)
with open(osp.join(data_root, f"../pre-process_dataset/{file_name}.pkl"), "wb") as f:
    pickle.dump(total_dict, f)
    
print("End")