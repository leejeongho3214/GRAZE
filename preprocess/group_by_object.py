"""Step 2 of the benchmark build: regroup data/hot3d/data/acc_ori.pkl by target object.

Writes data/hot3d/pre-process_dataset/acc_ori.pkl = {object name: {key: [one entry per action]}}.
Actions without frames are skipped; non-finite values raise an error.
"""
import argparse
import os
import os.path as osp
import pickle
import numpy as np

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data_root", default=osp.join(PROJECT_ROOT, "data/hot3d/data"),
                    help="folder that holds acc_ori.pkl from extract_hot3d_clips.py")
parser.add_argument("--save_root", default=None,
                    help="output folder (default: <data_root>/../pre-process_dataset)")
args = parser.parse_args()

data_root = args.data_root
save_root = args.save_root if args.save_root is not None else osp.join(data_root, "../pre-process_dataset")

from collections import defaultdict
total_dict = defaultdict(lambda: defaultdict(list))
skipped_empty_samples = []

file_name = "acc_ori"

with open(os.path.join(data_root, f"{file_name}.pkl"), "rb") as f:
    data = pickle.load(f)

sample_keys = [
    "obj_pose_rts",
    "all_obj_pose_rts",
    "all_obj_names",
    "cam_poses",
    "hand_pose_manos",
    "action_labels",
    "gaze",
    "act_id",
]

def find_nonfinite_path(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            found = find_nonfinite_path(item, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = find_nonfinite_path(item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.dtype.kind not in "fc":
        return None
    nonfinite = np.argwhere(~np.isfinite(array))
    if len(nonfinite) == 0:
        return None
    return f"{path}{nonfinite[0].tolist()}"

for idx, value in enumerate(data['meta']):
    if len(value) == 0:
        act_id = data.get("act_id", [[]])[idx]
        skipped_empty_samples.append({"sample_idx": idx, "act_id": act_id})
        continue
    target_obj_name = value[0][1]
    for key in sample_keys:
        if key in data:
            nonfinite_path = find_nonfinite_path(data[key][idx], key)
            if nonfinite_path is not None:
                act_id = data.get("act_id", [[]])[idx]
                raise RuntimeError(
                    "Non-finite value found while preprocessing | "
                    f"sample_idx={idx} | obj={target_obj_name} | "
                    f"meta={value[0]} | act_id={act_id} | key={key} | "
                    f"path={nonfinite_path}"
                )
            total_dict[target_obj_name][key].append(data[key][idx])
    total_dict[target_obj_name]['meta'].append(data['meta'][idx][0])
    
total_dict = {obj_name: dict(obj_data) for obj_name, obj_data in total_dict.items()}            

os.makedirs(save_root, exist_ok=True)
with open(osp.join(save_root, f"{file_name}.pkl"), "wb") as f:
    pickle.dump(total_dict, f)
    
if skipped_empty_samples:
    print("Skipped empty samples while preprocessing:")
    for item in skipped_empty_samples:
        print(f"sample_idx={item['sample_idx']} | act_id={item['act_id']}")
    print(f"Total skipped empty samples: {len(skipped_empty_samples)}")

print("End")
