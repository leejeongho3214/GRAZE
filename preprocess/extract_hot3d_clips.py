"""Step 1 of the benchmark build: HOT3D recordings + grasp annotations -> data/hot3d/data/acc_ori.pkl.

For every annotated grasp action (one row of benchmark/hot3d_grasp/clip_annotations.csv) it reads
frames [grasping_start, grasping_end) of the HOT3D sequence and stores, per frame, the target object
pose, the poses of all scene objects, both MANO hands (124-d: flag, wrist translation, full pose,
betas per hand), the RGB camera extrinsics and the eye-gaze ray in world coordinates.

Needs the HOT3D toolkit (https://github.com/facebookresearch/hot3d, commit aae496b) with
preprocess/hot3d_toolkit.patch applied (it makes the MANO layer return the full pose), and
projectaria_tools. Run it with the toolkit's `hot3d/` folder on PYTHONPATH or pass --hot3d_toolkit:

    python preprocess/extract_hot3d_clips.py \
        --hot3d_toolkit /path/to/hot3d/hot3d \
        --hot3d_root /path/to/hot3d_dataset \
        --mano_dir data/mano/mano_v1_2/models \
        --annotations benchmark/hot3d_grasp/clip_annotations.csv \
        --out data/hot3d/data/acc_ori.pkl
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--hot3d_toolkit", default=None,
                     help="folder of the HOT3D toolkit that contains dataset_api.py (added to sys.path)")
_parser.add_argument("--hot3d_root", required=True,
                     help="folder with the HOT3D sequences <P00xx_xxxxxxxx>/ (object models are read from <hot3d_root>/../assets)")
_parser.add_argument("--assets", default=None, help="HOT3D object library folder (default: <hot3d_root>/../assets)")
_parser.add_argument("--mano_dir", default=os.path.join(_PROJECT_ROOT, "data/mano/mano_v1_2/models"),
                     help="folder with MANO_LEFT.pkl / MANO_RIGHT.pkl")
_parser.add_argument("--annotations", default=os.path.join(_PROJECT_ROOT, "benchmark/hot3d_grasp/clip_annotations.csv"),
                     help="grasp annotation sheet (.csv or .xlsx)")
_parser.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "data/hot3d/data/acc_ori.pkl"))
_parser.add_argument("--sequences", nargs="*", default=None,
                     help="only process these sequence IDs (for testing; default: all annotated sequences)")
_args = _parser.parse_args()
if _args.hot3d_toolkit is not None:
    sys.path.insert(0, os.path.abspath(_args.hot3d_toolkit))

from typing import Dict, List, Optional

import pandas as pd
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.stream_id import StreamId

import json
import glob
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.mps import get_eyegaze_point_at_depth

from dataclasses import dataclass

hot3d_dataset_path = _args.hot3d_root
object_library_path = _args.assets if _args.assets is not None else os.path.join(hot3d_dataset_path, "../assets")
object_library = load_object_library(object_library_folderpath=object_library_path)

mano_hand_model_path = _args.mano_dir
mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)

@dataclass
class GraspAnnotation:
    sequence_id: str
    obj_name: str
    contact_point: int
    reviewed_prompt: Optional[str]
    grasp_start: int
    grasp_end: int
    action_ID: int

def load_annotations(path: str) -> List[GraspAnnotation]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    required_columns = {
        "sequence_ID",
        "obj_name",
        "contact_point",
        "reviewed_prompt",
        "grasping_start",
        "grasping_end",
        "action_ID"
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in annotation sheet: {sorted(missing)}")
    annotations: List[GraspAnnotation] = []
    for row in df.itertuples(index=False):
        if pd.isna(row.sequence_ID) or pd.isna(row.obj_name):
            continue
        try:
            grasp_start = int(row.grasping_start)
            grasp_end = int(row.grasping_end)
        except (TypeError, ValueError):
            continue
        action_ID = int(row.action_ID)
        annotations.append(
            GraspAnnotation(
                sequence_id=str(row.sequence_ID),
                obj_name=str(row.obj_name),
                contact_point=int(row.contact_point) if not pd.isna(row.contact_point) else -1,
                reviewed_prompt=None if pd.isna(row.reviewed_prompt) else str(row.reviewed_prompt),
                grasp_start=grasp_start,
                grasp_end=grasp_end,
                action_ID=action_ID
            )
        )
    return annotations


def get_seqeunce(seq_name):
    sequence_path = os.path.join(hot3d_dataset_path, seq_name)
    
    if not os.path.exists(sequence_path) or not os.path.exists(object_library_path):
        print("Invalid input sequence or library path.")
        print("Please do update the path to VALID values for your system.")
        raise

    hot3d_data_provider = Hot3dDataProvider(
        sequence_folder=sequence_path,
        object_library=object_library,
        mano_hand_model=mano_hand_model,)
    
    return hot3d_data_provider


total_act_id = 0

default_annotation_file = _args.annotations

_loaded_annotations = load_annotations(default_annotation_file)
annotation_groups: Dict[str, List[GraspAnnotation]] = defaultdict(list)
for annotation in _loaded_annotations:
    annotation_groups[annotation.sequence_id].append(annotation)
annotations = dict(annotation_groups)
if _args.sequences:
    annotations = {seq: rows for seq, rows in annotations.items() if seq in set(_args.sequences)}

total_obj_pose_rts = []
total_all_obj_pose_rts = []
total_all_obj_names = []
total_hand_pose_manos = []
total_cam_poses = []
total_action_labels = []
total_gaze = []
total_meta = []
total_act_id = []
missing_data_reasons = defaultdict(set)
reason_messages = {
    "frame_index_out_of_range": "Frame index out of range for action ID(s)",
    "hand_data_provider": "Hand data provider missing for action ID(s)",
    "hand_pose_timestamp": "Hand pose missing at timestamp for action ID(s)",
    "object_pose_timestamp": "Object pose timestamp missing for action ID(s)",
    "object_pose_missing": "Object pose missing for action ID(s)",
    "device_pose_provider": "Device pose provider missing for action ID(s)",
    "headset_pose_missing": "Headset pose missing at timestamp for action ID(s)",
    "gaze_missing": "Gaze missing for action ID(s)",
    "gaze_nonfinite": "Gaze contains NaN/Inf for action ID(s)",
}
filled_single_frame_gaps = defaultdict(int)
dropped_scene_objects = {}


def to_numpy(value):
    if value is None:
        raise ValueError("Cannot convert None to numpy array")
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.array(value)


def assert_finite(name, value, seq_name, obj_name, action_id, frame_index, hand=None):
    array = np.asarray(value)
    hand_info = f" | hand={hand}" if hand is not None else ""
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Non-numeric value found while building acc_ori.pkl | "
            f"seq={seq_name} | obj={obj_name} | action_id={action_id} | "
            f"frame_index={frame_index}{hand_info} | key={name} | "
            f"dtype={array.dtype} | shape={array.shape}"
        ) from exc
    if np.isfinite(array).all():
        return
    bad_idx = np.argwhere(~np.isfinite(array))
    first_bad = bad_idx[0].tolist() if bad_idx.size else None
    raise RuntimeError(
        "Non-finite value found while building acc_ori.pkl | "
        f"seq={seq_name} | obj={obj_name} | action_id={action_id} | "
        f"frame_index={frame_index}{hand_info} | key={name} | "
        f"shape={array.shape} | first_bad_index={first_bad}"
    )


def is_finite(value):
    try:
        return np.isfinite(np.asarray(value, dtype=np.float64)).all()
    except (TypeError, ValueError):
        return False


def copy_frame_record(record):
    return {
        "obj_pose_rt": record["obj_pose_rt"].copy(),
        "all_obj_pose_by_id": {
            object_id: object_pose_rt.copy()
            for object_id, object_pose_rt in record["all_obj_pose_by_id"].items()
        },
        "hand_pose_mano": record["hand_pose_mano"].copy(),
        "cam_pose": record["cam_pose"].copy(),
        "action_label": record["action_label"],
        "gaze": [record["gaze"][0].copy(), record["gaze"][1].copy()],
        "meta": list(record["meta"]),
        "act_id": record["act_id"],
    }


def append_frame_record(
    record,
    obj_pose_rts,
    hand_pose_manos,
    cam_poses,
    action_labels,
    gaze,
    meta,
    act_id,
):
    obj_pose_rts.append(record["obj_pose_rt"])
    hand_pose_manos.append(record["hand_pose_mano"])
    cam_poses.append(record["cam_pose"])
    action_labels.append(record["action_label"])
    gaze.append(record["gaze"])
    meta.append(record["meta"])
    act_id.append(record["act_id"])


def object_name_from_id(object_id):
    return object_library.object_id_to_name_dict.get(
        str(object_id),
        object_library.object_id_to_name_dict.get(object_id, str(object_id)),
    )


def object_id_sort_key(object_id):
    try:
        return (0, int(object_id))
    except (TypeError, ValueError):
        return (1, str(object_id))


def object_id_value(object_id):
    try:
        return int(object_id)
    except (TypeError, ValueError):
        return str(object_id)


def build_complete_scene_objects(output_records):
    if not output_records:
        return [], [], []

    present_object_ids = [set(record["all_obj_pose_by_id"]) for record in output_records]
    complete_object_ids = set.intersection(*present_object_ids)
    all_seen_object_ids = set.union(*present_object_ids)

    complete_object_ids = sorted(complete_object_ids, key=object_id_sort_key)
    dropped_object_ids = sorted(all_seen_object_ids - set(complete_object_ids), key=object_id_sort_key)

    all_obj_pose_rts = []
    all_obj_names = []
    for record in output_records:
        frame_obj_pose_rts = []
        frame_obj_names = []
        for object_id in complete_object_ids:
            numeric_object_id = object_id_value(object_id)
            frame_obj_pose_rts.append(record["all_obj_pose_by_id"][object_id])
            frame_obj_names.append([numeric_object_id, object_name_from_id(object_id)])
        all_obj_pose_rts.append(frame_obj_pose_rts)
        all_obj_names.append(frame_obj_names)

    return all_obj_pose_rts, all_obj_names, dropped_object_ids


for seq_name, seq_data in tqdm(annotations.items()):
    hot3d_data_provider = get_seqeunce(seq_name)
    hand_data_provider = hot3d_data_provider.mano_hand_data_provider

    device_data_provider = hot3d_data_provider.device_data_provider
    timestamps = device_data_provider.get_sequence_timestamps()
    T_device_cpf = hot3d_data_provider.device_data_provider.get_device_calibration().get_transform_device_cpf()
 
    for seq_dict in seq_data:
        obj_name = seq_dict.obj_name
        obj_id = object_library.object_name_to_id_dict[obj_name]
        
        obj_pose_rts = []
        all_obj_pose_rts = []
        all_obj_names = []
        hand_pose_manos = []
        meta = []
        gaze = []
        cam_poses = []
        action_labels = []
        act_id = []
        frame_records = []
        output_records = []

        def mark_missing(reason):
            missing_data_reasons[reason].add(seq_dict.action_ID)
            frame_records.append(None)

        for p in range(seq_dict.grasp_start, seq_dict.grasp_end):
            if p < 0 or p >= len(timestamps):
                mark_missing("frame_index_out_of_range")
                continue
            time_ns = timestamps[p]
            
            hand_poses_with_dt = None
            if hand_data_provider is None:
                mark_missing("hand_data_provider")
                continue
            
            hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
                acceptable_time_delta=0,
                timestamp_ns=time_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )

            if hand_poses_with_dt is None:
                mark_missing("hand_pose_timestamp")
                continue
                
            hand_pose_collection = hand_poses_with_dt.pose3d_collection
            hand_pose_mano_data = np.zeros(124)

            for hand_pose_data in hand_pose_collection.poses.values():
                # Retrieve the handedness of the hand (i.e Left or Right)
                handedness_label = hand_pose_data.handedness_label()

                # Skeleton/Joints landmark representation (for LEFT hand)
                if hand_pose_data.is_left_hand():
                    hand_mesh_vertices, landmarks, mano_output, _ = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                    full_pose = to_numpy(mano_output.full_pose).flatten()
                    hand_transl = np.asarray(
                        hand_pose_data.wrist_pose.to_matrix()[:3, 3], dtype=np.float64
                    )
                    assert_finite("left_full_pose", full_pose, seq_name, obj_name, seq_dict.action_ID, p, "left")
                    assert_finite("left_transl", hand_transl, seq_name, obj_name, seq_dict.action_ID, p, "left")
                    assert_finite("mano_shape_params", hand_data_provider._mano_shape_params, seq_name, obj_name, seq_dict.action_ID, p, "left")
                    hand_pose_mano_data[0] = 1
                    hand_pose_mano_data[1:4] = np.array(hand_transl)
                    hand_pose_mano_data[4:52] = np.array(full_pose)
                    hand_pose_mano_data[52:62] = hand_data_provider._mano_shape_params
                    
                if hand_pose_data.is_right_hand():
                    hand_mesh_vertices, landmarks, _, mano_output = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                    full_pose = to_numpy(mano_output.full_pose).flatten()
                    hand_transl = np.asarray(
                        hand_pose_data.wrist_pose.to_matrix()[:3, 3], dtype=np.float64
                    )
                    assert_finite("right_full_pose", full_pose, seq_name, obj_name, seq_dict.action_ID, p, "right")
                    assert_finite("right_transl", hand_transl, seq_name, obj_name, seq_dict.action_ID, p, "right")
                    assert_finite("mano_shape_params", hand_data_provider._mano_shape_params, seq_name, obj_name, seq_dict.action_ID, p, "right")
                    hand_pose_mano_data[62] = 1
                    hand_pose_mano_data[63:66] = np.array(hand_transl)
                    hand_pose_mano_data[66:114] = np.array(full_pose)
                    hand_pose_mano_data[114:124] = hand_data_provider._mano_shape_params
                    
            object_pose_dt = hot3d_data_provider.object_pose_data_provider.get_pose_at_timestamp(
                acceptable_time_delta=0,
                    timestamp_ns=time_ns,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                )
            if object_pose_dt is None:
                mark_missing("object_pose_timestamp")
                continue

            if obj_id not in object_pose_dt.pose3d_collection.poses:
                mark_missing("object_pose_missing")
                continue

            all_obj_pose_by_id = {}
            for scene_obj_id, scene_pose_data in object_pose_dt.pose3d_collection.poses.items():
                try:
                    numeric_scene_obj_id = int(scene_obj_id)
                except (TypeError, ValueError):
                    continue
                scene_T = scene_pose_data.T_world_object.to_matrix()
                scene_obj_pose_rt = np.concatenate(([numeric_scene_obj_id], scene_T.flatten()))
                if is_finite(scene_obj_pose_rt):
                    all_obj_pose_by_id[str(scene_obj_id)] = scene_obj_pose_rt
            
            pose_data = object_pose_dt.pose3d_collection.poses[obj_id]
            T = pose_data.T_world_object.to_matrix()
            
            device_pose_provider = hot3d_data_provider.device_pose_data_provider
            if device_pose_provider is None:
                mark_missing("device_pose_provider")
                continue

            headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
                acceptable_time_delta=0,
            timestamp_ns=time_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
                )
            if headset_pose3d_with_dt is None:
                mark_missing("headset_pose_missing")
                continue
            headset_pose3d = headset_pose3d_with_dt.pose3d
            [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(
            StreamId("214-1")
            )
            
            eye_gaze = device_data_provider.get_eye_gaze(timestamp_ns=time_ns)
            if eye_gaze is None:
                mark_missing("gaze_missing")
                continue
            if not is_finite([eye_gaze.yaw, eye_gaze.pitch]):
                mark_missing("gaze_nonfinite")
                continue
            gaze_vector_in_cpf = get_eyegaze_point_at_depth(
                eye_gaze.yaw, eye_gaze.pitch, depth_m=1.0  # only the direction is used, so the depth is arbitrary
            )
            
            gaze_origin = (headset_pose3d.T_world_device @ T_device_cpf @ np.array([0, 0, 0]))
            gaze_target = (headset_pose3d.T_world_device @ T_device_cpf @ gaze_vector_in_cpf)
            gaze_direction = gaze_target - gaze_origin
            if not is_finite([gaze_origin, gaze_direction]):
                mark_missing("gaze_nonfinite")
                continue
                        
            world_camera_pose = headset_pose3d.T_world_device @ extrinsics

            obj_pose_rt = np.concatenate(([int(obj_id)], T.flatten()))
            cam_pose = extrinsics.to_matrix()
            assert_finite("obj_pose_rt", np.concatenate(([int(obj_id)], T.flatten())), seq_name, obj_name, seq_dict.action_ID, p)
            assert_finite("hand_pose_mano_data", hand_pose_mano_data, seq_name, obj_name, seq_dict.action_ID, p)
            assert_finite("cam_pose", cam_pose, seq_name, obj_name, seq_dict.action_ID, p)
            assert_finite("gaze", [gaze_origin, gaze_direction], seq_name, obj_name, seq_dict.action_ID, p)

            frame_records.append(
                {
                    "obj_pose_rt": obj_pose_rt,
                    "all_obj_pose_by_id": all_obj_pose_by_id,
                    "hand_pose_mano": hand_pose_mano_data,
                    "cam_pose": cam_pose,  ## world -> camera
                    "action_label": seq_dict.reviewed_prompt,
                    "gaze": [gaze_origin, gaze_direction],
                    "meta": [seq_name, obj_name],
                    "act_id": seq_dict.action_ID,
                }
            )

        i = 0
        while i < len(frame_records):
            if frame_records[i] is not None:
                append_frame_record(
                    frame_records[i],
                    obj_pose_rts,
                    hand_pose_manos,
                    cam_poses,
                    action_labels,
                    gaze,
                    meta,
                    act_id,
                )
                output_records.append(frame_records[i])
                i += 1
                continue

            run_start = i
            while i < len(frame_records) and frame_records[i] is None:
                i += 1
            run_end = i
            run_length = run_end - run_start
            if run_length != 1:
                continue

            fill_source = None
            if run_start > 0 and frame_records[run_start - 1] is not None:
                fill_source = frame_records[run_start - 1]
            elif run_end < len(frame_records) and frame_records[run_end] is not None:
                fill_source = frame_records[run_end]

            if fill_source is None:
                continue

            filled_record = copy_frame_record(fill_source)
            append_frame_record(
                filled_record,
                obj_pose_rts,
                hand_pose_manos,
                cam_poses,
                action_labels,
                gaze,
                meta,
                act_id,
            )
            output_records.append(filled_record)
            filled_single_frame_gaps[seq_dict.action_ID] += 1

        all_obj_pose_rts, all_obj_names, dropped_object_ids = build_complete_scene_objects(output_records)
        if dropped_object_ids:
            dropped_scene_objects[seq_dict.action_ID] = [
                [object_id_value(object_id), object_name_from_id(object_id)]
                for object_id in dropped_object_ids
            ]
        
        total_obj_pose_rts.append(obj_pose_rts)
        total_all_obj_pose_rts.append(all_obj_pose_rts)
        total_all_obj_names.append(all_obj_names)
        total_hand_pose_manos.append(hand_pose_manos)
        total_meta.append(meta)
        total_gaze.append(gaze)
        total_cam_poses.append(cam_poses)
        total_action_labels.append(action_labels)
        total_act_id.append(act_id)
            
save_dict = {
    "obj_pose_rts": total_obj_pose_rts,
    "all_obj_pose_rts": total_all_obj_pose_rts,
    "all_obj_names": total_all_obj_names,
    "cam_poses": total_cam_poses,
    "hand_pose_manos": total_hand_pose_manos,
    "action_labels": total_action_labels,
    "meta": total_meta,
    "gaze": total_gaze,
    "act_id": total_act_id,
}

save_folder = os.path.dirname(os.path.abspath(_args.out))
os.makedirs(save_folder, exist_ok=True)
save_name = os.path.basename(_args.out)

with open(os.path.join(save_folder, f"{save_name}"), "wb") as f:
    pickle.dump(save_dict, f)

print(f"Data saved successfully. {save_folder}/{save_name}")
if missing_data_reasons:
    print("Summary of missing data by reason before one-frame filling:")
    for reason, action_ids in missing_data_reasons.items():
        message = reason_messages.get(reason, reason)
        print(f"{message}: {sorted(action_ids)}")
if filled_single_frame_gaps:
    total_filled = sum(filled_single_frame_gaps.values())
    print(f"Filled single-frame gaps by copying an adjacent valid frame: {total_filled}")
    print(f"Action ID(s) with filled single-frame gaps: {dict(sorted(filled_single_frame_gaps.items()))}")
if dropped_scene_objects:
    total_dropped = sum(len(object_ids) for object_ids in dropped_scene_objects.values())
    print(
        "Dropped scene objects with incomplete pose tracks: "
        f"{total_dropped} object track(s) across {len(dropped_scene_objects)} action(s)"
    )
    preview_items = list(sorted(dropped_scene_objects.items()))[:20]
    print(f"First dropped scene object action(s): {dict(preview_items)}")
