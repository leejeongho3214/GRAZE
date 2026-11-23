# HOT3D-Grasp annotations

Annotations of the HOT3D-Grasp benchmark. They refer to sequences and object models of
[HOT3D](https://facebookresearch.github.io/hot3d/), which you must obtain from the HOT3D
website under its license. No HOT3D data, poses or meshes are included here.

| File | Content |
|---|---|
| `clip_annotations.csv` | One row per annotated grasp action (1,044 rows): `action_ID`, HOT3D `sequence_ID`, `obj_name`, `contact_point` (frame), `reviewed_prompt` (verb, part, object, hand), `grasping_start`, `grasping_end` (frames, end exclusive). This is the sheet that `preprocess/extract_hot3d_clips.py` reads to build `acc_ori.pkl`. Cellphone and keyboard actions are dropped later by the preprocessing. |
| `threefold_assignment.json` | `object_folds`: the objects of folds obj0/obj1/obj2. `subject_folds`: the HOT3D participants of the subject folds. |
| `manifest.json` | Pool size (915 clips), clips per participant, train/test clip counts and held-out objects of every fold. |
| `labels_merged.json` | Part labels for 31 objects: `{object: {part: [point indices]}}`, indices into the 1,024-point cloud of `obj.pkl` (deterministic farthest point sampling from vertex 0 of the HOT3D mesh). |
| `excluded_clips.json` | Test clips without ground-truth hand-object contact, dropped from all metrics. Index = position in `<fold>_test.pkl`. |
| `text.json` | Prompt templates and paraphrases of the clips. |

## License

These annotation files are released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
They annotate HOT3D sequences and object models, so you must also comply with the
[HOT3D license](https://www.projectaria.com/datasets/hot3d/license/) when you use them.
