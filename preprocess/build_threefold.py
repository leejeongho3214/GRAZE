#!/usr/bin/env python
"""Build the subject / object 3-fold splits from the pooled HOT3D clips.

Pools gaze_train + gaze_test (915 clips), recovers each clip's subject from its MANO betas
(scripts/build_subject_split.py), and for every fold k of benchmark/hot3d_grasp/threefold_assignment.json writes

    data/hot3d/dataset/threefold/{obj,subj}{k}_train.pkl   (clips of the other two folds)
    data/hot3d/dataset/threefold/{obj,subj}{k}_test.pkl    (clips of fold k)

in the same dict-of-lists format as gaze_train.pkl, plus manifest.json.  Object folds hold out
whole objects (unseen-object test); subject folds hold out whole people.
"""
import collections
import json
import os
import os.path as osp
import pickle
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from build_subject_split import (  # noqa: E402
    DATASET_DIR, INSTANCE_JSON, build_beta_index, load_pool, resolve_subjects, subset,
)

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
ASSIGN = osp.join(PROJECT_ROOT, "benchmark/hot3d_grasp/threefold_assignment.json")
OUT_DIR = osp.join(osp.dirname(DATASET_DIR), "threefold")


def main():
    folds = json.load(open(ASSIGN))
    by_hash, mean_table, mean_subjects = build_beta_index()
    pool, n = load_pool(["gaze_train", "gaze_test"])
    subjects, fallback = resolve_subjects(pool["rhand_beta"], by_hash, mean_table, mean_subjects)
    pool["subject"] = subjects
    inst = json.load(open(INSTANCE_JSON))
    objects = [inst[str(int(i))]["instance_name"] for i in pool["object_idx"]]
    with open(osp.join(DATASET_DIR, "gaze_train.pkl"), "rb") as f:
        reference = dict(pickle.load(f))
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"pool_clips": n, "beta_fallback_matches": fallback,
                "clips_per_subject": dict(sorted(collections.Counter(subjects).items())), "folds": {}}
    for kind, key, groups in (("obj", objects, folds["object_folds"]), ("subj", subjects, folds["subject_folds"])):
        assert sorted(set().union(*map(set, groups))) == sorted(set(key)), f"{kind} folds do not cover the pool"
        for k, held in enumerate(groups):
            held = set(held)
            test = [i for i in range(n) if key[i] in held]
            train = [i for i in range(n) if key[i] not in held]
            name = f"{kind}{k}"
            for split, idx in (("train", train), ("test", test)):
                with open(osp.join(OUT_DIR, f"{name}_{split}.pkl"), "wb") as f:
                    pickle.dump(subset(pool, idx, reference), f)
            manifest["folds"][name] = dict(
                held_out=sorted(held), train_clips=len(train), test_clips=len(test),
                test_objects=sorted({objects[i] for i in test}),
                unseen_test_objects=sorted({objects[i] for i in test} - {objects[i] for i in train}))
            print(f"{name}: train {len(train)} / test {len(test)} clips, held out {sorted(held)}")
    json.dump(manifest, open(osp.join(OUT_DIR, "manifest.json"), "w"), indent=1)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
