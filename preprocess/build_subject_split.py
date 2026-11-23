"""Build the subject-disjoint HOT3D benchmark from the existing split pickles.

The shipped ``gaze_train``/``gaze_test`` pair is an object-wise random 10% split:
sibling clips from one recording session end up on both sides, and the ``subject``
field stored in those pickles is a constant string (a path-parsing bug in
``preprocess/preprocessing_hot3d.py`` that is now fixed). Rerunning the full
preprocessing takes hours of GPU time, so this script instead recovers the true
subject of every processed clip and re-partitions the same sample pool by person.

Subject recovery uses the MANO shape parameters: ``acc_ori.pkl`` groups clips by
recording (``meta[i] == [recording_name, object_name]``) and every distinct beta
sequence belongs to exactly one subject, which was verified over all 965 motions.
Clips whose beta bytes do not hash-match fall back to a nearest-neighbour lookup
on the mean beta vector.

Usage:
    python scripts/build_subject_split.py                 # write the pickles
    python scripts/build_subject_split.py --dry-run       # report the split only
"""

import argparse
import collections
import hashlib
import json
import os
import os.path as osp
import pickle

import numpy as np

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATASET_DIR = osp.join(PROJECT_ROOT, "data/hot3d/dataset/ori_dataset")
ACC_PATH = osp.join(PROJECT_ROOT, "data/hot3d/pre-process_dataset/acc_ori.pkl")
INSTANCE_JSON = osp.join(PROJECT_ROOT, "data/hot3d/previous/instance.json")

# Objects dropped during preprocessing; excluded here so the beta index matches.
EXCLUDED_OBJECTS = ("cellphone", "keyboard")
# See SUBJECT_SPLIT_TEST_SUBJECTS in preprocess/preprocessing_hot3d.py.
DEFAULT_TEST_SUBJECTS = ("P0012", "P0015")
# A fallback match is accepted only when the nearest clip of the winning subject
# is at least this many times closer than the nearest clip of any other subject.
# Subject beta centroids sit ~1.0 apart while within-subject spread stays below
# 0.08, so the true subject wins by a wide margin whenever the match is real.
FALLBACK_MIN_MARGIN = 3.0


def _hash(array, decimals=4):
    values = np.round(np.asarray(array, dtype=np.float64), decimals)
    return hashlib.md5(np.ascontiguousarray(values).tobytes()).hexdigest()


def build_beta_index():
    """Map beta-sequence hashes (and mean beta vectors) to their subject."""
    with open(ACC_PATH, "rb") as f:
        acc = pickle.load(f)

    by_hash = {}
    means, mean_subjects = [], []
    for object_name, value_list in acc.items():
        if object_name in EXCLUDED_OBJECTS:
            continue
        for index, meta in enumerate(value_list["meta"]):
            subject = str(meta[0]).split("_")[0]
            betas = np.asarray(value_list["hand_pose_manos"][index])[:, 114:124]
            digest = _hash(betas)
            previous = by_hash.setdefault(digest, subject)
            if previous != subject:
                raise RuntimeError(
                    f"Beta signature {digest} maps to both {previous} and {subject}; "
                    "the shape parameters are not subject-specific after all."
                )
            means.append(betas.mean(axis=0))
            mean_subjects.append(subject)
    return by_hash, np.asarray(means, dtype=np.float64), mean_subjects


def resolve_subjects(rhand_betas, by_hash, mean_table, mean_subjects):
    subjects, fallback_count = [], 0
    for betas in rhand_betas:
        subject = by_hash.get(_hash(betas))
        if subject is None:
            # A hash miss just means the clip was trimmed to a different frame
            # count than the source motion; the shape itself still identifies
            # the person, so fall back to nearest mean beta with a margin check.
            query = np.asarray(betas, dtype=np.float64).mean(axis=0)
            distances = np.linalg.norm(mean_table - query, axis=1)
            nearest = int(np.argmin(distances))
            subject = mean_subjects[nearest]
            other = distances[np.asarray(mean_subjects) != subject]
            runner_up = float(other.min()) if other.size else np.inf
            if distances[nearest] * FALLBACK_MIN_MARGIN > runner_up:
                raise RuntimeError(
                    "Could not attribute a clip to a subject: nearest "
                    f"{subject} at {distances[nearest]:.4f} but another subject "
                    f"sits at {runner_up:.4f}."
                )
            fallback_count += 1
        subjects.append(subject)
    return subjects, fallback_count


def load_pool(source_names):
    """Concatenate the source splits back into one sample pool."""
    pool, sample_count = None, 0
    for name in source_names:
        with open(osp.join(DATASET_DIR, f"{name}.pkl"), "rb") as f:
            part = dict(pickle.load(f))
        length = len(part["object_idx"])
        if pool is None:
            pool = {key: list(value) for key, value in part.items()}
        else:
            missing = set(pool) ^ set(part)
            if missing:
                raise RuntimeError(f"Split {name} has mismatched keys: {sorted(missing)}")
            for key, value in part.items():
                pool[key].extend(list(value))
        sample_count += length
        print(f"  loaded {name}: {length} samples")
    for key, value in pool.items():
        if len(value) != sample_count:
            raise RuntimeError(f"Field '{key}' has {len(value)} entries, expected {sample_count}")
    return pool, sample_count


def subset(pool, indices, reference):
    """Rebuild a split dict, restoring each field's original container type."""
    output = {}
    for key, values in pool.items():
        selected = [values[i] for i in indices]
        original = reference[key]
        if isinstance(original, np.ndarray):
            if original.dtype == object:
                array = np.empty(len(selected), dtype=object)
                for position, item in enumerate(selected):
                    array[position] = item
                output[key] = array
            elif original.dtype.kind in "US":
                # Let numpy size the string width: the recovered subject ids are
                # wider than the placeholder values the reference field held.
                output[key] = np.array(selected)
            else:
                output[key] = np.array(selected, dtype=original.dtype)
        else:
            output[key] = selected
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["gaze_train", "gaze_test"],
        help="Split pickles to merge back into one pool before re-splitting.",
    )
    parser.add_argument("--out-prefix", default="gaze_subj")
    parser.add_argument("--test-subjects", nargs="+", default=list(DEFAULT_TEST_SUBJECTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    test_subjects = set(args.test_subjects)
    print(f"Test subjects: {sorted(test_subjects)}")

    print("Indexing subjects by MANO shape parameters...")
    by_hash, mean_table, mean_subjects = build_beta_index()
    print(f"  {len(by_hash)} beta signatures over {len(mean_subjects)} source motions")

    print("Loading source splits...")
    pool, sample_count = load_pool(args.sources)

    subjects, fallback_count = resolve_subjects(
        pool["rhand_beta"], by_hash, mean_table, mean_subjects
    )
    print(
        f"Resolved {sample_count} clips to {len(set(subjects))} subjects "
        f"({fallback_count} via nearest-beta fallback)"
    )
    unknown = test_subjects - set(subjects)
    if unknown:
        raise SystemExit(f"Requested test subjects are absent from the pool: {sorted(unknown)}")

    # Overwrite the bogus constant subject field with the recovered identity.
    pool["subject"] = subjects

    with open(INSTANCE_JSON) as f:
        instances = json.load(f)
    object_names = [
        instances[str(int(index))]["instance_name"] for index in pool["object_idx"]
    ]

    train_indices = [i for i, s in enumerate(subjects) if s not in test_subjects]
    test_indices = [i for i, s in enumerate(subjects) if s in test_subjects]
    train_objects = {object_names[i] for i in train_indices}
    test_objects = {object_names[i] for i in test_indices}

    print(
        f"train {len(train_indices)} clips / {len(train_objects)} objects | "
        f"test {len(test_indices)} clips / {len(test_objects)} objects "
        f"({100.0 * len(test_indices) / sample_count:.1f}%)"
    )
    missing = train_objects - test_objects
    if missing:
        print(f"  objects absent from test: {sorted(missing)}")
    unseen = test_objects - train_objects
    if unseen:
        print(f"  WARNING: objects in test but never trained on: {sorted(unseen)}")
    print("  clips per subject:", dict(sorted(collections.Counter(subjects).items())))

    manifest = {
        "sources": args.sources,
        "test_subjects": sorted(test_subjects),
        "train_clips": len(train_indices),
        "test_clips": len(test_indices),
        "train_objects": sorted(train_objects),
        "test_objects": sorted(test_objects),
        "clips_per_subject": dict(sorted(collections.Counter(subjects).items())),
        "beta_fallback_matches": fallback_count,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    with open(osp.join(DATASET_DIR, f"{args.sources[0]}.pkl"), "rb") as f:
        reference = dict(pickle.load(f))

    os.makedirs(DATASET_DIR, exist_ok=True)
    for suffix, indices in (("train", train_indices), ("test", test_indices)):
        path = osp.join(DATASET_DIR, f"{args.out_prefix}_{suffix}.pkl")
        with open(path, "wb") as f:
            pickle.dump(subset(pool, indices, reference), f)
        print(f"wrote {path} ({len(indices)} samples)")

    manifest_path = osp.join(DATASET_DIR, f"{args.out_prefix}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
