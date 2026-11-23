"""Optional helper for step 1: run extract_hot3d_clips.py in parallel and merge the outputs.

extract_hot3d_clips.py takes about 1.5 min per HOT3D sequence (about 3.3 h for all 132). It can be
split over processes with --sequences. Each output is a dict of per-action lists, ordered by
sequence (sequences in order of first appearance in the annotation sheet). This script

  1. writes N contiguous chunks of that sequence order (one text file of sequence IDs per shard):

        python preprocess/merge_acc_shards.py split --num_shards 10 --out_dir shards/

  2. after `extract_hot3d_clips.py --sequences $(cat shards/shardK.txt) --out shards/shardK.pkl`
     has run for every K, concatenates the shard outputs in annotation order:

        python preprocess/merge_acc_shards.py merge --out data/hot3d/data/acc_ori.pkl shards/shard*.pkl

The merged file holds the same actions, arrays and values as a single-process run.
"""
import argparse
import os
import pickle

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ANN = os.path.join(_PROJECT_ROOT, "benchmark/hot3d_grasp/clip_annotations.csv")


def sequence_order(annotation_path):
    """Sequence IDs in order of first appearance, as extract_hot3d_clips.py groups them."""
    if annotation_path.lower().endswith(".csv"):
        df = pd.read_csv(annotation_path)
    else:
        df = pd.read_excel(annotation_path)
    order = []
    for seq, obj in zip(df["sequence_ID"], df["obj_name"]):
        if pd.isna(seq) or pd.isna(obj):
            continue
        if str(seq) not in order:
            order.append(str(seq))
    return order


def split(args):
    order = sequence_order(args.annotations)
    n = args.num_shards
    os.makedirs(args.out_dir, exist_ok=True)
    size, rest = divmod(len(order), n)
    start = 0
    for k in range(n):
        end = start + size + (1 if k < rest else 0)
        with open(os.path.join(args.out_dir, f"shard{k:02d}.txt"), "w") as f:
            f.write(" ".join(order[start:end]) + "\n")
        start = end
    print(f"{len(order)} sequences -> {n} shard lists in {args.out_dir}")


def shard_sequences(data):
    seqs = []
    for meta in data["meta"]:
        if len(meta):
            seq = meta[0][0]
            if not seqs or seqs[-1] != seq:
                seqs.append(seq)
    return seqs


def merge(args):
    order = sequence_order(args.annotations)
    pos = {s: i for i, s in enumerate(order)}
    shards = []
    for path in args.shards:
        with open(path, "rb") as f:
            data = pickle.load(f)
        seqs = shard_sequences(data)
        if not seqs:
            print(f"skipping {path}: no actions")
            continue
        shards.append((pos[seqs[0]], path, data, seqs))
    shards.sort(key=lambda x: x[0])

    keys = list(shards[0][2].keys())
    merged = {k: [] for k in keys}
    merged_seqs = []
    for _, path, data, seqs in shards:
        if list(data.keys()) != keys:
            raise ValueError(f"{path}: keys differ from the first shard")
        for k in keys:
            merged[k].extend(data[k])
        merged_seqs.extend(seqs)

    positions = [pos[s] for s in merged_seqs]
    if positions != sorted(positions) or len(set(merged_seqs)) != len(merged_seqs):
        raise ValueError("shards do not hold contiguous chunks of the annotation sequence order "
                         "(use the lists written by `split`)")
    missing = [s for s in order if s not in set(merged_seqs)]
    if missing:
        print(f"warning: {len(missing)} annotated sequences have no actions in the shards: {missing[:10]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(merged, f)
    print(f"{len(shards)} shards, {len(merged_seqs)} sequences, {len(merged[keys[0]])} actions -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", default=_DEFAULT_ANN, help="grasp annotation sheet (.csv or .xlsx)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_split = sub.add_parser("split", help="write N contiguous sequence lists")
    p_split.add_argument("--num_shards", type=int, required=True)
    p_split.add_argument("--out_dir", required=True)
    p_merge = sub.add_parser("merge", help="concatenate shard outputs in annotation order")
    p_merge.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "data/hot3d/data/acc_ori.pkl"))
    p_merge.add_argument("shards", nargs="+")
    args = parser.parse_args()
    split(args) if args.cmd == "split" else merge(args)


if __name__ == "__main__":
    main()
