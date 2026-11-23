#!/usr/bin/env python
"""Aggregate evaluator outputs into the HOT3D-Grasp Table 1 row (GSR, Part Acc, PCP, ID, Sim, SimD).

Per sample (one generated clip of one seed), from the per-frame CSV of eval/run_eval_gpu.sh:
  GSR     = Con_pass == 1 on the last 5 frames AND final-frame ID_max_mm <= 10
            AND final-frame object lift along the HOT3D world up >= 5 mm (Object_lift_y_m)
  Part    = final-frame Part_Acc;  PCP = final-frame PCP
  ID      = mean of ID_max_mm over the frames that penetrate (ID_mean_mm > 0); samples
            without penetration are left out of the ID mean
  Sim     = SimStable_2cm, SimD = sim_disp_cm (from eval/hot3d/visualize-code/sim_displacement.py)
The clip index of a sample is its sample_idx minus the smallest sample_idx of its seed in that
file. Clips listed in benchmark/hot3d_grasp/excluded_clips.json (no ground-truth contact) are
dropped. Per fold: GSR, Sim and SimD are sample means; Part, PCP and ID are per-seed means,
then averaged over seeds. The row is the plain mean of the three folds.

Example:
  python scripts/make_table1.py \
      --csv obj0 results/graze_obj0.csv --csv obj1 results/graze_obj1.csv --csv obj2 results/graze_obj2.csv \
      --sim obj0 results/sim_obj0.csv  --sim obj1 results/sim_obj1.csv  --sim obj2 results/sim_obj2.csv
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
NUM_TEST_CLIPS = {"obj0": 302, "obj1": 311, "obj2": 302}
COLS = ["file_name", "text", "sample_idx", "seed", "frame_idx", "ID_mean_mm", "ID_max_mm",
        "Con_pass", "Object_lift_y_m", "Part_Acc", "PCP"]
NUMERIC = COLS[2:]


def per_sample(csv_path, fold, gsr_frames=5, id_max_mm=10.0, lift_m=0.005):
    d = pd.read_csv(csv_path, usecols=COLS, keep_default_na=False, na_values=[""], low_memory=False)
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = []
    for file_name, df in d.groupby("file_name", sort=False):
        df = df.sort_values("frame_idx", kind="stable")
        g = df.groupby("sample_idx")
        last = g.tail(1).set_index("sample_idx").sort_index()
        con5 = g.Con_pass.apply(lambda s: bool((s.tail(gsr_frames) == 1).all())).reindex(last.index)
        pen = df[(df.ID_mean_mm > 0) & df.ID_max_mm.notna() & ~np.isclose(df.ID_max_mm.fillna(0), 0.0)]
        id_mean = pen.groupby("sample_idx").ID_max_mm.mean().reindex(last.index)
        o = pd.DataFrame(dict(
            file=file_name, sample_idx=last.index.values, seed=last.seed.values, text=last.text.values,
            con5=con5.values.astype(float), id_final=last.ID_max_mm.values, lift=last.Object_lift_y_m.values,
            part=last.Part_Acc.values, pcp=last.PCP.values, id=id_mean.values))
        o["gsr"] = ((o.con5 == 1) & (o.id_final <= id_max_mm) & (o.lift >= lift_m)).astype(float)
        o["clip"] = o.sample_idx - o.groupby("seed").sample_idx.transform("min")
        out.append(o)
    o = pd.concat(out, ignore_index=True)
    n = NUM_TEST_CLIPS[fold]
    sizes = o.groupby(["file", "seed"]).size()
    if not (sizes == n).all():
        print(f"[warn] {fold}: expected {n} clips per seed, got {sizes.to_dict()}")
    if o.groupby(["seed", "clip"]).size().max() != 1:
        raise RuntimeError(f"{fold}: a (seed, clip) pair appears twice; pass each seed once")
    return o


def attach_sim(o, sim_paths):
    s = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in sim_paths], ignore_index=True)
    s["file"] = s.file_name.str.replace(r"\.pkl$", "", regex=True)
    o["file_key"] = o.file.str.replace(r"\.pkl$", "", regex=True)
    sim, simd = [], []
    for key, idx in o.groupby("file_key").groups.items():
        ss = s[s.file.map(osp.basename) == osp.basename(key)].set_index("sample_idx")
        sub = o.loc[idx]
        if len(ss) == len(sub) and ss.index.min() > 0 and sub.sample_idx.min() == 0:
            ss.index = ss.index - ss.index.min()
        o.loc[idx, "sim"] = ss.SimStable_2cm.reindex(sub.sample_idx).values.astype(float)
        o.loc[idx, "simd"] = ss.sim_disp_cm.reindex(sub.sample_idx).values.astype(float)
    return o


def aggregate(o):
    by_seed = o.groupby("seed")
    r = dict(GSR=100 * o.gsr.mean(), PartAcc=100 * by_seed.part.mean().mean(), PCP=100 * by_seed.pcp.mean().mean(),
             ID=by_seed.id.mean().mean(), samples=int(len(o)), clips=int(o["clip"].nunique()), seeds=int(o.seed.nunique()))
    if "sim" in o and o.sim.notna().any():
        r.update(Sim=100 * o.sim.mean(), SimD=o.simd.mean())
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs=2, action="append", metavar=("FOLD", "CSV"), required=True,
                    help="evaluator per-frame CSV for a fold (repeat for several files of one fold)")
    ap.add_argument("--sim", nargs=2, action="append", metavar=("FOLD", "CSV"), default=[],
                    help="sim_displacement.py CSV for a fold (optional)")
    ap.add_argument("--excluded", default=osp.join(ROOT, "benchmark/hot3d_grasp/excluded_clips.json"))
    ap.add_argument("--keep-all", action="store_true", help="do not drop the excluded clips (915-clip numbers)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    excluded = json.load(open(args.excluded))["object_folds"]
    csvs, sims = defaultdict(list), defaultdict(list)
    for fold, p in args.csv:
        csvs[fold].append(p)
    for fold, p in args.sim:
        sims[fold].append(p)

    per_fold = {}
    for fold in sorted(csvs):
        o = pd.concat([per_sample(p, fold) for p in csvs[fold]], ignore_index=True)
        if sims.get(fold):
            o = attach_sim(o, sims[fold])
        if not args.keep_all:
            o = o[~o["clip"].isin(excluded.get(fold, []))]
        per_fold[fold] = aggregate(o)

    keys = [k for k in ("GSR", "PartAcc", "PCP", "ID", "Sim", "SimD") if all(k in r for r in per_fold.values())]
    mean = {k: float(np.mean([per_fold[f][k] for f in per_fold])) for k in keys}
    print("fold   " + "  ".join(f"{k:>8s}" for k in keys) + "   clips  seeds")
    for f, r in per_fold.items():
        print(f"{f:6s} " + "  ".join(f"{r[k]:8.2f}" for k in keys) + f"   {r['clips']:5d}  {r['seeds']:5d}")
    print("mean   " + "  ".join(f"{mean[k]:8.2f}" for k in keys))
    if args.json_out:
        os.makedirs(osp.dirname(osp.abspath(args.json_out)), exist_ok=True)
        json.dump(dict(mean=mean, per_fold=per_fold), open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
