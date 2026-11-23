<div align="center">

# 👁️ GRAZE

### Text-Free Grasp Synthesis for Unseen Objects from Human Gaze

<sub><b>GR</b>asp from g<b>AZE</b> · look at it, and the hand grasps it</sub>

<br>

<a href="https://leejeongho3214.github.io/GRAZE/"><img src="https://img.shields.io/badge/Project-Page-2f5fd0?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project page"></a>
<a href="https://leejeongho3214.github.io/GRAZE/#video"><img src="https://img.shields.io/badge/Video-Watch-14894a?style=for-the-badge&logo=youtube&logoColor=white" alt="Video"></a>
<a href="https://arxiv.org/abs/2611.21104"><img src="https://img.shields.io/badge/arXiv-2611.21104-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="#pretrained-models"><img src="https://img.shields.io/badge/Code-available-2f8f5b?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>

[Jeongho Lee](https://github.com/leejeongho3214)<sup>1</sup> · Boeun Kim<sup>1</sup> · Hyung Jin Chang<sup>2</sup> · Jungchan Cho<sup>3</sup> · Sang-Il Choi<sup>1</sup>

<sup>1</sup>Dankook University · <sup>2</sup>University of Birmingham · <sup>3</sup>Gachon University

</div>

<br>

<p align="center"><a href="https://leejeongho3214.github.io/GRAZE/#video"><img src="docs/static/images/preview.gif" width="100%" alt="GRAZE video preview (click for the full video)"></a></p>

GRAZE generates a hand–object grasp motion for an object it has never seen. Its inputs are the
object's point cloud, the object's initial pose, and the egocentric gaze rays of the first 30%
of the clip. No text prompt names the object or the part.

This repository contains the code to train GRAZE, generate grasps, and score them on
**HOT3D-Grasp**. HOT3D-Grasp is our grasp-centered benchmark built on
[HOT3D](https://facebookresearch.github.io/hot3d/): 912 clips of 31 objects in three
object-disjoint folds.

<p align="center"><img src="docs/static/images/overview.jpg" width="100%" alt="GRAZE overview"></p>

## Contents

- [Installation](#installation)
- [Assets](#assets)
- [Data preparation](#data-preparation)
- [Pretrained models](#pretrained-models)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Expected results](#expected-results)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Tested on Linux with Python 3.8, PyTorch 1.13.0, CUDA 11.6 and PyTorch3D 0.7.2.

```bash
git clone https://github.com/leejeongho3214/GRAZE.git
cd GRAZE
conda env create -f environment.yml
conda activate graze
# PyTorch3D 0.7.2 (see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c pytorch3d pytorch3d=0.7.2
```

Weights & Biases logging is off by default. To turn it on, add `use_wandb=true` to a training command.

## Assets

The following files are not distributed with this repository. Put them in `assets/` and `data/`:

| File | Source | Used for |
|---|---|---|
| `assets/bps.pt` | [`dataset/GRAB_HANDS/bps.pt`](https://github.com/facebookresearch/diffh2o/blob/main/dataset/GRAB_HANDS/bps.pt) in the [DiffH2O](https://github.com/facebookresearch/diffh2o) repository (1,024 basis points, key `obj`) | BPS object encoding and GeoGaze. The released checkpoints work only with this exact basis. |
| `assets/part_fps_hand_index_100.npy` | [`assets/part_fps_hand_index_100.npy`](https://github.com/RosettaWYzhang/BimArt/blob/main/assets/part_fps_hand_index_100.npy) in the [BimArt](https://github.com/RosettaWYzhang/BimArt) repository | The 100 MANO vertices per hand that the model predicts |
| `data/mano/mano_v1_2/models/MANO_{LEFT,RIGHT}.pkl` | [MANO](https://mano.is.tue.mpg.de/) (registration required) | Hand model |

```bash
wget -O assets/bps.pt \
  https://github.com/facebookresearch/diffh2o/raw/main/dataset/GRAB_HANDS/bps.pt
wget -O assets/part_fps_hand_index_100.npy \
  https://github.com/RosettaWYzhang/BimArt/raw/main/assets/part_fps_hand_index_100.npy
sha256sum assets/bps.pt assets/part_fps_hand_index_100.npy
# b1ab4b0c73fd3c65c350bbdaad495cac374282480c53aedd26ee2045f38a0e7c  assets/bps.pt
# 6c73242365294fc029be8a6025ebc70aa0f49255798d7e681279d3c06d61d327  assets/part_fps_hand_index_100.npy
```

## Data preparation

HOT3D-Grasp is built from the HOT3D dataset. HOT3D data and meshes cannot be
redistributed, so you have to download HOT3D yourself and rebuild the benchmark files.
This repository ships only our own annotations, in [`benchmark/hot3d_grasp/`](benchmark/hot3d_grasp):

| File | Content |
|---|---|
| `clip_annotations.csv` | Manually annotated grasp actions: HOT3D sequence, object, contact frame, prompt (part and hand), grasp start and end frames |
| `threefold_assignment.json` | Objects of the three object folds (and of the three subject folds) |
| `manifest.json` | Clip counts and held-out objects per fold |
| `labels_merged.json` | Part labels of the 31 objects, as indices into the 1,024-point object clouds (used for Part Acc. and PCP) |
| `excluded_clips.json` | The 3 test clips without ground-truth contact, which every metric drops (915 → 912 clips) |
| `text.json` | Prompt templates of the clips (only the hand side is read at test time) |

Expected layout under `data/`:

```
data/
├── mano/mano_v1_2/models/            MANO_LEFT.pkl, MANO_RIGHT.pkl
└── hot3d/
    ├── previous/                     HOT3D object models <id>.glb + instance.json (read by the loader)
    ├── obj/                          <id>.ply  (HOT3D models exported by trimesh; intermediate)
    ├── pre-process/                  <id>.ply  (models simplified to <= 4,000 vertices, read by the data loader)
    ├── object_mesh/                  <id>.ply + instance.json (HOT3D models with merged seam vertices; obj.pkl and the evaluator)
    ├── data/acc_ori.pkl              per-action hand / object / camera / gaze tracks extracted from HOT3D
    ├── pre-process_dataset/acc_ori.pkl   the same tracks, grouped by object
    └── dataset/
        ├── obj.pkl                   1,024-point clouds + normals (farthest point sampling)
        ├── ori_dataset/gaze_{train,test}.pkl   intermediate, read only by build_threefold.py
        └── threefold/obj{0,1,2}_{train,test}.pkl, manifest.json
```

### Building the benchmark from HOT3D

You need the full HOT3D dataset (the sequences `P00xx_xxxxxxxx/` and the object library `assets/`)
and the [HOT3D toolkit](https://github.com/facebookresearch/hot3d) with
[`projectaria_tools`](https://github.com/facebookresearch/projectaria_tools). We used toolkit commit
`aae496b` with `projectaria_tools` 1.5.1 (the toolkit's `pixi.lock` pins 1.5.2) on Python 3.10.
Step 1 needs a small patch to the toolkit, so that its MANO layer returns the smplx MANO output
(whose `hand_pose` step 1 stores):

```bash
git clone https://github.com/facebookresearch/hot3d.git && cd hot3d
git checkout aae496b
git apply /path/to/GRAZE/preprocess/hot3d_toolkit.patch
cd ..
# environment for step 1 (Python 3.10; numpy must stay < 2 for chumpy, which unpickles MANO)
pip install projectaria-tools==1.5.1 vrs pillow scipy numpy==1.23.5 smplx torch pandas openpyxl tqdm
pip install --no-build-isolation chumpy==0.70
```

Set `HOT3D=/path/to/hot3d_dataset` (the folder that holds the sequences; the object library is
expected at `$HOT3D/../assets`). Then, from the repository root:

```bash
# 1. Per-action tracks from the HOT3D recordings and clip_annotations.csv
#    -> data/hot3d/data/acc_ori.pkl   (HOT3D toolkit environment, 5 to 20 s per sequence, under 45 min in all)
python preprocess/extract_hot3d_clips.py --hot3d_toolkit /path/to/hot3d/hot3d \
    --hot3d_root $HOT3D --mano_dir data/mano/mano_v1_2/models \
    --annotations benchmark/hot3d_grasp/clip_annotations.csv --out data/hot3d/data/acc_ori.pkl
#    or in parallel (about 1 to 4 min in 10 shards): write 10 sequence lists, run one process per list, merge in annotation order
#    python preprocess/merge_acc_shards.py split --num_shards 10 --out_dir shards
#    for k in 00 01 ... 09: python preprocess/extract_hot3d_clips.py <same flags> \
#        --sequences $(cat shards/shard$k.txt) --out shards/shard$k.pkl
#    python preprocess/merge_acc_shards.py merge --out data/hot3d/data/acc_ori.pkl shards/shard*.pkl
# 2. Group by object -> data/hot3d/pre-process_dataset/acc_ori.pkl
python preprocess/group_by_object.py
# 3. Object models -> data/hot3d/previous/, data/hot3d/obj/*.ply, data/hot3d/pre-process/*.ply
python preprocess/convert_object_meshes.py --step library --assets $HOT3D/../assets
python preprocess/convert_object_meshes.py --step export  --assets $HOT3D/../assets
pip install pymeshlab==2021.10     # Python <= 3.9; any separate environment works
python preprocess/convert_object_meshes.py --step simplify --assets $HOT3D/../assets
#    data/hot3d/object_mesh/*.ply (+ HOT3D's assets/instance.json): see the note below (TODO)
# 4. 1,024-point object clouds -> data/hot3d/dataset/obj.pkl   (GPU)
python preprocess/preprocessing_hot3d.py --step object
# 5. Clips with contact and gaze maps -> data/hot3d/dataset/ori_dataset/gaze_{train,test}.pkl   (GPU)
python preprocess/preprocessing_hot3d.py --step clips
# 6. Pool both splits (915 clips) and write the 3-fold splits -> data/hot3d/dataset/threefold/
python preprocess/build_threefold.py
```

We checked these steps against the files used in the paper. With the paper's `obj.pkl` (and
therefore its `object_mesh/`, see the TODO below; steps 2 and 3 reproduce the other inputs exactly),
step 1 (run in 10 shards and merged) followed by steps 2, 5 and 6 gives the same six object-fold
splits (`obj{0,1,2}_{train,test}`) as the paper: the same 915 clips (613 / 302, 604 / 311,
613 / 302 per fold) in the same order, with identical actions, objects, parts, subjects, frame
counts, object poses, camera poses, gaze rays and left/right contact flags (`is_lhand` /
`is_rhand`, 0 flips), and finger poses and hand joints equal up to float32 rounding (at most 1.5e-7
in `x_lhand` / `x_rhand`, 2.4e-7 m in the joints). The subject folds (`subj*`) match in the same way.
What is left comes from recomputing the hand-object distances, which agree only to about 1e-4 m
(most likely `torch.cdist` in float32 on a different GPU and PyTorch version). In 140 of the 915 test clips
the point-level contact lists (`*cf_idx` / `*cov_idx` / `*chj_idx`) gain or lose one to three
hand-object pairs, and all 162 such pairs lie within 1e-5 m of the 2 cm contact threshold (0.019993
to 0.020000 m). The set of contact frames, and so the first contact frame, is identical in all
clips. The stored per-pair distances (`*dist_value`) agree to 1.5e-4 m where both files have the
pair, and are shifted where a pair was gained or lost; the loader turns them into `*dist_map`, which
is only used as a contact mask. The vertex-level lists (`*_ver_idx`, median about 105,000 pairs per
clip) differ in 793 of the 915 test clips, by a median of 5 pairs (at most 52, under 0.1 %). The
affordance maps, which are built from the vertex-level lists, differ in 91 test clips (at most
0.029), all in (object, prompt) groups with a vertex-level difference.
The per-frame gaze maps differ only for the 10 actions whose HOT3D eye gaze is missing (NaN) in some
frames (actions 54, 368, 432, 701, 820, 856, 992, 1118, 1119, 1362; one test and two training clips
each across the three folds). The paper-era extraction kept one gaze entry per annotated frame for
these actions, so their `gaze_map` / `score_map` were computed from a gaze list longer than the clip
(and, in 6 of the 10, offset from the poses after the first missing frame). The released model does
not read `gaze_map` or `score_map` (it uses the gaze rays). Step 2 gives a byte-identical file, and
steps 3 `export` / `simplify` give byte-identical meshes (33 / 33 and 34 / 34 files). Step 4 gives the same point clouds, normals and FPS indices for all 32 objects
as long as `object_mesh/` is the same (three objects, `coffee_pot`, `puzzle_toy` and `dino_toy`, have
a few NaN normals in the released file too; the dictionary order differs, which nothing depends on).

**Hand-pose convention of step 1.** `extract_hot3d_clips.py` stores each hand's 45-d finger pose
without the MANO mean pose (smplx `hand_pose`), and drops frames with a missing hand, object or
headset pose or gaze, as the extraction behind the paper's splits did. The HOT3D toolkit's MANO
layer uses `flat_hand_mean=False`, so its `full_pose` already contains the mean; step 5 adds the
mean back exactly once (`build_mano_aa(flat_hand=False)`). An earlier released version of this
script, not used for the paper's splits, stored `full_pose` (mean added twice downstream, about
2.8 cm joint error, contact flags flipped in 75 test clips) and filled single-frame gaps with a copy of the neighbouring frame (62
frames in 61 actions, the reason why 25 `obj0_test` clips were one frame longer). Only that variant
writes an `acc_ori.pkl` with the mean included; it is kept behind `--keep_hand_mean` and
`--fill_single_frame_gaps` for reference, and step 5 refuses an `acc_ori.pkl` whose finger poses
include the mean. Because the contact flags match, the rebuilt splits have the same 3 test clips
without ground-truth contact, so `excluded_clips.json` and the 912-clip benchmark apply to them.

**`object_mesh/` (TODO).** The script that wrote `data/hot3d/object_mesh/*.ply` was not recovered.
Step 4 also reads `object_mesh/instance.json`: copy HOT3D's `assets/instance.json` there unchanged
(`previous/instance.json` lacks one of the 32 meshed objects).
For 30 of the 32 objects these meshes contain exactly the vertices of the HOT3D models after
merging duplicated seam vertices, but in a different vertex and face order (the faces are the
HOT3D faces regrouped into two or three blocks); the other two objects differ slightly. Merging the seam vertices of the HOT3D
models (for example with Open3D's `remove_duplicated_vertices`) gives the same vertex set, but
farthest point sampling then picks different points for most objects, so `obj.pkl` and the part
labels in `labels_merged.json` (indices into those points) no longer match. Until this step is
released, the released `obj.pkl` order cannot be rebuilt from HOT3D alone.

Training, inference and evaluation read only `obj.pkl`, the `threefold/` files, the meshes and
`previous/instance.json`; `data/acc_ori.pkl`, `pre-process_dataset/` and `ori_dataset/` are needed
only to rebuild the splits.

Fold `obj<k>` tests on the objects of fold k and trains on the other two folds
(302 / 311 / 302 test clips, 613 / 604 / 613 training clips). The training loader skips
clips without hand annotations (obj0: 613 → 611).

## Pretrained models

There is one checkpoint per object fold. Each is the epoch-4000 model used in the paper, with the optimizer state removed:

| Fold | File |
|---|---|
| obj0 | `graze_obj0.pth` |
| obj1 | `graze_obj1.pth` |
| obj2 | `graze_obj2.pth` |

The checkpoints are attached to the [v1.0 release](https://github.com/leejeongho3214/GRAZE/releases/tag/v1.0):

```bash
for k in 0 1 2; do
  wget -P checkpoints https://github.com/leejeongho3214/GRAZE/releases/download/v1.0/graze_obj${k}.pth
done
(cd checkpoints && sha256sum -c SHA256SUMS)
```

Check them against [`checkpoints/SHA256SUMS`](checkpoints/SHA256SUMS). Each checkpoint stores the model weights, the
model settings and the motion normalization statistics.

## Training

The defaults in `configs/` are the settings of the released model. To train one fold:

```bash
bash scripts/train.sh obj0        # = python graze/train.py fold=obj0
```

This command trains for 4,000 epochs with batch size 64 and AdamW (lr 1e-4), and writes
`outputs/graze_obj0/epoch_4000.pth`. It takes about 14 h on one GPU. Training is not
bit-wise deterministic. At the start of training, the motion normalization statistics are
computed from the training split. They are stored in every checkpoint.

## Inference

```bash
bash scripts/test.sh obj0 checkpoints/graze_obj0.pth
```

This command does the following:

- It samples 10 motions per test clip (seeds 0–9, DDPM with 1,000 steps).
- It fits MANO to the predicted hand points.
- It runs the contact refinement (300 steps).

As in training, the gaze of the first 30% of each clip's valid frames is kept and the rest is
zeroed. The output is `results/graze_obj0.pkl`.

Runtime: one seed of fold obj0 (302 clips) took 52 min on one shared GPU, so the full
10 seeds of one fold take about 5–9 h. To run fewer seeds, append
`texthom.exp.num_test_seeds=<n>`.

As in the paper, the model is also given the initial object pose, the clip length and the
active hands. The active hands are read from the hand words of the prompt.

## Evaluation

```bash
bash scripts/eval.sh obj0         # per-frame metrics (GPU) + physics simulation (PyBullet)
```

The evaluator in `eval/` reads `eval/data/obj.pkl` and `eval/data/object_mesh/`. Link them to your data:

```bash
mkdir -p eval/data
ln -s ../../data/hot3d/dataset/obj.pkl eval/data/obj.pkl
ln -s ../../data/hot3d/object_mesh eval/data/object_mesh
```

After all three folds are scored, build the table row:

```bash
python scripts/make_table1.py \
  --csv obj0 results/graze_obj0.csv --csv obj1 results/graze_obj1.csv --csv obj2 results/graze_obj2.csv \
  --sim obj0 results/sim_obj0.csv  --sim obj1 results/sim_obj1.csv  --sim obj2 results/sim_obj2.csv
```

The simulation step (Sim. / SimD) needs PyBullet (`pybullet==3.2.6`, in `requirements.txt`).
Scoring one seed takes about 5 min (metrics) plus a few seconds (simulation).

Each metric is computed per generated sample and then averaged. The row is the mean of the three folds.

- **GSR** (grasp success): all three conditions must hold:
  - contact on each of the last 5 frames,
  - final penetration depth ≤ 1 cm,
  - the object lifted ≥ 5 mm along the HOT3D world up `u = [0.0946, −0.6165, 0.7816]`.
- **Part Acc.** / **PCP**: whether the final contact lies on the target part.
- **ID**: penetration depth (mm), averaged over penetrating frames.
- **Sim.** / **SimD**: a physics simulation with the hands fixed. Sim. is the share of grasps whose
  object moves less than 2 cm, and SimD is the mean object displacement (cm).

## Expected results

HOT3D-Grasp, 912 clips, 10 samples per clip (paper Table 1):

| | GSR ↑ | Part Acc. ↑ | PCP ↑ | ID ↓ | Sim. ↑ | SimD ↓ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| GRAZE | 66.3 | 49.3 | 50.1 | 6.7 | 74.5 | 2.04 |

Per fold (GSR / Part Acc. / PCP): obj0 53.6 / 56.0 / 56.9, obj1 68.8 / 51.7 / 51.1, obj2 76.6 / 40.2 / 42.1.

## Repository layout

```
graze/train.py, graze/test.py      training and inference (+ MANO fit and refinement)
configs/                           Hydra configs (texthom/texthom.yaml = model and experiment)
lib/networks/texthom.py            gaze-conditioned diffusion Transformer
lib/networks/diffusion.py          DDPM
lib/utils/texthom_train_helpers.py GeoGaze (BPS ray closeness + alignment), targets, MANO fit, refinement
lib/datasets/h2o.py                HOT3D-Grasp data loader (MotionHOT3D)
preprocess/                        benchmark construction
eval/                              metrics (run_eval_gpu.sh) and physics simulation
scripts/                           train / test / eval wrappers and the table script
benchmark/hot3d_grasp/             our annotations and splits
docs/                              project page (GitHub Pages)
```

The code started from [Text2HOI](https://github.com/JunukCha/Text2HOI), so some module and
config names keep the `texthom` prefix. The code also contains switches for the paper's
ablations. Their defaults are the settings of the released model.

Implementation details that are easy to miss:

- The object BPS feature and the ray closeness use nearest points on the object mesh vertices.
  The displacement alignment uses nearest points of the 1,024-point object cloud.
- Condition dropout (p = 0.1) is applied to the gaze tokens and, independently, to the object condition.
- The hand output heads also see the encoded object token.
- The MANO fit optimizes shape, but the refined and saved hands use the mean shape.
- Training visits the clips in a fixed order, with no shuffling or re-weighting.
- The contact refinement uses 256 object points (with normals) sampled from the 1,024-point cloud.
- `use_lift_gate` builds an auxiliary head that no loss trains. It exists only so that the
  released checkpoints load.

## Citation

```bibtex
@article{lee2026graze,
  title   = {{GRAZE}: Text-Free Grasp Synthesis for Unseen Objects from Human Gaze},
  author  = {Lee, Jeongho and Kim, Boeun and Chang, Hyung Jin and Cho, Jungchan and Choi, Sang-Il},
  journal = {arXiv preprint arXiv:2611.21104},
  year    = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). Parts of it come from
other projects under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The following are not part of this repository and keep their own terms: HOT3D, MANO, the
DiffH2O/GRAB BPS basis, and the BimArt hand-vertex indices.

Other parts of the release have their own licenses:

- **Pretrained checkpoints** (`graze_obj{0,1,2}.pth`):
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). They were trained on HOT3D and
  depend on the DiffH2O BPS basis (CC BY-NC 4.0).
- **HOT3D-Grasp annotations** (`benchmark/hot3d_grasp/`):
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). They annotate HOT3D, so
  you must also comply with the [HOT3D license](https://www.projectaria.com/datasets/hot3d/license/). See
  [benchmark/hot3d_grasp/README.md](benchmark/hot3d_grasp/README.md).

## Acknowledgements

This code builds on [Text2HOI](https://github.com/JunukCha/Text2HOI). We thank the authors of:

- [HOT3D](https://facebookresearch.github.io/hot3d/), [MANO](https://mano.is.tue.mpg.de/),
  [smplx](https://github.com/vchoutas/smplx) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d);
- [DiffH2O](https://github.com/facebookresearch/diffh2o) and [GRAB](https://grab.is.tue.mpg.de/), for the BPS basis;
- [BimArt](https://github.com/RosettaWYzhang/BimArt), for the hand-vertex subset and the
  post-optimization design that our contact refinement follows;
- [ObMan](https://github.com/hassony2/obman_train), for the simulation protocol.
