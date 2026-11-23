# Third-party notices

The original code of this repository is released under the MIT License (see `LICENSE`).
It includes or adapts code from the projects below, which keep their own licenses.

## Included or adapted code

| Component | Where | License |
|---|---|---|
| [Text2HOI](https://github.com/JunukCha/Text2HOI) (Junuk Cha, 2024) | Code base of `graze/`, `lib/`, `configs/`, `constants/`, `preprocess/` | MIT |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) `rotation_conversions.py` (Meta Platforms) | `eval/hot3d/data_loaders/pytorch3d_rotation/rotation_conversions.py`; snippets in `lib/utils/rot.py` and `eval/hot3d/rot.py` | BSD-3-Clause (header kept in the file) |
| [Kornia](https://github.com/kornia/kornia) | rotation conversion snippets in `lib/utils/rot.py` and `eval/hot3d/rot.py` (marked "borrowed from kornia") | Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0) |
| [pyquaternion](https://github.com/KieranWynn/pyquaternion) | quaternion snippet in `lib/utils/rot.py` and `eval/hot3d/rot.py` | MIT |
| [Ceres Solver](http://ceres-solver.org) `rotation.h` | rotation snippet in `lib/utils/rot.py` and `eval/hot3d/rot.py` | BSD-3-Clause |
| [smplx](https://github.com/vchoutas/smplx) (MPI) | `ModelOutput`-style dataclasses and the `MANO_C.forward` body in `lib/models/mano.py` and `eval/hot3d/mano.py` are adapted from smplx (inherited from Text2HOI); smplx itself is a pip dependency | smplx license: MPI non-commercial scientific research license (not MIT-compatible for commercial use) |
| [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) (Fei Xia) | `STNkd` / `PointNetfeat` in `lib/networks/pointnet.py` (inherited from Text2HOI) | MIT |
| [improved-diffusion](https://github.com/openai/improved-diffusion) / [guided-diffusion](https://github.com/openai/guided-diffusion) (OpenAI) | `betas_for_alpha_bar` in `lib/networks/diffusion.py` (inherited from Text2HOI) | MIT |
| [BimArt](https://github.com/RosettaWYzhang/BimArt) | The contact refinement (`mano_refine_mode=bimart_ramp`, `lib/utils/texthom_train_helpers.py`) follows BimArt's post-optimization design (penetration, projection and acceleration terms, AdamW). The code is a re-implementation. | BimArt: CC BY-NC 4.0 |
| [ObMan](https://github.com/hassony2/obman_train) / [GraspTTA](https://github.com/hwjiang1510/GraspTTA) | `eval/hot3d/visualize-code/sim_displacement.py` re-implements the simulation-displacement protocol and uses its parameters | ObMan: GPL-3.0 (no code copied) |

## Not included (download separately, own terms)

| Component | Terms |
|---|---|
| [HOT3D](https://facebookresearch.github.io/hot3d/) data and object models | HOT3D license (no redistribution of the data) |
| [MANO](https://mano.is.tue.mpg.de/) models | MANO license (non-commercial research) |
| BPS basis `GRAB_HANDS/bps.pt` ([DiffH2O](https://github.com/facebookresearch/diffh2o) / [GRAB](https://grab.is.tue.mpg.de/)) | DiffH2O: CC BY-NC 4.0, GRAB: GRAB license |
| `part_fps_hand_index_100.npy` ([BimArt](https://github.com/RosettaWYzhang/BimArt)) | CC BY-NC 4.0 |

Python dependencies (PyTorch, PyTorch3D, smplx, trimesh, Hydra, PyBullet, ...) are installed
from their own distributions under their own licenses.
