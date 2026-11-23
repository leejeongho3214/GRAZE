#!/usr/bin/env python3
"""Physics simulation displacement (SimDisp) for generated hand-object grasps.

Protocol
--------
This follows the "simulation displacement" metric of

* Hasson et al., "Learning joint reconstruction of hands and manipulated
  objects" (ObMan), CVPR 2019, Sec. 5.1: "the average displacement of the
  object's center of mass in a simulated environment [Bullet] assuming the
  hand is fixed and the object is subjected to gravity", with details taken
  from Tzionas et al., IJCV 2016; and
* Jiang et al., "Hand-Object Contact Consistency Reasoning for Human Grasps
  Generation" (GraspTTA), ICCV 2021, Sec. 4.2, which reuses the ObMan code.

The defaults below reproduce the released reference implementation,
``obman_train/mano_train/simulation/simulate.py::run_simulation`` (copied
verbatim as ``GraspTTA/metric/simulate.py``):

====================  =====================================================
time step             1/240 s (``fixedTimeStep``)
duration              35 steps (= 0.146 s; *not* 1 s)
solver iterations     150 (``numSolverIterations``)
gravity               9.8 m/s^2 along "down" (here ``-u``, u = HOT3D world up)
hand                  static (mass 0) concave triangle mesh
                      (``GEOM_FORCE_CONCAVE_TRIMESH``), no decomposition
object                dynamic, V-HACD convex decomposition, mass 1 kg
                      (fixed, not volume based), centre of mass at the mean
                      of the object vertices, inertia computed by Bullet
                      from the collision shape
friction              lateral 3 (hand) / 3 (object)
restitution           0 (hand) / 0.5 (object)
V-HACD                resolution 1000, concavity 0.001, alpha 0.05, beta 0,
                      planeDownsampling 4, convexhullDownsampling 4, pca 0,
                      mode 0, maxNumVerticesPerCH 64, minVolumePerCH 1e-4
metric                || final base (COM) position - initial COM ||
====================  =====================================================

Conventions inherited from the reference code (stated explicitly because the
papers do not discuss them):

* Initial interpenetration is not treated specially: the start state is
  simulated as-is and the solver's depenetration impulses move the object.
  ObMan warns that stable simulated grasps can also come from balanced
  collision forces, so SimDisp should be read together with penetration
  (ID/IV).
* An object that does not touch any hand simply falls; its free-fall
  displacement is counted in the mean like any other sample (no exclusion,
  no clamping).  With the reference duration this is about 10.7 cm.  GraspTTA
  assigns 0.10 m when the simulation raises an exception; we record NaN plus
  an error string instead and report the failure count.
* Stable rate: fraction of samples with displacement below a threshold
  (2 cm requested here; GraspTTA filters its grasps at 3 cm).

Optional deviations (off by default): ``mass_mode="volume"`` (mass and
principal inertia from the mesh volume at constant density, COM at the
volumetric centroid), ``hand_shape="vhacd"`` (convex-decomposed hand),
other step counts.

Frames: the evaluator's GSR checks penetration and lift at the final valid
frame ``nframes - 1``; SimDisp uses the same frame by default (``frame="last"``).
Both hand and object are taken in the world frame produced by the evaluator,
and gravity is ``-u`` with HOT3D world up ``u = [0.0946, -0.6165, 0.7816]``.

This module avoids importing torch at import time so that worker processes
stay light.  PyBullet is loaded from ``eval_metric_server_package/third_party_py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
THIRD_PARTY_DIR = os.path.join(PACKAGE_ROOT, "third_party_py")
if os.path.isdir(THIRD_PARTY_DIR) and THIRD_PARTY_DIR not in sys.path:
    sys.path.append(THIRD_PARTY_DIR)

HOT3D_WORLD_UP = (0.0946, -0.6165, 0.7816)
DEFAULT_CACHE_DIR = os.path.join(PACKAGE_ROOT, "data", "sim_vhacd_cache")
NAN = float("nan")


@dataclass
class SimConfig:
    time_step: float = 1.0 / 240.0
    num_steps: int = 35
    solver_iterations: int = 150
    gravity: float = 9.8
    hand_friction: float = 3.0
    object_friction: float = 3.0
    hand_restitution: float = 0.0
    object_restitution: float = 0.5
    mass_mode: str = "fixed"  # "fixed" (ObMan) or "volume"
    object_mass: float = 1.0
    density: float = 1000.0  # kg/m^3, used only when mass_mode == "volume"
    hand_shape: str = "concave"  # "concave" (ObMan) or "vhacd"
    vhacd_resolution: int = 1000
    vhacd_concavity: float = 0.001
    vhacd_alpha: float = 0.05
    vhacd_beta: float = 0.0
    vhacd_plane_downsampling: int = 4
    vhacd_convexhull_downsampling: int = 4
    vhacd_pca: int = 0
    vhacd_mode: int = 0
    vhacd_max_vertices_per_ch: int = 64
    vhacd_min_volume_per_ch: float = 0.0001
    stable_threshold_cm: float = 2.0
    contact_threshold_m: float = 0.005  # only for the diagnostic touch flag
    cache_dir: str = DEFAULT_CACHE_DIR

    def vhacd_kwargs(self) -> dict:
        return dict(
            concavity=float(self.vhacd_concavity),
            alpha=float(self.vhacd_alpha),
            beta=float(self.vhacd_beta),
            minVolumePerCH=float(self.vhacd_min_volume_per_ch),
            resolution=int(self.vhacd_resolution),
            maxNumVerticesPerCH=int(self.vhacd_max_vertices_per_ch),
            planeDownsampling=int(self.vhacd_plane_downsampling),
            convexhullDownsampling=int(self.vhacd_convexhull_downsampling),
            pca=int(self.vhacd_pca),
            mode=int(self.vhacd_mode),
        )


def stable_column_name(threshold_cm: float) -> str:
    return f"SimStable_{float(threshold_cm):g}cm"


# --------------------------------------------------------------------------
# Geometry helpers (numpy only)
# --------------------------------------------------------------------------


def _save_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as handle:
        for v in np.asarray(vertices, dtype=np.float64):
            handle.write("v %.8f %.8f %.8f\n" % (v[0], v[1], v[2]))
        for f in np.asarray(faces, dtype=np.int64) + 1:
            handle.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> list[float]:
    m = np.asarray(rot, dtype=np.float64)
    # Re-orthonormalize to avoid quaternion drift from numerical noise.
    u, _, vt = np.linalg.svd(m)
    m = u @ vt
    if np.linalg.det(m) < 0:
        u[:, -1] *= -1
        m = u @ vt
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return (q / np.linalg.norm(q)).tolist()


def _mesh_key(vertices: np.ndarray, faces: np.ndarray, extra: str) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(vertices, dtype=np.float32)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(faces, dtype=np.int64)).tobytes())
    h.update(extra.encode())
    return h.hexdigest()[:20]


def _import_pybullet():
    import pybullet as p  # noqa: WPS433 (deferred import)

    return p


def _run_vhacd(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: str,
    config: SimConfig,
) -> None:
    """V-HACD through PyBullet's built-in ``p.vhacd`` (atomic write)."""
    p = _import_pybullet()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out_path)) as tmp:
        src = os.path.join(tmp, "in.obj")
        dst = os.path.join(tmp, "out.obj")
        log = os.path.join(tmp, "log.txt")
        _save_obj(src, vertices, faces)
        p.vhacd(src, dst, log, **config.vhacd_kwargs())
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            raise RuntimeError(f"V-HACD produced no output for {out_path}")
        os.replace(dst, out_path)


def _count_hulls(obj_path: str) -> int:
    count = 0
    with open(obj_path) as handle:
        for line in handle:
            if line.startswith("o "):
                count += 1
    return count


def prepare_object_asset(
    object_key: str,
    vertices_local: np.ndarray,
    faces: np.ndarray,
    config: SimConfig,
) -> dict:
    """Body-frame mesh, V-HACD file (disk cached), COM, mass and inertia.

    The body frame has its origin at the centre of mass.  In "fixed" mode
    (ObMan) the axes are the object's canonical axes and Bullet computes the
    inertia from the collision shape.  In "volume" mode the axes are the
    principal axes of the solid mesh and the principal moments are set
    explicitly.
    """
    vertices_local = np.asarray(vertices_local, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    info: dict = {"object": object_key}
    body_axes = np.eye(3)
    inertia_diag = None
    mass = float(config.object_mass)
    if config.mass_mode == "fixed":
        com_local = vertices_local.mean(axis=0)
    elif config.mass_mode == "volume":
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices_local, faces=faces, process=False)
        solid = mesh
        info["watertight"] = bool(mesh.is_watertight)
        if not mesh.is_watertight or not np.isfinite(mesh.volume) or mesh.volume <= 0:
            solid = mesh.convex_hull
            info["volume_source"] = "convex_hull"
        else:
            info["volume_source"] = "mesh"
        solid = solid.copy()
        solid.density = float(config.density)
        mass = float(solid.mass)
        com_local = np.asarray(solid.center_mass, dtype=np.float64)
        moments, axes = trimesh.inertia.principal_axis(solid.moment_inertia)
        axes = np.asarray(axes, dtype=np.float64)  # rows are axes
        body_axes = axes.T  # columns: principal axes in the local frame
        if np.linalg.det(body_axes) < 0:
            body_axes[:, 2] *= -1
        inertia_diag = [float(x) for x in moments]
        info["volume_m3"] = float(solid.volume)
    else:
        raise ValueError(f"unknown mass_mode: {config.mass_mode}")

    body_vertices = (vertices_local - com_local[None]) @ body_axes
    extra = json.dumps(
        {"vhacd": config.vhacd_kwargs(), "mode": config.mass_mode}, sort_keys=True
    )
    key = _mesh_key(body_vertices, faces, extra)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in object_key)
    vhacd_path = os.path.join(config.cache_dir, f"{safe}_{key}.obj")
    if not os.path.exists(vhacd_path):
        _run_vhacd(body_vertices, faces, vhacd_path, config)
    info.update(
        {
            "vhacd_path": vhacd_path,
            "com_local": com_local.tolist(),
            "body_axes": body_axes.tolist(),
            "mass": mass,
            "inertia_diag": inertia_diag,
            "num_hulls": _count_hulls(vhacd_path),
        }
    )
    return info


# --------------------------------------------------------------------------
# Simulation (one job)
# --------------------------------------------------------------------------

_WORKER_CLIENT = None
_WORKER_CONFIG: Optional[SimConfig] = None


def _worker_init(config_dict: dict) -> None:
    global _WORKER_CLIENT, _WORKER_CONFIG
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    p = _import_pybullet()
    _WORKER_CONFIG = SimConfig(**config_dict)
    _WORKER_CLIENT = p.connect(p.DIRECT)


def _hand_collision_shape(p, cid, verts, faces, config: SimConfig) -> int:
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if config.hand_shape == "concave":
        return p.createCollisionShape(
            p.GEOM_MESH,
            vertices=verts.tolist(),
            indices=faces.reshape(-1).tolist(),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=cid,
        )
    if config.hand_shape == "vhacd":
        key = _mesh_key(verts, faces, json.dumps(config.vhacd_kwargs(), sort_keys=True))
        path = os.path.join(config.cache_dir, "hands", f"hand_{key}.obj")
        if not os.path.exists(path):
            _run_vhacd(verts, faces, path, config)
        return p.createCollisionShape(p.GEOM_MESH, fileName=path, physicsClientId=cid)
    raise ValueError(f"unknown hand_shape: {config.hand_shape}")


def simulate_job(job: dict, config: Optional[SimConfig] = None, client=None) -> dict:
    """Simulate one sample.

    ``job`` keys:
      key:          anything hashable/serialisable identifying the sample
      hands:        list of (vertices [V,3] world, faces [F,3]) static hands
      asset:        output of :func:`prepare_object_asset`
      obj_rot:      [3,3] local->world rotation of the object
      obj_trans:    [3] local->world translation
      gravity_dir:  [3] unit vector pointing *down* in the world frame
    """
    p = _import_pybullet()
    config = config or _WORKER_CONFIG or SimConfig()
    cid = client if client is not None else _WORKER_CLIENT
    own_client = False
    if cid is None:
        cid = p.connect(p.DIRECT)
        own_client = True
    t0 = time.time()
    out = {
        "key": job.get("key"),
        "sim_disp_m": NAN,
        "rot_deg": NAN,
        "init_min_dist_m": NAN,
        "init_max_pen_m": NAN,
        "init_touch": None,
        "error": "",
    }
    try:
        p.resetSimulation(physicsClientId=cid)
        p.setPhysicsEngineParameter(
            numSolverIterations=int(config.solver_iterations),
            fixedTimeStep=float(config.time_step),
            physicsClientId=cid,
        )
        g = np.asarray(job["gravity_dir"], dtype=np.float64)
        g = g / np.linalg.norm(g) * float(config.gravity)
        p.setGravity(float(g[0]), float(g[1]), float(g[2]), physicsClientId=cid)

        hand_ids = []
        for verts, faces in job.get("hands", []):
            verts = np.asarray(verts, dtype=np.float64)
            if verts.size == 0 or not np.all(np.isfinite(verts)):
                raise ValueError("non-finite or empty hand vertices")
            shape = _hand_collision_shape(p, cid, verts, faces, config)
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                physicsClientId=cid,
            )
            p.changeDynamics(
                body,
                -1,
                lateralFriction=float(config.hand_friction),
                restitution=float(config.hand_restitution),
                physicsClientId=cid,
            )
            hand_ids.append(body)

        asset = job["asset"]
        rot = np.asarray(job["obj_rot"], dtype=np.float64).reshape(3, 3)
        trans = np.asarray(job["obj_trans"], dtype=np.float64).reshape(3)
        com_local = np.asarray(asset["com_local"], dtype=np.float64)
        body_axes = np.asarray(asset["body_axes"], dtype=np.float64)
        com_world = rot @ com_local + trans
        body_rot_world = rot @ body_axes
        obj_shape = p.createCollisionShape(
            p.GEOM_MESH, fileName=asset["vhacd_path"], physicsClientId=cid
        )
        obj_body = p.createMultiBody(
            baseMass=float(asset["mass"]),
            baseCollisionShapeIndex=obj_shape,
            basePosition=com_world.tolist(),
            baseOrientation=_rotmat_to_quat_xyzw(body_rot_world),
            physicsClientId=cid,
        )
        dyn = dict(
            lateralFriction=float(config.object_friction),
            restitution=float(config.object_restitution),
        )
        if asset.get("inertia_diag") is not None:
            dyn["localInertiaDiagonal"] = list(asset["inertia_diag"])
        p.changeDynamics(obj_body, -1, physicsClientId=cid, **dyn)

        start_pos, start_orn = p.getBasePositionAndOrientation(
            obj_body, physicsClientId=cid
        )
        # Diagnostics on the initial state (do not affect the simulation).
        min_dist = NAN
        max_pen = 0.0
        if hand_ids:
            p.performCollisionDetection(physicsClientId=cid)
            dists = []
            for hb in hand_ids:
                pts = p.getClosestPoints(
                    obj_body, hb, distance=1.0, physicsClientId=cid
                )
                dists.extend(float(pt[8]) for pt in pts)
            if dists:
                min_dist = float(min(dists))
                max_pen = float(max(0.0, -min_dist))
        out["init_min_dist_m"] = min_dist
        out["init_max_pen_m"] = max_pen if hand_ids else NAN
        out["init_touch"] = bool(
            np.isfinite(min_dist) and min_dist <= float(config.contact_threshold_m)
        )

        for _ in range(int(config.num_steps)):
            p.stepSimulation(physicsClientId=cid)
        end_pos, end_orn = p.getBasePositionAndOrientation(
            obj_body, physicsClientId=cid
        )
        out["sim_disp_m"] = float(
            np.linalg.norm(np.asarray(end_pos) - np.asarray(start_pos))
        )
        dq = p.getDifferenceQuaternion(start_orn, end_orn)
        out["rot_deg"] = float(
            math.degrees(2.0 * math.acos(min(1.0, abs(float(dq[3])))))
        )
    except Exception as ex:  # keep the batch alive; reported as NaN
        out["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        out["time_s"] = time.time() - t0
        if own_client:
            p.disconnect(physicsClientId=cid)
    return out


def _worker_run(job: dict) -> dict:
    return simulate_job(job)


def run_simulations(
    jobs: Sequence[dict],
    config: Optional[SimConfig] = None,
    workers: int = 8,
    progress: bool = True,
) -> list[dict]:
    """Run ``jobs`` (see :func:`simulate_job`) on ``workers`` CPU processes.

    Returns one result dict per job, in input order.  ``sim_disp_m`` is the
    object centre-of-mass displacement in metres.
    """
    config = config or SimConfig()
    jobs = list(jobs)
    if not jobs:
        return []
    workers = max(1, int(workers))
    if workers == 1:
        _worker_init(asdict(config))
        it = map(_worker_run, jobs)
        results = []
        for i, r in enumerate(it):
            results.append(r)
            if progress and (i + 1) % 100 == 0:
                print(f"[SimDisp] {i + 1}/{len(jobs)}", flush=True)
        return results
    ctx = mp.get_context("spawn")
    results: list[dict] = [None] * len(jobs)  # type: ignore[list-item]
    chunk = max(1, min(16, len(jobs) // (workers * 4) or 1))
    with ctx.Pool(workers, initializer=_worker_init, initargs=(asdict(config),)) as pool:
        indexed = [dict(job, _idx=i) for i, job in enumerate(jobs)]
        done = 0
        for r_job, r in zip(
            indexed, pool.imap(_worker_run, indexed, chunksize=chunk)
        ):
            results[r_job["_idx"]] = r
            done += 1
            if progress and (done % 200 == 0 or done == len(jobs)):
                print(f"[SimDisp] {done}/{len(jobs)}", flush=True)
    return results


# --------------------------------------------------------------------------
# Job construction from evaluator data (numpy inputs)
# --------------------------------------------------------------------------


def resolve_frame_index(frame_spec, nframes: int) -> int:
    """``"last"`` -> nframes-1; integers index from the start (negative: end)."""
    nframes = int(nframes)
    if frame_spec in (None, "", "last"):
        return nframes - 1
    idx = int(frame_spec)
    if idx < 0:
        idx = nframes + idx
    return int(min(max(idx, 0), nframes - 1))


def gravity_dir_from_up(up=HOT3D_WORLD_UP) -> np.ndarray:
    u = np.asarray(up, dtype=np.float64).reshape(3)
    return -u / np.linalg.norm(u)


def make_job(
    key,
    hands: list,
    asset: dict,
    obj_rot: np.ndarray,
    obj_trans: np.ndarray,
    up=HOT3D_WORLD_UP,
) -> dict:
    return {
        "key": key,
        "hands": [
            (
                np.asarray(v, dtype=np.float32),
                np.asarray(f, dtype=np.int32),
            )
            for v, f in hands
        ],
        "asset": asset,
        "obj_rot": np.asarray(obj_rot, dtype=np.float64).reshape(3, 3),
        "obj_trans": np.asarray(obj_trans, dtype=np.float64).reshape(3),
        "gravity_dir": gravity_dir_from_up(up),
    }


class AssetCache:
    """Per-object assets keyed by evaluator object key (V-HACD on disk)."""

    def __init__(self, config: SimConfig):
        self.config = config
        self._assets: dict[str, dict] = {}

    def get(self, object_key: str, obj_mesh) -> dict:
        asset = self._assets.get(object_key)
        if asset is None:
            asset = prepare_object_asset(
                object_key,
                np.asarray(obj_mesh.vertices),
                np.asarray(obj_mesh.faces),
                self.config,
            )
            self._assets[object_key] = asset
        return asset

    @property
    def assets(self) -> dict:
        return dict(self._assets)


class EvaluatorJobCollector:
    """Hook object called from ``eval_metric.write_new_metric_per_frame_csv``.

    It receives the evaluator's own world-frame MANO vertices and object pose,
    so scored hands, betas and meshes are exactly the evaluator's.
    """

    def __init__(self, config: SimConfig, frame_spec="last", up=HOT3D_WORLD_UP):
        self.config = config
        self.frame_spec = frame_spec
        self.up = up
        self.assets = AssetCache(config)
        self.stable_column = stable_column_name(config.stable_threshold_cm)
        self.jobs: list[dict] = []
        self.meta: list[dict] = []

    def add(
        self,
        *,
        file_name: str,
        sample_idx: int,
        obj_key: str,
        obj_mesh,
        obj_params_frame_fn,
        nframes: int,
        hand_parts: list,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """``hand_parts``: list of (name, verts_seq [T,V,3] world, faces)."""
        fi = resolve_frame_index(self.frame_spec, nframes)
        rot, trans = obj_params_frame_fn(fi)
        hands = [
            (np.asarray(seq)[fi], faces)
            for _name, seq, faces in hand_parts
            if np.asarray(seq).ndim == 3 and fi < np.asarray(seq).shape[0]
        ]
        asset = self.assets.get(obj_key, obj_mesh)
        self.jobs.append(make_job((file_name, sample_idx), hands, asset, rot, trans, self.up))
        meta = {
            "file_name": file_name,
            "sample_idx": int(sample_idx),
            "object": obj_key,
            "sim_frame_idx": int(fi),
            "frames_total": int(nframes),
            "hands": "+".join(name for name, _s, _f in hand_parts),
        }
        if extra_meta:
            meta.update(extra_meta)
        self.meta.append(meta)


# --------------------------------------------------------------------------
# Evaluator CSV post-processing
# --------------------------------------------------------------------------


def fill_evaluator_csv(
    csv_path: str,
    results_by_key: dict,
    threshold_cm: float,
) -> int:
    """Fill ``SimDisp_cm`` / ``SimStable_*cm`` on each sample's final row.

    The evaluator writes the two columns empty; only rows whose
    ``(file_name, sample_idx)`` was simulated in this run are filled, so
    resumed rows keep their values.
    """
    stable_col = stable_column_name(threshold_cm)
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    last_row_idx: dict = {}
    for i, row in enumerate(rows):
        key = (str(row.get("file_name", "")), str(row.get("sample_idx", "")))
        try:
            fidx = int(row.get("frame_idx", -1))
        except (TypeError, ValueError):
            fidx = -1
        prev = last_row_idx.get(key)
        if prev is None or fidx > int(rows[prev].get("frame_idx", -1)):
            last_row_idx[key] = i
    filled = 0
    for key, res in results_by_key.items():
        idx = last_row_idx.get((str(key[0]), str(key[1])))
        if idx is None:
            continue
        disp = res.get("sim_disp_m")
        if disp is None or not np.isfinite(disp):
            rows[idx]["SimDisp_cm"] = ""
            rows[idx][stable_col] = ""
        else:
            rows[idx]["SimDisp_cm"] = f"{float(disp) * 100.0:.6f}"
            rows[idx][stable_col] = str(int(float(disp) * 100.0 < float(threshold_cm)))
        filled += 1
    tmp = csv_path + ".simdisp_tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(c, "") for c in header])
    os.replace(tmp, csv_path)
    return filled


# --------------------------------------------------------------------------
# Standalone CLI on evaluator input pickles
# --------------------------------------------------------------------------


def _load_evaluator_objects(base, obj_pkl_path: str) -> dict:
    """Same object meshes as ``eval_metric.main`` (mesh, or proxy fallback)."""
    import torch  # noqa: F401  (eval_metric already depends on it)

    object_model = base.ObjectModel(obj_pkl_path)
    meshes = {}
    proxy_cache = {}
    for k, pc in object_model.obj_pcs.items():
        key = str(k).lower()
        pc_np = np.asarray(pc, dtype=np.float32)
        obj_path_value = object_model.obj_path.get(k)
        mesh = None
        if obj_path_value is not None:
            mesh = base._load_object_mesh(obj_pkl_path, obj_path_value, pc_np, object_key=key)
        if mesh is None:
            mesh = base._get_or_build_proxy_mesh_from_object_pc(
                pc_np, object_key=key, proxy_cache=proxy_cache
            )
        if mesh is not None:
            meshes[key] = mesh
    return meshes


def _sample_seed(sample: dict) -> str:
    meta = sample.get("object_meta") if isinstance(sample.get("object_meta"), dict) else {}
    return str(meta.get("seed", meta.get("source_seed", meta.get("test_seed", ""))))


def _iter_eval_samples(base, input_paths, use_gt: bool):
    """Yield (file_name, sample_idx, sample) with the evaluator's numbering."""
    global_idx = 0
    for path in input_paths:
        file_name = os.path.basename(path)
        if use_gt:
            file_name = os.path.splitext(file_name)[0] + "_gt.pkl"
        for record in base._load_items_from_path(path):
            for sample in base._iter_samples_from_record(record):
                idx = global_idx
                global_idx += 1
                if use_gt:
                    if sample.get("gt_obj_params") is None:
                        continue
                    sample = dict(sample)
                    sample["obj_params"] = sample["gt_obj_params"]
                    for side in ("lhand_params", "rhand_params"):
                        if sample.get("gt_" + side) is not None:
                            sample[side] = sample["gt_" + side]
                    if sample.get("gt_object"):
                        sample["object"] = sample["gt_object"]
                yield file_name, idx, sample


def collect_jobs_from_pickles(args, config: SimConfig) -> tuple[list[dict], list[dict], dict]:
    sys.path.insert(0, SCRIPT_DIR)
    import eval_metric as base  # heavy import only in the CLI parent

    meshes = _load_evaluator_objects(base, os.path.abspath(args.obj_pkl))
    l_layer = base.build_mano_aa(is_rhand=False, flat_hand=False)
    r_layer = base.build_mano_aa(is_rhand=True, flat_hand=False)
    collector = EvaluatorJobCollector(config, frame_spec=args.frame, up=args.up)
    seeds = None if not args.seed else {str(s) for s in args.seed}
    idx_filter = None
    if args.sample_idx:
        idx_filter = {int(x) for v in args.sample_idx for x in str(v).split(",") if x}
    per_file_counts: dict[str, int] = {}
    for file_name, sample_idx, sample in _iter_eval_samples(base, args.input, args.gt):
        if seeds is not None and _sample_seed(sample) not in seeds:
            continue
        if idx_filter is not None and sample_idx not in idx_filter:
            continue
        if args.max_samples and per_file_counts.get(file_name, 0) >= args.max_samples:
            continue
        text = str(sample.get("text", ""))
        obj_key = str(sample.get("object") or base._extract_object_key(text)).strip().lower()
        meta = sample.get("object_meta")
        if obj_key not in meshes and isinstance(meta, dict) and meta.get("object_name"):
            obj_key = str(meta["object_name"]).strip().lower()
        mesh = meshes.get(obj_key)
        if mesh is None:
            continue
        use_left, use_right = base._selected_hands(text)
        if not use_left and not use_right:
            continue
        nframes = base._sequence_length(sample["obj_params"])
        if use_left:
            nframes = min(nframes, base._sequence_length(sample["lhand_params"]))
        if use_right:
            nframes = min(nframes, base._sequence_length(sample["rhand_params"]))
        if nframes <= 0:
            continue
        frames = np.arange(int(nframes), dtype=np.int64)
        hand_parts = []
        try:
            if use_left:
                v, _j, f = base.process_hand_result(
                    l_layer, base._slice_frame_indices(sample["lhand_params"], frames)
                )
                hand_parts.append(("left", base._to_numpy(v), base._to_numpy(f)))
            if use_right:
                v, _j, f = base.process_hand_result(
                    r_layer, base._slice_frame_indices(sample["rhand_params"], frames)
                )
                hand_parts.append(("right", base._to_numpy(v), base._to_numpy(f)))
        except Exception as ex:
            print(f"[SimDisp WARN] skip {file_name}#{sample_idx}: {ex}")
            continue
        fi = resolve_frame_index(args.frame, nframes)
        trans, rot = base._object_pose_trans_rot(
            base._slice_frame_indices(sample["obj_params"], np.asarray([fi]))
        )
        if args.drop_hands:
            hand_parts = []
        elif args.hand_offset_cm:
            # Sanity case: rigidly move every hand away from the object COM.
            com = rot @ np.asarray(mesh.vertices).mean(axis=0) + trans
            moved = []
            for name, seq, faces in hand_parts:
                d = seq[fi].mean(axis=0) - com
                d = d / max(np.linalg.norm(d), 1e-9)
                moved.append((name, seq + d[None, None] * args.hand_offset_cm / 100.0, faces))
            hand_parts = moved
        collector.add(
            file_name=file_name,
            sample_idx=sample_idx,
            obj_key=obj_key,
            obj_mesh=mesh,
            obj_params_frame_fn=lambda _fi, rot=rot, trans=trans: (rot, trans),
            nframes=nframes,
            hand_parts=hand_parts,
            extra_meta={"seed": _sample_seed(sample), "text": text},
        )
        per_file_counts[file_name] = per_file_counts.get(file_name, 0) + 1
    return collector.jobs, collector.meta, collector.assets.assets


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.{digits}f}"


def summarize(rows: list[dict], threshold_cm: float, extra_thresholds=(3.0,)) -> list[dict]:
    """Per file and seed, then mean +- SD across seeds (sample SD, ddof=1)."""
    out = []
    by_file: dict[str, list[dict]] = {}
    for row in rows:
        by_file.setdefault(row["file_name"], []).append(row)

    def _stats(group):
        d = np.array([r["sim_disp_cm"] for r in group], dtype=np.float64)
        ok = np.isfinite(d)
        dd = d[ok]
        touch = np.array([bool(r["init_touch"]) for r in group])
        res = {
            "n": int(len(group)),
            "failed": int((~ok).sum()),
            "SimDisp_cm_mean": float(dd.mean()) if dd.size else NAN,
            "SimDisp_cm_median": float(np.median(dd)) if dd.size else NAN,
            f"SimStable_{threshold_cm:g}cm_pct": float((dd < threshold_cm).mean() * 100) if dd.size else NAN,
            "init_touch_pct": float(touch[ok].mean() * 100) if dd.size else NAN,
            "SimDisp_cm_mean_touching": float(d[ok & touch].mean()) if (ok & touch).any() else NAN,
            "time_s_per_sample": float(np.mean([r["time_s"] for r in group])),
        }
        for t in extra_thresholds:
            res[f"SimStable_{t:g}cm_pct"] = float((dd < t).mean() * 100) if dd.size else NAN
        return res

    for file_name, group in by_file.items():
        by_seed: dict[str, list[dict]] = {}
        for r in group:
            by_seed.setdefault(str(r.get("seed", "")), []).append(r)
        seed_stats = {s: _stats(g) for s, g in sorted(by_seed.items())}
        for s, st in seed_stats.items():
            out.append({"file_name": file_name, "seed": s, **st})
        if len(seed_stats) > 1:
            agg = {"file_name": file_name, "seed": "mean+-SD"}
            for k in next(iter(seed_stats.values())):
                vals = np.array([st[k] for st in seed_stats.values()], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                agg[k] = (
                    f"{vals.mean():.3f} +- {vals.std(ddof=1):.3f}" if vals.size > 1 else NAN
                )
            out.append(agg)
    return out


def write_outputs(results, meta, args, config: SimConfig, assets: dict, wall_s: float) -> None:
    rows = []
    for m, r in zip(meta, results):
        disp_cm = r["sim_disp_m"] * 100.0 if np.isfinite(r["sim_disp_m"]) else NAN
        rows.append(
            {
                **m,
                "sim_disp_cm": disp_cm,
                stable_column_name(config.stable_threshold_cm): (
                    int(disp_cm < config.stable_threshold_cm) if np.isfinite(disp_cm) else ""
                ),
                "rot_deg": r["rot_deg"],
                "init_touch": r["init_touch"],
                "init_min_dist_mm": r["init_min_dist_m"] * 1000.0,
                "init_max_pen_mm": r["init_max_pen_m"] * 1000.0,
                "time_s": r["time_s"],
                "error": r["error"],
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            )
    summary = summarize(rows, config.stable_threshold_cm)
    md = args.out_md or os.path.splitext(args.out_csv)[0] + "_summary.md"
    keys = [k for k in (summary[0].keys() if summary else []) if k not in ("file_name", "seed")]
    lines = [
        "# Simulation displacement (SimDisp) summary",
        "",
        f"- inputs: {', '.join(args.input)}{' (GT params)' if args.gt else ''}",
        f"- frame: {args.frame}; gravity: -u, u={list(args.up)}",
        f"- config: {json.dumps({k: v for k, v in asdict(config).items() if k != 'cache_dir'})}",
        f"- samples: {len(rows)}; workers: {args.workers}; wall time: {wall_s:.1f} s "
        f"({wall_s / max(1, len(rows)) * 1000:.1f} ms/sample wall)",
        "",
        "| file | seed | " + " | ".join(keys) + " |",
        "|" + "---|" * (len(keys) + 2),
    ]
    for s in summary:
        lines.append(
            f"| {s['file_name']} | {s['seed']} | "
            + " | ".join(v if isinstance(v, str) else _fmt(v) for v in (s[k] for k in keys))
            + " |"
        )
    lines += ["", "## Objects", "", "| object | hulls | mass kg | watertight |", "|---|---|---|---|"]
    for k, a in sorted(assets.items()):
        lines.append(f"| {k} | {a['num_hulls']} | {a['mass']:.4g} | {a.get('watertight', '')} |")
    with open(md, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("## Objects")]))
    print(f"[SimDisp] wrote {args.out_csv} and {md}")


def add_config_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    d = SimConfig()
    a = parser.add_argument
    a(f"--{prefix}steps", type=int, default=d.num_steps,
      help="Simulation steps (default 35 = ObMan/GraspTTA).")
    a(f"--{prefix}time-step", type=float, default=d.time_step)
    a(f"--{prefix}mass-mode", choices=("fixed", "volume"), default=d.mass_mode,
      help="fixed: 1 kg, vertex-mean COM (ObMan); volume: density x volume.")
    a(f"--{prefix}density", type=float, default=d.density)
    a(f"--{prefix}hand-shape", choices=("concave", "vhacd"), default=d.hand_shape)
    a(f"--{prefix}vhacd-resolution", type=int, default=d.vhacd_resolution)
    a(f"--{prefix}stable-cm", type=float, default=d.stable_threshold_cm)
    a(f"--{prefix}cache-dir", default=d.cache_dir)


def config_from_args(args, prefix: str = "") -> SimConfig:
    g = lambda name: getattr(args, (prefix + name).replace("-", "_"))  # noqa: E731
    return SimConfig(
        num_steps=int(g("steps")),
        time_step=float(g("time_step")),
        mass_mode=str(g("mass_mode")),
        density=float(g("density")),
        hand_shape=str(g("hand_shape")),
        vhacd_resolution=int(g("vhacd_resolution")),
        stable_threshold_cm=float(g("stable_cm")),
        cache_dir=os.path.abspath(str(g("cache_dir"))),
    )


def parse_up(text: str) -> tuple:
    vals = tuple(float(x) for x in str(text).split(","))
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("--up needs three comma-separated values")
    return vals


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", action="append", required=True,
                        help="Evaluator input pickle (repeatable).")
    parser.add_argument("--obj-pkl", default=os.path.join(PACKAGE_ROOT, "data", "obj.pkl"))
    parser.add_argument("--gt", action="store_true",
                        help="Use gt_obj_params / gt_*hand_params of the samples.")
    parser.add_argument("--seed", action="append", default=None,
                        help="Keep only these inference seeds (object_meta seed).")
    parser.add_argument("--sample-idx", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="Per input file.")
    parser.add_argument("--frame", default="last",
                        help="'last' (GSR final frame, default) or a frame index (negative from end).")
    parser.add_argument("--up", type=parse_up, default=HOT3D_WORLD_UP,
                        help="World up vector; gravity is -up.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--drop-hands", action="store_true", help="Sanity: object alone.")
    parser.add_argument("--hand-offset-cm", type=float, default=0.0,
                        help="Sanity: move hands away from the object COM by this distance.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    t0 = time.time()
    jobs, meta, assets = collect_jobs_from_pickles(args, config)
    t_collect = time.time() - t0
    print(f"[SimDisp] {len(jobs)} jobs, {len(assets)} objects, collect {t_collect:.1f} s")
    t1 = time.time()
    results = run_simulations(jobs, config, workers=args.workers)
    wall = time.time() - t1
    write_outputs(results, meta, args, config, assets, wall)


if __name__ == "__main__":
    main()
