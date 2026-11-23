import os
os.environ.setdefault("HAND_FLAGS_FROM_PROMPT", "1")
import os.path as osp
import sys

# isort: off
# Keep this bootstrap before project imports (formatters may run isort separately).
PROJECT_ROOT = osp.dirname(osp.abspath(osp.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# isort: on

# isort: split

import json
import pickle
import random
from copy import deepcopy

import hydra
import numpy as np
import torch
import tqdm
from easydict import EasyDict as edict
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from lib.datasets.datasets import get_dataloader
from lib.models.mano import build_mano_aa
from lib.utils.eval import (
    get_valid_mask_bunch,
)
from lib.utils.model_utils import (
    build_model_and_diffusion,
    build_pointnetfeat,
)
from lib.utils.motion_normalizer import MotionNormalizer
from lib.utils.proc import (
    load_bps_basis,
    proc_obj_feat_final_train,
)
from lib.utils.proc_output import (
    get_hand_joints_w_tip,
    get_NN,
    get_interior,
    get_hand_verts,
    get_transformed_obj_pc,
)
from lib.utils.rot import rot6d_to_rotmat
from lib.utils.seed import seed_everything
from lib.utils.target_condition import ClipTextEncoder, target_condition_texts, uses_text_condition
from lib.utils.texthom_train_helpers import (
    build_gaze_origin_tokens_for_texthom,
    _compute_last_frame_mesh_penetration_metrics_per_hand,
    _is_relative_gaze_mlp_mode,
    _is_temporal_gaze_token_mode,
    build_gaze_alignment_temporal_mask,
    apply_gaze_token_weights,
    apply_null_gaze_condition,
    build_bimart_refinement_mesh_cache,
    build_gaze_condition_feature_for_texthom,
    build_bps_correspondence_cache,
    build_relative_gaze_mlp_for_texthom,
    compute_bps_feature_from_mesh_cache_for_texthom,
    configure_gaze_token_fusion_for_mode,
    compute_bps_gaze_ray_closeness_map_from_mesh_cache_for_texthom,
    compute_tail_mask,
    get_bps_correspondence_source,
    gaze_token_dim_for_model,
    get_texthom_gaze_condition_dim,
    gaze_condition_requires_bps,
    load_hand_sample_indices_for_texthom,
    load_object_mesh_bps_cache,
    repeat_initial_object_pose_for_gaze_condition,
    select_object_pose_for_gaze_condition,
    restore_point_token_outputs_for_texthom,
    restore_hand_point_token_outputs_with_object_pose_for_texthom,
    load_object_mesh_cache,
    resolve_partwise_bps_context,
)


def _resolve_eval_hands(text_entry, is_lhand_value, is_rhand_value):
    text_lower = str(text_entry).lower()
    if "both hands" in text_lower:
        return True, True
    if "left hand" in text_lower and "right hand" not in text_lower:
        return True, False
    if "right hand" in text_lower and "left hand" not in text_lower:
        return False, True
    return bool(is_lhand_value), bool(is_rhand_value)


def _resolve_test_data_name(config, task_overrides):
    data_name_was_overridden = any(
        str(override).lstrip("+").startswith("dataset.data_name=")
        for override in task_overrides
    )
    if data_name_was_overridden:
        return str(config.dataset.data_name)
    return str(
        getattr(config.texthom.exp, "test_data_name", "ori_dataset/gaze_test")
    )


def _get_target_object_indices(item, batch_size, nobj, device):
    raw_idx = item.get("target_obj_idx", item.get("target_object_idx"))
    if raw_idx is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if torch.is_tensor(raw_idx):
        idx = raw_idx.to(device=device, dtype=torch.long).view(-1)
    else:
        idx = torch.as_tensor(raw_idx, dtype=torch.long, device=device).view(-1)
    if idx.numel() == 1:
        idx = idx.expand(batch_size)
    return idx[:batch_size].clamp(0, nobj - 1)


def _gather_target_object_slot(tensor, target_indices):
    if not torch.is_tensor(tensor) or tensor.dim() < 2:
        return tensor
    batch_idx = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch_idx, target_indices.to(device=tensor.device)]


def _object_names_per_sample(item, batch_size, nobj):
    names = item.get("candidate_obj_names")
    if names is None:
        obj_name = item.get("obj_name", item.get("object_name"))
        context_name = item.get("context_obj_name")
        if nobj == 2 and context_name is not None:
            names = list(zip(obj_name, context_name))
        else:
            names = obj_name
    if names is None:
        return [[None] * nobj for _ in range(batch_size)]

    if (
        nobj > 1
        and isinstance(names, (list, tuple))
        and len(names) == nobj
        and all(isinstance(col, (list, tuple)) and len(col) == batch_size for col in names)
    ):
        names = list(zip(*names))

    rows = []
    for b in range(batch_size):
        sample_names = names[b] if isinstance(names, (list, tuple)) else names
        if isinstance(sample_names, (list, tuple)):
            row = list(sample_names)
        else:
            row = [sample_names] * nobj
        if len(row) < nobj:
            row = row + [row[-1] if row else None] * (nobj - len(row))
        rows.append([str(name) if name is not None else None for name in row[:nobj]])
    return rows


def _target_object_names(item, batch_size, nobj, target_indices):
    rows = _object_names_per_sample(item, batch_size, nobj)
    target_cpu = target_indices.detach().cpu().tolist()
    return [
        rows[b][int(target_cpu[b])] if rows[b] and int(target_cpu[b]) < len(rows[b]) else None
        for b in range(batch_size)
    ]


def _rotate_obj_vectors(obj_params, vectors, dataset_name):
    bs, nframes = obj_params.shape[:2]
    obj_rot6d = obj_params[..., 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(bs, nframes, 3, 3)
    if dataset_name == "grab":
        return torch.einsum("btij,bki->btkj", obj_rotmat, vectors)
    return torch.einsum("btij,bkj->btki", obj_rotmat, vectors)


def _normalize_gaze_tensor(gaze):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        return gaze.squeeze(-1), True
    return gaze, False


def _active_gaze_target_obj(obj_pc, gaze_map_frame, origin, direction, obj_points_cam):
    active = gaze_map_frame > 0
    if bool(active.any().item()):
        return obj_pc[active].mean(dim=0)

    direction = direction / direction.norm().clamp_min(1e-8)
    rel = obj_points_cam - origin.unsqueeze(0)
    ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
    closest = origin.unsqueeze(0) + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
    dist = (obj_points_cam - closest).norm(dim=-1)
    return obj_pc[dist.argmin()]


def _build_gaze_down_variants(
    gaze, gaze_map, x_obj, obj_pc, down_ratios, shift_origin=False
):
    """Lower the object-local gaze target y and rebuild gaze directions."""
    if gaze_map is None:
        raise ValueError("gaze_down_sweep requires item['gaze_map'] to rebuild shifted ray maps.")

    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    ratios = torch.as_tensor(down_ratios, device=gaze.device, dtype=gaze.dtype)
    variant_gaze = []
    variant_gaze_map = []
    batch_size, nframes = x_obj.shape[:2]

    for ratio in ratios:
        gaze_out = gaze_base.clone()
        map_out = torch.zeros_like(gaze_map)
        for b in range(batch_size):
            y_min = obj_pc[b, :, 1].min()
            y_max = obj_pc[b, :, 1].max()
            height = (y_max - y_min).clamp_min(1e-8)
            for t in range(nframes):
                rot = rot6d_to_rotmat(x_obj[b, t, 3:9].reshape(1, 6)).reshape(3, 3)
                trans = x_obj[b, t, :3]
                obj_points_cam = torch.matmul(obj_pc[b], rot.transpose(0, 1)) + trans
                origin = gaze_base[b, t, 0]
                direction = gaze_base[b, t, 1]
                direction = direction / direction.norm().clamp_min(1e-8)

                # Choose a target on the original raw ray at the depth of the
                # nearest object point. This keeps ratio=0 exactly equal to the
                # input gaze direction; larger ratios change only target local-y.
                rel = obj_points_cam - origin.unsqueeze(0)
                ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
                closest = (
                    origin.unsqueeze(0)
                    + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
                )
                dist = (obj_points_cam - closest).norm(dim=-1)
                forward = ray_t > 0
                nearest_idx = torch.where(
                    forward, dist, torch.full_like(dist, torch.inf)
                ).argmin() if bool(forward.any().item()) else dist.argmin()
                target_t = ray_t[nearest_idx].clamp_min(1e-4)
                target_cam = origin + target_t * direction
                target_obj = torch.matmul(rot.transpose(0, 1), target_cam - trans)
                shifted_target_obj = target_obj.clone()
                shifted_target_obj[1] = target_obj[1] * (1.0 - ratio) + y_min * ratio
                shifted_origin = origin
                if shift_origin:
                    origin_obj = torch.matmul(rot.transpose(0, 1), origin - trans)
                    shifted_origin_obj = origin_obj.clone()
                    shifted_origin_obj[1] = origin_obj[1] - ratio * height
                    shifted_origin = torch.matmul(rot, shifted_origin_obj) + trans
                shifted_target = torch.matmul(rot, shifted_target_obj) + trans
                shifted_direction = shifted_target - shifted_origin
                shifted_direction = shifted_direction / shifted_direction.norm().clamp_min(1e-8)
                gaze_out[b, t, 0] = shifted_origin
                gaze_out[b, t, 1] = shifted_direction

                rel = obj_points_cam - shifted_origin.unsqueeze(0)
                ray_t = (rel * shifted_direction.unsqueeze(0)).sum(dim=-1)
                closest = (
                    shifted_origin.unsqueeze(0)
                    + ray_t.clamp_min(0.0).unsqueeze(-1) * shifted_direction
                )
                dist = (obj_points_cam - closest).norm(dim=-1)
                active = (dist <= 0.01) & (ray_t > 0)
                if not bool(active.any().item()):
                    topk = min(64, dist.numel())
                    active_idx = torch.topk(dist, k=topk, largest=False).indices
                    map_out[b, t, active_idx] = 1.0
                else:
                    map_out[b, t, active] = 1.0
        if had_last_dim:
            gaze_out = gaze_out.unsqueeze(-1)
        variant_gaze.append(gaze_out)
        variant_gaze_map.append(map_out)

    return torch.cat(variant_gaze, dim=0), torch.cat(variant_gaze_map, dim=0)


_PROMPT_PART_ALIASES = {("whiteboard_eraser", "long edge"): "side"}


def _prompt_part_key(text, object_name):
    """Target part named in the prompt, mapped to its labels_merged.json key."""
    value = str(text).lower()
    if not value.startswith("grab ") or " of " not in value:
        return ""
    part = value[len("grab "):value.index(" of ")].strip()
    return _PROMPT_PART_ALIASES.get((str(object_name).lower(), part), part)


def _canonical_part_point(obj_pc_local, indices):
    """Labeled point closest to the part centroid (object-local coordinates)."""
    pts = obj_pc_local[indices]
    centroid = pts.mean(dim=0, keepdim=True)
    return pts[(pts - centroid).norm(dim=-1).argmin()]


def _redirect_rays_to_target(gaze_base_b, x_obj_b, obj_pc_b, target_local, map_template_b):
    """Keep each frame's recorded origin, point the ray at a fixed object-local
    target that follows the object pose, and rebuild the raw gaze map."""
    gaze_out = gaze_base_b.clone()
    map_out = torch.zeros_like(map_template_b)
    for t in range(x_obj_b.shape[0]):
        rot = rot6d_to_rotmat(x_obj_b[t, 3:9].reshape(1, 6)).reshape(3, 3)
        trans = x_obj_b[t, :3]
        obj_points_cam = torch.matmul(obj_pc_b, rot.transpose(0, 1)) + trans
        origin = gaze_base_b[t, 0]
        target_cam = torch.matmul(rot, target_local) + trans
        direction = target_cam - origin
        direction = direction / direction.norm().clamp_min(1e-8)
        gaze_out[t, 0] = origin
        gaze_out[t, 1] = direction
        rel = obj_points_cam - origin.unsqueeze(0)
        ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
        closest = origin.unsqueeze(0) + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
        dist = (obj_points_cam - closest).norm(dim=-1)
        active = (dist <= 0.01) & (ray_t > 0)
        if bool(active.any().item()):
            map_out[t, active] = 1.0
        else:
            nearest = torch.topk(dist, k=min(64, dist.numel()), largest=False).indices
            map_out[t, nearest] = 1.0
    return gaze_out, map_out


def _build_gaze_part_swap_variants(
    gaze, gaze_map, x_obj, obj_pc, text, object_names, labels, include_same=True
):
    """Per clip: recorded gaze ("orig"), a synthetic ray to the prompt part's
    canonical point ("same"), and one synthetic ray per other labeled part
    ("swap"). Returns gaze, gaze_map, source row index, rewritten prompts, kinds.
    The prompt is rewritten to the variant's target part so the standard
    evaluator scores Part Acc against the NEW target; a tag is appended after
    the sentence (it contains no hand words and no " of ")."""
    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    out_gaze, out_map, src_idx, out_text, kinds = [], [], [], [], []
    for b in range(x_obj.shape[0]):
        object_name = str(object_names[b])
        parts = labels.get(object_name, {})
        prompt_key = _prompt_part_key(text[b], object_name)
        variants = [("orig", prompt_key, None)]
        if include_same and prompt_key in parts:
            variants.append(("same", prompt_key, parts[prompt_key]))
        for part_key in parts:
            if part_key != prompt_key:
                variants.append(("swap", part_key, parts[part_key]))
        for kind, part_key, indices in variants:
            if indices is None:
                gaze_b, map_b = gaze_base[b], gaze_map[b]
            else:
                index_tensor = torch.as_tensor(indices, device=obj_pc.device, dtype=torch.long)
                target = _canonical_part_point(obj_pc[b], index_tensor)
                gaze_b, map_b = _redirect_rays_to_target(
                    gaze_base[b], x_obj[b], obj_pc[b], target, gaze_map[b]
                )
            out_gaze.append(gaze_b)
            out_map.append(map_b)
            src_idx.append(b)
            kinds.append(kind)
            words = part_key.replace("_", " ")
            original = str(text[b])
            marker = original.lower().index(" of ") if " of " in original.lower() else len(original)
            rewritten = f"Grab {words}{original[marker:]}" if kind != "orig" else original
            out_text.append(f"{rewritten} [gaze_swap={kind}:{prompt_key}->{part_key}]")
    gaze_out = torch.stack(out_gaze, dim=0)
    if had_last_dim:
        gaze_out = gaze_out.unsqueeze(-1)
    return gaze_out, torch.stack(out_map, dim=0), src_idx, out_text, kinds


def _part_shift_axis(obj_pc_local, indices):
    """Principal axis of a labeled part (object-local), sign fixed so the axis points
    toward +y (or +x when nearly horizontal); returns axis, centroid, extent."""
    pts = obj_pc_local[indices].double()
    mu = pts.mean(dim=0)
    u = torch.linalg.svd(pts - mu, full_matrices=False)[2][0]
    u = u / u.norm()
    ref = u[1] if abs(float(u[1])) > 0.2 else u[0]
    if float(ref) < 0:
        u = -u
    proj = (pts - mu) @ u
    return u.float(), mu.float(), float(proj.max() - proj.min())


def _build_gaze_part_shift_variants(
    gaze, x_obj, obj_pc, text, object_names, labels, pairs, frac=0.30, hit_m=0.02
):
    """(Analysis) within-part gaze shift. Object and prompt stay fixed; on frames
    whose recorded ray hits the prompt part (≤ hit_m, first-frame pose, the pose the
    model's gaze features use) the hit point is moved by ∓/± frac·extent along the
    part's principal axis, snapped back onto the part, and the ray is re-aimed there
    from the recorded origin. Clips whose (object, part) is not in `pairs` are dropped.
    Returns gaze, source row index, tagged prompts, kinds, per-clip shift info."""
    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    out_gaze, src_idx, out_text, kinds, info = [], [], [], [], []
    for b in range(x_obj.shape[0]):
        object_name = str(object_names[b])
        part = _prompt_part_key(text[b], object_name)
        if f"{object_name}:{part}" not in pairs or part not in labels.get(object_name, {}):
            continue
        idx = torch.as_tensor(sorted(labels[object_name][part]), device=obj_pc.device, dtype=torch.long)
        u, mu, extent = _part_shift_axis(obj_pc[b], idx)
        delta = frac * extent
        rot = rot6d_to_rotmat(x_obj[b, 0, 3:9].reshape(1, 6)).reshape(3, 3)
        trans = x_obj[b, 0, :3]
        pc_cam = torch.matmul(obj_pc[b], rot.transpose(0, 1)) + trans
        part_local = obj_pc[b][idx]
        in_part = torch.zeros(obj_pc.shape[1], dtype=torch.bool, device=obj_pc.device)
        in_part[idx] = True
        origin = gaze_base[b, :, 0]
        direction = gaze_base[b, :, 1] / gaze_base[b, :, 1].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rel = pc_cam.unsqueeze(0) - origin.unsqueeze(1)                     # (T,N,3)
        ray_t = (rel * direction.unsqueeze(1)).sum(-1).clamp_min(0.0)
        dist = (rel - ray_t.unsqueeze(-1) * direction.unsqueeze(1)).norm(dim=-1)
        hit_d, hit = dist.min(dim=1)
        frames = ((hit_d <= hit_m) & in_part[hit]).nonzero().flatten().tolist()
        for kind, sign in (("orig", 0.0), ("minus", -1.0), ("plus", 1.0)):
            g = gaze_base[b].clone()
            if sign != 0.0:
                for t in frames:
                    target = obj_pc[b][hit[t]] + sign * delta * u
                    target = part_local[(part_local - target).norm(dim=-1).argmin()]
                    target_cam = torch.matmul(rot, target) + trans
                    v = target_cam - g[t, 0]
                    g[t, 1] = v / v.norm().clamp_min(1e-8)
            out_gaze.append(g)
            src_idx.append(b)
            kinds.append(kind)
            out_text.append(f"{text[b]} [gaze_shift={kind}:{part}]")
        info.append(dict(object=object_name, part=part, axis=u.tolist(), center=mu.tolist(),
                         extent_m=extent, delta_m=delta, frames_shifted=len(frames), T=int(x_obj.shape[1])))
    if not out_gaze:
        return None, [], [], [], info
    gaze_out = torch.stack(out_gaze, dim=0)
    if had_last_dim:
        gaze_out = gaze_out.unsqueeze(-1)
    return gaze_out, src_idx, out_text, kinds, info


def _build_gaze_body_lr_variants(gaze, x_obj, obj_pc, text, object_names, labels, objects, angle_deg=40.0, hit_m=0.02, only=None):
    """Body left/right gaze shift (analysis). For each clip of `objects`, the frames whose recorded
    ray passes within hit_m of the object (first-frame pose, as the gaze features use) are re-aimed at a
    point on the body at mid-height: facing the camera ("center"), or rotated -/+angle_deg about the body
    axis ("left"/"right", camera-right = +). Other frames keep the recorded ray. Object/prompt unchanged."""
    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    out_gaze, src_idx, out_text, kinds, info = [], [], [], [], []
    for b in range(x_obj.shape[0]):
        name = str(object_names[b])
        if name not in objects or "body" not in labels.get(name, {}) or "rim" not in labels.get(name, {}):
            continue
        pc = obj_pc[b]
        body = pc[torch.as_tensor(sorted(labels[name]["body"]), device=pc.device)]
        rim = pc[torch.as_tensor(sorted(labels[name]["rim"]), device=pc.device)]
        c = body.mean(0); up = rim.mean(0) - c; up = up / up.norm()
        radial = (body - c) - ((body - c) @ up)[:, None] * up
        r = radial.norm(dim=-1).mean()
        rot = rot6d_to_rotmat(x_obj[b, 0, 3:9].reshape(1, 6)).reshape(3, 3); trans = x_obj[b, 0, :3]
        pc_cam = torch.matmul(pc, rot.transpose(0, 1)) + trans
        origin = gaze_base[b, :, 0]; direction = gaze_base[b, :, 1] / gaze_base[b, :, 1].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rel = pc_cam.unsqueeze(0) - origin.unsqueeze(1); ray_t = (rel * direction.unsqueeze(1)).sum(-1).clamp_min(0.0)
        dist = (rel - ray_t.unsqueeze(-1) * direction.unsqueeze(1)).norm(dim=-1)
        frames = (dist.min(dim=1).values <= hit_m).nonzero().flatten().tolist()
        o_local = torch.matmul(rot.transpose(0, 1), origin.median(dim=0).values - trans)
        a = o_local - c; a = a - (a @ up) * up; a = a / a.norm()                      # body -> viewer, horizontal
        side = torch.linalg.cross(up, a); side = side / side.norm()
        cam_right = torch.matmul(rot.transpose(0, 1), torch.tensor([1.0, 0.0, 0.0], device=pc.device))
        if float(side @ cam_right) < 0:
            side = -side
        targets = {}
        for kind, deg in (("center", 0.0), ("left", -angle_deg), ("right", angle_deg)):
            th = torch.tensor(deg * np.pi / 180.0, device=pc.device)
            want = c + r * (torch.cos(th) * a + torch.sin(th) * side)
            targets[kind] = body[(body - want).norm(dim=-1).argmin()]
        # only=<kind>: emit a single variant so each condition runs in its own pass with an identical batch
        # layout and RNG stream (variants stacked in one batch diverge numerically by row position).
        for kind in ((only,) if only else ("center", "left", "right")):
            g = gaze_base[b].clone()
            tgt_cam = torch.matmul(rot, targets[kind]) + trans
            for t in frames:
                v = tgt_cam - g[t, 0]; g[t, 1] = v / v.norm().clamp_min(1e-8)
            out_gaze.append(g); src_idx.append(b); kinds.append(kind)
            # tag must not contain hand words: the evaluator selects the scored hands from the prompt text
            out_text.append(f"{text[b]} [gaze_lr={dict(center='mid', left='neg', right='pos')[kind]}:body]")
        info.append(dict(object=name, axis=side.tolist(), up=up.tolist(), center=c.tolist(), radius_m=float(r),
                         angle_deg=angle_deg, targets={k: v.tolist() for k, v in targets.items()},
                         frames_redirected=len(frames), T=int(x_obj.shape[1])))
    if not out_gaze:
        return None, [], [], [], info
    gaze_out = torch.stack(out_gaze, dim=0)
    if had_last_dim:
        gaze_out = gaze_out.unsqueeze(-1)
    return gaze_out, src_idx, out_text, kinds, info


_VERTICAL_TOP_PARTS = ("top", "cap", "lid", "rim")


def _build_gaze_vertical_variants(gaze, x_obj, obj_pc, text, object_names, labels, objects, levels, hit_m=0.02):
    """Vertical gaze shift on tall objects (analysis). Up axis = centroid of the top-side part
    (top/cap/lid/rim) minus that of the bottom part (or body when no bottom part is labeled). For
    each level q the target is the object point at height quantile q of the object's extent along
    up that faces the recorded eye most (first-frame pose, as the gaze features use); frames whose
    recorded ray passes within hit_m of the object are re-aimed there, other frames keep the
    recorded ray. Variant "orig" keeps every recorded ray. Object/prompt unchanged."""
    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    out_gaze, src_idx, out_text, kinds, info = [], [], [], [], []
    for b in range(x_obj.shape[0]):
        name = str(object_names[b])
        parts = labels.get(name, {})
        top_key = next((k for k in _VERTICAL_TOP_PARTS if k in parts), None)
        if name not in objects or top_key is None:
            continue
        bottom_key = "bottom" if "bottom" in parts else "body"
        pc = obj_pc[b]
        top = pc[torch.as_tensor(sorted(parts[top_key]), device=pc.device)].mean(0)
        bottom = pc[torch.as_tensor(sorted(parts[bottom_key]), device=pc.device)].mean(0)
        up = top - bottom; up = up / up.norm()
        c = pc.mean(0); h = (pc - c) @ up; h_min, h_max = float(h.min()), float(h.max()); height = h_max - h_min
        rot = rot6d_to_rotmat(x_obj[b, 0, 3:9].reshape(1, 6)).reshape(3, 3); trans = x_obj[b, 0, :3]
        pc_cam = torch.matmul(pc, rot.transpose(0, 1)) + trans
        origin = gaze_base[b, :, 0]; direction = gaze_base[b, :, 1] / gaze_base[b, :, 1].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rel = pc_cam.unsqueeze(0) - origin.unsqueeze(1); ray_t = (rel * direction.unsqueeze(1)).sum(-1).clamp_min(0.0)
        dist = (rel - ray_t.unsqueeze(-1) * direction.unsqueeze(1)).norm(dim=-1)
        frames = (dist.min(dim=1).values <= hit_m).nonzero().flatten().tolist()
        o_local = torch.matmul(rot.transpose(0, 1), origin.median(dim=0).values - trans)
        a = o_local - c; a = a - (a @ up) * up; a = a / a.norm()                      # object -> viewer, horizontal
        band = max(0.01, 0.04 * height)
        targets, target_h = {}, {}
        for q in levels:
            want = h_min + q * height
            near = (h - want).abs()
            cand = (near <= band).nonzero().flatten()
            if cand.numel() == 0:
                cand = near.argmin().view(1)
            pick = cand[((pc[cand] - c) @ a).argmax()]
            kind = f"q{int(round(100 * q)):02d}"
            targets[kind] = pc[pick]; target_h[kind] = float(h[pick] - h_min)
        for kind in ["orig"] + list(targets):
            g = gaze_base[b].clone()
            if kind != "orig":
                tgt_cam = torch.matmul(rot, targets[kind]) + trans
                for t in frames:
                    v = tgt_cam - g[t, 0]; g[t, 1] = v / v.norm().clamp_min(1e-8)
            out_gaze.append(g); src_idx.append(b); kinds.append(kind)
            # tag must not contain hand words: the evaluator selects the scored hands from the prompt text
            out_text.append(f"{text[b]} [gaze_v={kind}]")
        info.append(dict(object=name, up=up.tolist(), center=c.tolist(), h_min=h_min, height_m=height,
                         top_part=top_key, bottom_part=bottom_key, view=a.tolist(),
                         targets={k: v.tolist() for k, v in targets.items()}, target_height_m=target_h,
                         frames_redirected=len(frames), T=int(x_obj.shape[1])))
    if not out_gaze:
        return None, [], [], [], info
    gaze_out = torch.stack(out_gaze, dim=0)
    if had_last_dim:
        gaze_out = gaze_out.unsqueeze(-1)
    return gaze_out, src_idx, out_text, kinds, info


def _rotation_about_axis(axis, deg):
    """Rodrigues rotation matrix for a unit axis (3,) and an angle in degrees."""
    th = torch.tensor(float(deg) * np.pi / 180.0, device=axis.device, dtype=axis.dtype)
    k = axis / axis.norm()
    K = torch.stack([torch.stack([0 * k[0], -k[2], k[1]]), torch.stack([k[2], 0 * k[0], -k[0]]),
                     torch.stack([-k[1], k[0], 0 * k[0]])])
    return torch.eye(3, device=axis.device, dtype=axis.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _build_gaze_origin_rotation_variants(gaze, x_obj, obj_pc, text, object_names, labels, objects, angles,
                                         axis_mode="azimuth", hit_m=0.02, surface_m=0.005, durations=None):
    """Gaze-origin rotation (analysis): move the eye around the object instead of only re-aiming the ray.
    The rotation is about the object's up axis (azimuth) or about the horizontal axis normal to the
    eye->object view direction (elevation, + = eye moves up), through the object centre, in the
    first-frame object pose (the pose the gaze features use). Two variants per angle:
      origin-only "o<deg>": every frame's origin is rotated; frames whose recorded ray passes within hit_m
        of the object are re-aimed at their recorded first-surface hit point (nearest-the-eye object point
        within surface_m of the ray, else the point closest to the ray) - same target, new viewpoint; the
        other frames' directions are rotated with the origin.
      rigid "r<deg>": origin and direction of every frame are rotated together, so the hit point moves
        around the object with the viewpoint.
    Up axis as in _build_gaze_vertical_variants. Variant "orig" keeps the recorded rays. With 0 in `angles`,
    "r+0" repeats "orig" (row-position noise floor) and "o+0" only re-aims the hit frames at their recorded
    hit (control for the hit sharpening of the "o" variants). Eye median and hit counts use valid frames only."""
    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    out_gaze, src_idx, out_text, kinds, info = [], [], [], [], []
    for b in range(x_obj.shape[0]):
        name = str(object_names[b])
        parts = labels.get(name, {})
        top_key = next((k for k in _VERTICAL_TOP_PARTS if k in parts), None)
        if name not in objects or top_key is None:
            continue
        bottom_key = "bottom" if "bottom" in parts else "body"
        pc = obj_pc[b]
        up = (pc[torch.as_tensor(sorted(parts[top_key]), device=pc.device)].mean(0)
              - pc[torch.as_tensor(sorted(parts[bottom_key]), device=pc.device)].mean(0))
        up = up / up.norm(); c = pc.mean(0)
        rot = rot6d_to_rotmat(x_obj[b, 0, 3:9].reshape(1, 6)).reshape(3, 3); trans = x_obj[b, 0, :3]
        pc_cam = torch.matmul(pc, rot.transpose(0, 1)) + trans
        c_cam = torch.matmul(rot, c) + trans; up_cam = torch.matmul(rot, up)
        origin = gaze_base[b, :, 0]; direction = gaze_base[b, :, 1] / gaze_base[b, :, 1].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rel = pc_cam.unsqueeze(0) - origin.unsqueeze(1); ray_t = (rel * direction.unsqueeze(1)).sum(-1).clamp_min(0.0)
        dist = (rel - ray_t.unsqueeze(-1) * direction.unsqueeze(1)).norm(dim=-1)                   # (T, N)
        n_valid = int(durations[b]) if durations is not None else dist.shape[0]
        hit_frames = (dist.min(dim=1).values <= hit_m)
        hit_frames[n_valid:] = False
        # first-surface hit: nearest-the-eye point within surface_m of the ray, else the point closest to the ray
        on_surface = dist <= surface_m
        hit_idx = torch.where(on_surface, ray_t, torch.full_like(ray_t, float("inf"))).argmin(dim=1)
        hit_idx = torch.where(on_surface.any(dim=1), hit_idx, dist.argmin(dim=1))
        eye = origin[:n_valid].median(dim=0).values
        view = c_cam - eye; view = view - (view @ up_cam) * up_cam; view = view / view.norm()
        axis = up_cam if axis_mode == "azimuth" else torch.linalg.cross(up_cam, view)   # (up x view): + lifts the eye
        axis = axis / axis.norm()
        variants = [("orig", None, None)] + [(f"{m}{int(a):+d}", m, a) for a in angles for m in ("o", "r")]
        for kind, mode, deg in variants:
            g = gaze_base[b].clone()
            if mode is not None and not (mode == "r" and deg == 0):   # r+0 stays a bit-exact copy of orig
                Rm = _rotation_about_axis(axis, deg)
                new_o = torch.matmul(origin - c_cam, Rm.transpose(0, 1)) + c_cam
                new_d = torch.matmul(gaze_base[b, :, 1], Rm.transpose(0, 1))
                g[:, 0] = new_o; g[:, 1] = new_d
                if mode == "o":
                    for t in hit_frames.nonzero().flatten().tolist():
                        v = pc_cam[hit_idx[t]] - new_o[t]; g[t, 1] = v / v.norm().clamp_min(1e-8)
            out_gaze.append(g); src_idx.append(b); kinds.append(kind)
            # tag must not contain hand words: the evaluator selects the scored hands from the prompt text
            out_text.append(f"{text[b]} [gaze_rot={axis_mode[:2]}:{kind}]")
        up_local = up; view_local = torch.matmul(rot.transpose(0, 1), view)
        info.append(dict(object=name, axis_mode=axis_mode, up=up_local.tolist(), view=view_local.tolist(), center=c.tolist(),
                         angles=[float(a) for a in angles], frames_hit=int(hit_frames.sum()), T=n_valid,
                         eye_to_center_m=float((c_cam - eye).norm()),
                         eye_elevation_deg=float(np.degrees(np.arcsin(float(((eye - c_cam) / (eye - c_cam).norm()) @ up_cam))))))
    if not out_gaze:
        return None, [], [], [], info
    gaze_out = torch.stack(out_gaze, dim=0)
    if had_last_dim:
        gaze_out = gaze_out.unsqueeze(-1)
    return gaze_out, src_idx, out_text, kinds, info

def _gaze_keep_window_mask(window, frac, valid_mask_obj, ldist_map, rdist_map):
    """Frames whose gaze tokens are kept by texthom.exp.gaze_keep_window; the others are
    zeroed, i.e. set to the No-gaze value. pre_contact = start .. first GT contact frame inclusive
    (clips without contact keep every frame); first / last = first / last ceil(frac * valid frames)."""
    valid = valid_mask_obj.to(torch.bool)
    if window == "pre_contact":
        return build_gaze_alignment_temporal_mask(
            valid_mask_obj, ldist_map=ldist_map, rdist_map=rdist_map, temporal_scope="until_contact"
        ).to(torch.bool)
    n = valid.long().sum(dim=1, keepdim=True)
    k = torch.ceil(frac * n.float()).long()
    idx = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
    if window == "first":
        return valid & (idx < k)
    if window == "last":
        return valid & (idx >= n - k)
    raise ValueError(f"Unknown texthom.exp.gaze_keep_window={window!r}; expected all, pre_contact, first or last.")

def _repeat_tensor_for_variants(tensor, repeat_count):
    if tensor is None:
        return None
    return tensor.repeat_interleave(repeat_count, dim=0)


def _repeat_list_for_variants(values, repeat_count, suffixes=None):
    out = []
    for value in values:
        for idx in range(repeat_count):
            suffix = "" if suffixes is None else suffixes[idx]
            out.append(f"{value}{suffix}")
    return out


def _normalize_gaze_noise_mode(mode):
    mode = str(mode).strip().lower()
    aliases = {
        "": "none",
        "clean": "none",
        "off": "none",
        "rotate": "rotation",
        "rot": "rotation",
        "translate": "translation",
        "trans": "translation",
        "frame_drop": "frame_hold",
        "drop_frame": "frame_hold",
        "temporal_drop": "frame_hold",
        "angle": "angular",
        "dropout": "random_drop",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("none", "rotation", "angular", "translation", "frame_hold", "random_drop"):
        raise ValueError(
            f"Unknown texthom.exp.gaze_noise_mode={mode!r}; expected "
            "'none', 'rotation', 'angular', 'translation', 'frame_hold', or 'random_drop'."
        )
    return mode


def _hold_every_nth_frame(tensor, interval):
    if tensor is None:
        return None
    interval = int(interval)
    if interval < 2:
        raise ValueError("gaze_frame_hold_interval must be at least 2.")
    output = tensor.clone()
    frame_indices = torch.arange(
        interval - 1,
        tensor.shape[1],
        interval,
        device=tensor.device,
    )
    if frame_indices.numel() > 0:
        output[:, frame_indices] = output[:, frame_indices - 1]
    return output


def _drop_frames_randomly(tensor, prob, generator=None):
    """Tracker dropout: each frame after the first is lost with probability `prob` and holds the
    last kept frame (frame_hold does the same on a fixed every-n-th grid)."""
    B, T = tensor.shape[:2]
    drop = torch.rand((B, T), device=tensor.device, generator=generator) < float(prob)
    drop[:, 0] = False
    t = torch.arange(T, device=tensor.device).expand(B, T)
    src = torch.where(drop, torch.zeros_like(t), t).cummax(dim=1).values
    return torch.gather(tensor, 1, src.view(B, T, *([1] * (tensor.dim() - 2))).expand_as(tensor))


def _shift_gaze_in_time(gaze, nframes, shift):
    """gaze'[t] = gaze[clamp(t - shift, 0, n - 1)] within each clip's n valid frames (edge padding).
    shift > 0 delays the gaze relative to the motion; frames past n are left as they are (masked)."""
    shift = int(shift)
    if shift == 0:
        return gaze
    B, T = gaze.shape[:2]
    t = torch.arange(T, device=gaze.device).unsqueeze(0)
    n = torch.as_tensor(nframes, device=gaze.device).reshape(-1, 1).long().clamp(1, T)
    src = torch.minimum((t - shift).clamp(min=0), n - 1)
    src = torch.where(t < n, src, t.expand(B, T))
    return torch.gather(gaze, 1, src.view(B, T, *([1] * (gaze.dim() - 2))).expand_as(gaze))


def _gaze_direction_to_object_center(gaze, x_obj, obj_cent):
    """Viewpoint-only control: every recorded origin is kept and every direction points at the
    object centre (point-cloud centroid obj_cent placed with the initial pose, c = R0 cent + p0)."""
    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    B = x_obj.shape[0]
    R0 = rot6d_to_rotmat(x_obj[:, 0, 3:9].reshape(-1, 6)).reshape(B, 3, 3).to(gaze_base)
    centre = torch.einsum("bij,bj->bi", R0, obj_cent.reshape(B, 3).to(gaze_base)) + x_obj[:, 0, :3].to(gaze_base)
    output = gaze_base.clone()
    to_centre = centre[:, None, :] - output[:, :, 0]
    output[:, :, 1] = to_centre / to_centre.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return output.unsqueeze(-1) if had_last_dim else output


def _apply_gaze_noise(
    gaze,
    mode,
    generator=None,
    rotation_std_deg=2.0,
    translation_std_m=0.005,
    frame_hold_interval=5,
    frame_drop_prob=0.2,
):
    """Apply test-only corruption to camera-frame gaze rays."""
    mode = _normalize_gaze_noise_mode(mode)
    if mode == "none":
        return gaze
    if mode == "frame_hold":
        return _hold_every_nth_frame(gaze, frame_hold_interval)
    if mode == "random_drop":
        return _drop_frames_randomly(gaze, frame_drop_prob, generator)

    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    if gaze_base.dim() != 4 or gaze_base.shape[2:] != (2, 3):
        raise ValueError(
            f"Expected gaze shape (B,T,2,3) or (B,T,2,3,1), got "
            f"{tuple(gaze.shape)}."
        )
    output = gaze_base.clone()

    if mode in ("rotation", "angular"):
        rotation_std_deg = float(rotation_std_deg)
        if rotation_std_deg < 0:
            raise ValueError("gaze_rotation_std_deg must be non-negative.")
        direction = output[:, :, 1]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis = torch.randn(
            direction.shape,
            device=direction.device,
            dtype=direction.dtype,
            generator=generator,
        )
        if mode == "angular":
            # axis perpendicular to the ray: the direction turns by exactly |angle|, so the RMS
            # angular error equals rotation_std_deg ("rotation" uses a free axis: RMS ~0.82 x std)
            axis = axis - (axis * direction).sum(dim=-1, keepdim=True) * direction
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angle = torch.randn(
            (*direction.shape[:2], 1),
            device=direction.device,
            dtype=direction.dtype,
            generator=generator,
        )
        angle = angle * np.deg2rad(rotation_std_deg)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotated = (
            direction * cos_angle
            + torch.cross(axis, direction, dim=-1) * sin_angle
            + axis
            * (axis * direction).sum(dim=-1, keepdim=True)
            * (1.0 - cos_angle)
        )
        output[:, :, 1] = rotated / rotated.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    else:
        translation_std_m = float(translation_std_m)
        if translation_std_m < 0:
            raise ValueError("gaze_translation_std_m must be non-negative.")
        translation = torch.randn(
            output[:, :, 0].shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        # Parallel ray shift: move the attended region without another rotation.
        output[:, :, 0] += translation * translation_std_m

    return output.unsqueeze(-1) if had_last_dim else output


def _rebuild_gaze_map_for_rays(gaze, x_obj, obj_pc, template):
    """Rebuild the raw object-point gaze map after ray corruption."""
    if template is None:
        return None
    gaze_base, _ = _normalize_gaze_tensor(gaze)
    rebuilt = torch.zeros_like(template)
    batch_size, frame_count = x_obj.shape[:2]
    for b in range(batch_size):
        for t in range(frame_count):
            rot = rot6d_to_rotmat(x_obj[b, t, 3:9].reshape(1, 6)).reshape(3, 3)
            obj_points_cam = torch.matmul(obj_pc[b], rot.transpose(0, 1))
            obj_points_cam = obj_points_cam + x_obj[b, t, :3]
            origin = gaze_base[b, t, 0]
            direction = gaze_base[b, t, 1]
            direction = direction / direction.norm().clamp_min(1e-8)
            rel = obj_points_cam - origin.unsqueeze(0)
            ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
            closest = (
                origin.unsqueeze(0)
                + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
            )
            distance = (obj_points_cam - closest).norm(dim=-1)
            active = (distance <= 0.01) & (ray_t > 0)
            if bool(active.any().item()):
                rebuilt[b, t, active] = 1.0
            else:
                nearest = torch.topk(
                    distance,
                    k=min(64, distance.numel()),
                    largest=False,
                ).indices
                rebuilt[b, t, nearest] = 1.0
    return rebuilt


def _config_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ("true", "1", "yes", "y", "on"):
            return True
        if value_lower in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


def _compute_exterior_contact_frame_mask_per_hand(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    obj_pc_normal_org,
    dataset_name,
    valid_mask_hand,
    valid_mask_obj,
    contact_threshold=0.01,
    contact_min_keypoints=2,
):
    bs, nframes = pred_obj.shape[:2]
    hand_joints = get_hand_joints_w_tip(pred_hand, hand_layer)
    transf_obj_pc = get_transformed_obj_pc(pred_obj, obj_pc_org, dataset_name)

    flat_hand_joints = hand_joints.reshape(bs * nframes, -1, 3)
    flat_obj_pc = transf_obj_pc.reshape(bs * nframes, -1, 3)

    nn_dist, _ = get_NN(flat_hand_joints, flat_obj_pc)
    nn_dist = nn_dist.sqrt().view(bs, nframes, -1)

    close_contact_mask = nn_dist < contact_threshold
    return (close_contact_mask.sum(dim=2) >= contact_min_keypoints) & torch.logical_and(
        valid_mask_obj, valid_mask_hand
    )


def _resolve_output_representation(config):
    output_representation = str(
        getattr(config.texthom.model, "output_representation", "point")
    ).lower()
    aliases = {
        "points": "point",
        "point_token": "point",
        "point_tokens": "point",
        "mano": "pose",
        "mano_pose": "pose",
        "param": "pose",
        "params": "pose",
    }
    output_representation = aliases.get(output_representation, output_representation)
    if output_representation not in ("point", "pose"):
        raise ValueError(
            f"Unknown texthom.model.output_representation={output_representation!r}; "
            "expected 'point' or 'pose'."
        )
    config.texthom.model.output_representation = output_representation
    config.texthom.model.use_point_token_output = output_representation == "point"
    return output_representation


def _max_tail_joint_penetration_depth_per_hand(
    hand_joints,
    obj_pc,
    obj_pc_normal,
    tail_mask,
):
    batch_size, nframes = tail_mask.shape
    depths = torch.zeros(
        batch_size,
        nframes,
        device=hand_joints.device,
        dtype=hand_joints.dtype,
    )
    if not tail_mask.any():
        return depths

    flat_tail_mask = tail_mask.reshape(-1)
    flat_hand_joints = hand_joints.reshape(batch_size * nframes, -1, 3)
    flat_obj_pc = obj_pc.reshape(batch_size * nframes, -1, 3)
    flat_obj_pc_normal = obj_pc_normal.reshape(batch_size * nframes, -1, 3)

    tail_hand_joints = flat_hand_joints[flat_tail_mask]
    tail_obj_pc = flat_obj_pc[flat_tail_mask]
    tail_obj_pc_normal = flat_obj_pc_normal[flat_tail_mask]

    joint_nn_dist, joint_nn_idx = get_NN(tail_hand_joints, tail_obj_pc)
    joint_is_interior = get_interior(
        tail_obj_pc_normal,
        tail_obj_pc,
        tail_hand_joints,
        joint_nn_idx,
    )
    joint_depth = torch.where(
        joint_is_interior,
        joint_nn_dist.sqrt(),
        torch.zeros_like(joint_nn_dist),
    )
    depths[tail_mask] = joint_depth.max(dim=1).values
    return depths


def _run_single_seed(config):
    test_seed = seed_everything(
        getattr(config.texthom.exp, "seed", 0),
        deterministic=bool(getattr(config.texthom.exp, "deterministic", False)),
    )
    print(
        f"Using seed={test_seed} "
        f"deterministic={bool(getattr(config.texthom.exp, 'deterministic', False))}"
    )

    print(f"Test dataset split: {config.dataset.data_name}")
    config.batch_size = int(getattr(config.texthom.exp, "test_batch_size", 64))
    save_pre_post_comparison = _config_bool(
        getattr(config.texthom.exp, "save_pre_post_comparison", False),
        default=False,
    )
    max_test_batches = int(getattr(config.texthom.exp, "max_test_batches", 0))
    max_comparison_samples = int(
        getattr(config.texthom.exp, "max_comparison_samples", 0)
    )
    pre_post_comparison_records = []
    save_raw_point_tokens = _config_bool(
        getattr(config.texthom.exp, "save_raw_point_tokens", False), False
    )
    raw_point_records = []
    raw_batch_sink = [] if save_raw_point_tokens else None
    gaze_noise_mode = _normalize_gaze_noise_mode(
        getattr(config.texthom.exp, "gaze_noise_mode", "none")
    )
    gaze_rotation_std_deg = float(
        getattr(config.texthom.exp, "gaze_rotation_std_deg", 2.0)
    )
    gaze_translation_std_m = float(
        getattr(config.texthom.exp, "gaze_translation_std_m", 0.005)
    )
    gaze_frame_hold_interval = int(
        getattr(config.texthom.exp, "gaze_frame_hold_interval", 5)
    )
    gaze_frame_drop_prob = float(getattr(config.texthom.exp, "gaze_frame_drop_prob", 0.2))
    gaze_time_shift = int(getattr(config.texthom.exp, "gaze_time_shift", 0) or 0)
    gaze_direction_mode = str(getattr(config.texthom.exp, "gaze_direction_mode", "recorded") or "recorded").lower()
    assert gaze_direction_mode in ("recorded", "to_center"), gaze_direction_mode
    if gaze_time_shift != 0 or gaze_direction_mode != "recorded" or gaze_noise_mode == "random_drop":
        print(f"[gaze_ctrl] time_shift={gaze_time_shift} direction_mode={gaze_direction_mode} "
              f"frame_drop_prob={gaze_frame_drop_prob:g}")
    gaze_noise_generator = torch.Generator(device="cuda")
    gaze_noise_generator.manual_seed(int(test_seed) + 104729)
    print(
        "Test gaze noise: "
        f"mode={gaze_noise_mode}, rotation_std={gaze_rotation_std_deg:g}deg, "
        f"translation_std={gaze_translation_std_m:g}m, "
        f"frame_hold_interval={gaze_frame_hold_interval}"
    )

    config.texthom.exp.mano_fit_use_post_opt_losses = _config_bool(
        getattr(config.texthom.exp, "mano_fit_use_post_opt_losses", True),
        default=True,
    )
    print(
        "MANO post-optimization: "
        f"{'ON' if config.texthom.exp.mano_fit_use_post_opt_losses else 'OFF'}"
    )
    print(f"MANO refine mode: {getattr(config.texthom.exp, 'mano_refine_mode', 'contact_ramp')}")
    
    target_object_name = getattr(config.texthom.exp, "target_object_name", None)

    if target_object_name is not None:
        target_object_name = str(target_object_name)

    data_config = config.dataset
    dataset_name = data_config.name

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False).cuda()
    _, data_loader = get_dataloader(
        "Motion" + dataset_name, config, data_config, test=True
    )

    object_feature_type = str(
        getattr(config.texthom.model, "object_feature_type", "bps")
    ).lower()
    use_pointnet_object_feature = object_feature_type == "pointnet"
    if object_feature_type not in ("bps", "pointnet"):
        raise ValueError(
            f"Unknown texthom.model.object_feature_type={object_feature_type!r}; "
            "expected 'bps' or 'pointnet'."
        )
    gaze_condition_mode = str(
        getattr(config.texthom.model, "gaze_condition_mode", "alignment")
    ).lower()
    configure_gaze_token_fusion_for_mode(config)
    needs_bps = (
        not use_pointnet_object_feature
        or gaze_condition_requires_bps(gaze_condition_mode)
    )
    if use_pointnet_object_feature and gaze_condition_requires_bps(gaze_condition_mode):
        raise ValueError(
            "PointNet object ablation cannot use the BPS-dependent gaze mode "
            f"{gaze_condition_mode!r}. Use "
            "gaze_condition_mode=object_ray_closeness_temporal so gaze is "
            "computed directly from the gaze ray and object point cloud."
        )
    if needs_bps:
        bps_path = getattr(
            config.texthom.exp,
            "bps_path",
            "assets/bps.pt",
        )
        if not osp.isabs(bps_path):
            bps_path = osp.abspath(osp.join(PROJECT_ROOT, bps_path))
        bps_basis = load_bps_basis(bps_path, device="cuda")
        part_label_map, num_bps_parts = resolve_partwise_bps_context(config)
        bps_count = int(bps_basis.shape[0])
    else:
        bps_basis = None
        part_label_map = None
        num_bps_parts = 1
        bps_count = int(getattr(config.texthom.model, "gaze_map_dim", 1024))
        print("BPS disabled: object and gaze conditions use PointNet/object PC only.")
    object_bps_feature_mode = str(
        getattr(config.texthom.model, "object_bps_feature_mode", "displacement")
    ).lower()
    if object_bps_feature_mode in ("distance", "dist", "scalar", "norm"):
        object_bps_dim_per_token = 1
    elif object_bps_feature_mode in ("displacement", "xyz", "vector", "nearest_delta", "delta"):
        object_bps_dim_per_token = 3
    else:
        raise ValueError(
            f"Unknown texthom.model.object_bps_feature_mode={object_bps_feature_mode!r}; "
            "expected 'distance' or 'displacement'."
        )
    use_relative_gaze_mlp = _is_relative_gaze_mlp_mode(gaze_condition_mode)
    use_contact_condition = gaze_condition_mode in (
        "contact_map",
        "cov_map",
        "gt_contact_map",
        "bps_contact_map",
        "bps_cov_map",
        "raw_contact_map",
        "raw_cov_map",
    )
    use_gaze_closeness = bool(getattr(config.texthom.exp, "use_gaze_closeness", False))
    if (
        use_contact_condition
        or _is_temporal_gaze_token_mode(gaze_condition_mode)
        or gaze_condition_mode in (
            "direction_with_origin_xyz",
            "direction_alignment_origin_xyz",
            "alignment_direction_origin_xyz",
            "bps_direction_origin_xyz",
            "gaze_direction_origin_xyz",
            "alignment_combined",
            "combined_alignment",
            "origin_direction_alignment",
            "direction_origin_alignment",
            "bps_alignment_combined",
            "gaze_alignment_combined",
            "method1",
            "gaze_method1",
            "ray_distance_map",
            "object_ray_distance",
            "object_point_ray_distance",
            "ray_closeness_map",
            "object_ray_closeness_map",
            "method2",
            "gaze_method2",
            "bps_ray_distance_map",
            "bps_ray_closeness",
            "bps_ray_closeness_map",
            "bps_gaze_ray_closeness",
            "bps_gaze_ray_closeness_map",
        )
    ):
        use_gaze_closeness = False
    config.texthom.model.use_gaze_alignment = gaze_condition_mode in (
        "alignment",
        "bps_alignment",
        "gaze_alignment",
        "direction_with_origin_xyz",
        "direction_alignment_origin_xyz",
        "alignment_direction_origin_xyz",
        "bps_direction_origin_xyz",
        "gaze_direction_origin_xyz",
        "alignment_combined",
        "combined_alignment",
        "origin_direction_alignment",
        "direction_origin_alignment",
        "bps_alignment_combined",
        "gaze_alignment_combined",
        "method1",
        "gaze_method1",
        "ray_distance_map",
        "object_ray_distance",
        "object_point_ray_distance",
        "ray_closeness_map",
        "object_ray_closeness_map",
        "method2",
        "gaze_method2",
        "bps_ray_distance_map",
        "bps_ray_closeness",
        "bps_ray_closeness_map",
        "bps_gaze_ray_closeness",
        "bps_gaze_ray_closeness_map",
    ) or _is_temporal_gaze_token_mode(gaze_condition_mode)
    gaze_condition_dim = get_texthom_gaze_condition_dim(
        config, num_bps_parts, bps_count
    )
    object_feature_dim = (
        int(getattr(config.texthom.model, "pointnet_obj_dim", 1024))
        if use_pointnet_object_feature
        else num_bps_parts * bps_count * object_bps_dim_per_token
    )
    config.texthom.model.obj_global_dim = object_feature_dim
    if _is_temporal_gaze_token_mode(gaze_condition_mode):
        config.texthom.model.obj_dim = object_feature_dim
        config.texthom.model.gaze_token_dim = gaze_token_dim_for_model(config, gaze_condition_dim)
    else:
        config.texthom.model.obj_dim = (
            object_feature_dim
            + gaze_condition_dim
            + (num_bps_parts * bps_count if use_gaze_closeness else 0)
        )
        config.texthom.model.gaze_token_dim = 0
    print(
        "Using object/gaze condition dims: "
        f"num_bps_parts={num_bps_parts}, bps_count={bps_count}, "
        f"object_bps_feature_mode={object_bps_feature_mode}, "
        f"object_feature_dim={object_feature_dim}, "
        f"gaze_condition_dim={gaze_condition_dim}, "
        f"obj_dim={config.texthom.model.obj_dim}."
    )
    output_representation = _resolve_output_representation(config)
    use_point_token_output = output_representation == "point"
    predict_object_pose = bool(
        getattr(config.texthom.model, "predict_object_pose", True)
    )
    config.texthom.model.predict_object_pose = predict_object_pose
    hand_sample_indices = None
    if use_point_token_output:
        hand_sample_indices = load_hand_sample_indices_for_texthom(
            config, device="cuda"
        )
        include_hand_object_dirvec = True
        config.texthom.model.include_hand_object_dirvec = True
        canonicalize_point_targets = bool(
            getattr(config.texthom.model, "canonicalize_point_targets", True)
        )
        config.texthom.model.canonicalize_hand_point_targets = canonicalize_point_targets
        config.texthom.model.canonicalize_object_point_targets = canonicalize_point_targets
        use_obj_scale = bool(getattr(config.texthom.model, "use_obj_scale", True))
        use_obj_centroid = bool(
            getattr(config.texthom.model, "use_obj_centroid", False)
        )
        config.texthom.model.use_obj_scale = use_obj_scale
        config.texthom.model.use_obj_centroid = use_obj_centroid
        point_coord_dim = int(hand_sample_indices.numel()) * 3
        config.texthom.model.hand_nfeats = point_coord_dim * 2
        config.texthom.model.obj_nfeats = 9 if predict_object_pose else point_coord_dim
    else:
        use_obj_scale = bool(getattr(config.texthom.model, "use_obj_scale", True))
        use_obj_centroid = bool(
            getattr(config.texthom.model, "use_obj_centroid", False)
        )
    texthom, diffusion = build_model_and_diffusion(
        config, lhand_layer, rhand_layer, test=True
    )
    motion_normalizer = None
    if use_point_token_output and predict_object_pose:
        normalization_checkpoint = torch.load(
            config.texthom.exp.weight_path, map_location="cpu"
        )
        normalization_state = normalization_checkpoint.get("motion_normalization")
        if normalization_state is None:
            raise RuntimeError(
                "Hybrid hand-point/object-pose checkpoint is missing "
                "motion_normalization statistics. Retrain or use a compatible checkpoint."
            )
        motion_normalizer = MotionNormalizer.from_state_dict(normalization_state)
    use_obj_scale = bool(getattr(config.texthom.model, "use_obj_scale", True))
    use_obj_centroid = bool(
        getattr(config.texthom.model, "use_obj_centroid", False)
    )
    relative_gaze_mlp = (
        build_relative_gaze_mlp_for_texthom(config, output_dim=gaze_condition_dim).cuda()
        if use_relative_gaze_mlp
        else None
    )
    if relative_gaze_mlp is not None:
        ckpt = torch.load(config.texthom.exp.weight_path, map_location="cuda")
        relative_gaze_state = ckpt.get("relative_gaze_mlp")
        if relative_gaze_state is None:
            raise RuntimeError(
                "Checkpoint does not contain `relative_gaze_mlp` state, "
                "but gaze_condition_mode uses relative_gaze_mlp."
            )
        relative_gaze_mlp.load_state_dict(relative_gaze_state)
        relative_gaze_mlp.eval()
        for param in relative_gaze_mlp.parameters():
            param.requires_grad = False
    pointnet = None
    if use_pointnet_object_feature:
        config.pointfeat.global_feat = True
        pointnet = build_pointnetfeat(config, test=False)
        ckpt = torch.load(config.texthom.exp.weight_path, map_location="cuda")
        pointnet_state = ckpt.get("pointnet_model")
        if pointnet_state is None:
            raise RuntimeError(
                "PointNet object feature mode requires `pointnet_model` in the TextHOM checkpoint."
            )
        pointnet.load_state_dict(pointnet_state)
        pointnet.eval()
        for param in pointnet.parameters():
            param.requires_grad = False
        print(
            "Using PointNet object feature instead of BPS: "
            f"dim {object_feature_dim}."
        )
    bps_correspondence_source = get_bps_correspondence_source(config)
    mesh_bps_cache = (
        load_object_mesh_bps_cache(
            config,
            device="cuda",
            align_to_obj_pc_norm=True,
        )
        if (not use_pointnet_object_feature and bps_correspondence_source == "object_mesh")
        else None
    )
    mesh_bps_correspondence_cache = (
        None
        if use_pointnet_object_feature
        else build_bps_correspondence_cache(
            config,
            bps_basis,
            mesh_cache=mesh_bps_cache,
            device="cuda",
        )
    )
    if part_label_map is not None:
        print("Part-wise BPS enabled; BPS correspondences remain dynamic.")
    clip_text_encoder = ClipTextEncoder(config) if uses_text_condition(config) else None
    obj_meshes = load_object_mesh_cache(data_config)
    refinement_obj_meshes = build_bimart_refinement_mesh_cache(
        obj_meshes,
        max_faces=int(
            getattr(config.texthom.exp, "mano_refine_mesh_max_faces", 2000)
        ),
    )
    hand_nfeats = config.texthom.model.hand_nfeats
    obj_nfeats = config.texthom.model.obj_nfeats
    save_list = []
    numerical_events = []
    evaluated_sample_count = 0
    dataset_cursor = 0
    gaze_down_sweep = bool(getattr(config.texthom.exp, "gaze_down_sweep", False))
    gaze_down_sample_idx = int(getattr(config.texthom.exp, "gaze_down_sample_idx", 29))
    gaze_down_ratios = list(
        getattr(config.texthom.exp, "gaze_down_ratios", [0.0, 0.25, 0.5, 0.75])
    )
    gaze_down_shift_origin = bool(
        getattr(config.texthom.exp, "gaze_down_shift_origin", False)
    )
    gaze_down_processed = False
    # --- (analysis) gaze target intervention: redirect every recorded ray to a
    # canonical point of another labeled part; noise is shared across the batch so
    # variants of one clip are paired.
    gaze_part_swap = bool(getattr(config.texthom.exp, "gaze_part_swap", False))
    gaze_part_swap_labels_path = str(
        getattr(config.texthom.exp, "gaze_part_swap_labels", "labels_merged.json")
    )
    gaze_part_swap_include_same = bool(
        getattr(config.texthom.exp, "gaze_part_swap_include_same", True)
    )
    gaze_part_swap_min_parts = int(getattr(config.texthom.exp, "gaze_part_swap_min_parts", 2))
    gaze_part_swap_labels = None
    if gaze_part_swap:
        with open(gaze_part_swap_labels_path, "r", encoding="utf-8") as label_file:
            gaze_part_swap_labels = json.load(label_file)
        print(
            f"[gaze_part_swap] labels={gaze_part_swap_labels_path} "
            f"objects={len(gaze_part_swap_labels)} include_same={gaze_part_swap_include_same} "
            f"min_parts={gaze_part_swap_min_parts} shared_noise=True"
        )

    # --- (analysis) within-part gaze shift: same object, same prompt, gaze hit moved
    # along the prompt part's principal axis; shared noise pairs the three variants.
    gaze_part_shift = bool(getattr(config.texthom.exp, "gaze_part_shift", False))
    gaze_part_shift_pairs = set(str(getattr(config.texthom.exp, "gaze_part_shift_pairs", "")).split(",")) - {""}
    gaze_part_shift_frac = float(getattr(config.texthom.exp, "gaze_part_shift_frac", 0.30))
    gaze_part_shift_labels = None
    gaze_part_shift_info = []
    gaze_body_lr = bool(getattr(config.texthom.exp, "gaze_body_lr", False))
    gaze_body_lr_objects = set(str(getattr(config.texthom.exp, "gaze_body_lr_objects", "mug_white,mug_patterned")).split(",")) - {""}
    gaze_body_lr_angle = float(getattr(config.texthom.exp, "gaze_body_lr_angle", 40.0))
    gaze_body_lr_only = str(getattr(config.texthom.exp, "gaze_body_lr_only", "") or "") or None
    if gaze_body_lr:
        gaze_part_shift = True   # reuse the variant plumbing below
        gaze_part_shift_pairs = {"__body_lr__"}
        print(f"[gaze_body_lr] objects={sorted(gaze_body_lr_objects)} angle={gaze_body_lr_angle} shared_noise=True")
    gaze_vertical = bool(getattr(config.texthom.exp, "gaze_vertical", False))
    gaze_vertical_objects = set(str(getattr(config.texthom.exp, "gaze_vertical_objects", "carton_milk,flask,vase")).split(",")) - {""}
    gaze_vertical_levels = [float(v) for v in str(getattr(config.texthom.exp, "gaze_vertical_levels", "0.05,0.275,0.5,0.725,0.95")).split(",") if v]
    if gaze_vertical:
        assert not gaze_body_lr, "gaze_vertical and gaze_body_lr are exclusive"
        gaze_part_shift = True   # reuse the variant plumbing below
        gaze_part_shift_pairs = {"__vertical__"}
        print(f"[gaze_vertical] objects={sorted(gaze_vertical_objects)} levels={gaze_vertical_levels} shared_noise=True")
    gaze_origin_rot = str(getattr(config.texthom.exp, "gaze_origin_rot", "") or "").lower()   # azimuth | elevation
    gaze_origin_rot_objects = set(str(getattr(config.texthom.exp, "gaze_origin_rot_objects", "carton_milk,flask,vase")).split(",")) - {""}
    gaze_origin_rot_angles = [float(v) for v in str(getattr(config.texthom.exp, "gaze_origin_rot_angles", "-60,-30,30,60")).split(",") if v]
    if gaze_origin_rot:
        assert gaze_origin_rot in ("azimuth", "elevation"), gaze_origin_rot
        assert not (gaze_body_lr or gaze_vertical), "gaze_origin_rot excludes gaze_body_lr / gaze_vertical"
        gaze_part_shift = True   # reuse the variant plumbing below
        gaze_part_shift_pairs = {"__origin_rot__"}
        print(f"[gaze_origin_rot] mode={gaze_origin_rot} objects={sorted(gaze_origin_rot_objects)} angles={gaze_origin_rot_angles} shared_noise=True")
    gaze_keep_window = str(getattr(config.texthom.exp, "gaze_keep_window", "all") or "all").lower()
    gaze_keep_frac = float(getattr(config.texthom.exp, "gaze_keep_frac", 0.30))
    gaze_keep_info = []
    if gaze_keep_window != "all":
        print(f"[gaze_keep_window] window={gaze_keep_window} frac={gaze_keep_frac} (other frames' gaze tokens = 0)")
    if gaze_part_shift:
        with open(str(getattr(config.texthom.exp, "gaze_part_swap_labels", "labels_merged.json")), "r", encoding="utf-8") as label_file:
            gaze_part_shift_labels = json.load(label_file)
        assert gaze_part_shift_pairs, "gaze_part_shift=true needs texthom.exp.gaze_part_shift_pairs=obj:part,..."
        print(f"[gaze_part_shift] pairs={sorted(gaze_part_shift_pairs)} frac={gaze_part_shift_frac} shared_noise=True")

    with torch.no_grad():
        for batch_idx, item in enumerate(
            tqdm.tqdm(data_loader, desc="LOADING DATA"),
            start=1,
        ):
            if max_test_batches > 0 and batch_idx > max_test_batches:
                break
            if (
                save_pre_post_comparison
                and max_comparison_samples > 0
                and len(pre_post_comparison_records) >= max_comparison_samples
            ):
                break
            original_batch_size = item["x_obj"].shape[0]
            dataset_indices = torch.arange(
                dataset_cursor,
                dataset_cursor + original_batch_size,
                dtype=torch.long,
                device="cuda",
            )
            dataset_cursor += original_batch_size
            x_obj = item["x_obj"].cuda()
            x_lhand_gt = item["x_lhand"].cuda()
            x_rhand_gt = item["x_rhand"].cuda()
            obj_pc_org = item["obj_pc"].cuda()
            obj_pc_normal_org = item["obj_pc_normal"].cuda()
            gaze = item["gaze"].cuda()
            part_id = item["part_id"].cuda() if "part_id" in item else None
            ldist_map = item["ldist_map"].cuda()
            rdist_map = item["rdist_map"].cuda()
            source_gaze_map = item["gaze_map"].cuda()
            gaze_map = (
                source_gaze_map
                if gaze_condition_mode
                in ("gaze_map", "raw_gaze_map", "dataset_gaze_map")
                else None
            )
            contact_map = item["cov_map"].cuda() if use_contact_condition else None
            cam_pose = item.get("cam_pose")
            normalized_obj_pc = item["normalized_obj_pc"].cuda()
            obj_cent = item["obj_cent"].cuda()
            obj_scale = item["obj_scale"].cuda()
            duration = item["nframes"].cuda()
            is_lhand, is_rhand = item["is_lhand"].cuda(), item["is_rhand"].cuda()
            text = item["text"]
            batch_size = x_obj.shape[0]
            nobj = x_obj.shape[1] if x_obj.dim() == 4 else 1
            target_object_indices = _get_target_object_indices(
                item, batch_size, nobj, x_obj.device
            )
            object_names = _target_object_names(
                item, batch_size, nobj, target_object_indices
            )
            if x_obj.dim() == 4:
                x_obj = _gather_target_object_slot(x_obj, target_object_indices)
                obj_pc_org = _gather_target_object_slot(
                    obj_pc_org, target_object_indices
                )
                obj_pc_normal_org = _gather_target_object_slot(
                    obj_pc_normal_org, target_object_indices
                )
                normalized_obj_pc = _gather_target_object_slot(
                    normalized_obj_pc, target_object_indices
                )
                obj_cent = _gather_target_object_slot(
                    obj_cent, target_object_indices
                )
                obj_scale = _gather_target_object_slot(
                    obj_scale, target_object_indices
                )
            max_nframes = x_obj.shape[1]
            if target_object_name is not None:
                keep_indices = [
                    idx
                    for idx, object_name in enumerate(object_names)
                    if object_name == target_object_name
                ]
                if not keep_indices:
                    continue
                keep_tensor = torch.as_tensor(
                    keep_indices, dtype=torch.long, device=x_obj.device
                )
                x_obj = x_obj.index_select(0, keep_tensor)
                x_lhand_gt = x_lhand_gt.index_select(0, keep_tensor)
                x_rhand_gt = x_rhand_gt.index_select(0, keep_tensor)
                obj_pc_org = obj_pc_org.index_select(0, keep_tensor)
                obj_pc_normal_org = obj_pc_normal_org.index_select(0, keep_tensor)
                gaze = gaze.index_select(0, keep_tensor)
                if part_id is not None:
                    part_id = part_id.index_select(0, keep_tensor)
                ldist_map = ldist_map.index_select(0, keep_tensor)
                rdist_map = rdist_map.index_select(0, keep_tensor)
                if gaze_map is not None:
                    gaze_map = gaze_map.index_select(0, keep_tensor)
                if contact_map is not None:
                    contact_map = contact_map.index_select(0, keep_tensor)
                normalized_obj_pc = normalized_obj_pc.index_select(0, keep_tensor)
                obj_cent = obj_cent.index_select(0, keep_tensor)
                obj_scale = obj_scale.index_select(0, keep_tensor)
                duration = duration.index_select(0, keep_tensor)
                is_lhand = is_lhand.index_select(0, keep_tensor)
                is_rhand = is_rhand.index_select(0, keep_tensor)
                if torch.is_tensor(cam_pose):
                    cam_pose = cam_pose.index_select(0, keep_tensor.to(cam_pose.device))
                text = [text[idx] for idx in keep_indices]
                object_names = [object_names[idx] for idx in keep_indices]
                dataset_indices = dataset_indices.index_select(0, keep_tensor)
                source_gaze_map = source_gaze_map.index_select(0, keep_tensor)
            batch_size = x_obj.shape[0]

            gaze_down_ratio_values = None
            if gaze_down_sweep:
                keep_indices = torch.nonzero(
                    dataset_indices == gaze_down_sample_idx, as_tuple=False
                ).flatten()
                if keep_indices.numel() == 0:
                    continue
                if gaze_down_processed:
                    break
                keep_tensor = keep_indices.to(device=x_obj.device, dtype=torch.long)
                x_obj = x_obj.index_select(0, keep_tensor)
                x_lhand_gt = x_lhand_gt.index_select(0, keep_tensor)
                x_rhand_gt = x_rhand_gt.index_select(0, keep_tensor)
                obj_pc_org = obj_pc_org.index_select(0, keep_tensor)
                obj_pc_normal_org = obj_pc_normal_org.index_select(0, keep_tensor)
                normalized_obj_pc = normalized_obj_pc.index_select(0, keep_tensor)
                obj_cent = obj_cent.index_select(0, keep_tensor)
                obj_scale = obj_scale.index_select(0, keep_tensor)
                duration = duration.index_select(0, keep_tensor)
                is_lhand = is_lhand.index_select(0, keep_tensor)
                is_rhand = is_rhand.index_select(0, keep_tensor)
                gaze = gaze.index_select(0, keep_tensor)
                if part_id is not None:
                    part_id = part_id.index_select(0, keep_tensor)
                ldist_map = ldist_map.index_select(0, keep_tensor)
                rdist_map = rdist_map.index_select(0, keep_tensor)
                source_gaze_map = source_gaze_map.index_select(0, keep_tensor)
                if gaze_map is not None:
                    gaze_map = gaze_map.index_select(0, keep_tensor)
                if contact_map is not None:
                    contact_map = contact_map.index_select(0, keep_tensor)
                if torch.is_tensor(cam_pose):
                    cam_pose = cam_pose.index_select(0, keep_tensor.to(cam_pose.device))
                text = [text[int(idx)] for idx in keep_indices.detach().cpu().tolist()]
                object_names = [
                    object_names[int(idx)] for idx in keep_indices.detach().cpu().tolist()
                ]
                dataset_indices = dataset_indices.index_select(0, keep_tensor)

                repeat_count = len(gaze_down_ratios)
                gaze, shifted_gaze_map = _build_gaze_down_variants(
                    gaze,
                    source_gaze_map,
                    x_obj,
                    obj_pc_org,
                    gaze_down_ratios,
                    shift_origin=gaze_down_shift_origin,
                )
                if gaze_map is not None:
                    gaze_map = shifted_gaze_map
                ldist_map = _repeat_tensor_for_variants(ldist_map, repeat_count)
                rdist_map = _repeat_tensor_for_variants(rdist_map, repeat_count)
                x_obj = _repeat_tensor_for_variants(x_obj, repeat_count)
                x_lhand_gt = _repeat_tensor_for_variants(x_lhand_gt, repeat_count)
                x_rhand_gt = _repeat_tensor_for_variants(x_rhand_gt, repeat_count)
                obj_pc_org = _repeat_tensor_for_variants(obj_pc_org, repeat_count)
                obj_pc_normal_org = _repeat_tensor_for_variants(
                    obj_pc_normal_org, repeat_count
                )
                normalized_obj_pc = _repeat_tensor_for_variants(
                    normalized_obj_pc, repeat_count
                )
                obj_cent = _repeat_tensor_for_variants(obj_cent, repeat_count)
                obj_scale = _repeat_tensor_for_variants(obj_scale, repeat_count)
                duration = _repeat_tensor_for_variants(duration, repeat_count)
                is_lhand = _repeat_tensor_for_variants(is_lhand, repeat_count)
                is_rhand = _repeat_tensor_for_variants(is_rhand, repeat_count)
                if contact_map is not None:
                    contact_map = _repeat_tensor_for_variants(contact_map, repeat_count)
                if torch.is_tensor(cam_pose):
                    cam_pose = _repeat_tensor_for_variants(cam_pose, repeat_count)
                ratio_suffixes = [
                    f" [gaze_down={float(ratio):.2f}]" for ratio in gaze_down_ratios
                ]
                text = _repeat_list_for_variants(text, repeat_count, ratio_suffixes)
                object_names = _repeat_list_for_variants(object_names, repeat_count)
                dataset_indices = _repeat_tensor_for_variants(dataset_indices, repeat_count)
                gaze_down_ratio_values = [
                    float(ratio)
                    for _ in range(keep_indices.numel())
                    for ratio in gaze_down_ratios
                ]
                gaze_down_processed = True
                batch_size = x_obj.shape[0]

            if gaze_part_swap:
                keep_indices = [
                    idx for idx in range(batch_size)
                    if len(gaze_part_swap_labels.get(str(object_names[idx]), {}))
                    >= gaze_part_swap_min_parts
                ]
                if not keep_indices:
                    continue
                keep_tensor = torch.as_tensor(keep_indices, device=x_obj.device, dtype=torch.long)

                def _sel(tensor):
                    return tensor.index_select(0, keep_tensor.to(tensor.device)) if torch.is_tensor(tensor) else tensor

                (x_obj, x_lhand_gt, x_rhand_gt, obj_pc_org, obj_pc_normal_org, normalized_obj_pc,
                 obj_cent, obj_scale, duration, is_lhand, is_rhand, gaze, ldist_map, rdist_map,
                 source_gaze_map, dataset_indices) = [
                    _sel(t_) for t_ in (
                        x_obj, x_lhand_gt, x_rhand_gt, obj_pc_org, obj_pc_normal_org, normalized_obj_pc,
                        obj_cent, obj_scale, duration, is_lhand, is_rhand, gaze, ldist_map, rdist_map,
                        source_gaze_map, dataset_indices)
                ]
                gaze_map = _sel(gaze_map) if gaze_map is not None else None
                contact_map = _sel(contact_map) if contact_map is not None else None
                cam_pose = _sel(cam_pose)
                text = [text[idx] for idx in keep_indices]
                object_names = [object_names[idx] for idx in keep_indices]

                gaze, swapped_map, swap_src, text, swap_kinds = _build_gaze_part_swap_variants(
                    gaze, source_gaze_map, x_obj, obj_pc_org, text, object_names,
                    gaze_part_swap_labels, include_same=gaze_part_swap_include_same,
                )
                src_tensor = torch.as_tensor(swap_src, device=x_obj.device, dtype=torch.long)

                def _rep(tensor):
                    return tensor.index_select(0, src_tensor.to(tensor.device)) if torch.is_tensor(tensor) else tensor

                (x_obj, x_lhand_gt, x_rhand_gt, obj_pc_org, obj_pc_normal_org, normalized_obj_pc,
                 obj_cent, obj_scale, duration, is_lhand, is_rhand, ldist_map, rdist_map,
                 dataset_indices) = [
                    _rep(t_) for t_ in (
                        x_obj, x_lhand_gt, x_rhand_gt, obj_pc_org, obj_pc_normal_org, normalized_obj_pc,
                        obj_cent, obj_scale, duration, is_lhand, is_rhand, ldist_map, rdist_map,
                        dataset_indices)
                ]
                contact_map = _rep(contact_map) if contact_map is not None else None
                cam_pose = _rep(cam_pose)
                source_gaze_map = swapped_map
                if gaze_map is not None:
                    gaze_map = swapped_map
                object_names = [object_names[idx] for idx in swap_src]
                batch_size = x_obj.shape[0]
                print(
                    f"[gaze_part_swap] batch {batch_idx}: {len(keep_indices)} clips -> "
                    f"{batch_size} rows ({swap_kinds.count('orig')} orig, "
                    f"{swap_kinds.count('same')} same, {swap_kinds.count('swap')} swap)"
                )

            if gaze_part_shift:
                if gaze_body_lr:
                    shifted, shift_src, shift_text, shift_kinds, shift_info = _build_gaze_body_lr_variants(
                        gaze, x_obj, obj_pc_org, text, object_names, gaze_part_shift_labels,
                        gaze_body_lr_objects, angle_deg=gaze_body_lr_angle, only=gaze_body_lr_only,
                    )
                elif gaze_origin_rot:
                    shifted, shift_src, shift_text, shift_kinds, shift_info = _build_gaze_origin_rotation_variants(
                        gaze, x_obj, obj_pc_org, text, object_names, gaze_part_shift_labels,
                        gaze_origin_rot_objects, gaze_origin_rot_angles, axis_mode=gaze_origin_rot,
                        durations=duration,
                    )
                elif gaze_vertical:
                    shifted, shift_src, shift_text, shift_kinds, shift_info = _build_gaze_vertical_variants(
                        gaze, x_obj, obj_pc_org, text, object_names, gaze_part_shift_labels,
                        gaze_vertical_objects, gaze_vertical_levels,
                    )
                else:
                    shifted, shift_src, shift_text, shift_kinds, shift_info = _build_gaze_part_shift_variants(
                        gaze, x_obj, obj_pc_org, text, object_names, gaze_part_shift_labels,
                        gaze_part_shift_pairs, frac=gaze_part_shift_frac,
                    )
                if shifted is None:
                    continue
                for rec in shift_info:
                    rec["batch_idx"] = batch_idx
                gaze_part_shift_info.extend(shift_info)
                src_tensor = torch.as_tensor(shift_src, device=x_obj.device, dtype=torch.long)

                def _rep(tensor):
                    return tensor.index_select(0, src_tensor.to(tensor.device)) if torch.is_tensor(tensor) else tensor

                (x_obj, x_lhand_gt, x_rhand_gt, obj_pc_org, obj_pc_normal_org, normalized_obj_pc,
                 obj_cent, obj_scale, duration, is_lhand, is_rhand, ldist_map, rdist_map,
                 dataset_indices, source_gaze_map) = [
                    _rep(t_) for t_ in (
                        x_obj, x_lhand_gt, x_rhand_gt, obj_pc_org, obj_pc_normal_org, normalized_obj_pc,
                        obj_cent, obj_scale, duration, is_lhand, is_rhand, ldist_map, rdist_map,
                        dataset_indices, source_gaze_map)
                ]
                gaze = shifted
                contact_map = _rep(contact_map) if contact_map is not None else None
                cam_pose = _rep(cam_pose)
                if gaze_map is not None or gaze_noise_mode == "frame_hold":   # unused otherwise; a slow per-frame loop
                    source_gaze_map = _rebuild_gaze_map_for_rays(gaze, x_obj, obj_pc_org, source_gaze_map)
                if gaze_map is not None:
                    gaze_map = source_gaze_map
                if part_id is not None and torch.is_tensor(part_id):
                    part_id = _rep(part_id)
                text = shift_text
                object_names = [object_names[idx] for idx in shift_src]
                batch_size = x_obj.shape[0]
                print(f"[gaze_part_shift] batch {batch_idx}: {len(shift_info)} clips -> {batch_size} rows")

            if gaze_time_shift != 0 or gaze_direction_mode != "recorded":
                assert gaze_map is None, "gaze_time_shift / gaze_direction_mode rewrite rays only (no gaze_map modes)"
                gaze = _shift_gaze_in_time(gaze, duration, gaze_time_shift)
                if gaze_direction_mode == "to_center":
                    gaze = _gaze_direction_to_object_center(gaze, x_obj, obj_cent)
            if gaze_noise_mode != "none":
                gaze = _apply_gaze_noise(
                    gaze,
                    gaze_noise_mode,
                    generator=gaze_noise_generator,
                    rotation_std_deg=gaze_rotation_std_deg,
                    translation_std_m=gaze_translation_std_m,
                    frame_hold_interval=gaze_frame_hold_interval,
                    frame_drop_prob=gaze_frame_drop_prob,
                )
                if gaze_noise_mode == "frame_hold":
                    source_gaze_map = _hold_every_nth_frame(
                        source_gaze_map, gaze_frame_hold_interval
                    )
                    if gaze_map is not None:
                        gaze_map = source_gaze_map
                elif gaze_map is not None:
                    gaze_map = _rebuild_gaze_map_for_rays(
                        gaze,
                        x_obj,
                        obj_pc_org,
                        source_gaze_map,
                    )
                    source_gaze_map = gaze_map

            (
                valid_mask_lhand,
                valid_mask_rhand,
                valid_mask_obj,
            ) = get_valid_mask_bunch(is_lhand, is_rhand, max_nframes, duration)
            if gaze_condition_mode in (
                "alignment",
                "bps_alignment",
                "gaze_alignment",
                "direction_with_origin_xyz",
                "direction_alignment_origin_xyz",
                "alignment_direction_origin_xyz",
                "bps_direction_origin_xyz",
                "gaze_direction_origin_xyz",
                "alignment_combined",
                "combined_alignment",
                "origin_direction_alignment",
                "direction_origin_alignment",
                "bps_alignment_combined",
                "gaze_alignment_combined",
                "method1",
                "gaze_method1",
                "ray_distance_map",
                "object_ray_distance",
                "object_point_ray_distance",
                "ray_closeness_map",
                "object_ray_closeness_map",
                "method2",
                "gaze_method2",
                "bps_ray_distance_map",
                "bps_ray_closeness",
                "bps_ray_closeness_map",
                "bps_gaze_ray_closeness",
                "bps_gaze_ray_closeness_map",
            ) or _is_temporal_gaze_token_mode(gaze_condition_mode):
                gaze_condition_valid_mask = build_gaze_alignment_temporal_mask(
                    valid_mask_obj,
                    ldist_map=ldist_map,
                    rdist_map=rdist_map,
                    temporal_scope=getattr(
                        config.texthom.model,
                        "gaze_alignment_temporal_scope",
                        "all",
                    ),
                )
            else:
                gaze_condition_valid_mask = valid_mask_obj

            with torch.no_grad():
                if use_pointnet_object_feature:
                    obj_feat = pointnet(normalized_obj_pc)
                else:
                    obj_feat = compute_bps_feature_from_mesh_cache_for_texthom(
                        normalized_obj_pc,
                        bps_basis,
                        object_names=object_names,
                        mesh_cache=mesh_bps_cache,
                        mesh_bps_correspondence_cache=mesh_bps_correspondence_cache,
                        part_label_map=part_label_map,
                        bbox_margin=float(
                            getattr(config.dataset, "mesh_part_bbox_margin", 0.03)
                        ),
                        feature_mode=object_bps_feature_mode,
                    )
                gaze_condition_x_obj = select_object_pose_for_gaze_condition(config, x_obj)
                gaze_score = build_gaze_condition_feature_for_texthom(
                    config,
                    gaze,
                    gaze_map,
                    gaze_condition_x_obj,
                    normalized_obj_pc,
                    bps_basis,
                    obj_cent,
                    obj_scale,
                    gaze_condition_valid_mask,
                    object_names=object_names,
                    mesh_cache=mesh_bps_cache,
                    part_label_map=part_label_map,
                    bbox_margin=float(
                        getattr(config.dataset, "mesh_part_bbox_margin", 0.03)
                    ),
                    target_dim=gaze_condition_dim,
                    contact_map=contact_map,
                    relative_gaze_mlp=relative_gaze_mlp,
                )
                gaze_score = apply_gaze_token_weights(config, apply_null_gaze_condition(config, gaze_score))
                if gaze_keep_window != "all":
                    keep = _gaze_keep_window_mask(gaze_keep_window, gaze_keep_frac, valid_mask_obj, ldist_map, rdist_map)
                    assert not use_gaze_closeness and gaze_score.dim() >= 3 and gaze_score.shape[:2] == keep.shape, \
                        f"gaze_keep_window needs per-frame gaze tokens (B, T, ...); got {tuple(gaze_score.shape)}"
                    gaze_score = gaze_score * keep.view(*keep.shape, *([1] * (gaze_score.dim() - 2))).to(gaze_score.dtype)
                    n_valid = valid_mask_obj.to(torch.bool).sum(dim=1)
                    for i in range(keep.shape[0]):
                        gaze_keep_info.append(dict(batch_idx=batch_idx, row=i, text=str(text[i]), n_valid=int(n_valid[i]),
                                                   n_kept=int(keep[i].sum()), first_kept=int(keep[i].nonzero().min()) if keep[i].any() else -1,
                                                   last_kept=int(keep[i].nonzero().max()) if keep[i].any() else -1))
                    print(f"[gaze_keep_window] batch {batch_idx}: kept {float(keep.sum()) / float(n_valid.sum().clamp_min(1)):.3f} of valid frames")
                if use_gaze_closeness:
                    gaze_closeness = compute_bps_gaze_ray_closeness_map_from_mesh_cache_for_texthom(
                        gaze,
                        gaze_condition_x_obj,
                        normalized_obj_pc,
                        bps_basis,
                        obj_cent,
                        obj_scale,
                        valid_mask_obj,
                        object_names=object_names,
                        mesh_cache=mesh_bps_cache,
                        part_label_map=part_label_map,
                        bbox_margin=float(
                            getattr(config.dataset, "mesh_part_bbox_margin", 0.03)
                        ),
                        sigma=float(getattr(config.texthom.exp, "gaze_distance_sigma", 0.05)),
                    )
                    gaze_closeness = apply_null_gaze_condition(config, gaze_closeness)
                    contact_feat = torch.cat([gaze_score, gaze_closeness], dim=1)
                else:
                    contact_feat = gaze_score

            obj_feat_final = proc_obj_feat_final_train(
                contact_feat,
                obj_scale,
                obj_cent,
                obj_feat,
                use_obj_scale=use_obj_scale,
                use_obj_centroid=use_obj_centroid,
            )
            if isinstance(obj_feat_final, dict) and (bool(getattr(config.texthom.model, "global_gaze_origin", False)) or "origin" in str(
                    getattr(config.texthom.model, "gaze_memory_extras", "none"))):
                obj_feat_final["gaze_origin"] = build_gaze_origin_tokens_for_texthom(gaze, x_obj)
            if isinstance(obj_feat_final, dict) and bool(getattr(config.texthom.model, "use_part_label", False)):
                if part_id is None:
                    raise ValueError("use_part_label=True but the batch has no part_id")
                obj_feat_final["part_id"] = part_id
            if isinstance(obj_feat_final, dict) and clip_text_encoder is not None:
                obj_feat_final["text_emb"] = clip_text_encoder(target_condition_texts(
                    config, text, object_names, part_id if part_id is not None else [0] * len(text)))

            replace_gt = None
            if str(getattr(config.texthom.exp, "replace_sampling", "none")).startswith("gt_"):
                from lib.utils.texthom_train_helpers import build_hybrid_motion_targets_for_texthom
                assert predict_object_pose and use_point_token_output, "gt_* oracles need a point-token + object-pose model"
                replace_gt = build_hybrid_motion_targets_for_texthom(
                    x_lhand_gt, x_rhand_gt, x_obj, obj_pc_org, lhand_layer, rhand_layer,
                    dataset_name, hand_sample_indices, obj_pc_top_idx=None,
                    include_hand_object_dirvec=True,
                    canonicalize_hand_targets=bool(getattr(config.texthom.model, "canonicalize_point_targets", True)),
                )
                if motion_normalizer is not None:
                    replace_gt = motion_normalizer.normalize(*replace_gt)
            coarse_x_lhand, coarse_x_rhand, coarse_x_obj = diffusion.sampling(
                texthom,
                obj_feat_final,
                max_nframes,
                hand_nfeats,
                obj_nfeats,
                valid_mask_lhand,
                valid_mask_rhand,
                valid_mask_obj,
                device=torch.device("cuda"),
                enc_text=None,
                shared_noise_across_batch=gaze_down_sweep or gaze_part_swap or gaze_part_shift,
                gt_x0=replace_gt,
            )
            if motion_normalizer is not None:
                coarse_x_lhand, coarse_x_rhand, coarse_x_obj = (
                    motion_normalizer.denormalize(
                        coarse_x_lhand, coarse_x_rhand, coarse_x_obj
                    )
                )
            batch_diagnostics = []
            post_optimization_applied = False
            if use_point_token_output:
                point_token_lhand = coarse_x_lhand
                point_token_rhand = coarse_x_rhand
                point_token_obj = coarse_x_obj

                def _restore_point_token_variant(
                    refine_mode,
                    diagnostics_target,
                    return_pre_post,
                ):
                    return restore_hand_point_token_outputs_with_object_pose_for_texthom(
                        point_token_lhand,
                        point_token_rhand,
                        point_token_obj,
                        obj_pc_org,
                        lhand_layer,
                        rhand_layer,
                        hand_sample_indices,
                        dataset_name,
                        obj_pc_normal=obj_pc_normal_org,
                        include_hand_object_dirvec=True,
                        mano_fit_iters=int(
                            getattr(config.texthom.exp, "mano_fit_iters", 80)
                        ),
                        mano_fit_lr=float(
                            getattr(config.texthom.exp, "mano_fit_lr", 0.05)
                        ),
                        canonicalize_hand_targets=bool(
                            getattr(config.texthom.model, "canonicalize_point_targets", True)
                        ),
                        canonicalize_object_targets=bool(
                            getattr(config.texthom.model, "canonicalize_point_targets", True)
                        ),
                        initial_obj_pose=x_obj[:, :1],
                        object_pose_is_relative=predict_object_pose,
                        mano_fit_pose_reg_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_pose_reg_weight",
                                1e-4,
                            )
                        ),
                        mano_fit_temporal_reg_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_temporal_reg_weight",
                                1e-3,
                            )
                        ),
                        mano_fit_shape_reg_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_shape_reg_weight",
                                1e-4,
                            )
                        ),
                        mano_fit_trans_temporal_reg_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_trans_temporal_reg_weight",
                                1e-3,
                            )
                        ),
                        mano_fit_projection_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_projection_weight",
                                0.0,
                            )
                        ),
                        mano_fit_projection_obj_points=int(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_projection_obj_points",
                                256,
                            )
                        ),
                        mano_fit_penetration_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_penetration_weight",
                                0.0,
                            )
                        ),
                        mano_fit_penetration_obj_points=int(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_penetration_obj_points",
                                256,
                            )
                        ),
                        mano_fit_acceleration_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_acceleration_weight",
                                0.0,
                            )
                        ),
                        mano_fit_post_opt_steps=int(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_post_opt_steps",
                                0,
                            )
                        ),
                        mano_fit_use_post_opt_losses=_config_bool(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_use_post_opt_losses",
                                True,
                            ),
                            default=True,
                        ),
                        mano_fit_optimize_shape=bool(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_optimize_shape",
                                True,
                            )
                        ),
                        mano_fit_use_root_alignment=bool(
                            getattr(
                                config.texthom.exp,
                                "mano_fit_use_root_alignment",
                                True,
                            )
                        ),
                        diagnostics=diagnostics_target,
                        object_names=object_names,
                        object_meshes=refinement_obj_meshes,
                        is_lhand=is_lhand,
                        is_rhand=is_rhand,
                        valid_mask_lhand=valid_mask_lhand,
                        valid_mask_rhand=valid_mask_rhand,
                        valid_mask_obj=valid_mask_obj,
                        mano_refine_steps=int(
                            getattr(config.texthom.exp, "mano_refine_steps", 100)
                        ),
                        mano_refine_lr=float(
                            getattr(config.texthom.exp, "mano_refine_lr", 5e-4)
                        ),
                        mano_refine_penetration_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_penetration_weight",
                                10.0,
                            )
                        ),
                        mano_refine_projection_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_projection_weight",
                                100.0,
                            )
                        ),
                        mano_refine_acceleration_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_acceleration_weight",
                                1000.0,
                            )
                        ),
                        mano_refine_contact_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_contact_weight",
                                100.0,
                            )
                        ),
                        mano_refine_contact_threshold=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_contact_threshold",
                                0.02,
                            )
                        ),
                        mano_refine_fallback_contact_frames=int(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_fallback_contact_frames",
                                10,
                            )
                        ),
                        mano_refine_transition_frames=int(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_transition_frames",
                                5,
                            )
                        ),
                        mano_refine_velocity_preserve_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_velocity_preserve_weight",
                                0.0,
                            )
                        ),
                        mano_refine_hand_object_penetration_weight=float(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_hand_object_penetration_weight",
                                0.0,
                            )
                        ),
                        mano_refine_mode=str(refine_mode),
                        mano_refine_max_frames=int(
                            getattr(
                                config.texthom.exp,
                                "mano_refine_max_frames",
                                20,
                            )
                        ),
                        return_pre_post_opt=return_pre_post,
                        raw_points_sink=raw_batch_sink,
                        progress_prefix=f"batch {batch_idx}/{len(data_loader)}",
                    )
                restore_result = _restore_point_token_variant(
                    str(getattr(config.texthom.exp, "mano_refine_mode", "contact_ramp")),
                    batch_diagnostics,
                    save_pre_post_comparison,
                )
                if save_pre_post_comparison:
                    (
                        coarse_x_lhand,
                        coarse_x_rhand,
                        coarse_x_obj,
                        pre_post_x_lhand,
                        pre_post_x_rhand,
                        pre_post_x_obj,
                    ) = restore_result
                    for sample_idx in range(coarse_x_obj.shape[0]):
                        sample_duration = int(duration[sample_idx].item())
                        pre_post_comparison_records.append(
                            {
                                "dataset_index": int(dataset_indices[sample_idx].item()),
                                "text": str(text[sample_idx]),
                                "object_name": str(object_names[sample_idx]),
                                "duration": sample_duration,
                                "is_lhand": bool(is_lhand[sample_idx].item()),
                                "is_rhand": bool(is_rhand[sample_idx].item()),
                                "object_points": obj_pc_org[sample_idx].detach().cpu(),
                                "object_pose": pre_post_x_obj[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "pre_lhand": pre_post_x_lhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "pre_rhand": pre_post_x_rhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "post_lhand": coarse_x_lhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "post_rhand": coarse_x_rhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                            }
                        )
                else:
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj = restore_result
                post_optimization_applied = bool(
                    config.texthom.exp.mano_fit_use_post_opt_losses
                )
                if save_raw_point_tokens and raw_batch_sink:
                    raw = raw_batch_sink.pop()
                    raw_batch_sink.clear()
                    for sample_idx in range(coarse_x_obj.shape[0]):
                        sample_duration = int(duration[sample_idx].item())
                        raw_point_records.append(
                            {
                                "dataset_index": int(dataset_indices[sample_idx].item()),
                                "text": str(text[sample_idx]),
                                "object_name": str(object_names[sample_idx]),
                                "duration": sample_duration,
                                "is_lhand": bool(is_lhand[sample_idx].item()),
                                "is_rhand": bool(is_rhand[sample_idx].item()),
                                "object_points": obj_pc_org[sample_idx].detach().cpu(),
                                "obj_pose": raw["obj_pose"][sample_idx, :sample_duration],
                                "lhand_points": raw["lhand_points"][sample_idx, :sample_duration],
                                "rhand_points": raw["rhand_points"][sample_idx, :sample_duration],
                                "lhand_dirvec": None if raw.get("lhand_dirvec") is None else raw["lhand_dirvec"][sample_idx, :sample_duration],
                                "rhand_dirvec": None if raw.get("rhand_dirvec") is None else raw["rhand_dirvec"][sample_idx, :sample_duration],
                            }
                        )
                if batch_diagnostics:
                    for event in batch_diagnostics:
                        event.update(
                            {
                                "sample_start": evaluated_sample_count + 1,
                                "sample_end": evaluated_sample_count + batch_size,
                                "object_names": list(object_names),
                                "texts": list(text),
                            }
                        )
                    numerical_events.extend(batch_diagnostics)

            finite_outputs = {
                "left_hand": torch.isfinite(coarse_x_lhand)
                .reshape(coarse_x_lhand.shape[0], -1)
                .all(dim=1),
                "right_hand": torch.isfinite(coarse_x_rhand)
                .reshape(coarse_x_rhand.shape[0], -1)
                .all(dim=1),
                "object": torch.isfinite(coarse_x_obj)
                .reshape(coarse_x_obj.shape[0], -1)
                .all(dim=1),
            }
            invalid_samples = []
            for local_idx in range(coarse_x_obj.shape[0]):
                invalid_parts = [
                    name
                    for name, finite_mask in finite_outputs.items()
                    if not bool(finite_mask[local_idx].item())
                ]
                if invalid_parts:
                    invalid_samples.append(
                        {
                            "sample": evaluated_sample_count + local_idx + 1,
                            "local_index": local_idx,
                            "object_name": object_names[local_idx],
                            "text": text[local_idx],
                            "invalid_parts": invalid_parts,
                        }
                    )
            if invalid_samples:
                failure_event = {
                    "stage": "final_output_validation",
                    "reason": "non_finite_output",
                    "action": "abort_without_saving_incomplete_results",
                    "samples": invalid_samples,
                }
                numerical_events.append(failure_event)
                save_name = (
                    config.texthom.exp.save_name
                    if config.texthom.exp.save_name
                    else config.texthom.exp.name
                )
                info_path = f"{save_name}_numerical_info.json"
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "test_seed": test_seed,
                            "status": "failed",
                            "evaluated_samples_before_failure": evaluated_sample_count,
                            "events": numerical_events,
                        },
                        f,
                        indent=2,
                    )
                raise RuntimeError(
                    "Non-finite evaluation output detected. "
                    f"Details were written to {info_path}; incomplete PKL was not saved."
                )

            def _append_batch_result(
                target_save_list,
                pred_lhand,
                pred_rhand,
                pred_obj,
                variant_diagnostics,
                post_optimization_applied,
            ):
                contact_threshold = 0.02
                contact_min_keypoints = int(
                    getattr(config.texthom.exp, "eval_contact_min_keypoints", 2)
                )
                last_k = 5
                penetration_max_depth = 0.02
                lhand_contact_frame_mask = _compute_exterior_contact_frame_mask_per_hand(
                    pred_lhand,
                    lhand_layer,
                    pred_obj,
                    obj_pc_org,
                    obj_pc_normal_org,
                    dataset_name,
                    valid_mask_lhand,
                    valid_mask_obj,
                    contact_threshold=contact_threshold,
                    contact_min_keypoints=contact_min_keypoints,
                )
                rhand_contact_frame_mask = _compute_exterior_contact_frame_mask_per_hand(
                    pred_rhand,
                    rhand_layer,
                    pred_obj,
                    obj_pc_org,
                    obj_pc_normal_org,
                    dataset_name,
                    valid_mask_rhand,
                    valid_mask_obj,
                    contact_threshold=contact_threshold,
                    contact_min_keypoints=contact_min_keypoints,
                )
                tail_mask_obj = compute_tail_mask(valid_mask_obj, last_k=last_k)
                has_full_tail_obj = tail_mask_obj.sum(dim=1) == last_k

                end_contact_list = []
                eval_hands_list = []
                lhand_joints = get_hand_joints_w_tip(pred_lhand, lhand_layer)
                rhand_joints = get_hand_joints_w_tip(pred_rhand, rhand_layer)
                transf_obj_pc = get_transformed_obj_pc(
                    pred_obj, obj_pc_org, dataset_name
                )
                transf_obj_pc_normal = _rotate_obj_vectors(
                    pred_obj, obj_pc_normal_org, dataset_name
                )
                left_tail_penetration_depth = _max_tail_joint_penetration_depth_per_hand(
                    lhand_joints,
                    transf_obj_pc,
                    transf_obj_pc_normal,
                    tail_mask_obj,
                )
                right_tail_penetration_depth = _max_tail_joint_penetration_depth_per_hand(
                    rhand_joints,
                    transf_obj_pc,
                    transf_obj_pc_normal,
                    tail_mask_obj,
                )
                left_last_id_max = None
                right_last_id_max = None
                left_last_id_valid = None
                right_last_id_valid = None
                if obj_meshes is not None:
                    _, left_last_id_max, _, _, left_last_id_valid = (
                        _compute_last_frame_mesh_penetration_metrics_per_hand(
                            pred_lhand,
                            lhand_layer,
                            pred_obj,
                            obj_meshes,
                            object_names,
                            valid_mask_lhand,
                            valid_mask_obj=valid_mask_obj,
                        )
                    )
                    _, right_last_id_max, _, _, right_last_id_valid = (
                        _compute_last_frame_mesh_penetration_metrics_per_hand(
                            pred_rhand,
                            rhand_layer,
                            pred_obj,
                            obj_meshes,
                            object_names,
                            valid_mask_rhand,
                            valid_mask_obj=valid_mask_obj,
                        )
                    )
                joint_penetration_depth_list = []
                for i in range(pred_obj.shape[0]):
                    use_left_eval, use_right_eval = _resolve_eval_hands(
                        text[i], is_lhand[i].item(), is_rhand[i].item()
                    )
                    eval_hands_list.append(
                        {"left": bool(use_left_eval), "right": bool(use_right_eval)}
                    )
                    sample_contact_frame_mask = torch.zeros_like(
                        lhand_contact_frame_mask[i], dtype=torch.bool
                    )
                    if use_left_eval:
                        sample_contact_frame_mask |= lhand_contact_frame_mask[i]
                    if use_right_eval:
                        sample_contact_frame_mask |= rhand_contact_frame_mask[i]
                    sample_end_contact = bool(
                        ((sample_contact_frame_mask | (~tail_mask_obj[i])).all().item())
                        and bool(has_full_tail_obj[i].item())
                    )
                    end_contact_list.append(sample_end_contact)
                    tail_indices = torch.nonzero(
                        tail_mask_obj[i], as_tuple=False
                    ).flatten()
                    max_penetration_depth = torch.zeros(
                        last_k, dtype=pred_obj.dtype
                    )
                    used_mesh_penetration = False
                    if obj_meshes is not None and tail_indices.numel() > 0:
                        sample_depth = torch.tensor(0.0, dtype=pred_obj.dtype)
                        if (
                            use_left_eval
                            and left_last_id_max is not None
                            and bool(left_last_id_valid[i].item())
                        ):
                            sample_depth = torch.maximum(
                                sample_depth, left_last_id_max[i].detach().cpu()
                            )
                            used_mesh_penetration = True
                        if (
                            use_right_eval
                            and right_last_id_max is not None
                            and bool(right_last_id_valid[i].item())
                        ):
                            sample_depth = torch.maximum(
                                sample_depth, right_last_id_max[i].detach().cpu()
                            )
                            used_mesh_penetration = True
                    if used_mesh_penetration and tail_indices.numel() > 0:
                        max_penetration_depth[: tail_indices.numel()] = sample_depth
                    elif tail_indices.numel() > 0:
                        if use_left_eval:
                            max_penetration_depth[: tail_indices.numel()] = torch.maximum(
                                max_penetration_depth[: tail_indices.numel()],
                                left_tail_penetration_depth[
                                    i, tail_indices
                                ].detach().cpu(),
                            )
                        if use_right_eval:
                            max_penetration_depth[: tail_indices.numel()] = torch.maximum(
                                max_penetration_depth[: tail_indices.numel()],
                                right_tail_penetration_depth[
                                    i, tail_indices
                                ].detach().cpu(),
                            )
                    joint_penetration_depth_list.append(
                        [float(v) for v in max_penetration_depth.tolist()]
                    )

                target_save_list.append(
                    [
                        [
                            pred_obj[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_obj.shape[0])
                        ],
                        [
                            pred_lhand[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_lhand.shape[0])
                        ],
                        [
                            pred_rhand[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_rhand.shape[0])
                        ],
                        text,
                        [
                            {
                                "contact_success": bool(end_contact_list[i]),
                                "contact_fail": not bool(end_contact_list[i]),
                                "success": bool(
                                    end_contact_list[i]
                                    and all(
                                        v <= penetration_max_depth
                                        for v in joint_penetration_depth_list[i]
                                    )
                                ),
                                "eval_hands": eval_hands_list[i],
                                "post_optimization_applied": bool(
                                    post_optimization_applied
                                ),
                                "numerical_events_applied": bool(variant_diagnostics),
                                "numerical_events": variant_diagnostics,
                                "numerical_fallback_applied": False,
                                "numerical_fallback_events": [],
                                "joint_penetration_max_depth_threshold_m": penetration_max_depth,
                                "joint_penetration_max_depth_last_frames": joint_penetration_depth_list[
                                    i
                                ],
                            }
                            for i in range(pred_obj.shape[0])
                        ],
                        gaze.detach().cpu(),
                        cam_pose.detach().cpu(),
                        [
                            x_obj[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_obj.shape[0])
                        ],
                        [
                            x_lhand_gt[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_lhand.shape[0])
                        ],
                        [
                            x_rhand_gt[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_rhand.shape[0])
                        ],
                        [
                            {
                                "object_name": (
                                    object_names[i] if i < len(object_names) else None
                                ),
                                "dataset_name": dataset_name,
                                "obj_pc_org": obj_pc_org[i].detach().cpu(),
                                "dataset_sample_idx": int(dataset_indices[i].item())
                                if i < dataset_indices.shape[0]
                                else None,
                                "gaze_down_ratio": (
                                    gaze_down_ratio_values[i]
                                    if gaze_down_ratio_values is not None
                                    and i < len(gaze_down_ratio_values)
                                    else None
                                ),
                                "gaze_down_shift_origin": gaze_down_shift_origin,
                                "gaze_noise_mode": gaze_noise_mode,
                                "gaze_rotation_std_deg": gaze_rotation_std_deg,
                                "gaze_translation_std_m": gaze_translation_std_m,
                                "gaze_frame_hold_interval": gaze_frame_hold_interval,
                                "test_seed": test_seed,
                            }
                            for i in range(pred_obj.shape[0])
                        ],
                    ]
                )

            _append_batch_result(
                save_list,
                coarse_x_lhand,
                coarse_x_rhand,
                coarse_x_obj,
                batch_diagnostics,
                post_optimization_applied,
            )
            evaluated_sample_count += coarse_x_obj.shape[0]

    if gaze_down_sweep and not gaze_down_processed:
        raise RuntimeError(
            f"gaze_down_sweep requested sample index {gaze_down_sample_idx}, "
            f"but it was not found in dataset split {config.dataset.data_name}."
        )

    if gaze_keep_window != "all":
        with open(f"{config.texthom.exp.save_name or config.texthom.exp.name}_window_info.json", "w", encoding="utf-8") as f:
            json.dump(gaze_keep_info, f, indent=1)
    if gaze_part_shift:
        shift_save = config.texthom.exp.save_name or config.texthom.exp.name
        with open(f"{shift_save}_shift_info.json", "w", encoding="utf-8") as f:
            json.dump(gaze_part_shift_info, f, indent=1)
    return save_list, pre_post_comparison_records, raw_point_records


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    task_overrides = HydraConfig.get().overrides.task
    config = edict(OmegaConf.to_object(config))
    gaze_noise_mode = _normalize_gaze_noise_mode(
        getattr(config.texthom.exp, "gaze_noise_mode", "none")
    )
    config.dataset.data_name = _resolve_test_data_name(config, task_overrides)
    print(
        "Null gaze condition: "
        f"{'ON' if bool(getattr(config.texthom.model, 'null_gaze_condition', False)) else 'OFF'}"
    )

    base_seed = int(getattr(config.texthom.exp, "seed", 0))
    num_test_seeds = int(getattr(config.texthom.exp, "num_test_seeds", 10))

    test_seeds = list(range(base_seed, base_seed + num_test_seeds))
    print(f"Running {num_test_seeds} diffusion samples with seeds: {test_seeds}")
    save_list = []
    raw_point_records_all = []
    pre_post_comparison_records = []
    save_pre_post_comparison = bool(
        getattr(config.texthom.exp, "save_pre_post_comparison", False)
    )

    for run_idx, test_seed in enumerate(test_seeds, start=1):
        print(f"\n=== Test seed {run_idx}/{num_test_seeds}: {test_seed} ===")
        run_config = deepcopy(config)
        run_config.texthom.exp.seed = test_seed
        seed_save_list, seed_comparison_records, seed_raw_records = _run_single_seed(run_config)
        save_list.extend(seed_save_list)
        for rec in seed_raw_records:
            rec["seed"] = int(run_config.texthom.exp.seed)
        raw_point_records_all.extend(seed_raw_records)
        pre_post_comparison_records.extend(seed_comparison_records)

    save_name = (
        config.texthom.exp.save_name
        if config.texthom.exp.save_name
        else (
            f"{config.texthom.exp.name}_gaze_noise_{gaze_noise_mode}"
            if gaze_noise_mode != "none"
            else config.texthom.exp.name
        )
    )
    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(save_list, f)

    if save_pre_post_comparison:
        comparison_path = f"{save_name}_pre_post_comparison.pkl"
        with open(comparison_path, "wb") as f:
            pickle.dump(pre_post_comparison_records, f)
        print(f"Saved pre/post comparison at {comparison_path}")

    if raw_point_records_all:
        with open(f"{save_name}_rawpoints.pkl", "wb") as f:
            pickle.dump(raw_point_records_all, f)
        print(f"Saved raw point tokens at {save_name}_rawpoints.pkl ({len(raw_point_records_all)} records)")
    print(f"Saved at {save_name}.pkl")


if __name__ == "__main__":
    main()
