import os.path as osp
import shutil
import sys

# Keep this bootstrap before project imports (formatters may run isort separately).
PROJECT_ROOT = osp.dirname(osp.abspath(osp.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# isort: split

from lib.utils.train_texthom_utils import (
    TeeIO,
    get_current_git_branch,
    save_code_snapshot,
)
from lib.utils.proc import (
    load_bps_basis,
    proc_obj_feat_final_train,
)
from lib.utils.seed import seed_everything
from lib.utils.file import (
    make_model_result_folder,
    wandb_login,
)
from lib.utils.metric import AverageMeter
from lib.utils.texthom_config import (
    get_texthom_paths,
    move_batch_to_cuda,
)
from lib.utils.model_utils import (
    build_model_and_diffusion,
    build_pointnetfeat,
)
from lib.datasets.datasets import get_dataloader
from lib.models.mano import build_mano_aa
from lib.utils.texthom_train_helpers import (
    build_gaze_origin_tokens_for_texthom,
    _is_relative_gaze_mlp_mode,
    _is_temporal_gaze_token_mode,
    build_gaze_alignment_temporal_mask,
    apply_gaze_token_weights,
    apply_null_gaze_condition,
    build_relative_gaze_mlp_for_texthom,
    build_bps_correspondence_cache,
    build_gaze_condition_feature_for_texthom,
    build_point_token_motion_targets_for_texthom,
    build_hybrid_motion_targets_for_texthom,
    compute_bps_feature_from_mesh_cache_for_texthom,
    configure_gaze_token_fusion_for_mode,
    get_bps_correspondence_source,
    gaze_token_dim_for_model,
    get_texthom_gaze_condition_dim,
    gaze_condition_requires_bps,
    load_hand_sample_indices_for_texthom,
    load_object_mesh_bps_cache,
    repeat_initial_object_pose_for_gaze_condition,
    select_object_pose_for_gaze_condition,
    resolve_partwise_bps_context,
)
from lib.utils.target_condition import (
    ClipTextEncoder,
    target_condition_texts,
    uses_text_condition,
)
from lib.utils.motion_normalizer import (
    MaskedFeatureStats,
    MotionNormalizer,
    finalize_motion_normalizer,
)
import torch.optim as optim
import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from datetime import datetime
import hydra
import numpy as np
import tqdm
import os


def _gaze_direction_to_object_center(gaze, x_obj, obj_cent):
    """Same rewrite as test_texthom._gaze_direction_to_object_center: origin kept, direction -> object centre (initial pose)."""
    from lib.utils.rot import rot6d_to_rotmat
    squeeze = gaze.dim() == 5 and gaze.shape[-1] == 1
    g = gaze.squeeze(-1) if squeeze else gaze
    B = x_obj.shape[0]
    R0 = rot6d_to_rotmat(x_obj[:, 0, 3:9].reshape(-1, 6)).reshape(B, 3, 3).to(g)
    centre = torch.einsum("bij,bj->bi", R0, obj_cent.reshape(B, 3).to(g)) + x_obj[:, 0, :3].to(g)
    out = g.clone()
    to_centre = centre[:, None, :] - out[:, :, 0]
    out[:, :, 1] = to_centre / to_centre.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return out.unsqueeze(-1) if squeeze else out


def _train_gaze_keep_window_mask(window, frac, valid_mask_obj, ldist_map, rdist_map):
    """Train-time copy of test_texthom._gaze_keep_window_mask (first-30% gaze training: train with the same
    gaze window as the main test protocol). Kept frames = True; the rest get gaze tokens = 0 (No-gaze value).
    first / last = first / last ceil(frac * valid frames); pre_contact = start .. first GT contact frame."""
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
    raise ValueError(f"Unknown texthom.exp.train_gaze_keep_window={window!r}; expected all, pre_contact, first or last.")


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


@torch.no_grad()
def _compute_hybrid_motion_normalizer(
    dataloader,
    lhand_layer,
    rhand_layer,
    hand_sample_indices,
    dataset_name,
    include_hand_object_dirvec=True,
    canonicalize_hand_targets=True,
    min_std=1e-4,
):
    point_count = int(hand_sample_indices.numel())
    hand_dim = point_count * (6 if include_hand_object_dirvec else 3)
    left_stats = MaskedFeatureStats(hand_dim)
    right_stats = MaskedFeatureStats(hand_dim)
    object_stats = MaskedFeatureStats(9)
    stat_keys = [
        "x_lhand",
        "x_rhand",
        "x_obj",
        "obj_pc",
        "valid_mask_lhand",
        "valid_mask_rhand",
        "valid_mask_obj",
    ]
    for item in tqdm.tqdm(dataloader, desc="COMPUTING MOTION NORMALIZATION"):
        batch = move_batch_to_cuda(item, stat_keys)
        target_lhand, target_rhand, target_obj = build_hybrid_motion_targets_for_texthom(
            batch["x_lhand"],
            batch["x_rhand"],
            batch["x_obj"],
            batch["obj_pc"],
            lhand_layer,
            rhand_layer,
            dataset_name,
            hand_sample_indices,
            include_hand_object_dirvec=include_hand_object_dirvec,
            canonicalize_hand_targets=canonicalize_hand_targets,
        )
        left_stats.update(target_lhand, batch["valid_mask_lhand"])
        right_stats.update(target_rhand, batch["valid_mask_rhand"])
        object_stats.update(target_obj, batch["valid_mask_obj"])
    return finalize_motion_normalizer(
        left_stats, right_stats, object_stats, min_std=min_std
    )


def _load_or_compute_motion_normalizer(
    stats_path,
    checkpoint_path,
    compute_fn,
):
    if osp.exists(stats_path):
        state = torch.load(stats_path, map_location="cpu")
        print(f"Load motion normalization statistics from {stats_path}")
        return MotionNormalizer.from_state_dict(state)
    if checkpoint_path and osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("motion_normalization")
        if state is not None:
            print(f"Load motion normalization statistics from {checkpoint_path}")
            normalizer = MotionNormalizer.from_state_dict(state)
            torch.save(normalizer.state_dict(), stats_path)
            return normalizer
    normalizer = compute_fn()
    torch.save(normalizer.state_dict(), stats_path)
    print(f"Saved motion normalization statistics to {stats_path}")
    return normalizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    config = OmegaConf.to_object(config)
    config = edict(config)
    train_seed = seed_everything(
        getattr(config.texthom.exp, "seed", 0),
        deterministic=bool(getattr(config.texthom.exp, "deterministic", False)),
    )
    print(
        f"Using seed={train_seed} "
        f"deterministic={bool(getattr(config.texthom.exp, 'deterministic', False))}"
    )
    print(
        "Null gaze condition: "
        f"{'ON' if bool(getattr(config.texthom.model, 'null_gaze_condition', False)) else 'OFF'}"
    )

    train_gaze_keep_window = str(getattr(config.texthom.exp, "train_gaze_keep_window", "all") or "all").lower()
    train_gaze_keep_frac = float(getattr(config.texthom.exp, "train_gaze_keep_frac", 0.30))
    train_gaze_keep_logged = 0
    if train_gaze_keep_window not in ("all", "first", "last", "pre_contact"):
        raise ValueError(f"Unknown texthom.exp.train_gaze_keep_window={train_gaze_keep_window!r}")
    if train_gaze_keep_window != "all":
        print(f"[train_gaze_keep_window] {train_gaze_keep_window} {train_gaze_keep_frac}", flush=True)
    # (ablation) head-only control: train (and test) with every gaze direction pointed at the object centre, origin kept
    train_gaze_direction_mode = str(getattr(config.texthom.exp, "train_gaze_direction_mode", "recorded") or "recorded").lower()
    assert train_gaze_direction_mode in ("recorded", "to_center"), train_gaze_direction_mode
    if train_gaze_direction_mode != "recorded":
        print(f"[train_gaze_direction_mode] {train_gaze_direction_mode}", flush=True)

    train_data_name = getattr(config.texthom.exp, "train_data_name", None)
    if train_data_name:
        config.dataset.data_name = train_data_name

    wandb = wandb_login(config, config.texthom.exp, relogin=False)
    (
        model_name,
        save_root,
        data_config,
        lambda_simple,
        nepoch,
    ) = get_texthom_paths(config)

    model_folder = make_model_result_folder(save_root)
    log_path = osp.join(model_folder, "log.txt")
    reset = bool(getattr(config, "reset", False))

    # training state init
    cur_loss = 9999
    best_loss = 9999
    best_epoch = 0
    start_epoch = 0
    best_eval_success = float("-inf")
    best_eval_contact = 0.0
    resume_checkpoint = None

    if reset and osp.isdir(model_folder):
        for filename in os.listdir(model_folder):
            path = osp.join(model_folder, filename)
            if osp.isfile(path) or osp.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        print("Reset texthom: cleared existing checkpoints/logs.")

    os.makedirs(model_folder, exist_ok=True)
    git_branch = get_current_git_branch(PROJECT_ROOT)
    if (not osp.exists(log_path)) or osp.getsize(log_path) == 0:
        cmd_line = "python " + " ".join(sys.argv)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "w") as f:
            f.write(f"[{timestamp}] cmd: {cmd_line}\n")
            f.write(f"[{timestamp}] git_branch: {git_branch}\n")
    log_stream = open(log_path, "a", buffering=1)
    sys.stdout = TeeIO(sys.stdout, log_stream)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False).cuda()

    _, dataloader = get_dataloader(
        "Motion" + data_config.name, config, data_config, test=True,
        shuffle=bool(getattr(config, "train_shuffle", False)))
    print(f"[loader] train_shuffle={bool(getattr(config, 'train_shuffle', False))}")

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
        print(
            f"Using GRAB BPS basis from {bps_path} "
            f"with shape {tuple(bps_basis.shape)}"
        )
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
    use_contact_condition = gaze_condition_mode in (
        "contact_map",
        "cov_map",
        "gt_contact_map",
        "bps_contact_map",
        "bps_cov_map",
        "raw_contact_map",
        "raw_cov_map",
    )
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
    use_obj_scale = bool(getattr(config.texthom.model, "use_obj_scale", True))
    use_obj_centroid = bool(
        getattr(config.texthom.model, "use_obj_centroid", False)
    )
    config.texthom.model.use_obj_scale = use_obj_scale
    config.texthom.model.use_obj_centroid = use_obj_centroid
    canonicalize_point_targets = bool(
        getattr(config.texthom.model, "canonicalize_point_targets", True)
    )
    canonicalize_hand_point_targets = canonicalize_point_targets
    canonicalize_object_point_targets = canonicalize_point_targets
    config.texthom.model.canonicalize_hand_point_targets = canonicalize_point_targets
    config.texthom.model.canonicalize_object_point_targets = canonicalize_point_targets
    if use_point_token_output:
        point_coord_dim = int(hand_sample_indices.numel()) * 3
        config.texthom.model.hand_nfeats = point_coord_dim * 2
        config.texthom.model.obj_nfeats = 9 if predict_object_pose else point_coord_dim
    else:
        config.texthom.model.hand_nfeats = 99
        config.texthom.model.obj_nfeats = 9
    if use_point_token_output:
        print(
            "Using point-token TextHOM targets: "
            f"{hand_sample_indices.numel()} points per entity, "
            f"hand feature dim {config.texthom.model.hand_nfeats}, "
            f"object representation {'relative 9D pose' if predict_object_pose else 'point tokens'} "
            f"with dim {config.texthom.model.obj_nfeats}. "
            f"Hand targets are {'canonicalized' if canonicalize_hand_point_targets else 'world-space'}. "
            f"Object targets are {'canonicalized' if canonicalize_object_point_targets else 'world-space'}. "
            f"Object scale condition is {'enabled' if use_obj_scale else 'disabled'}; "
            f"object centroid condition is {'enabled' if use_obj_centroid else 'disabled'}. "
            f"Legacy relative-orientation loss is disabled; hybrid object pose uses "
            f"translation/geodesic rotation losses. Hand points and direction vectors "
            f"are supervised by simple loss only."
        )
    else:
        print(
            "Using MANO-parameter TextHOM targets: "
            f"hand feature dim {config.texthom.model.hand_nfeats}, "
            f"object pose dim {config.texthom.model.obj_nfeats}."
        )
    motion_normalizer = None
    if use_point_token_output and predict_object_pose:
        stats_path = osp.join(model_folder, "motion_normalization_stats.pt")
        configured_resume_path = getattr(config.texthom.exp, "weight_path", None)
        default_resume_path = osp.join(model_folder, "best_model.pth")
        stats_checkpoint_path = (
            configured_resume_path
            if configured_resume_path and osp.exists(configured_resume_path)
            else default_resume_path
        )
        motion_normalizer = _load_or_compute_motion_normalizer(
            stats_path,
            stats_checkpoint_path,
            lambda: _compute_hybrid_motion_normalizer(
                dataloader,
                lhand_layer,
                rhand_layer,
                hand_sample_indices,
                data_config.name,
                include_hand_object_dirvec=include_hand_object_dirvec,
                canonicalize_hand_targets=canonicalize_hand_point_targets,
                min_std=float(
                    getattr(config.texthom.model, "motion_normalization_min_std", 1e-4)
                ),
            ),
        )
        print(
            "Hybrid motion normalization enabled: "
            f"left std [{motion_normalizer.left_std.min():.4g}, {motion_normalizer.left_std.max():.4g}], "
            f"right std [{motion_normalizer.right_std.min():.4g}, {motion_normalizer.right_std.max():.4g}], "
            f"object std [{motion_normalizer.object_std.min():.4g}, {motion_normalizer.object_std.max():.4g}]."
        )
    texthom, diffusion = build_model_and_diffusion(
        config, lhand_layer, rhand_layer)
    relative_gaze_mlp = (
        build_relative_gaze_mlp_for_texthom(config, output_dim=gaze_condition_dim).cuda()
        if use_relative_gaze_mlp
        else None
    )
    if relative_gaze_mlp is not None:
        print(
            "Using object-relative raw gaze MLP condition: "
            f"sequence length={getattr(config.texthom.model, 'relative_gaze_sequence_length', 100)}, "
            f"feature dim={gaze_condition_dim}."
        )
    pointnet = None
    if use_pointnet_object_feature:
        config.pointfeat.global_feat = True
        pointnet = build_pointnetfeat(config, test=False)
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
    if clip_text_encoder is not None:
        print("Text condition: frozen CLIP sentence vector -> embed_text "
              f"(template={getattr(config.texthom.model, 'text_condition_template', 'action')}, "
              f"slot={getattr(config.texthom.model, 'target_condition_slot', 'emb')})")

    resume_path_cfg = config.texthom.exp.weight_path
    best_model_path = osp.join(model_folder, "best_model.pth")
    latest_model_path = osp.join(model_folder, "latest_model.pth")
    resume_path = resume_path_cfg if resume_path_cfg else best_model_path
    if (
        osp.exists(latest_model_path)
        and (not resume_path_cfg or osp.abspath(resume_path_cfg) == osp.abspath(best_model_path))
    ):
        print(f"Found latest training checkpoint: {latest_model_path}")
        resume_path = latest_model_path
    if resume_path_cfg and (not osp.exists(resume_path)) and osp.exists(best_model_path):
        print(
            f"Configured texthom weight_path not found: {resume_path_cfg}. "
            f"Resume from {best_model_path} instead."
        )
        resume_path = best_model_path

    if osp.exists(resume_path):
        print(f"Resume texthom from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cuda")
        resume_checkpoint = ckpt
        try:
            texthom.load_state_dict(ckpt["model"])
            if use_pointnet_object_feature and pointnet is not None:
                pointnet_state = ckpt.get("pointnet_model")
                if pointnet_state is None:
                    raise RuntimeError(
                        "Checkpoint does not contain `pointnet_model` state."
                    )
                pointnet.load_state_dict(pointnet_state)
            if relative_gaze_mlp is not None:
                relative_gaze_state = ckpt.get("relative_gaze_mlp")
                if relative_gaze_state is None:
                    raise RuntimeError(
                        "Checkpoint does not contain `relative_gaze_mlp` state, "
                        "but gaze_condition_mode uses relative_gaze_mlp."
                    )
                relative_gaze_mlp.load_state_dict(relative_gaze_state)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint. This usually means the checkpoint was trained "
                "with a different object feature layout, such as BPS vs PointNet "
                "or old 1-part dimensions. Use a fresh exp.name/weight_path."
            ) from exc
        start_epoch = ckpt.get("epoch", -1) + 1
        best_epoch = ckpt.get("best_eval_epoch", ckpt.get("best_epoch", start_epoch))
        best_loss = ckpt.get("best_loss", ckpt.get("loss", best_loss))
        best_eval_success = ckpt.get("best_eval_success", best_eval_success)
        best_eval_contact = ckpt.get("best_eval_contact", best_eval_contact)
    else:
        print("No pretrained texthom found, training from scratch")

    trainable_params = list(texthom.parameters())
    if pointnet is not None:
        trainable_params += list(pointnet.parameters())
    if relative_gaze_mlp is not None:
        trainable_params += list(relative_gaze_mlp.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.texthom.exp.lr)
    if resume_checkpoint is not None:
        optimizer_state = resume_checkpoint.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("Restored optimizer state from checkpoint")
        else:
            print(
                "Checkpoint has no optimizer state; continuing with a newly "
                "initialized optimizer."
            )

    lambda_object_translation = float(
        getattr(config.texthom.exp, "lambda_obj_trans", 1.0)
    )
    lambda_object_rotation = float(
        getattr(config.texthom.exp, "lambda_obj_rot", 0.1)
    )
    lambda_contact = float(
        getattr(config.texthom.exp, "lambda_contact", 0.0)
    )
    print(
        "Using paper training objective only: "
        f"L = {lambda_simple:.4f} * L_rec + "
        f"{lambda_object_translation:.4f} * L_trans + "
        f"{lambda_object_rotation:.4f} * L_rot."
    )

    nepoch = int(np.ceil(nepoch / 50.0) * 50)

    total_epochs = nepoch
    log_epoch_freq = int(getattr(config.texthom.exp, "log_epoch_freq", 20))
    cuda_keys = [
        "x_lhand",
        "x_rhand",
        "x_obj",
        "obj_pc",
        "normalized_obj_pc",
        "obj_scale",
        "obj_cent",
        "ldist_map",
        "rdist_map",
        "valid_mask_lhand",
        "valid_mask_rhand",
        "valid_mask_obj",
        "gaze",
    ]
    if gaze_condition_mode in ("gaze_map", "raw_gaze_map", "dataset_gaze_map"):
        cuda_keys.append("gaze_map")
    if use_contact_condition:
        cuda_keys.append("cov_map")
    if start_epoch >= total_epochs:
        print(
            f"Checkpoint epoch {start_epoch} already exceeds target {total_epochs}, nothing to train."
        )
        return

    snapshot_dir = osp.join(model_folder, "code_snapshot")
    if reset or not osp.isdir(snapshot_dir):
        save_code_snapshot(PROJECT_ROOT, snapshot_dir)

    def save_checkpoint(epoch, save_path, epoch_loss):
        checkpoint = {
            "model": texthom.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            # Keep `loss` for compatibility with older loading code.
            "loss": best_loss,
            "epoch_loss": epoch_loss,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_eval_epoch": best_epoch,
            "best_eval_success": best_eval_success,
            "best_eval_contact": best_eval_contact,
            "use_contact_map": use_contact_condition,
            "gaze_condition_mode": gaze_condition_mode,
            "null_gaze_condition": bool(
                getattr(config.texthom.model, "null_gaze_condition", False)
            ),
            "gaze_token_fusion": str(
                getattr(config.texthom.model, "gaze_token_fusion", "add")
            ),
            "cross_attn_order": str(
                getattr(config.texthom.model, "cross_attn_order", "hand_object")
            ),
            "stream_mask": str(getattr(config.texthom.model, "stream_mask", "none")),
            "gazeflow_every_layer": bool(getattr(config.texthom.model, "gazeflow_every_layer", False)),
            "cascade_mode": str(getattr(config.texthom.model, "cascade_mode", "none")),
            "gaze_memory_extras": str(getattr(config.texthom.model, "gaze_memory_extras", "none")),
            "use_part_label": bool(getattr(config.texthom.model, "use_part_label", False)),
            "use_text_condition": bool(getattr(config.texthom.model, "use_text_condition", False)),
            "target_condition_slot": str(getattr(config.texthom.model, "target_condition_slot", "emb")),
            "text_condition_template": str(getattr(config.texthom.model, "text_condition_template", "action")),
            "text_max_len": int(getattr(config.texthom.model, "text_max_len", 20) or 20),
            "gaze_both_fusion": str(getattr(config.texthom.model, "gaze_both_fusion", "tokens")),
            "mask_empty_gaze_tokens": bool(getattr(config.texthom.model, "mask_empty_gaze_tokens", False)),
            "gaze_token_target": str(
                getattr(config.texthom.model, "gaze_token_target", "object")
            ),
            "use_obj_scale": bool(
                getattr(config.texthom.model, "use_obj_scale", True)
            ),
            "use_obj_centroid": bool(
                getattr(config.texthom.model, "use_obj_centroid", False)
            ),
            "lambda_reconstruction": float(lambda_simple),
            "lambda_object_translation": lambda_object_translation,
            "lambda_object_rotation": lambda_object_rotation,
            "seed": int(getattr(config.texthom.exp, "seed", 0)),
            "deterministic": bool(getattr(config.texthom.exp, "deterministic", False)),
            "object_bps_feature_mode": object_bps_feature_mode,
            "gaze_alignment_method": str(
                getattr(config.texthom.model, "gaze_alignment_method", "direction")
            ),
            "gaze_alignment_temporal_scope": str(
                getattr(config.texthom.model, "gaze_alignment_temporal_scope", "all")
            ),
            "gaze_ray_distance_sigma": float(
                getattr(config.texthom.model, "gaze_ray_distance_sigma", 0.05)
            ),
            "gaze_ray_distance_sigma_method1": float(
                getattr(config.texthom.model, "gaze_ray_distance_sigma_method1", 0.05)
            ),
            "gaze_ray_distance_sigma_method2": float(
                getattr(config.texthom.model, "gaze_ray_distance_sigma_method2", 0.16)
            ),
            "gaze_ray_transform": str(
                getattr(config.texthom.model, "gaze_ray_transform", "gaussian")
            ),
            "gaze_ray_inv_scale_cm": float(
                getattr(config.texthom.model, "gaze_ray_inv_scale_cm", 4.0)
            ),
            "gaze_ray_center": float(
                getattr(config.texthom.model, "gaze_ray_center", 0.0)
            ),
            "object_feature_type": object_feature_type,
            "output_representation": output_representation,
            "use_point_token_output": use_point_token_output,
            "predict_object_pose": predict_object_pose,
            "object_pose_is_relative": bool(
                use_point_token_output and predict_object_pose
            ),
        }
        if motion_normalizer is not None:
            checkpoint["motion_normalization"] = motion_normalizer.state_dict()
        if pointnet is not None:
            checkpoint["pointnet_model"] = pointnet.state_dict()
        if relative_gaze_mlp is not None:
            checkpoint["relative_gaze_mlp"] = relative_gaze_mlp.state_dict()
        torch.save(checkpoint, save_path)

    with tqdm.tqdm(
        range(start_epoch, total_epochs),
        total=total_epochs,
        initial=start_epoch,
    ) as pbar:
        for epoch in pbar:
            texthom.train()
            if pointnet is not None:
                pointnet.train()
            if relative_gaze_mlp is not None:
                relative_gaze_mlp.train()
            loss_meter = AverageMeter()
            loss_simple_meter = AverageMeter()
            loss_obj_trans_meter = AverageMeter()
            loss_obj_rot_meter = AverageMeter()

            for item in dataloader:
                raw_object_names = item.get("object_name", item.get("obj_name"))
                if raw_object_names is None:
                    object_names = None
                elif isinstance(raw_object_names, (list, tuple)):
                    object_names = [
                        str(name) if name is not None else None
                        for name in raw_object_names
                    ]
                else:
                    object_names = [str(raw_object_names)] * item["x_obj"].shape[0]
                obj_pc_top_idx = None
                batch_cuda = move_batch_to_cuda(item, cuda_keys)
                x_lhand = batch_cuda["x_lhand"]
                x_rhand = batch_cuda["x_rhand"]
                x_obj = batch_cuda["x_obj"]
                obj_pc_org = batch_cuda["obj_pc"]
                normalized_obj_pc = batch_cuda["normalized_obj_pc"]
    
                obj_scale = batch_cuda["obj_scale"]
                obj_cent = batch_cuda["obj_cent"]
                valid_mask_lhand = batch_cuda["valid_mask_lhand"]
                valid_mask_rhand = batch_cuda["valid_mask_rhand"]
                valid_mask_obj = batch_cuda["valid_mask_obj"]
                ldist_map = batch_cuda["ldist_map"]
                rdist_map = batch_cuda["rdist_map"]
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

                bs = x_obj.shape[0]
                if use_pointnet_object_feature:
                    obj_feat = pointnet(normalized_obj_pc)
                else:
                    with torch.no_grad():
                        obj_feat = compute_bps_feature_from_mesh_cache_for_texthom(
                            normalized_obj_pc,
                            bps_basis,
                            object_names=object_names,
                            mesh_cache=mesh_bps_cache,
                            mesh_bps_correspondence_cache=mesh_bps_correspondence_cache,
                            part_label_map=part_label_map,
                            bbox_margin=float(getattr(config.dataset, "mesh_part_bbox_margin", 0.03)),
                            feature_mode=object_bps_feature_mode,
                        )
                gaze_condition_x_obj = select_object_pose_for_gaze_condition(config, x_obj)
                if train_gaze_direction_mode == "to_center":
                    batch_cuda["gaze"] = _gaze_direction_to_object_center(batch_cuda["gaze"], x_obj, obj_cent)
                if relative_gaze_mlp is None:
                    with torch.no_grad():
                        gaze_score = build_gaze_condition_feature_for_texthom(
                            config,
                            batch_cuda["gaze"],
                            batch_cuda.get("gaze_map"),
                            gaze_condition_x_obj,
                            normalized_obj_pc,
                            bps_basis,
                            obj_cent,
                            obj_scale,
                            gaze_condition_valid_mask,
                            object_names=object_names,
                            mesh_cache=mesh_bps_cache,
                            part_label_map=part_label_map,
                            bbox_margin=float(getattr(config.dataset, "mesh_part_bbox_margin", 0.03)),
                            target_dim=gaze_condition_dim,
                            contact_map=batch_cuda.get("cov_map"),
                        )
                        gaze_score = apply_gaze_token_weights(config, apply_null_gaze_condition(config, gaze_score))
                        contact_feat = gaze_score
                else:
                    gaze_score = build_gaze_condition_feature_for_texthom(
                        config,
                        batch_cuda["gaze"],
                        batch_cuda.get("gaze_map"),
                        gaze_condition_x_obj,
                        normalized_obj_pc,
                        bps_basis,
                        obj_cent,
                        obj_scale,
                        gaze_condition_valid_mask,
                        object_names=object_names,
                        mesh_cache=mesh_bps_cache,
                        part_label_map=part_label_map,
                        bbox_margin=float(getattr(config.dataset, "mesh_part_bbox_margin", 0.03)),
                        target_dim=gaze_condition_dim,
                        contact_map=batch_cuda.get("cov_map"),
                        relative_gaze_mlp=relative_gaze_mlp,
                    )
                    gaze_score = apply_gaze_token_weights(config, apply_null_gaze_condition(config, gaze_score))
                    contact_feat = gaze_score
                if train_gaze_keep_window != "all":
                    # same mask and same place as test_texthom (after null-gaze / token weights, before the
                    # model's condition dropout, which still zeroes the whole gaze condition with p=cond_mask_prob)
                    keep = _train_gaze_keep_window_mask(train_gaze_keep_window, train_gaze_keep_frac,
                                                        valid_mask_obj, ldist_map, rdist_map)
                    assert torch.is_tensor(gaze_score) and gaze_score.dim() >= 3 and gaze_score.shape[:2] == keep.shape, \
                        f"train_gaze_keep_window needs per-frame gaze tokens (B, T, ...); got {getattr(gaze_score, 'shape', type(gaze_score))}"
                    gaze_score = gaze_score * keep.view(*keep.shape, *([1] * (gaze_score.dim() - 2))).to(gaze_score.dtype)
                    contact_feat = gaze_score
                    if train_gaze_keep_logged < 3:
                        train_gaze_keep_logged += 1
                        n_valid_frames = float(valid_mask_obj.to(torch.bool).sum().clamp_min(1))
                        nz = (gaze_score.detach().reshape(*gaze_score.shape[:2], -1).abs().sum(-1) > 0) & valid_mask_obj.to(torch.bool)
                        print(f"[train_gaze_keep_window] step {train_gaze_keep_logged}: kept {float(keep.sum()) / n_valid_frames:.3f} "
                              f"of valid frames; frames with nonzero gaze {float(nz.sum()) / n_valid_frames:.3f}; "
                              f"gaze_score {tuple(gaze_score.shape)}", flush=True)

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
                    obj_feat_final["gaze_origin"] = build_gaze_origin_tokens_for_texthom(batch_cuda["gaze"], x_obj)
                if isinstance(obj_feat_final, dict) and bool(getattr(config.texthom.model, "use_part_label", False)):
                    obj_feat_final["part_id"] = item["part_id"].cuda()
                if isinstance(obj_feat_final, dict) and clip_text_encoder is not None:
                    obj_feat_final["text_emb"] = clip_text_encoder(
                        target_condition_texts(config, item["text"], object_names, item["part_id"]))
                if use_point_token_output:
                    if predict_object_pose:
                        target_lhand, target_rhand, target_obj = (
                            build_hybrid_motion_targets_for_texthom(
                                x_lhand,
                                x_rhand,
                                x_obj,
                                obj_pc_org,
                                lhand_layer,
                                rhand_layer,
                                data_config.name,
                                hand_sample_indices,
                                obj_pc_top_idx=obj_pc_top_idx,
                                include_hand_object_dirvec=include_hand_object_dirvec,
                                canonicalize_hand_targets=canonicalize_hand_point_targets,
                            )
                        )
                    else:
                        target_lhand, target_rhand, target_obj = build_point_token_motion_targets_for_texthom(
                            x_lhand,
                            x_rhand,
                            x_obj,
                            obj_pc_org,
                            lhand_layer,
                            rhand_layer,
                            data_config.name,
                            hand_sample_indices,
                            obj_pc_top_idx=obj_pc_top_idx,
                            include_hand_object_dirvec=include_hand_object_dirvec,
                            canonicalize_hand_targets=canonicalize_hand_point_targets,
                            canonicalize_object_targets=canonicalize_object_point_targets,
                        )
                else:
                    target_lhand = x_lhand
                    target_rhand = x_rhand
                    target_obj = x_obj
                if motion_normalizer is not None:
                    target_lhand, target_rhand, target_obj = motion_normalizer.normalize(
                        target_lhand, target_rhand, target_obj
                    )
                pred_lhand, pred_rhand, pred_obj, losses_dict = diffusion(
                    texthom,
                    target_lhand,
                    target_rhand,
                    target_obj,
                    obj_feat_final,
                    get_losses=True,
                    valid_mask_lhand=valid_mask_lhand,
                    valid_mask_rhand=valid_mask_rhand,
                    valid_mask_obj=valid_mask_obj,
                    motion_normalizer=motion_normalizer,
                )
                simple_loss = losses_dict["simple_loss"]
                object_translation_loss = losses_dict["object_translation_loss"]
                object_rotation_loss = losses_dict["object_rotation_loss"]
                contact_loss = losses_dict.get("contact_consistency_loss", 0.0 * simple_loss)

                losses = (
                    lambda_simple * simple_loss
                    + lambda_object_translation * object_translation_loss
                    + lambda_object_rotation * object_rotation_loss
                    + lambda_contact * contact_loss
                )

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_meter.update(losses.item(), bs)
                loss_simple_meter.update(simple_loss.item(), bs)
                loss_obj_trans_meter.update(object_translation_loss.item(), bs)
                loss_obj_rot_meter.update(object_rotation_loss.item(), bs)

            cur_loss = loss_meter.avg
            is_best = cur_loss < best_loss
            if is_best:
                best_loss = cur_loss
                best_epoch = epoch
                save_checkpoint(
                    epoch,
                    osp.join(model_folder, "best_model.pth"),
                    cur_loss,
                )

            # Always keep a fully resumable checkpoint for the most recent epoch.
            save_checkpoint(
                epoch,
                osp.join(model_folder, "latest_model.pth"),
                cur_loss,
            )

            checkpoint_freq = int(getattr(config.texthom.exp, "checkpoint_freq", 500))
            if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                periodic_path = osp.join(model_folder, f"epoch_{epoch + 1:04d}.pth")
                save_checkpoint(epoch, periodic_path, cur_loss)

            log_payload = {
                "loss": cur_loss,
                "simple_loss": loss_simple_meter.avg,
                "object_translation_loss": loss_obj_trans_meter.avg,
                "object_rotation_loss": loss_obj_rot_meter.avg,
            }
            wandb.log(log_payload)
            log_line = (
                f"{model_name} | Train loss: {cur_loss:.4f}"
            )
            pbar.set_description(log_line)
            should_log_epoch = ((epoch + 1) % log_epoch_freq == 0) or (epoch == total_epochs - 1)
            if should_log_epoch:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_line_with_time = (
                    f"[{timestamp}] [{git_branch}] "
                    f"Epoch {epoch + 1}/{total_epochs} | "
                    f"Train loss: {cur_loss:.4f}"
                )
                with open(log_path, "a") as f:
                    f.write(log_line_with_time + "\n")
    wandb.finish()


if __name__ == "__main__":
    main()
