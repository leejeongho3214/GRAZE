import os
import os.path as osp

import torch
import torch.nn as nn

from lib.networks.texthom import TextHOM
from lib.networks.diffusion import Diffusion
from lib.networks.pointnet import PointNetfeat

def _print_separator():
    print("-" * 60)

def init_weights_to_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def build_model_and_diffusion(args, lhand_layer, rhand_layer, test=False):

    args_diffusion = args.diffusion
    obj_diffusion_lag = int(getattr(args.texthom.model, "obj_diffusion_lag", 0))
    replace_sampling = str(getattr(args.texthom.exp, "replace_sampling", "none"))
    cfg_scale = float(getattr(args.texthom.exp, "cfg_scale", 1.0))
    sampler = str(getattr(args.texthom.exp, "sampler", "ddpm") or "ddpm")
    sample_steps = int(getattr(args.texthom.exp, "sample_steps", 0) or 0)
    contact_consistency = str(getattr(args.texthom.exp, "contact_consistency", "none"))

    texthom_args = dict(args.texthom.model)
    texthom_args.pop("use_gaze_alignment", None)
    texthom_args.pop("gaze_condition_mode", None)
    texthom_args.pop("null_gaze_condition", None)
    texthom_args.pop("gaze_alignment_method", None)
    texthom_args.pop("gaze_alignment_temporal_scope", None)
    texthom_args.pop("gaze_ray_distance_sigma", None)
    texthom_args.pop("gaze_ray_distance_sigma_method1", None)
    texthom_args.pop("gaze_ray_distance_sigma_method2", None)
    texthom_args.pop("gaze_ray_weight", None)
    texthom_args.pop("gaze_align_weight", None)
    texthom_args.pop("gaze_ray_frame_normalize", None)
    texthom_args.pop("gaze_ray_norm_min_peak", None)
    texthom_args.pop("gaze_object_pose_mode", None)
    texthom_args.pop("gaze_ray_transform", None)
    texthom_args.pop("gaze_ray_inv_scale_cm", None)
    texthom_args.pop("gaze_ray_center", None)
    texthom_args.pop("gaze_both_fusion", None)
    texthom_args.pop("gaze_gate_gamma", None)
    texthom_args.pop("gaze_ray_only_fixes", None)
    texthom_args.pop("gaze_ray_temporal_accumulate", None)
    texthom_args.pop("gaze_align_rescale", None)
    texthom_args.pop("gaze_map_dim", None)
    texthom_args.pop("contact_map_dim", None)
    texthom_args.pop("relative_gaze_feat_dim", None)
    texthom_args.pop("relative_gaze_sequence_length", None)
    texthom_args.pop("relative_gaze_hidden_dim", None)
    texthom_args.pop("relative_gaze_dropout", None)
    texthom_args.pop("output_representation", None)
    texthom_args.pop("use_point_token_output", None)
    texthom_args.pop("point_token_count", None)
    texthom_args.pop("predict_object_pose", None)
    texthom_args.pop("canonicalize_point_targets", None)
    texthom_args.pop("include_hand_object_dirvec", None)
    texthom_args.pop("canonicalize_hand_point_targets", None)
    texthom_args.pop("canonicalize_object_point_targets", None)
    texthom_args.pop("object_feature_type", None)
    texthom_args.pop("object_bps_feature_mode", None)
    texthom_args.pop("pointnet_obj_dim", None)
    texthom_args.pop("motion_normalization_min_std", None)
    texthom_args.pop("text_condition_template", None)
    texthom_args.pop("text_max_len", None)

    checkpoints = None
    if test:
        weight_path = args.texthom.exp.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        checkpoints = torch.load(weight_path)
        checkpoint_fusion = checkpoints.get("gaze_token_fusion")
        if checkpoint_fusion is None and "gaze_film.weight" in checkpoints["model"]:
            checkpoint_fusion = "film"
        if checkpoint_fusion is not None:
            configured_fusion = str(texthom_args.get("gaze_token_fusion", "add")).lower()
            checkpoint_fusion = str(checkpoint_fusion).lower()
            if configured_fusion != checkpoint_fusion:
                print(
                    "Override texthom.model.gaze_token_fusion from "
                    f"{configured_fusion} to {checkpoint_fusion} for checkpoint compatibility."
                )
                texthom_args["gaze_token_fusion"] = checkpoint_fusion
                args.texthom.model.gaze_token_fusion = checkpoint_fusion
        checkpoint_use_scale = checkpoints.get("use_obj_scale")
        checkpoint_use_centroid = checkpoints.get("use_obj_centroid")
        embed_obj_weight = checkpoints["model"].get("embed_obj.weight")
        if embed_obj_weight is not None and (
            checkpoint_use_scale is None or checkpoint_use_centroid is None
        ):
            obj_dim = int(texthom_args.get("obj_dim", 1024))
            metadata_dim = int(embed_obj_weight.shape[1]) - obj_dim
            if metadata_dim not in (0, 1, 3, 4):
                raise ValueError(
                    "Cannot infer object metadata condition from checkpoint: "
                    f"embed_obj input dim={embed_obj_weight.shape[1]}, obj_dim={obj_dim}."
                )
            checkpoint_use_scale = metadata_dim in (1, 4)
            checkpoint_use_centroid = metadata_dim in (3, 4)

        # Do not let the deprecated combined flag override the independent flags.
        texthom_args.pop("use_obj_scale_centroid", None)
        if hasattr(args.texthom.model, "use_obj_scale_centroid"):
            del args.texthom.model["use_obj_scale_centroid"]

        for key, checkpoint_value, default in (
            ("use_obj_scale", checkpoint_use_scale, True),
            ("use_obj_centroid", checkpoint_use_centroid, False),
        ):
            if checkpoint_value is None:
                continue
            checkpoint_value = bool(checkpoint_value)
            configured_value = bool(texthom_args.get(key, default))
            if configured_value != checkpoint_value:
                print(
                    f"Override texthom.model.{key} from {configured_value} "
                    f"to {checkpoint_value} for checkpoint compatibility."
                )
            texthom_args[key] = checkpoint_value
            args.texthom.model[key] = checkpoint_value

        for key in ("stream_mask", "cascade_mode", "gaze_memory_extras", "gaze_both_fusion", "mask_empty_gaze_tokens", "use_part_label", "gazeflow_every_layer",
                    "use_text_condition", "target_condition_slot", "text_condition_template", "text_max_len"):
            if key not in checkpoints:
                continue
            ckpt_v = checkpoints[key]
            cfg_v = texthom_args.get(key, args.texthom.model.get(key, None))
            if cfg_v is not None and str(cfg_v).lower() != str(ckpt_v).lower():
                print(f"Override texthom.model.{key} from {cfg_v} to {ckpt_v} (checkpoint metadata).")
            if key not in ("gaze_both_fusion", "text_condition_template", "text_max_len"):
                texthom_args[key] = ckpt_v
            args.texthom.model[key] = ckpt_v

        for key in ("gaze_ray_transform", "gaze_ray_inv_scale_cm", "gaze_ray_center"):
            if key not in checkpoints:
                continue
            ckpt_v = checkpoints[key]
            cfg_v = args.texthom.model.get(key, None)
            if cfg_v is not None and str(cfg_v).lower() != str(ckpt_v).lower():
                print(f"Override texthom.model.{key} from {cfg_v} to {ckpt_v} (checkpoint metadata).")
            args.texthom.model[key] = ckpt_v

    texthom = TextHOM(**texthom_args)
    diffusion = Diffusion(
        lhand_layer=lhand_layer, 
        rhand_layer=rhand_layer, 
        obj_diffusion_lag=obj_diffusion_lag,
        replace_sampling=replace_sampling,
        cfg_scale=cfg_scale,
        sampler=sampler,
        sample_steps=sample_steps,
        contact_consistency=contact_consistency,
        **args_diffusion
    )
    texthom = texthom.cuda()
    diffusion = diffusion.cuda()
    print("build texthom, diffusion")
    
    if test:
        weight_path = args.texthom.exp.weight_path
        print(f"load texthom, {weight_path}")
        texthom.load_state_dict(checkpoints["model"])
        print(f"epoch : {checkpoints['epoch']}, loss : {checkpoints['loss']:.5f}")
        _print_separator()
        texthom.eval()
        diffusion.eval()
        for p in texthom.parameters():
            p.requires_grad = False
            
    return texthom, diffusion

def build_pointnetfeat(args, test=False):
    args_pointfeat = args.pointfeat
    point_encoder = PointNetfeat(**args_pointfeat)
    point_encoder = point_encoder.cuda()
    print("build point encoder")
    
    if test:
        weight_path = args.contact.weight_path if args.contact.epoch is None else osp.join(args.contact.save_root, args.contact.name, f"best_model.pth")
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load point encoder, {weight_path}")

        checkpoints = torch.load(weight_path)
        state_dict = checkpoints.get("model") or checkpoints.get("pointnet_model")
        assert state_dict is not None, "point encoder checkpoint needs either `model` or `pointnet_model` state dict"
        point_encoder.load_state_dict(state_dict)
        if "epoch" in checkpoints:
            if "best_loss" in checkpoints and "best_epoch" in checkpoints:
                print(
                    f"epoch : {checkpoints['epoch']}, "
                    f"best loss : {checkpoints['best_loss']:.0f} ({checkpoints['best_epoch']})"
                )
            elif "loss" in checkpoints:
                print(f"epoch : {checkpoints['epoch']}, loss : {checkpoints['loss']:.5f}")
        _print_separator()
        point_encoder.eval()
        for p in point_encoder.parameters():
            p.requires_grad = False
            
    return point_encoder
