from typing import Tuple
import os.path as osp

import numpy as np

from omegaconf import DictConfig


def get_texthom_paths(config: DictConfig) -> Tuple[str, str]:
    """Return the model name and save root defined in the texthom config."""
    texthom_config = config.texthom.exp

    model_name = texthom_config.name
    save_root = texthom_config.save_root
    lambda_simple = texthom_config.lambda_simple
    
    data_config = config.dataset
    # Text2HOI convention: iteration / (data_num / text_num) epochs.  GRAZE sets
    # `epochs` directly (4000 = the value the convention gives for iteration=31050).
    epochs = getattr(texthom_config, "epochs", None)
    if epochs is not None:
        nepoch = float(epochs)
    else:
        nepoch = texthom_config.iteration / (data_config.data_num/data_config.text_num)
    
    return model_name, save_root, data_config, lambda_simple, nepoch

def move_batch_to_cuda(batch, keys):
    """Move the requested tensors in a batch to CUDA."""
    return {key: batch[key].cuda() for key in keys}


def apply_text_guidance_dropout(text, guidance_uncodp):
    """Randomly replace text with empty strings for classifier-free guidance."""
    if guidance_uncodp <= 0:
        return text
    if isinstance(text, str):
        return "" if np.random.rand(1) < guidance_uncodp else text
    return ["" if np.random.rand(1) < guidance_uncodp else t for t in text]


def build_uncond_text(text, batch_size=None):
    """Build unconditional text prompts matching the input batch shape."""
    if isinstance(text, str):
        size = batch_size if batch_size is not None else 1
        return [""] * size
    return [""] * len(text)


def resolve_bps_distance_stats_path(config):
    explicit_path = getattr(config.texthom.exp, "bps_distance_stats_path", None)
    if explicit_path:
        if osp.isabs(explicit_path):
            return explicit_path
        return osp.join(osp.dirname(osp.abspath(osp.dirname(__file__))), "..", explicit_path)

    data_config = config.dataset
    if getattr(data_config, "name", None) == "hot3d":
        base_dir = getattr(data_config, "data_root", None)
    else:
        data_path = getattr(data_config, "data_path", None)
        base_dir = osp.dirname(data_path) if data_path else getattr(data_config, "root", None)

    if base_dir is None:
        return None
    if not osp.isabs(base_dir):
        base_dir = osp.join(osp.dirname(osp.abspath(osp.dirname(__file__))), "..", base_dir)
    return osp.join(base_dir, "bps_distance_stats.npz")
