"""Same-backbone non-gaze target conditions (ablation): CLIP sentence vectors for use_text_condition.

The encoder is the frozen CLIP that the original Text2HOI uses (configs/clip, lib/networks/clip.py).
Prompts are fixed per clip, so encodings are cached by string.
"""
import torch

from constants.part_constants import TEXTHOM_PART_VOCAB


def uses_text_condition(config):
    return bool(getattr(config.texthom.model, "use_text_condition", False))


def target_condition_texts(config, texts, object_names, part_ids):
    """action: the dataset sentence the Text2HOI baseline is given at test time (raw `action`);
    part_object: the paper's template "Grab the {part} of the {object}." (no hand side)."""
    template = str(getattr(config.texthom.model, "text_condition_template", "action")).lower()
    if template == "action":
        return [str(t) for t in texts]
    if template == "part_object":
        ids = part_ids.reshape(-1).tolist() if torch.is_tensor(part_ids) else list(part_ids)
        names = [TEXTHOM_PART_VOCAB[int(i) - 1].replace("_", " ") if int(i) > 0 else "part" for i in ids]   # long_edge -> long edge, as in action
        return [f"Grab the {p} of the {o}." for p, o in zip(names, object_names)]
    raise ValueError(f"Unknown text_condition_template={template!r}; expected action or part_object")


class ClipTextEncoder:
    def __init__(self, config, device="cuda"):
        from lib.networks.clip import load_and_freeze_clip
        version = str(getattr(getattr(config, "clip", None), "clip_version", None) or "ViT-B/32")
        self.device = torch.device(device)
        self.model = load_and_freeze_clip(version).to(self.device)
        if next(self.model.parameters()).device.type == "cpu":
            self.model = self.model.float()   # fp16 CLIP weights have no CPU kernels
        self.cache = {}
        self.max_text_len = int(getattr(config.texthom.model, "text_max_len", 20) or 20)

    @torch.no_grad()
    def __call__(self, texts):
        from lib.networks.clip import tokenize
        for t in sorted({t for t in texts if t not in self.cache}):
            # one string per call: the vector does not depend on which batch first saw the prompt
            self.cache[t] = self.model.encode_text(tokenize([t], device=self.device, max_text_len=self.max_text_len)).float()[0]
        return torch.stack([self.cache[t] for t in texts], dim=0)
