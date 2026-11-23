import numpy as np

import torch
import torch.nn as nn
from constants.part_constants import TEXTHOM_PART_VOCAB

class TextHOM(nn.Module):
    def __init__(self, hand_nfeats=99, obj_nfeats=9, latent_dim=512, 
                ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                activation="gelu", clip_dim=512, obj_dim=1024,
                cond_mask_prob=0.1, use_cond_fc=True, 
                use_obj_scale=True,
                use_obj_centroid=False,
                use_obj_scale_centroid=None,
                use_frame_pos=True, 
                use_gaze_feat = False,
                use_inst_pos=True, 
                obj_global_dim=1024,
                gaze_token_dim=0,
                gaze_token_fusion="add",
                gaze_token_target="object",
                cross_attn_order="hand_object",
                obj_diffusion_lag=0,
                use_lift_gate=True,
                mask_empty_gaze_tokens=False,
                gaze_separate_cue_projection=False,
                cascade_mode="none",
                stream_mask="none",
                gaze_memory_extras="none",
                global_gaze_origin=False,
                use_part_label=False,
                use_text_condition=False,
                target_condition_slot="emb",
                gazeflow_every_layer=False,
                ):
        super().__init__()

        self.cond_mask_prob = cond_mask_prob
        self.use_gaze_feat = use_gaze_feat
        self.nfeats = hand_nfeats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.obj_dim = obj_dim

        self.input_feats_hand = hand_nfeats
        self.input_feats_obj = obj_nfeats

        self.use_cond_fc = use_cond_fc
        # `use_obj_scale_centroid` is retained only for loading legacy configs.
        # New experiments control scale and centroid independently.
        if use_obj_scale_centroid is not None:
            use_obj_scale = bool(use_obj_scale_centroid)
            use_obj_centroid = bool(use_obj_scale_centroid)
        self.use_obj_scale = bool(use_obj_scale)
        self.use_obj_centroid = bool(use_obj_centroid)
        self.use_frame_pos = use_frame_pos
        self.use_inst_pos = use_inst_pos
        self.obj_global_dim = obj_global_dim
        self.gaze_token_dim = int(gaze_token_dim)
        self.gaze_separate_cue_projection = bool(gaze_separate_cue_projection)
        self.gaze_token_fusion = str(gaze_token_fusion).lower()
        if self.gaze_token_fusion not in (
            "add",
            "additive",
            "film",
            "token",
            "separate",
            "separate_token",
            "cross",
            "cross_attn",
            "cross_attention",
            "append",
            "append_tokens",
            "sequence_append",
            "global_tokens",
        ):
            raise ValueError(
                f"Unknown gaze_token_fusion={gaze_token_fusion!r}; "
                "expected 'add', 'film', 'token', 'cross_attn', or 'append'."
            )
        self.gaze_token_target = str(gaze_token_target).lower()
        self.cross_attn_order = str(cross_attn_order).lower()
        self.use_lift_gate = bool(use_lift_gate)
        # When the Both condition stacks [ray, alignment] tokens, a frame whose
        # ray token is all zeros still projects to the layer bias and competes
        # for attention as a constant distractor. This drops such tokens from
        # the attention memory instead of feeding them as a constant.
        self.mask_empty_gaze_tokens = bool(mask_empty_gaze_tokens)
        if self.gaze_token_target not in ("object", "obj", "all"):
            raise ValueError(
                f"Unknown gaze_token_target={gaze_token_target!r}; expected 'object' or 'all'."
            )
        if self.cross_attn_order not in ("hand_object", "object_hand", "parallel"):
            raise ValueError(
                f"Unknown cross_attn_order={cross_attn_order!r}; expected "
                "'hand_object', 'object_hand', or 'parallel'."
            )
        
        ### Architecture
        self.init_fc_lhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_rhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_obj = InitFC(self.input_feats_obj, self.latent_dim)
        self.init_fc_gaze = (
            InitFC(self.gaze_token_dim, self.latent_dim)
            if self.gaze_token_dim > 0
            else None
        )
        # Two-token Both condition with one projection per cue (ray, alignment) and a
        # learned cue-type embedding (zero-initialised), so the two differently
        # distributed features stop sharing weights while staying separate keys.
        if self.gaze_token_dim > 0 and self.gaze_separate_cue_projection:
            self.init_fc_gaze_cues = nn.ModuleList(
                [InitFC(self.gaze_token_dim, self.latent_dim) for _ in range(2)]
            )
            self.gaze_cue_embed = nn.Parameter(torch.zeros(2, self.latent_dim))
        else:
            self.init_fc_gaze_cues = None
            self.gaze_cue_embed = None
        self.gaze_film = (
            nn.Linear(self.gaze_token_dim, self.latent_dim * 2)
            if self.gaze_token_dim > 0 and self.gaze_token_fusion == "film"
            else None
        )
        if self.gaze_token_dim > 0 and self._uses_cross_attention():
            self.gaze_cross_attn = nn.MultiheadAttention(
                self.latent_dim,
                self.num_heads,
                dropout=self.dropout,
            )
            self.gaze_cross_attn_dropout = nn.Dropout(self.dropout)
            self.gaze_cross_attn_norm = nn.LayerNorm(self.latent_dim)
            self.hand_object_cross_attn = nn.MultiheadAttention(
                self.latent_dim,
                self.num_heads,
                dropout=self.dropout,
            )
            self.hand_object_cross_attn_dropout = nn.Dropout(self.dropout)
            self.hand_object_cross_attn_norm = nn.LayerNorm(self.latent_dim)
        else:
            self.gaze_cross_attn = None
            self.gaze_cross_attn_dropout = None
            self.gaze_cross_attn_norm = None
            self.hand_object_cross_attn = None
            self.hand_object_cross_attn_dropout = None
            self.hand_object_cross_attn_norm = None

        self._flow_ctx = None
        self.gazeflow_every_layer = bool(gazeflow_every_layer)
        if self.gazeflow_every_layer:
            if self.gaze_cross_attn is None or str(cross_attn_order) != "hand_object":
                raise ValueError("gazeflow_every_layer needs gaze cross-attention with cross_attn_order=hand_object")
            def _mha():
                m = nn.MultiheadAttention(self.latent_dim, self.num_heads, dropout=self.dropout)
                nn.init.zeros_(m.out_proj.weight); nn.init.zeros_(m.out_proj.bias)
                return m
            self.layer_gaze_attn = _mha()
            self.layer_gaze_norm = nn.ModuleList([nn.LayerNorm(self.latent_dim) for _ in range(num_layers)])
            self.layer_obj_attn = _mha()
            self.layer_obj_norm = nn.ModuleList([nn.LayerNorm(self.latent_dim) for _ in range(num_layers)])
            self.layer_flow_dropout = nn.Dropout(self.dropout)

        self.lift_gate_mlp = (
            nn.Sequential(
                nn.Linear(self.latent_dim * 3, self.latent_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, 1),
            )
            if self.use_lift_gate
            else None
        )

        if not self.use_frame_pos:
            self.sequence_pos_encoder = OrgPositionalEncoding(self.latent_dim, self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        self.sequence_hand_encoder = HandObjectEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=self.num_layers
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.obj_diffusion_lag = int(obj_diffusion_lag)
        self.embed_timestep_lag = None
        if self.obj_diffusion_lag != 0:
            self.embed_timestep_lag = TimestepEmbedder(
                self.latent_dim, self.sequence_pos_encoder)
            nn.init.zeros_(self.embed_timestep_lag.time_embed[-1].weight)
            nn.init.zeros_(self.embed_timestep_lag.time_embed[-1].bias)

        self.cascade_mode = str(cascade_mode).lower()
        if self.cascade_mode not in ("none", "hand_first", "hand_first_strict", "object_first"):
            raise ValueError(f"Unknown cascade_mode={cascade_mode!r}; expected none|hand_first|hand_first_strict|object_first")
        self.cascade_pass_embed = (
            nn.Parameter(torch.zeros(1, 1, self.latent_dim)) if self.cascade_mode != "none" else None)

        self.stream_mask = str(stream_mask).lower()
        if self.stream_mask not in ("none", "hand_blind", "object_blind", "hand_blind_last2"):
            raise ValueError(f"Unknown stream_mask={stream_mask!r}")

        extras = str(gaze_memory_extras).lower()
        self.gaze_memory_extras = set(k for k in ("shape", "origin") if k in extras)
        assert extras in ("", "none") or self.gaze_memory_extras, f"Unknown gaze_memory_extras={gaze_memory_extras!r}"
        self.memory_type_embed = (nn.Parameter(torch.zeros(2, 1, self.latent_dim))
                                  if self.gaze_memory_extras else None)
        self.embed_gaze_origin = nn.Linear(6, self.latent_dim) if "origin" in self.gaze_memory_extras else None
        self.memory_extra_bias = (nn.Parameter(torch.full((2,), -8.0))
                                  if self.gaze_memory_extras else None)

        self.global_gaze_origin = bool(global_gaze_origin)
        self.embed_global_origin = (nn.Linear(6, self.latent_dim)
                                    if self.global_gaze_origin else None)

        self.use_part_label = bool(use_part_label)
        self.embed_part_label = (nn.Embedding(len(TEXTHOM_PART_VOCAB) + 1, self.latent_dim)
                                 if self.use_part_label else None)
        if self.embed_part_label is not None:
            nn.init.zeros_(self.embed_part_label.weight)
        self.use_text_condition = bool(use_text_condition)
        self.target_condition_slot = str(target_condition_slot).lower()
        if self.target_condition_slot not in ("emb", "memory"):
            raise ValueError(f"Unknown target_condition_slot={target_condition_slot!r}; expected emb or memory")
        if self.target_condition_slot == "memory" and not (self._uses_cross_attention() and self.gaze_token_dim > 0):
            raise ValueError("target_condition_slot=memory requires gaze_token_fusion=cross_attn (gaze_token_dim>0)")
        if self.gaze_memory_extras and not self._uses_cross_attention():
            raise ValueError("gaze_memory_extras requires gaze_token_fusion=cross_attn")
        if self.stream_mask != "none":
            if self._uses_separate_gaze_token() or self._uses_appended_gaze_tokens():
                raise ValueError("stream_mask supports only the 3-token (L,R,O) sequence (not gaze token fusion)")
            if self.stream_mask.startswith("hand_blind") and self.cross_attn_order == "object_hand":
                raise ValueError("hand_blind with cross_attn_order=object_hand leaks object state into the hands")
            if self.stream_mask == "object_blind" and self.cross_attn_order == "hand_object":
                raise ValueError("object_blind with cross_attn_order=hand_object leaks hand state into the object")

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        obj_metadata_dim = int(self.use_obj_scale) + 3 * int(self.use_obj_centroid)
        self.embed_obj = nn.Linear(
            self.obj_dim + obj_metadata_dim,
            self.latent_dim,
        )

        if use_cond_fc:
            self.out_fc_lhand = CondOutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = CondOutFC(self.input_feats_hand, self.latent_dim)
        else:
            self.out_fc_lhand = OutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = OutFC(self.input_feats_hand, self.latent_dim)
        self.out_fc_obj = OutFC(self.input_feats_obj, self.latent_dim)

    def _mem_bias(self, bias, query):
        """Expand a (1,S) key bias to the (L,S) 2-D attn_mask MultiheadAttention expects."""
        return None if bias is None else bias.expand(query.shape[0], -1)
    def _draw_cond_mask(self, bs, device):
        """Draw the condition-dropout mask once per forward so gaze features and the origin
        token are dropped together."""
        if not (self.training and self.cond_mask_prob > 0.):
            return None
        return torch.bernoulli(torch.ones(bs, device=device) * self.cond_mask_prob)

    def _mask_cond_with(self, cond, mask, force_mask=False):
        if force_mask:
            return torch.zeros_like(cond)
        if mask is None:
            return cond
        return cond * (1. - mask.view(*([cond.shape[0]] + [1] * (cond.dim() - 1))))
    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask_shape = [bs] + [1] * (cond.dim() - 1)
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(*mask_shape)
            return cond * (1. - mask)
        else:
            return cond

    def _split_obj_condition(self, obj_feat):
        if isinstance(obj_feat, dict):
            return obj_feat.get("global"), obj_feat.get("gaze"), obj_feat.get("gaze_origin")
        if isinstance(obj_feat, (tuple, list)):
            if len(obj_feat) == 2:
                return obj_feat[0], obj_feat[1], None
            raise ValueError(f"Expected obj_feat tuple/list of length 2, got {len(obj_feat)}")
        return obj_feat, None, None

    def _uses_separate_gaze_token(self):
        return self.gaze_token_fusion in ("token", "separate", "separate_token")

    def _uses_cross_attention(self):
        return self.gaze_token_fusion in ("cross", "cross_attn", "cross_attention")

    def _uses_appended_gaze_tokens(self):
        return self.gaze_token_fusion in (
            "append",
            "append_tokens",
            "sequence_append",
            "global_tokens",
        )

    def _apply_gaze_condition(self, token, gaze_feat):
        if self.gaze_token_fusion in ("add", "additive"):
            return token + self.init_fc_gaze(gaze_feat)
        scale, shift = self.gaze_film(gaze_feat).chunk(2, dim=-1)
        scale = scale.permute(1, 0, 2)
        shift = shift.permute(1, 0, 2)
        return token * (1.0 + scale) + shift

    def _embed_gaze_memory(self, gaze_feat):
        if gaze_feat.dim() == 3:
            return self.init_fc_gaze(gaze_feat), 1
        if gaze_feat.dim() == 4:
            batch_size, frame_count, token_count, feat_dim = gaze_feat.shape
            if self.init_fc_gaze_cues is not None and token_count == 2:
                cues = [
                    self.init_fc_gaze_cues[i](gaze_feat[:, :, i, :]) + self.gaze_cue_embed[i]
                    for i in range(2)
                ]
                # InitFC returns sequence-first (T, B, D) per cue; interleave cues frame-major
                # so row (t*2 + i) is frame t / cue i, exactly like the shared-projection reshape.
                stacked = torch.stack(cues, dim=1)  # (T, 2, B, D)
                return stacked.reshape(frame_count * token_count, batch_size, self.latent_dim), token_count
            gaze_feat = gaze_feat.reshape(batch_size, frame_count * token_count, feat_dim)
            return self.init_fc_gaze(gaze_feat), token_count
        raise ValueError(
            f"Expected gaze_feat shape (B,T,D) or (B,T,M,D), got {tuple(gaze_feat.shape)}."
        )

    def _directional_mask(self, seq_len, tokens_per_frame, device, blind):
        """(S,S) bool mask, True = blocked (query, key) pair. Sequence = [emb] + T x [L,R,O](+gaze)."""
        idx = torch.arange(seq_len, device=device)
        slot = (idx - 1) % tokens_per_frame
        is_emb = idx == 0
        is_hand = (~is_emb) & (slot < 2)
        is_obj = (~is_emb) & (slot == 2)
        if blind == "hand":
            q, k = (is_hand | is_emb), is_obj
        else:
            q, k = (is_obj | is_emb), is_hand
        return q[:, None] & k[None, :]

    def _directional_encode(self, xseq, key_padding_mask, tokens_per_frame):
        """Run nn.TransformerEncoder layer by layer with the directional mask."""
        S = xseq.shape[0]
        blind = "object" if self.stream_mask == "object_blind" else "hand"
        full = self._directional_mask(S, tokens_per_frame, xseq.device, blind)
        n = len(self.seqTransEncoder.layers)
        out = xseq
        for li, layer in enumerate(self.seqTransEncoder.layers):
            lifted = self.stream_mask == "hand_blind_last2" and li >= n - 2
            out = layer(out, src_mask=None if lifted else full, src_key_padding_mask=key_padding_mask)
        if self.seqTransEncoder.norm is not None:
            out = self.seqTransEncoder.norm(out)
        return out

    def forward(
        self, x_lhand, x_rhand,
        x_obj, obj_feat,
        timesteps,
        force_mask=False,
        valid_mask_lhand=None,
        valid_mask_rhand=None,
        valid_mask_obj=None,
        obj_timesteps=None
    ):
        kw = dict(force_mask=force_mask, valid_mask_lhand=valid_mask_lhand,
                  valid_mask_rhand=valid_mask_rhand, valid_mask_obj=valid_mask_obj,
                  obj_timesteps=obj_timesteps)
        if self.cascade_mode == "none":
            return self._forward_once(x_lhand, x_rhand, x_obj, obj_feat, timesteps, **kw)
        l1, r1, o1, g1 = self._forward_once(
            x_lhand, x_rhand, x_obj, obj_feat, timesteps,
            mask_object_tokens=(self.cascade_mode == "hand_first_strict"), **kw)
        if self.cascade_mode in ("hand_first", "hand_first_strict"):
            _, _, o2, g2 = self._forward_once(
                l1.detach(), r1.detach(), x_obj, obj_feat, timesteps, clean_branch="hand", **kw)
            return l1, r1, o2, g2
        l2, r2, _, g2 = self._forward_once(
            x_lhand, x_rhand, o1.detach(), obj_feat, timesteps, clean_branch="object", **kw)
        return l2, r2, o1, g2

    def _forward_once(
        self, x_lhand, x_rhand, 
        x_obj, obj_feat,
        timesteps, 
        force_mask=False,
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        valid_mask_obj=None,
        obj_timesteps=None,
        clean_branch=None,
        mask_object_tokens=False,
    ):
        # obj_feat -> [gaze_map, feature w. pointnet, obj s, obj c]
        bs = timesteps.shape[0] 
        self._flow_ctx = None
        cond_mask_shared = None
        obj_feat_global, gaze_feat, gaze_origin = self._split_obj_condition(obj_feat)
        emb = self.embed_timestep(timesteps)
        shape_vec = self.embed_obj(self.mask_cond(obj_feat_global, force_mask=force_mask))
        emb = emb + shape_vec
        if self.global_gaze_origin:
            if gaze_origin is None:
                raise ValueError("global_gaze_origin=True but obj_feat has no 'gaze_origin'")
            og_ = gaze_origin.to(dtype=emb.dtype)
            if og_.dim() == 3:
                if valid_mask_obj is not None:
                    w = valid_mask_obj.to(device=og_.device, dtype=og_.dtype).unsqueeze(-1)
                    og_ = (og_ * w).sum(1) / w.sum(1).clamp_min(1e-6)
                else:
                    og_ = og_.mean(1)
            if cond_mask_shared is None:
                cond_mask_shared = self._draw_cond_mask(bs, og_.device)
            og_ = self._mask_cond_with(og_, cond_mask_shared, force_mask)
            emb = emb + self.embed_global_origin(og_).reshape(1, bs, self.latent_dim)

        target_vec = None
        if self.use_part_label:
            pid = obj_feat.get("part_id") if isinstance(obj_feat, dict) else None
            if pid is None:
                raise ValueError("use_part_label=True but obj_feat has no 'part_id'")
            pid = torch.as_tensor(pid, device=emb.device).reshape(-1).long()
            if cond_mask_shared is None:
                cond_mask_shared = self._draw_cond_mask(bs, emb.device)
            if force_mask:
                pid = torch.zeros_like(pid)
            elif cond_mask_shared is not None:
                drop = cond_mask_shared.reshape(-1).to(torch.bool)
                pid = torch.where(drop, torch.zeros_like(pid), pid)
            part_vec = self.embed_part_label(pid).reshape(1, bs, self.latent_dim)
            if self.target_condition_slot == "memory":
                target_vec = part_vec
            else:
                emb = emb + part_vec
        if self.use_text_condition:
            txt = obj_feat.get("text_emb") if isinstance(obj_feat, dict) else None
            if txt is None:
                raise ValueError("use_text_condition=True but obj_feat has no 'text_emb'")
            txt = txt.to(device=emb.device, dtype=emb.dtype)
            if cond_mask_shared is None:
                cond_mask_shared = self._draw_cond_mask(bs, emb.device)
            txt = self._mask_cond_with(txt, cond_mask_shared, force_mask)
            text_vec = self.embed_text(txt).reshape(1, bs, self.latent_dim)
            if self.target_condition_slot == "memory":
                target_vec = text_vec if target_vec is None else target_vec + text_vec
            else:
                emb = emb + text_vec
        
        x_init_lhand = self.init_fc_lhand(x_lhand)
        x_init_rhand = self.init_fc_rhand(x_rhand)
        x_init_obj = self.init_fc_obj(x_obj)
        if mask_object_tokens:
            x_init_obj = torch.zeros_like(x_init_obj)
        if clean_branch == "hand":
            x_init_lhand = x_init_lhand + self.cascade_pass_embed
            x_init_rhand = x_init_rhand + self.cascade_pass_embed
        elif clean_branch == "object":
            x_init_obj = x_init_obj + self.cascade_pass_embed
        x_init_gaze = None
        gaze_memory = None
        appended_gaze_tokens = None
        gaze_empty_mask = None

        if gaze_feat is not None:
            if self.init_fc_gaze is None:
                raise ValueError("Received temporal gaze features but gaze_token_dim is 0.")
            if self._uses_appended_gaze_tokens():
                if gaze_feat.dim() != 3 or gaze_feat.shape[0] != x_obj.shape[0]:
                    raise ValueError(
                        "Appended gaze tokens require gaze_feat shape (B,M,D), got "
                        f"{tuple(gaze_feat.shape)}."
                )
                if cond_mask_shared is None:
                    cond_mask_shared = self._draw_cond_mask(bs, gaze_feat.device)
                gaze_feat = self._mask_cond_with(gaze_feat, cond_mask_shared, force_mask)
                appended_gaze_tokens = self.init_fc_gaze(gaze_feat)
            elif gaze_feat.shape[:2] != x_obj.shape[:2]:
                raise ValueError(
                    f"Expected gaze_feat shape (B,T,D) or (B,T,M,D) matching x_obj, got "
                    f"{tuple(gaze_feat.shape)} and x_obj {tuple(x_obj.shape)}."
                )
            else:
                if cond_mask_shared is None:
                    cond_mask_shared = self._draw_cond_mask(bs, gaze_feat.device)
                gaze_feat = self._mask_cond_with(gaze_feat, cond_mask_shared, force_mask)
                if gaze_feat.dim() == 4 and not self._uses_cross_attention():
                    if self._uses_separate_gaze_token():
                        # The non-CGIA token-fusion ablation keeps gaze
                        # information but has one gaze token per frame.
                        # Collapse the per-frame BPS token axis before the
                        # regular (B,T,D) gaze-token projection.
                        gaze_feat = gaze_feat.mean(dim=2)
                    else:
                        raise ValueError(
                            "4D gaze features (B,T,M,D) require either "
                            "gaze_token_fusion='cross_attn' or the separate "
                            "gaze-token fusion path."
                        )
                if self._uses_separate_gaze_token():
                    x_init_gaze = self.init_fc_gaze(gaze_feat)
                elif self._uses_cross_attention():
                    if self.mask_empty_gaze_tokens and gaze_feat.dim() in (3, 4):
                        gaze_empty_mask = (
                            gaze_feat.abs().amax(dim=-1) < 1e-6
                        ).reshape(gaze_feat.shape[0], -1)
                    gaze_memory, gaze_tokens_per_frame = self._embed_gaze_memory(gaze_feat)
                elif self.gaze_token_target == "all":
                    x_init_lhand = self._apply_gaze_condition(x_init_lhand, gaze_feat)
                    x_init_rhand = self._apply_gaze_condition(x_init_rhand, gaze_feat)
                    x_init_obj = self._apply_gaze_condition(x_init_obj, gaze_feat)
                else:
                    x_init_obj = self._apply_gaze_condition(x_init_obj, gaze_feat)

        if target_vec is not None:
            gaze_memory = target_vec.expand(x_obj.shape[1], -1, -1)
            gaze_tokens_per_frame = 1
            gaze_empty_mask = None

        if self.embed_timestep_lag is not None and obj_timesteps is not None:
            x_init_obj = x_init_obj + self.embed_timestep_lag(obj_timesteps)

        if x_init_gaze is None:
            x_init = torch.stack((x_init_lhand, x_init_rhand, x_init_obj), dim=1)
            tokens_per_frame = 3
        else:
            x_init = torch.stack(
                (x_init_lhand, x_init_rhand, x_init_obj, x_init_gaze), dim=1
            )
            tokens_per_frame = 4
        x_init = x_init.reshape(-1, bs, self.latent_dim)

        xseq = torch.cat((emb, x_init), dim=0)
        xseq = self.sequence_pos_encoder(xseq, tokens_per_frame=tokens_per_frame)
        xseq = self.sequence_hand_encoder(xseq, tokens_per_frame=tokens_per_frame)
        if appended_gaze_tokens is not None:
            gaze_token_count = appended_gaze_tokens.shape[0]
            gaze_positions = self.sequence_pos_encoder.pe[
                x_obj.shape[1] + 1 : x_obj.shape[1] + 1 + gaze_token_count
            ]
            gaze_role = self.sequence_hand_encoder.pe[
                len(self.sequence_hand_encoder.pe) * 3 // 4 :
                len(self.sequence_hand_encoder.pe) * 3 // 4 + 1
            ]
            appended_gaze_tokens = appended_gaze_tokens + gaze_positions + gaze_role
            appended_gaze_tokens = self.sequence_pos_encoder.dropout(
                appended_gaze_tokens
            )
            appended_gaze_tokens = self.sequence_hand_encoder.dropout(
                appended_gaze_tokens
            )
            xseq = torch.cat((xseq, appended_gaze_tokens), dim=0)
        if gaze_memory is not None:
            if self.gaze_cross_attn is None:
                raise ValueError(
                    "gaze_token_fusion uses cross-attention but cross-attention "
                    "modules were not initialized."
                )
            gaze_frame_count = x_obj.shape[1]
            if gaze_tokens_per_frame == 1:
                gaze_pos = self.sequence_pos_encoder.pe[1 : gaze_frame_count + 1]
            else:
                gaze_pos = self.sequence_pos_encoder.pe[1 : gaze_frame_count + 1]
                gaze_pos = (
                    gaze_pos.reshape(gaze_frame_count, 1, 1, self.latent_dim)
                    .expand(-1, gaze_tokens_per_frame, -1, -1)
                    .reshape(gaze_frame_count * gaze_tokens_per_frame, 1, self.latent_dim)
                )
            gaze_memory = gaze_memory + gaze_pos
            gaze_key_padding_mask = (
                ~valid_mask_obj.to(device=x_obj.device, dtype=torch.bool)
                if valid_mask_obj is not None
                else None
            )
            if gaze_key_padding_mask is not None and gaze_tokens_per_frame > 1:
                gaze_key_padding_mask = (
                    gaze_key_padding_mask.unsqueeze(-1)
                    .expand(-1, -1, gaze_tokens_per_frame)
                    .reshape(gaze_key_padding_mask.shape[0], -1)
                )
            if gaze_empty_mask is not None:
                if gaze_key_padding_mask is None:
                    gaze_key_padding_mask = gaze_empty_mask
                else:
                    gaze_key_padding_mask = gaze_key_padding_mask | gaze_empty_mask
                # never mask an entire row: attention over zero keys is NaN
                all_masked = gaze_key_padding_mask.all(dim=1, keepdim=True)
                gaze_key_padding_mask = gaze_key_padding_mask & ~all_masked
            gaze_attn_bias = None
            if self.gaze_memory_extras:
                bsz = gaze_memory.shape[1]
                if gaze_key_padding_mask is None:
                    gaze_key_padding_mask = torch.zeros(
                        (bsz, gaze_memory.shape[0]), dtype=torch.bool, device=gaze_memory.device)
                extra_tokens, extra_pads = [], []
                if "shape" in self.gaze_memory_extras:
                    extra_tokens.append(shape_vec.reshape(1, bsz, self.latent_dim) + self.memory_type_embed[0:1])
                    extra_pads.append(torch.zeros((bsz, 1), dtype=torch.bool, device=gaze_memory.device))
                if "origin" in self.gaze_memory_extras:
                    if gaze_origin is None:
                        raise ValueError("gaze_memory_extras includes 'origin' but obj_feat has no 'gaze_origin'")
                    gaze_origin = self._mask_cond_with(gaze_origin.to(dtype=gaze_memory.dtype), cond_mask_shared, force_mask)
                    org = self.embed_gaze_origin(gaze_origin).permute(1, 0, 2)  # (T,B,D)
                    org = org + self.sequence_pos_encoder.pe[1 : org.shape[0] + 1] + self.memory_type_embed[1:2]
                    extra_tokens.append(org)
                    extra_pads.append(~valid_mask_obj.to(device=gaze_memory.device, dtype=torch.bool)
                                      if valid_mask_obj is not None
                                      else torch.zeros((bsz, org.shape[0]), dtype=torch.bool, device=gaze_memory.device))
                gaze_memory = torch.cat([gaze_memory] + extra_tokens, dim=0)
                n_extra = sum(t.shape[0] for t in extra_tokens)
                parts = [torch.zeros(gaze_memory.shape[0] - n_extra, device=gaze_memory.device, dtype=gaze_memory.dtype)]
                if "shape" in self.gaze_memory_extras:
                    parts.append(self.memory_extra_bias[0].expand(1).to(gaze_memory.dtype))
                if "origin" in self.gaze_memory_extras:
                    parts.append(self.memory_extra_bias[1].expand(extra_tokens[-1].shape[0]).to(gaze_memory.dtype))
                gaze_attn_bias = torch.cat(parts).reshape(1, -1)
                gaze_key_padding_mask = torch.cat([gaze_key_padding_mask] + extra_pads, dim=1)
            motion_tokens = xseq[1:]
            left_idx = torch.arange(
                0, motion_tokens.shape[0], tokens_per_frame, device=motion_tokens.device
            )
            right_idx = torch.arange(
                1, motion_tokens.shape[0], tokens_per_frame, device=motion_tokens.device
            )
            hand_idx = torch.stack((left_idx, right_idx), dim=1).reshape(-1)
            object_idx = torch.arange(
                2, motion_tokens.shape[0], tokens_per_frame, device=motion_tokens.device
            )
            motion_tokens = motion_tokens.clone()

            if self.cross_attn_order == "hand_object":
                hand_query = motion_tokens.index_select(0, hand_idx)
                gaze_out, _ = self.gaze_cross_attn(
                    hand_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    attn_mask=self._mem_bias(gaze_attn_bias, hand_query),
                    need_weights=False,
                )
                updated_hands = self.gaze_cross_attn_norm(
                    hand_query + self.gaze_cross_attn_dropout(gaze_out)
                )
                motion_tokens.index_copy_(0, hand_idx, updated_hands)

                object_query = motion_tokens.index_select(0, object_idx)
                hand_padding_mask = None
                if valid_mask_lhand is not None and valid_mask_rhand is not None:
                    hand_padding_mask = ~torch.stack(
                        (
                            valid_mask_lhand.to(dtype=torch.bool),
                            valid_mask_rhand.to(dtype=torch.bool),
                        ),
                        dim=2,
                    ).reshape(bs, -1)
                object_out, _ = self.hand_object_cross_attn(
                    object_query,
                    updated_hands,
                    updated_hands,
                    key_padding_mask=hand_padding_mask,
                    need_weights=False,
                )
                updated_objects = self.hand_object_cross_attn_norm(
                    object_query + self.hand_object_cross_attn_dropout(object_out)
                )
                motion_tokens.index_copy_(0, object_idx, updated_objects)
                if self.gazeflow_every_layer:
                    self._flow_ctx = dict(gaze_memory=gaze_memory, gaze_kpm=gaze_key_padding_mask, gaze_bias=gaze_attn_bias,
                                          hand_idx=hand_idx, object_idx=object_idx, hand_kpm=hand_padding_mask)
            elif self.cross_attn_order == "object_hand":
                object_query = motion_tokens.index_select(0, object_idx)
                gaze_out, _ = self.gaze_cross_attn(
                    object_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    attn_mask=self._mem_bias(gaze_attn_bias, object_query),
                    need_weights=False,
                )
                updated_objects = self.gaze_cross_attn_norm(
                    object_query + self.gaze_cross_attn_dropout(gaze_out)
                )
                motion_tokens.index_copy_(0, object_idx, updated_objects)

                hand_query = motion_tokens.index_select(0, hand_idx)
                object_padding_mask = (
                    ~valid_mask_obj.to(device=x_obj.device, dtype=torch.bool)
                    if valid_mask_obj is not None
                    else None
                )
                hand_out, _ = self.hand_object_cross_attn(
                    hand_query,
                    updated_objects,
                    updated_objects,
                    key_padding_mask=object_padding_mask,
                    need_weights=False,
                )
                updated_hands = self.hand_object_cross_attn_norm(
                    hand_query + self.hand_object_cross_attn_dropout(hand_out)
                )
                motion_tokens.index_copy_(0, hand_idx, updated_hands)
            else:
                # Both branches read the same gaze memory and neither depends on
                # the other branch's update: Gaze -> {Hand, Object}.
                hand_query = motion_tokens.index_select(0, hand_idx)
                object_query = motion_tokens.index_select(0, object_idx)
                hand_out, _ = self.gaze_cross_attn(
                    hand_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    attn_mask=self._mem_bias(gaze_attn_bias, hand_query),
                    need_weights=False,
                )
                object_out, _ = self.hand_object_cross_attn(
                    object_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    attn_mask=self._mem_bias(gaze_attn_bias, object_query),
                    need_weights=False,
                )
                updated_hands = self.gaze_cross_attn_norm(
                    hand_query + self.gaze_cross_attn_dropout(hand_out)
                )
                updated_objects = self.hand_object_cross_attn_norm(
                    object_query + self.hand_object_cross_attn_dropout(object_out)
                )
                motion_tokens.index_copy_(0, hand_idx, updated_hands)
                motion_tokens.index_copy_(0, object_idx, updated_objects)
            xseq = torch.cat((xseq[:1], motion_tokens), dim=0)

        if valid_mask_obj is not None:
            bs = x_obj.shape[0]
            emb_mask = torch.ones((bs, 1), device=x_obj.device, dtype=bool)
            obj_key_mask = (torch.zeros_like(valid_mask_obj)
                            if mask_object_tokens else valid_mask_obj)
            frame_masks = [valid_mask_lhand, valid_mask_rhand, obj_key_mask]
            if tokens_per_frame == 4:
                frame_masks.append(valid_mask_obj)
            aug_mask_no_emb = torch.stack(frame_masks, dim=2)
            aug_mask_no_emb = aug_mask_no_emb.reshape(bs, -1)
            aug_mask = torch.cat([emb_mask, aug_mask_no_emb], dim=1)
            if appended_gaze_tokens is not None:
                gaze_valid = valid_mask_obj.to(dtype=torch.bool).any(
                    dim=1, keepdim=True
                )
                gaze_valid = gaze_valid.expand(-1, appended_gaze_tokens.shape[0])
                aug_mask = torch.cat((aug_mask, gaze_valid), dim=1)
            if self.gazeflow_every_layer and self._flow_ctx is not None:
                x_enc = self._encode_with_layer_flow(xseq, ~aug_mask)[1:]
            elif self.stream_mask == "none":
                x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]
            else:
                x_enc = self._directional_encode(xseq, ~aug_mask, tokens_per_frame)[1:]
        else:
            if self.gazeflow_every_layer and self._flow_ctx is not None:
                x_enc = self._encode_with_layer_flow(xseq, None)[1:]
            elif self.stream_mask == "none":
                x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=None)[1:]
            else:
                x_enc = self._directional_encode(xseq, None, tokens_per_frame)[1:]

        motion_token_count = x_obj.shape[1] * tokens_per_frame
        x_enc_motion = x_enc[:motion_token_count]
        x_enc_lhand = x_enc_motion[::tokens_per_frame]
        x_enc_rhand = x_enc_motion[1::tokens_per_frame]
        x_enc_obj = x_enc_motion[2::tokens_per_frame]

        head_obj = torch.zeros_like(x_enc_obj) if self.stream_mask == "hand_blind" else x_enc_obj
        pred_lhand = self.out_fc_lhand(x_enc_lhand, head_obj)
        pred_rhand = self.out_fc_rhand(x_enc_rhand, head_obj)
        pred_obj = self.out_fc_obj(x_enc_obj)

        pred_lift_gate = None
        if self.lift_gate_mlp is not None:
            gate_features = torch.cat(
                (x_enc_lhand, x_enc_rhand, x_enc_obj), dim=-1
            )
            pred_lift_gate = torch.sigmoid(
                self.lift_gate_mlp(gate_features).permute(1, 0, 2)
            )

        return pred_lhand, pred_rhand, pred_obj, pred_lift_gate


def _encode_with_layer_flow_impl(self, xseq, key_padding_mask):
    """(Ablation) unroll the TransformerEncoder and re-apply GazeFlow (hand <- gaze,
    object <- hand) after every layer. xseq: (S,B,D); index 0 is the condition token z,
    followed by the (L,R,O) x F motion tokens."""
    ctx = self._flow_ctx
    for l, layer in enumerate(self.seqTransEncoder.layers):
        xseq = layer(xseq, src_key_padding_mask=key_padding_mask)
        motion = xseq[1:].clone()
        hq = motion.index_select(0, ctx["hand_idx"])
        g, _ = self.layer_gaze_attn(hq, ctx["gaze_memory"], ctx["gaze_memory"], key_padding_mask=ctx["gaze_kpm"],
                                       attn_mask=self._mem_bias(ctx["gaze_bias"], hq), need_weights=False)
        hq = self.layer_gaze_norm[l](hq + self.layer_flow_dropout(g))
        motion.index_copy_(0, ctx["hand_idx"], hq)
        oq = motion.index_select(0, ctx["object_idx"])
        o, _ = self.layer_obj_attn(oq, hq, hq, key_padding_mask=ctx["hand_kpm"], need_weights=False)
        oq = self.layer_obj_norm[l](oq + self.layer_flow_dropout(o))
        motion.index_copy_(0, ctx["object_idx"], oq)
        xseq = torch.cat((xseq[:1], motion), dim=0)
    if self.seqTransEncoder.norm is not None:
        xseq = self.seqTransEncoder.norm(xseq)
    return xseq


TextHOM._encode_with_layer_flow = _encode_with_layer_flow_impl


class InitFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        return x


class OutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x
    

class CondOutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim*2, self.input_feats)
        
    def forward(self, x, cond):
        x = self.fc(torch.cat([x, cond], dim=2))
        x = x.permute(1, 0, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class OrgPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(OrgPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, tokens_per_frame=None):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, tokens_per_frame=None):
        if tokens_per_frame is None:
            tokens_per_frame = 3
        frame_count = (x.shape[0] - 1) // tokens_per_frame
        x[0] = x[0] + self.pe[0]
        frame_pe = self.pe[1 : frame_count + 1]
        for token_offset in range(tokens_per_frame):
            x[1 + token_offset :: tokens_per_frame] = (
                x[1 + token_offset :: tokens_per_frame] + frame_pe
            )
        return self.dropout(x)


class HandObjectEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(HandObjectEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, tokens_per_frame=None):
        if tokens_per_frame is None:
            tokens_per_frame = 3
        if tokens_per_frame == 4:
            role_positions = [
                1,
                len(self.pe) // 4,
                len(self.pe) // 2,
                len(self.pe) * 3 // 4,
            ]
        else:
            role_positions = [
                1,
                len(self.pe) // 3,
                len(self.pe) * 2 // 3,
            ]
        for token_offset, role_position in enumerate(role_positions):
            x[1 + token_offset :: tokens_per_frame] = (
                x[1 + token_offset :: tokens_per_frame]
                + self.pe[role_position : role_position + 1]
            )
        return self.dropout(x)
