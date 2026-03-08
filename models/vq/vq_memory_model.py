"""VQ-Memory Conditioned Local Pose Transformer.

Architecture overview
---------------------
1. Shared FrameEncoder (early-CNN)       → h_t for every frame
2. Two independent projections:
   - proj_vq(h_t)  → f_vq_t   (for VQ / codebook / memory)
   - proj_tf(h_t)  → f_tf_t   (for local transformer / pose)
3. Anchor VQ (stride-sampled only)       → z_q_j, code_id_j
4. Scan summary g (attention pooling of z_q)
5. Local transformer on f_tf window      → ctx
6. FiLM conditioning (g modulates ctx)
7. VQ memory cross-attention (ctx queries z_q)  → c_t
8. Pose head on (ctx + c_t)              → pred_local_T

Design principles:
- VQ is EMA-updated, never backprop through codebook embeddings.
- Main pose loss may shape ``proj_vq`` through the straight-through quantised codes.
- VQ memory and g have complementary roles (dict entries vs global overview).
- proj_vq and proj_tf are independent to avoid entangling their objectives.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from models.temporal.early_cnn import FrameEncoder
from models.temporal.temporal_transformer import TemporalPoseTransformer
from models.pose_heads.pose_head import LocalPoseHead, MultiIntervalHead
from models.vq.vq_tokenizer import VQTokenizerHead
from models.vq.scan_summary import ScanSummaryPool
from models.vq.memory_cross_attn import VQMemoryCrossAttn
from models.vq.film import FiLMConditioner
from models.vq.scan_geom_head import ScanGeomHead
from models.vq.scan_context import ScanContextEncoder


class VQMemoryPoseModel(nn.Module):
    """VQ-Memory Conditioned Local Pose Transformer.

    Parameters
    ----------
    backbone : str
        CNN backbone for FrameEncoder.
    in_channels : int
        Input channels (1 for greyscale US).
    backbone_dim : int
        Raw backbone output dim (1280 for EfficientNet-B0/B1).
        If token_dim != backbone_dim, encoder projects to token_dim.
    token_dim : int
        Dimension of h_t (shared early-CNN output). Also used as
        proj_tf output dim and transformer d_model.
    code_dim : int
        Dimension for VQ codebook entries (proj_vq output).
    codebook_size : int
        Number of VQ codebook entries.
    ema_decay : float
        EMA decay for codebook updates.
    anchor_stride : int
        Sample every anchor_stride-th frame as anchor for VQ.
    n_heads, n_layers, dim_feedforward : int
        Local transformer hyperparams.
    window_size : int
        Sliding-window attention span.
    dropout : float
        Transformer dropout.
    rotation_rep : str
        Rotation parameterisation for pose heads.
    aux_intervals : sequence of int
        Multi-interval auxiliary head intervals.
    pretrained_backbone : bool
        Use ImageNet pretrained backbone.
    memory_size : int
        Transformer-XL memory tokens (0 = disabled).
    pool_type : str
        Scan summary pooling: "attention" or "latent".
    n_latents : int
        Number of latent tokens for latent pooling.
    n_geom_waypoints : int
        Waypoints for geometry head.
    use_film : bool
        Apply FiLM conditioning with g.
    use_global_token : bool
        Inject g as additive global token.
    use_memory_cross_attn : bool
        Use VQ memory cross-attention.
    memory_n_heads : int
        Heads for memory cross-attention.
    scan_context_layers : int
        Number of bidirectional self-attention layers over anchor memories.
    scan_context_heads : int
        Number of attention heads in the scan context encoder.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 1,
        token_dim: int = 256,
        code_dim: int = 256,
        codebook_size: int = 512,
        ema_decay: float = 0.99,
        anchor_stride: int = 8,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        window_size: int = 64,
        dropout: float = 0.1,
        rotation_rep: str = "rot6d",
        aux_intervals: Sequence[int] = (2, 4, 8, 16),
        share_aux_decoder: bool = False,
        pretrained_backbone: bool = False,
        memory_size: int = 0,
        pool_type: str = "attention",
        n_latents: int = 8,
        n_geom_waypoints: int = 8,
        use_film: bool = True,
        use_global_token: bool = False,
        use_memory_cross_attn: bool = True,
        memory_n_heads: int = 4,
        scan_context_layers: int = 2,
        scan_context_heads: int = 4,
        encoder_input_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.rotation_rep = rotation_rep
        self.aux_intervals = list(aux_intervals)
        self.memory_size = max(0, int(memory_size))
        self.anchor_stride = anchor_stride
        self.use_film = use_film
        self.use_global_token = use_global_token
        self.use_memory_cross_attn = use_memory_cross_attn
        self.use_scan_context = scan_context_layers > 0

        # ── Stage 1: Shared backbone ──────────────────────────────────────
        self.encoder = FrameEncoder(
            backbone=backbone,
            in_channels=in_channels,
            token_dim=token_dim,
            pretrained=pretrained_backbone,
            input_size=encoder_input_size,
        )

        # ── Stage 2: Two independent projection heads ────────────────────
        # proj_vq projects to code_dim (for VQ — quantisable space)
        self.proj_vq = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, code_dim),
            nn.GELU(),
            nn.Linear(code_dim, code_dim),
            nn.LayerNorm(code_dim),
        )

        # proj_tf projects to token_dim (for transformer — geometric detail)
        self.proj_tf = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self._init_proj_weights()

        # ── Stage 3: VQ codebook ─────────────────────────────────────────
        self.vq = VQTokenizerHead(
            code_dim=code_dim,
            codebook_size=codebook_size,
            ema_decay=ema_decay,
        )

        # ── Stage 3.5: Scan-level contextualisation over anchor codes ────
        self.scan_context = ScanContextEncoder(
            d_model=code_dim,
            n_heads=scan_context_heads,
            n_layers=scan_context_layers,
            dim_feedforward=max(code_dim * 4, dim_feedforward),
            dropout=dropout,
        ) if self.use_scan_context else nn.Identity()

        # ── Stage 4: Scan summary pool ───────────────────────────────────
        self.summary_pool = ScanSummaryPool(
            d_in=code_dim,
            d_out=token_dim,  # g lives in transformer dim for FiLM
            pool_type=pool_type,
            n_latents=n_latents,
        )

        # ── Stage 5: Local transformer ───────────────────────────────────
        self.transformer = TemporalPoseTransformer(
            d_model=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            window_size=window_size,
            memory_size=memory_size,
        )

        # ── Stage 6: FiLM conditioning ───────────────────────────────────
        if use_film or use_global_token:
            self.film = FiLMConditioner(
                d_model=token_dim,
                d_cond=token_dim,
                use_film=use_film,
                use_global_token=use_global_token,
            )
        else:
            self.film = None

        # ── Stage 7: VQ memory cross-attention ───────────────────────────
        if use_memory_cross_attn:
            self.memory_cross_attn = VQMemoryCrossAttn(
                d_model=token_dim,
                d_memory=code_dim,
                n_heads=memory_n_heads,
                dropout=dropout,
            )
        else:
            self.memory_cross_attn = None

        # ── Stage 8: Pose heads ──────────────────────────────────────────
        # Input dim doubles because we concatenate ctx + c_t (if memory cross-attn)
        pose_input_dim = token_dim * 2 if use_memory_cross_attn else token_dim
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_input_dim),
            nn.Linear(pose_input_dim, token_dim),
            nn.GELU(),
        ) if use_memory_cross_attn else nn.Identity()

        self.local_head = LocalPoseHead(
            d_model=token_dim,
            d_hidden=token_dim,
            rotation_rep=rotation_rep,
        )
        self.aux_head = MultiIntervalHead(
            intervals=aux_intervals,
            d_model=token_dim,
            d_hidden=token_dim,
            rotation_rep=rotation_rep,
            share_decoder=share_aux_decoder,
        ) if aux_intervals else None

        # ── Stage 9: Scan geometry head (for L_geom) ────────────────────
        self.geom_head = ScanGeomHead(
            d_in=token_dim,
            n_waypoints=n_geom_waypoints,
        )

    def _init_proj_weights(self) -> None:
        """Xavier initialisation for projection heads."""
        for proj in [self.proj_vq, self.proj_tf]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def _extract_anchors(
        self,
        h_all: torch.Tensor,
        anchor_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract anchor frames at anchor_stride from full sequence.

        Parameters
        ----------
        h_all : (B, T, D) full sequence backbone features
        anchor_indices : optional pre-computed anchor indices

        Returns
        -------
        h_anchors : (B, M, D) anchor features
        anchor_idx : (M,) index tensor
        """
        B, T, D = h_all.shape
        if anchor_indices is not None:
            idx = anchor_indices
        else:
            idx = torch.arange(0, T, self.anchor_stride, device=h_all.device)
        h_anchors = h_all[:, idx]  # (B, M, D)
        return h_anchors, idx

    def encode_scan_anchors(
        self,
        scan_frames: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode entire scan and build VQ memory + summary g.

        This is designed to be called ONCE per scan (or cached),
        not per-window.

        Parameters
        ----------
        scan_frames : (B, T_scan, H, W) or (B, T_scan, C, H, W)
            Full scan frames.

        Returns
        -------
        dict with:
            z_q           : (B, M, code_dim)  quantised anchor codes
            z_q_detached  : (B, M, code_dim)  detached for g
            g             : (B, D)            scan summary
            anchor_idx    : (M,)              anchor frame indices
            commit_loss   : scalar
            vq_indices    : (B, M)            codebook indices
        """
        # 1. Encode all frames (chunked for memory)
        h_all = self.encoder.encode_sequence(scan_frames)  # (B, T, token_dim)

        # 2. Extract anchor features
        h_anchors, anchor_idx = self._extract_anchors(h_all)  # (B, M, token_dim)
        denom = max(1, h_all.shape[1] - 1)
        anchor_pos = anchor_idx.to(h_all.dtype) / float(denom)
        anchor_pos = anchor_pos.unsqueeze(0).expand(h_all.shape[0], -1)

        # 3. Project to VQ space
        f_vq = self.proj_vq(h_anchors)  # (B, M, code_dim)

        # 4. Quantise
        vq_out = self.vq(f_vq)  # z_q, indices, commit_loss, z_q_detached

        # 5. Contextualise anchor memory over the whole scan.
        z_ctx = self.contextualize_memory(vq_out["z_q"], mask=memory_mask)

        # 6. Pool contextualised anchor memory into the global summary.
        g = self.summary_pool(z_ctx, mask=memory_mask)  # (B, D)

        return {
            "z_q": vq_out["z_q"],
            "z_ctx": z_ctx,
            "z_q_detached": vq_out["z_q_detached"],
            "g": g,
            "anchor_idx": anchor_idx,
            "anchor_pos": anchor_pos,
            "commit_loss": vq_out["commit_loss"],
            "vq_indices": vq_out["indices"],
            "h_all": h_all,  # cache backbone features for reuse
        }

    def contextualize_memory(
        self,
        z_q: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contextualise anchor VQ codes over the whole scan."""
        if self.use_scan_context:
            return self.scan_context(z_q, mask=mask)
        return z_q

    def forward(
        self,
        frames: torch.Tensor,
        scan_vq_cache: dict[str, torch.Tensor] | None = None,
        scan_frames: torch.Tensor | None = None,
        memory: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> dict[str, torch.Tensor | dict[int, torch.Tensor]]:
        """
        Parameters
        ----------
        frames : (B, T, H, W) or (B, T, C, H, W)
            Window frames for local transformer (e.g. 128-frame window).
        scan_vq_cache : dict or None
            Pre-computed scan-level VQ cache from ``encode_scan_anchors()``.
            If None and scan_frames is given, computes it on the fly.
            If both are None, VQ/g/memory features are disabled (fallback
            to vanilla local transformer).
        scan_frames : (B, T_scan, H, W) or None
            Full scan frames — only used if scan_vq_cache is None.
        memory : (B, M_tf, D) or None
            Transformer-XL memory.
        pos_offset : int
            Positional encoding offset.

        Returns
        -------
        dict with keys:
            pred_local_T   : (B, T, 4, 4)
            pred_aux_T     : dict[Δ → (B, T, 4, 4)]
            tokens         : (B, T, D)
            ctx            : (B, T, D)    final context (after FiLM + memory)
            memory         : (B, M_tf, D) or None
            g              : (B, D)       scan summary (or None)
            pred_geom      : (B, n_wp, 6) geometry prediction (or None)
            commit_loss    : scalar (or 0)
            vq_indices     : (B, M_anchor) or None
            attn_weights   : (B, T, M_anchor) or None
        """
        # ── 1. Encode window frames ──────────────────────────────────────
        h_window = self.encoder.encode_sequence(frames)  # (B, T, token_dim)
        f_tf = self.proj_tf(h_window)                    # (B, T, token_dim)

        # ── 2. Build/retrieve scan VQ cache ──────────────────────────────
        has_vq = scan_vq_cache is not None or scan_frames is not None
        g = None
        vq_memory = None
        memory_mask = None
        memory_pos = None
        commit_loss = torch.tensor(0.0, device=frames.device)
        vq_indices = None
        pred_geom = None
        attn_weights = None

        if scan_vq_cache is not None:
            g = scan_vq_cache["g"]
            vq_memory = scan_vq_cache.get(
                "z_ctx",
                scan_vq_cache.get("z_q", scan_vq_cache["z_q_detached"]),
            )
            memory_mask = scan_vq_cache.get("memory_mask")
            memory_pos = scan_vq_cache.get("anchor_pos")
            commit_loss = scan_vq_cache.get("commit_loss", commit_loss)
            vq_indices = scan_vq_cache.get("vq_indices")
        elif scan_frames is not None:
            cache = self.encode_scan_anchors(scan_frames)
            g = cache["g"]
            vq_memory = cache.get("z_ctx", cache["z_q"])
            memory_pos = cache.get("anchor_pos")
            commit_loss = cache["commit_loss"]
            vq_indices = cache["vq_indices"]

        # ── 3. Local transformer ─────────────────────────────────────────
        ctx, new_memory = self.transformer(
            f_tf, memory=memory, pos_offset=pos_offset,
        )  # (B, T, D)

        # ── 4. FiLM conditioning with g ──────────────────────────────────
        if self.film is not None and g is not None:
            ctx = self.film(ctx, g)

        # ── 5. VQ memory cross-attention ─────────────────────────────────
        c_t = None
        if self.memory_cross_attn is not None and vq_memory is not None:
            c_t, attn_weights = self.memory_cross_attn(
                local_tokens=ctx,
                memory=vq_memory,
                memory_mask=memory_mask,
                memory_pos=memory_pos,
            )

        # ── 6. Merge ctx + c_t for pose head ────────────────────────────
        if c_t is not None:
            ctx_pose = self.pose_proj(torch.cat([ctx, c_t], dim=-1))
        else:
            ctx_pose = ctx

        # ── 7. Pose heads ────────────────────────────────────────────────
        pred_local_T = self.local_head(ctx_pose)  # (B, T, 4, 4)

        pred_aux_T: dict[int, torch.Tensor] = {}
        if self.aux_head is not None:
            pred_aux_T = self.aux_head(ctx_pose)

        # ── 8. Geometry head (only if g is available) ────────────────────
        if g is not None:
            pred_geom = self.geom_head(g)  # (B, n_wp, 6)

        return {
            "pred_local_T": pred_local_T,
            "pred_aux_T": pred_aux_T,
            "tokens": h_window,
            "ctx": ctx_pose,
            "memory": new_memory,
            "g": g,
            "pred_geom": pred_geom,
            "commit_loss": commit_loss,
            "vq_indices": vq_indices,
            "attn_weights": attn_weights,
        }
