import torch
import torch.nn.functional as F
from torch import nn

from .blocks import OverlapPatchEmbed, ShiftedBlock
from .probability import ProbabilityFusion, ProbabilitySampler, _parse_ops


class HJUNet(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=3,
        img_size=224,
        embed_dims=(128, 160, 256),
        decoder_dims=(160, 128, 32, 16),
        depths=(1, 1),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        decoder=None,
        probabilistic=None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.decoder_dims = tuple(decoder_dims)
        self.decoder_stages = len(self.decoder_dims)

        self.encoder1 = nn.Conv2d(self.in_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList(
            [ShiftedBlock(dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)]
        )
        self.block2 = nn.ModuleList(
            [ShiftedBlock(dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)]
        )
        self.dblock1 = nn.ModuleList(
            [ShiftedBlock(dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)]
        )
        self.dblock2 = nn.ModuleList(
            [ShiftedBlock(dim=embed_dims[0], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )

        self.decoder1 = nn.Conv2d(embed_dims[2], self.decoder_dims[0], 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(self.decoder_dims[0], self.decoder_dims[1], 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(self.decoder_dims[1], self.decoder_dims[2], 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(self.decoder_dims[2], self.decoder_dims[3], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(self.decoder_dims[3], self.decoder_dims[3], 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(self.decoder_dims[0])
        self.dbn2 = nn.BatchNorm2d(self.decoder_dims[1])
        self.dbn3 = nn.BatchNorm2d(self.decoder_dims[2])
        self.dbn4 = nn.BatchNorm2d(self.decoder_dims[3])
        self.dnorm3 = norm_layer(self.decoder_dims[0])
        self.dnorm4 = norm_layer(self.decoder_dims[1])

        self.decoder_type = "conv"
        decoder_cfg = decoder
        if isinstance(decoder_cfg, str):
            self.decoder_type = decoder_cfg.strip().lower()
        elif isinstance(decoder_cfg, dict):
            self.decoder_type = str(decoder_cfg.get("type", "conv")).strip().lower()

        self.tf_decoder = None
        self.tf_head = None
        if self.decoder_type in {"tf_decoder", "tfdecoder", "mmt"}:
            from ..decoder.tf_decoder import MultiScaleMaskedTransformerDecoder

            cfg = decoder_cfg if isinstance(decoder_cfg, dict) else {}
            hidden_dim = int(cfg.get("hidden_dim", 64))
            num_queries = int(cfg.get("num_queries", 16))
            nheads = int(cfg.get("nheads", 8))
            dim_feedforward = int(cfg.get("dim_feedforward", 64))
            dec_layers = int(cfg.get("dec_layers", 4))
            pre_norm = bool(cfg.get("pre_norm", False))
            enforce_input_project = bool(cfg.get("enforce_input_project", False))
            mask_dim = int(cfg.get("mask_dim", self.decoder_dims[-1]))

            self.tf_decoder = MultiScaleMaskedTransformerDecoder(
                in_channels=[self.decoder_dims[0], self.decoder_dims[1], self.decoder_dims[2], self.decoder_dims[3]],
                mask_classification=True,
                num_classes=int(num_classes),
                hidden_dim=hidden_dim,
                num_queries=num_queries,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                dec_layers=dec_layers,
                pre_norm=pre_norm,
                mask_dim=mask_dim,
                enforce_input_project=enforce_input_project,
            )
            self.tf_head = nn.Conv2d(num_queries, int(num_classes), kernel_size=1)
        else:
            self.final = nn.Conv2d(self.decoder_dims[3], num_classes, kernel_size=1)

        self.probabilistic_sampler = None
        self.probabilistic_layers = nn.ModuleDict()
        prob_cfg = probabilistic or {}
        if prob_cfg:
            sample_channels = int(prob_cfg.get("sample_channels", max(self.decoder_dims)))
            self.probabilistic_sampler = ProbabilitySampler(
                embed_dims[2],
                sample_channels,
                prior=prob_cfg.get("prior", "cbl_linear"),
                posterior=prob_cfg.get("posterior", "cbl_linear"),
                use_layernorm=prob_cfg.get("use_layernorm", False),
                norm_layer=norm_layer,
            )
            strategy_ops = _parse_ops(prob_cfg.get("strategy"))
            fusion_ops = _parse_ops(prob_cfg.get("fusion_type"))
            stage_ops = [op for op in strategy_ops + fusion_ops if op in {"add", "mul", "cat"}]
            if not stage_ops:
                stage_ops = ["add"]
            use_norm = prob_cfg.get("use_layernorm", False) or ("norm" in fusion_ops)
            raw_layers = prob_cfg.get("layers", [])
            for raw_idx in raw_layers:
                try:
                    idx = int(raw_idx)
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= self.decoder_stages:
                    stage_dim = self.decoder_dims[self.decoder_stages - idx]
                    self.probabilistic_layers[str(idx)] = ProbabilityFusion(
                        stage_dim,
                        sample_channels,
                        stage_ops,
                        use_norm,
                        norm_layer=norm_layer,
                    )

    def _apply_probability_layer(self, stage_idx, feature, sample):
        if sample is None:
            return feature
        module = self.probabilistic_layers[str(stage_idx)] if str(stage_idx) in self.probabilistic_layers else None
        if module is None:
            return feature
        return module(feature, sample)

    def forward(self, x):
        B = x.shape[0]

        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        prob_sample = self.probabilistic_sampler(out) if self.probabilistic_sampler is not None else None

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t4)
        out = self._apply_probability_layer(4, out, prob_sample)
        stage4 = out

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t3)
        out = self._apply_probability_layer(3, out, prob_sample)
        stage3 = out

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t2)
        out = self._apply_probability_layer(2, out, prob_sample)
        stage2 = out

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t1)
        out = self._apply_probability_layer(1, out, prob_sample)
        stage1 = out

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear"))
        stage0 = out

        if self.decoder_type in {"tf_decoder", "tfdecoder", "mmt"}:
            assert self.tf_decoder is not None and self.tf_head is not None
            tf_out = self.tf_decoder([stage4, stage3, stage2, stage1], stage0)
            return self.tf_head(tf_out["pred_masks"])

        return self.final(stage0)

