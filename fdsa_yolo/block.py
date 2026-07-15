"""FDSA-YOLO neck modules used in the JRS manuscript.

This file intentionally contains only SCFR, PFM, and FDSA modules. The
``P4P3_R16_ScaleAttn`` alias preserves compatibility with released checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


def _sf_resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize a source feature to the P3 reference grid with nearest interpolation."""
    return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="nearest")


class HFP_SCR_Gate(nn.Module):
    """Spatial-continuity frequency residual (SCFR) selection at P3."""

    def __init__(self, c1: int, c2: int, reduction: int = 4, alpha: float = 0.1, refine_mode: str = "base"):
        super().__init__()
        hidden = max(c2 // reduction, 8)
        self.proj = Conv(c1, c2, 1, 1)
        self.direction_h = nn.Conv2d(c2, c2, (1, 3), padding=(0, 1), groups=c2, bias=False)
        self.direction_v = nn.Conv2d(c2, c2, (3, 1), padding=(1, 0), groups=c2, bias=False)
        self.spatial_gate = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1, bias=True), nn.Sigmoid())
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine_mode = refine_mode
        self.local_refine = Conv(c2, c2, 3, 1, g=c2, act=False)
        if refine_mode == "sa":
            self.context_refine = Conv(c2, c2, 3, 1, g=c2, d=2, act=False)
            self.route = nn.Sequential(nn.Conv2d(3, 2, 1, bias=True), nn.Softmax(dim=1))
            self.expand = Conv(c2, c2, 1, 1, act=False)
            self.output_refine = Conv(c2, c2, 3, 1)
        elif refine_mode == "base":
            self.expand = Conv(c2, c2, 1, 1, act=False)
            self.output_refine = Conv(c2, c2, 3, 1)
        elif refine_mode != "lite":
            raise ValueError(f"Unsupported HFP_SCR_Gate refine_mode: {refine_mode}")
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        channels = x.shape[1]
        kernel = self.laplacian_kernel.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
        detail = F.conv2d(x, kernel, padding=1, groups=channels)
        magnitude = detail.abs()
        continuity = 0.5 * (self.direction_h(detail).abs() + self.direction_v(detail).abs())
        evidence = torch.cat(
            (magnitude.mean(1, keepdim=True), magnitude.amax(1, keepdim=True), continuity.mean(1, keepdim=True)), dim=1
        )
        gate = self.spatial_gate(evidence) * self.channel_gate(magnitude)
        local = self.local_refine(detail)
        if self.refine_mode == "sa":
            route = self.route(evidence)
            residual = self.expand(route[:, :1] * local + route[:, 1:2] * self.context_refine(detail))
        elif self.refine_mode == "base":
            residual = self.expand(local)
        else:
            residual = local
        enhanced = x + self.alpha.tanh() * gate * residual
        return self.output_refine(enhanced) if self.refine_mode != "lite" else enhanced


class P4P3_FRMLite(nn.Module):
    """Pre-fusion modulation (PFM) for P4-up and P3-skip features."""

    def __init__(self, channels: list[int], c2: int, reduction: int = 8, alpha: float = 0.05):
        super().__init__()
        c_in = sum(channels)
        hidden = max(c2 // reduction, 8)
        self.proj = Conv(c_in, c2, 1, 1, act=False) if c_in != c2 else nn.Identity()
        self.weight = nn.Sequential(Conv(c2, hidden, 1, 1), nn.Conv2d(hidden, c2, 1, bias=True), nn.Tanh())
        self.refine = Conv(c2, c2, 3, 1, g=max(c2 // 32, 1), act=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        nn.init.zeros_(self.weight[1].weight)
        nn.init.zeros_(self.weight[1].bias)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        p4, p3 = xs[0], xs[1]
        p4 = _sf_resize_like(p4, p3)
        base = self.proj(torch.cat((p4, p3), dim=1))
        return base + self.alpha.tanh() * self.weight(base) * self.refine(base)


class _FDSADynamicBase(nn.Module):
    """PFM correction modulated by four-source dynamic scale arbitration."""

    def __init__(self, channels: list[int], c2: int, reduction: int = 8, alpha: float = 0.05):
        super().__init__()
        hidden = max(c2 // reduction, 8)
        self.base = Conv(channels[0] + channels[1], c2, 1, 1, act=False)
        self.weight = nn.Sequential(Conv(c2, hidden, 1, 1), nn.Conv2d(hidden, c2, 1, bias=True), nn.Tanh())
        self.refine = Conv(c2, c2, 3, 1, g=max(c2 // 32, 1), act=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        nn.init.zeros_(self.weight[1].weight)
        nn.init.zeros_(self.weight[1].bias)
        self.scale_proj = nn.ModuleList(Conv(c, c2, 1, 1, act=False) for c in channels)
        self.scale_logits = nn.Sequential(
            nn.Conv2d(c2 * len(channels), hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, len(channels), 1, bias=True),
        )
        self.scale_out = nn.Conv2d(c2, c2, 1, bias=True)
        nn.init.zeros_(self.scale_logits[2].weight)
        nn.init.zeros_(self.scale_logits[2].bias)
        nn.init.zeros_(self.scale_out.weight)
        nn.init.zeros_(self.scale_out.bias)

    def _scale_residual(self, xs: list[torch.Tensor], p3: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        features = [_sf_resize_like(projection(x), p3) for projection, x in zip(self.scale_proj, xs)]
        descriptor = torch.cat([F.adaptive_avg_pool2d(feature, 1) for feature in features], dim=1)
        weights = self.scale_logits(descriptor).softmax(dim=1)
        fused = torch.zeros_like(base)
        for index, feature in enumerate(features):
            fused = fused + weights[:, index : index + 1] * feature
        return self.scale_out(fused - base)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        p4, p3 = xs[0], xs[1]
        p4 = _sf_resize_like(p4, p3)
        base = self.base(torch.cat((p4, p3), dim=1))
        correction = self.weight(base) * self.refine(base)
        scale_residual = self._scale_residual(xs, p3, base)
        scale_attention = 0.75 + 0.5 * scale_residual.sigmoid()
        return base + self.alpha.tanh() * scale_attention * correction


class P4P3_FDSA(_FDSADynamicBase):
    """Public FDSA module name used by the release YAML."""


# Checkpoint compatibility with the class name used during the experiments.
P4P3_R16_ScaleAttn = P4P3_FDSA
