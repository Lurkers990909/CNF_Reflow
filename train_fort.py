r"""
/*
//                            _ooOoo_
//                           o8888888o
//                           88" . "88
//                           (| -_- |)
//                            O\ = /O
//                        ____/`---'\____
//                      .   ' \\| |// `.
//                       / \\||| : |||// \
//                     / _||||| -:- |||||- \
//                       | | \\\ - /// | |
//                     | \_| ''\---/'' | |
//                      \ .-\__ `-` ___/-. /
//                   ___`. .' /--.--\ `. . __
//                ."" '< `.___\_<|>_/___.' >'"".
//               | | : `- \`.;`\ _ /`;.`/ - ` : | |
//                 \ \ `-. \_ __\ /__ _/ .-` / /
//         ======`-.____`-.___\_____/___.-`____.-'======
//                            `=---='
//
//         .............................................
//                  佛祖保佑             永无BUG
//          佛曰:
//                  写字楼里写字间，写字间里程序员；
//                  程序人员写程序，又拿程序换酒钱。
//                  酒醒只在网上坐，酒醉还来网下眠；
//                  酒醉酒醒日复日，网上网下年复年。
//                  但愿老死电脑间，不愿鞠躬老板前；
//                  奔驰宝马贵者趣，公交自行程序员。
//                  别人笑我忒疯癫，我笑自己命太贱；
//                  不见满街漂亮妹，哪个归得程序员？
"""
"""
train_fort.py
=============
FORT: Forward-Only Regression Training of Normalizing Flows
Paper: arXiv:2506.01158v1 (Rehman et al., 2025)

Modified version: training in the *generative* direction (Algorithm 1, §3.2)
- Sample z ~ N(0,I) in latent_shape.
- Teacher decodes z to target image x_target.
- Student NF (reverse pass) maps z to x_pred and returns log_det.
- Loss: ||x_pred - x_target||² + λr·(log_det)²
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.utils import save_image
from scipy.stats import wasserstein_distance

import lib.layers as flow_layers
import lib.utils as utils
from tabular_benchmarks import TABULAR_DATASETS, build_tabular_loaders
from train_misc import (
    build_model_tabular,
    count_parameters,
    create_regularization_fns,
    set_cnf_options,
    standard_normal_logprob,
)

torch.backends.cudnn.benchmark = True

IMAGE_DATASETS = {"mnist", "svhn", "cifar10"}
PEPTIDE_CANONICAL_DATASETS = {"aldp", "al3", "al4"}
PEPTIDE_DATASET_ALIASES = {"tetra": "al4"}
PEPTIDE_DATASETS = PEPTIDE_CANONICAL_DATASETS | set(PEPTIDE_DATASET_ALIASES)


def canonicalize_dataset_name(data_name: Optional[str]) -> Optional[str]:
    if data_name is None:
        return None
    normalized = str(data_name).lower()
    if normalized in PEPTIDE_DATASETS:
        return PEPTIDE_DATASET_ALIASES.get(normalized, normalized)
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def build_logger(logpath: str) -> logging.Logger:
    logger = logging.getLogger("fort")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    logger.handlers = []
    for h in [logging.FileHandler(logpath), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# EMA  — Appendix C.2: decay = 0.999
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential moving average of model parameters.
    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer.step():
        ema.update(model)
        # for evaluation/checkpointing:
        ema.store(model)     # save current (non-EMA) weights
        ema.apply(model)     # copy EMA weights into model
        # ... run evaluation ...
        ema.restore(model)   # restore non-EMA weights for next training step
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Store EMA weights as plain tensors (no grad, on same device as model)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.clone().detach().float()
            for k, v in model.state_dict().items()
        }
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module) -> None:
        """Back up current model weights before applying EMA."""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}

    def apply(self, model: nn.Module) -> None:
        """Load EMA weights into model for evaluation."""
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()},
            strict=False,
        )

    def restore(self, model: nn.Module) -> None:
        """Restore the backed-up (non-EMA) weights after evaluation."""
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Coupling network (conditioner CNN shared by RealNVP / NSF)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CouplingNet(nn.Module):
    """CNN conditioner: in_ch → out_ch with residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = 64, n_blocks: int = 2):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_blocks):
            layers.append(ResBlock(mid_ch))
        layers.append(nn.Conv2d(mid_ch, out_ch, 3, padding=1))
        self.net = nn.Sequential(*layers)
        # Zero-init output → identity-like start (Glow convention)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Invertible layers
# ─────────────────────────────────────────────────────────────────────────────

class LogitTransform(nn.Module):
    """Logit pre-processing: x ∈ [0,1] → y ∈ ℝ  (RealNVP/Glow standard)."""

    def __init__(self, alpha: float = 1e-6):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        alpha = self.alpha
        if not reverse:
            s = alpha + (1.0 - 2.0 * alpha) * x
            y = torch.log(s) - torch.log(1.0 - s)
            if logpx is None:
                return y
            logdet = (-torch.log(s - s * s) + math.log(1.0 - 2.0 * alpha)
                      ).reshape(x.shape[0], -1).sum(1, keepdim=True)
            return y, logpx + logdet
        else:
            sig = torch.sigmoid(x)
            y = (sig - alpha) / (1.0 - 2.0 * alpha)
            if logpx is None:
                return y
            # FIX: Clamp sig to avoid log(0) but keep gradients finite
            sig_clamped = sig.clamp(min=1e-7, max=1-1e-7)
            logdet = (torch.log(sig_clamped)
                      + torch.log(1.0 - sig_clamped)
                      - math.log(1.0 - 2.0 * alpha)
                      ).reshape(x.shape[0], -1).sum(1, keepdim=True)
            return y, logpx + logdet


class SqueezeLayer(nn.Module):
    """Spatial squeeze: (C,H,W) ↔ (C·f²,H/f,W/f). Log-det = 0."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        f = self.factor
        if not reverse:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H // f, f, W // f, f).permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(B, C * f * f, H // f, W // f)
        else:
            B, C, H, W = x.shape
            x = x.reshape(B, C // (f * f), f, f, H, W).permute(0, 1, 4, 2, 5, 3)
            x = x.reshape(B, C // (f * f), H * f, W * f)
        return (x, logpx) if logpx is not None else x


class ActNorm(nn.Module):
    """Activation normalisation (Kingma & Dhariwal, 2018)."""

    def __init__(self, channels: int):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

    def _initialize(self, x: torch.Tensor):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std  = x.std (dim=[0, 2, 3], keepdim=True).clamp(min=1e-6)
            self.loc.data.copy_(-mean)
            self.log_scale.data.copy_(-torch.log(std))
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        if not self.initialized:
            self._initialize(x)   # initialize on first real data
        H, W = x.shape[2], x.shape[3]
        if not reverse:
            y = (x + self.loc) * torch.exp(self.log_scale)
            if logpx is None:
                return y
            ld = (self.log_scale.sum() * H * W).expand(x.shape[0], 1)
            return y, logpx + ld
        else:
            y = x * torch.exp(-self.log_scale) - self.loc
            if logpx is None:
                return y
            ld = (self.log_scale.sum() * H * W).expand(x.shape[0], 1)
            return y, logpx - ld


class InvConv1x1(nn.Module):
    """Invertible 1×1 convolution (learned channel permutation)."""

    def __init__(self, channels: int):
        super().__init__()
        W, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        H, W_img = x.shape[2], x.shape[3]
        if not reverse:
            y = F.conv2d(x, self.weight.unsqueeze(2).unsqueeze(3))
            if logpx is None:
                return y
            _, ld = torch.linalg.slogdet(self.weight)
            return y, logpx + (ld * H * W_img).expand(x.shape[0], 1)
        else:
            y = F.conv2d(x, torch.linalg.inv(self.weight).unsqueeze(2).unsqueeze(3))
            if logpx is None:
                return y
            _, ld = torch.linalg.slogdet(self.weight)
            return y, logpx - (ld * H * W_img).expand(x.shape[0], 1)


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def checkerboard_mask(h: int, w: int, parity: int) -> torch.Tensor:
    mask = torch.zeros(h, w)
    mask[parity::2, 0::2] = 1.0
    mask[1 - parity::2, 1::2] = 1.0
    return mask.view(1, 1, h, w)


def channel_mask(channels: int, swap: bool) -> torch.Tensor:
    mask = torch.zeros(1, channels, 1, 1)
    half = channels // 2
    mask[:, half:] = 1.0 if swap else 0.0
    mask[:, :half]  = 0.0 if swap else 1.0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# ① RealNVP — Affine Coupling Layer  (Dinh et al., 2016)
# ─────────────────────────────────────────────────────────────────────────────

class AffineCouplingLayer(nn.Module):
    """
    y₂ = x₂ · exp(s(x₁)) + t(x₁)
    mask = 1 → passthrough (conditioner x₁)
    mask = 0 → transformed (x₂)
    log-scale clamped to [-2, 2] via tanh·2 for numerical stability.
    """

    def __init__(self, channels: int, mask: torch.Tensor,
                 mid_ch: int = 64, n_blocks: int = 2):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = CouplingNet(channels, channels * 2, mid_ch, n_blocks)

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        x_id = x * self.mask
        st    = self.net(x_id)
        log_s = torch.tanh(st[:, :x.shape[1]]) * 2.0
        t     = st[:, x.shape[1]:]

        if not reverse:
            y = x_id + (1.0 - self.mask) * (x * log_s.exp() + t)
            if logpx is None:
                return y
            ld = ((1.0 - self.mask) * log_s).reshape(x.shape[0], -1).sum(1, keepdim=True)
            return y, logpx + ld
        else:
            y = x_id + (1.0 - self.mask) * ((x - t) * (-log_s).exp())
            if logpx is None:
                return y
            ld = ((1.0 - self.mask) * log_s).reshape(x.shape[0], -1).sum(1, keepdim=True)
            return y, logpx - ld


# ─────────────────────────────────────────────────────────────────────────────
# ② NSF — Rational-Quadratic Spline Coupling Layer  (Durkan et al., 2019)
# ─────────────────────────────────────────────────────────────────────────────

_MIN_BIN   = 1e-3
_MIN_DERIV = 1e-3


def _rqs_forward(x: torch.Tensor, W: torch.Tensor, H: torch.Tensor,
                 D: torch.Tensor, tail: float) -> Tuple[torch.Tensor, torch.Tensor]:
    K   = W.shape[-1]
    w   = F.softmax(W, -1) * (2 * tail - K * _MIN_BIN) + _MIN_BIN
    h   = F.softmax(H, -1) * (2 * tail - K * _MIN_BIN) + _MIN_BIN
    d   = F.softplus(D) + _MIN_DERIV
    cw  = torch.cat([W.new_zeros(W.shape[0], 1), w.cumsum(-1)], -1) - tail
    ch  = torch.cat([H.new_zeros(H.shape[0], 1), h.cumsum(-1)], -1) - tail
    x_c = x.clamp(-(tail - 1e-6), tail - 1e-6)
    bi  = torch.searchsorted(cw[:, 1:].contiguous(), x_c.unsqueeze(-1)).squeeze(-1).clamp(0, K - 1)
    g   = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    s   = g(h, bi) / g(w, bi)
    dk, dkp1 = g(d[:, :-1], bi), g(d[:, 1:], bi)
    xi  = ((x_c - g(cw[:, :-1], bi)) / g(w, bi)).clamp(0., 1.)
    num   = g(h, bi) * (s * xi.pow(2) + dk * xi * (1 - xi))
    denom = s + (dkp1 + dk - 2 * s) * xi * (1 - xi)
    y   = g(ch[:, :-1], bi) + num / denom
    d_n = s.pow(2) * (dkp1 * xi.pow(2) + 2 * s * xi * (1 - xi) + dk * (1 - xi).pow(2))
    lj  = d_n.clamp(1e-8).log() - 2 * denom.abs().clamp(1e-8).log()
    out = (x < -tail) | (x > tail)
    return torch.where(out, x, y), torch.where(out, torch.zeros_like(lj), lj)


def _rqs_inverse(y: torch.Tensor, W: torch.Tensor, H: torch.Tensor,
                 D: torch.Tensor, tail: float) -> Tuple[torch.Tensor, torch.Tensor]:
    K   = W.shape[-1]
    w   = F.softmax(W, -1) * (2 * tail - K * _MIN_BIN) + _MIN_BIN
    h   = F.softmax(H, -1) * (2 * tail - K * _MIN_BIN) + _MIN_BIN
    d   = F.softplus(D) + _MIN_DERIV
    cw  = torch.cat([W.new_zeros(W.shape[0], 1), w.cumsum(-1)], -1) - tail
    ch  = torch.cat([H.new_zeros(H.shape[0], 1), h.cumsum(-1)], -1) - tail
    y_c = y.clamp(-(tail - 1e-6), tail - 1e-6)
    bi  = torch.searchsorted(ch[:, 1:].contiguous(), y_c.unsqueeze(-1)).squeeze(-1).clamp(0, K - 1)
    g   = lambda t, i: t.gather(-1, i.unsqueeze(-1)).squeeze(-1)
    bw, bh = g(w, bi), g(h, bi)
    s   = bh / bw
    dk, dkp1 = g(d[:, :-1], bi), g(d[:, 1:], bi)
    dy  = y_c - g(ch[:, :-1], bi)
    a   = bh * (s - dk)     + dy * (dkp1 + dk - 2 * s)
    b_  = bh * dk           - dy * (dkp1 + dk - 2 * s)
    c   = -s * dy
    xi  = (2 * c / (-b_ - (b_.pow(2) - 4 * a * c).clamp(0).sqrt()).clamp(max=-1e-8)).clamp(0., 1.)
    x_o = xi * bw + g(cw[:, :-1], bi)
    d_n = s.pow(2) * (dkp1 * xi.pow(2) + 2 * s * xi * (1 - xi) + dk * (1 - xi).pow(2))
    denom = s + (dkp1 + dk - 2 * s) * xi * (1 - xi)
    lj  = d_n.clamp(1e-8).log() - 2 * denom.abs().clamp(1e-8).log()
    out = (y < -tail) | (y > tail)
    return torch.where(out, y, x_o), torch.where(out, torch.zeros_like(lj), lj)


class RQSplineCouplingLayer(nn.Module):
    """NSF coupling layer: mask=1 → conditioner; mask=0 → spline transform."""

    def __init__(self, channels: int, mask: torch.Tensor,
                 K: int = 8, tail: float = 5.0, mid_ch: int = 64, n_blocks: int = 2):
        super().__init__()
        self.register_buffer("mask", mask)
        self.K, self.tail = K, tail
        self.net = CouplingNet(channels, channels * (3 * K + 1), mid_ch, n_blocks)

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        B, C, Hi, Wi = x.shape
        K = self.K
        params = self.net(x * self.mask).reshape(B, C, 3 * K + 1, Hi, Wi)
        prep   = lambda t: t.permute(0, 1, 3, 4, 2).reshape(-1, t.shape[2])
        Wf, Hf, Df = prep(params[:, :, :K]), prep(params[:, :, K:2*K]), prep(params[:, :, 2*K:])
        x_tf   = ((1.0 - self.mask) * x).reshape(-1)
        fn     = _rqs_forward if not reverse else _rqs_inverse
        y_f, lj_f = fn(x_tf, Wf, Hf, Df, self.tail)
        y  = x * self.mask + (1.0 - self.mask) * y_f.reshape(B, C, Hi, Wi)
        if logpx is None:
            return y
        ld = ((1.0 - self.mask) * lj_f.reshape(B, C, Hi, Wi)).reshape(B, -1).sum(1, keepdim=True)
        return (y, logpx + ld) if not reverse else (y, logpx - ld)


# ─────────────────────────────────────────────────────────────────────────────
# ③ Glow step: ActNorm + InvConv1x1 + AffineCoupling
# ─────────────────────────────────────────────────────────────────────────────

class GlowStep(nn.Module):
    def __init__(self, channels: int, mid_ch: int = 64, n_blocks: int = 2):
        super().__init__()
        self.actnorm  = ActNorm(channels)
        self.invconv  = InvConv1x1(channels)
        self.coupling = AffineCouplingLayer(
            channels, channel_mask(channels, swap=False), mid_ch, n_blocks)

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        if logpx is None:
            if not reverse:
                return self.coupling(self.invconv(self.actnorm(x)))
            else:
                return self.actnorm(self.invconv(self.coupling(x, reverse=True), reverse=True), reverse=True)
        if not reverse:
            x, logpx = self.actnorm(x, logpx)
            x, logpx = self.invconv(x, logpx)
            x, logpx = self.coupling(x, logpx)
        else:
            x, logpx = self.coupling(x, logpx, reverse=True)
            x, logpx = self.invconv(x, logpx, reverse=True)
            x, logpx = self.actnorm(x, logpx, reverse=True)
        return x, logpx


# ─────────────────────────────────────────────────────────────────────────────
# Sequential flow container
# ─────────────────────────────────────────────────────────────────────────────

class SequentialFlow(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.chain = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor,
                logpx: Optional[torch.Tensor] = None,
                reverse: bool = False) -> torch.Tensor:
        inds = range(len(self.chain) - 1, -1, -1) if reverse else range(len(self.chain))
        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        for i in inds:
            x, logpx = self.chain[i](x, logpx, reverse=reverse)
        return x, logpx


# ─────────────────────────────────────────────────────────────────────────────
# Flow builders
# ─────────────────────────────────────────────────────────────────────────────

def build_realnvp(data_shape: Tuple[int, ...], args: argparse.Namespace) -> SequentialFlow:
    """
    RealNVP for images:
      LogitTransform → 3× checkerboard-AffineCoupling (C,H,W)
      → Squeeze(2) → 3× channel-AffineCoupling (4C, H/2, W/2)
    """
    C, H, W = data_shape
    mc, nb  = args.mid_channels, args.coupling_blocks
    layers: List[nn.Module] = []
    if args.alpha > 0:
        layers.append(LogitTransform(args.alpha))
    for p in (0, 1, 0):
        layers.append(AffineCouplingLayer(C, checkerboard_mask(H, W, p).expand(1, C, H, W), mc, nb))
    layers.append(SqueezeLayer(2))
    C2 = C * 4
    for swap in (False, True, False):
        layers.append(AffineCouplingLayer(C2, channel_mask(C2, swap), mc, nb))
    return SequentialFlow(layers)


def build_nsf(data_shape: Tuple[int, ...], args: argparse.Namespace) -> SequentialFlow:
    """NSF: same topology as RealNVP but with RQ-spline coupling layers."""
    C, H, W = data_shape
    mc, nb  = args.mid_channels, args.coupling_blocks
    K, tail = args.spline_bins, args.spline_tail
    layers: List[nn.Module] = []
    if args.alpha > 0:
        layers.append(LogitTransform(args.alpha))
    for p in (0, 1, 0):
        layers.append(RQSplineCouplingLayer(C, checkerboard_mask(H, W, p).expand(1, C, H, W), K, tail, mc, nb))
    layers.append(SqueezeLayer(2))
    C2 = C * 4
    for swap in (False, True, False):
        layers.append(RQSplineCouplingLayer(C2, channel_mask(C2, swap), K, tail, mc, nb))
    return SequentialFlow(layers)


def build_glow(data_shape: Tuple[int, ...], args: argparse.Namespace) -> SequentialFlow:
    """Glow: L levels of (Squeeze → K × GlowStep)."""
    C, H, W = data_shape
    mc, nb  = args.mid_channels, args.coupling_blocks
    layers: List[nn.Module] = []
    if args.alpha > 0:
        layers.append(LogitTransform(args.alpha))
    for _ in range(args.glow_levels):
        layers.append(SqueezeLayer(2))
        C *= 4
        for _ in range(args.glow_steps):
            layers.append(GlowStep(C, mc, nb))
    return SequentialFlow(layers)


# ─────────────────────────────────────────────────────────────────────────────
# FORT loss (generative direction) — Algorithm 1  §3.2
# ─────────────────────────────────────────────────────────────────────────────
def fort_loss_generation(
    z:       torch.Tensor,   # latent noise  (B, *latent_shape)
    x_target: torch.Tensor,  # target image from teacher (B, *data_shape)
    model:    nn.Module,
    lambda_r: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FORT loss in the generative direction:
        x_pred, log_det = model(z, reverse=True, logpx=zeros)
        loss = MSE(x_pred, x_target) + lambda_r * (log_det)²
    Returns (total, mse.detach(), reg.detach()).

    Numerical note: clamp log_det to a reasonable range before squaring.
    """
    B = z.shape[0]
    x_pred, log_det = model(z, z.new_zeros(B, 1), reverse=True)

    mse = F.mse_loss(x_pred, x_target)
    if lambda_r > 0:
        # FIX: clamp log_det to [-50,50] to prevent extreme squared values
        log_det_clamped = log_det.squeeze().clamp(-50.0, 50.0)
        reg = lambda_r * (log_det_clamped ** 2).mean()
    else:
        reg = z.new_zeros(1).squeeze()

    return mse + reg, mse.detach(), reg.detach()


# ─────────────────────────────────────────────────────────────────────────────
# BPD evaluation (uses EMA weights via caller)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bits_per_dim(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Exact bits/dim via NF log-likelihood."""
    B = x.shape[0]
    z, log_det = model(x, x.new_zeros(B, 1))
    logpz       = standard_normal_logprob(z).reshape(B, -1).sum(1, keepdim=True)
    logpx_dim   = (logpz + log_det).mean() / x[0].numel()
    return -(logpx_dim - np.log(256.0)) / np.log(2.0)


def compute_tabular_nll(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Exact mean negative log-likelihood for vector-valued flows."""
    B = x.shape[0]
    z, delta_logp = model(x, x.new_zeros(B, 1))
    logpz = standard_normal_logprob(z).reshape(B, -1).sum(1, keepdim=True)
    logpx = logpz - delta_logp
    return -logpx.mean()


def is_image_dataset(data_name: str) -> bool:
    return data_name in IMAGE_DATASETS


def _vector_dataset_from_array(array: np.ndarray) -> TensorDataset:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    labels = torch.zeros(tensor.shape[0], dtype=torch.long)
    return TensorDataset(tensor, labels)


def _maybe_limit_dataset(dataset, max_train_samples: Optional[int]):
    if max_train_samples is None or max_train_samples <= 0 or len(dataset) <= max_train_samples:
        return dataset
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=max_train_samples, replace=False)
    return Subset(dataset, indices.tolist())


def _get_peptide_utils():
    import train_rf_pipeline as peptide_utils

    return peptide_utils


def build_tabular_realnvp(input_dim: int, args: argparse.Namespace) -> flow_layers.SequentialFlow:
    hidden_dims = tuple(int(part) for part in args.tabular_hidden_dims.split("-") if part)
    if not hidden_dims:
        raise ValueError("--tabular-hidden-dims must contain at least one hidden size")

    chain: List[nn.Module] = []
    for idx in range(args.tabular_depth):
        if args.tabular_glow:
            chain.append(flow_layers.BruteForceLayer(input_dim))
        chain.append(flow_layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=idx % 2 == 0))
        if args.tabular_batch_norm:
            chain.append(flow_layers.MovingBatchNorm1d(input_dim, bn_lag=args.tabular_bn_lag))
    return flow_layers.SequentialFlow(chain)


def build_student_flow(
    data_shape: Tuple[int, ...],
    data_kind: str,
    args: argparse.Namespace,
) -> nn.Module:
    if data_kind in {"tabular", "peptide"}:
        if args.flow != "realnvp":
            raise ValueError(f"{data_kind.capitalize()} FORT currently supports --flow realnvp only.")
        return build_tabular_realnvp(data_shape[0], args)

    builders = {"realnvp": build_realnvp, "nsf": build_nsf, "glow": build_glow}
    return builders[args.flow](data_shape, args)


def compute_validation_metric(x: torch.Tensor, model: nn.Module, data_kind: str) -> torch.Tensor:
    if data_kind in {"tabular", "peptide"}:
        return compute_tabular_nll(x, model)
    return compute_bits_per_dim(x, model)


def evaluate_loader_metric(loader, model: nn.Module, data_kind: str, cvt) -> float:
    values = []
    with torch.no_grad():
        for x, _ in loader:
            values.append(compute_validation_metric(cvt(x), model, data_kind).item())
    return float(np.mean(values))


def sample_exact_flow_with_logq(
    model: nn.Module,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    samples: List[torch.Tensor] = []
    logq_values: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            current = min(batch_size, n_samples - start)
            z = torch.randn(current, *latent_shape, device=device)
            logpz = standard_normal_logprob(z).reshape(current, -1).sum(1, keepdim=True)
            x, logq = model(z, logpz, reverse=True)
            samples.append(x.detach().cpu())
            logq_values.append(logq.reshape(current).detach().cpu())
    return torch.cat(samples, dim=0), torch.cat(logq_values, dim=0)


def evaluate_peptide_student(
    model: nn.Module,
    bundle,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    metric_samples: int,
    tw2_subsample: int,
    openmm_platform: str,
    artifact_prefix: str,
    logger: logging.Logger,
) -> Dict[str, object]:
    peptide_utils = _get_peptide_utils()
    backend = peptide_utils.OpenMMPeptideMetricBackend(bundle, platform=openmm_platform)
    reference = peptide_utils._reference_observables(bundle, backend)

    batch_size = min(max(1, metric_samples), 2048)
    start_time = time.time()
    proposal_states, proposal_logq = sample_exact_flow_with_logq(
        model,
        latent_shape=latent_shape,
        device=device,
        n_samples=metric_samples,
        batch_size=batch_size,
    )
    sampling_time = time.time() - start_time

    proposal_coords = bundle.denormalize_flat(proposal_states.numpy())
    proposal = peptide_utils._proposal_observables(proposal_coords, backend, cache_path=None)
    log_weights = -proposal["reduced_energies"] - proposal_logq.numpy()
    proposal_weights, _ = peptide_utils.normalize_log_weights(log_weights)

    ess = peptide_utils.compute_kish_ess(proposal_weights)
    e_w1 = wasserstein_distance(
        reference["reduced_energies"],
        proposal["reduced_energies"],
        u_weights=np.full(len(reference["reduced_energies"]), 1.0 / len(reference["reduced_energies"]), dtype=np.float64),
        v_weights=proposal_weights,
    )
    t_w2 = peptide_utils.estimate_torus_w2(
        proposal_torsions=proposal["torsions"],
        reference_torsions=reference["torsions"],
        proposal_weights=proposal_weights,
        subsample_size=tw2_subsample,
    )
    artifact_paths = peptide_utils.visualize_peptide_metrics(
        artifact_prefix=artifact_prefix,
        reference_energies=reference["reduced_energies"],
        proposal_energies=proposal["reduced_energies"],
        proposal_weights=proposal_weights,
        reference_torsions=reference["torsions"],
        proposal_torsions=proposal["torsions"],
    )

    logger.info(
        "Peptide metrics | ESS %.2f (ratio=%.6f) | E-W1 %.4f | T-W2 %.4f | %.3f ms/sample",
        ess,
        ess / float(metric_samples),
        e_w1,
        t_w2,
        sampling_time / max(metric_samples, 1) * 1000.0,
    )
    logger.info("Peptide figures saved: %s", artifact_paths)

    return {
        "ess": float(ess),
        "ess_ratio": float(ess / float(metric_samples)),
        "e_w1": float(e_w1),
        "t_w2": float(t_w2),
        "sampling_time": float(sampling_time),
        "time_per_sample_ms": float(sampling_time / max(metric_samples, 1) * 1000.0),
        "metric_samples": int(metric_samples),
        "artifacts": artifact_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mini-batch OT  (§3.3 — note: paper uses full-batch offline OT for molecules;
# mini-batch OT is used here as an online approximation for image experiments)
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_log(a: torch.Tensor, b: torch.Tensor,
                  C: torch.Tensor, eps: float, n_iter: int) -> torch.Tensor:
    """Log-domain Sinkhorn. Returns transport plan P : (n, m)."""
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    lK = -C / eps
    for _ in range(n_iter):
        f = eps * (a.log() - torch.logsumexp(lK + g.unsqueeze(0), dim=1))
        g = eps * (b.log() - torch.logsumexp(lK.T + f.unsqueeze(0), dim=1))
    return (f.unsqueeze(1) + lK + g.unsqueeze(0)).exp()


def ot_match(x0: torch.Tensor, x1: torch.Tensor,
             eps: float = 0.1, n_iter: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mini-batch OT coupling between prior samples x0 and data samples x1.
    Both are in data space (after optional logit pre-processing).
    Returns (x0_perm, x1): x0 permuted so that x0_perm[i] ↔ x1[i].

    Cost matrix: squared L2 in standardised space (both sides standardised
    independently so their scales are comparable).
    """
    n   = x0.shape[0]
    x0f = x0.reshape(n, -1).detach().float()
    x1f = x1.reshape(n, -1).detach().float()
    x0s = x0f / x0f.std().clamp(1e-6)
    x1s = x1f / x1f.std().clamp(1e-6)
    C   = torch.cdist(x0s, x1s, p=2).pow(2)
    a   = x0f.new_ones(n) / n
    b   = x1f.new_ones(n) / n
    with torch.no_grad():
        P = _sinkhorn_log(a, b, C, eps, n_iter)
    pi  = P.argmax(dim=1)          # x0[i] → x1[pi[i]]
    return x0, x1[pi]              # keep x0 fixed, permute x1


# ─────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ─────────────────────────────────────────────────────────────────────────────

def _dequant(x: torch.Tensor) -> torch.Tensor:
    return (x * 255.0 + torch.rand_like(x)) / 256.0


def get_dataset(args: argparse.Namespace):
    if args.data in TABULAR_DATASETS:
        bundle, train_loader, val_loader, test_loader = build_tabular_loaders(
            args.data,
            train_batch_size=args.batch_size,
            eval_batch_size=args.test_batch_size,
            num_workers=0,
            data_root=args.data_root,
        )
        train_set = _maybe_limit_dataset(train_loader.dataset, args.max_train_samples)
        return train_set, val_loader, test_loader, (bundle.input_dim,), "tabular", None

    if args.data in PEPTIDE_DATASETS:
        peptide_utils = _get_peptide_utils()
        bundle = peptide_utils.load_peptide_bundle(args.data, args.data_root)
        train_set = _vector_dataset_from_array(bundle.train_flat)
        train_set = _maybe_limit_dataset(train_set, args.max_train_samples)
        val_loader = DataLoader(
            _vector_dataset_from_array(bundle.val_flat),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
        test_loader = DataLoader(
            _vector_dataset_from_array(bundle.test_flat),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
        return train_set, val_loader, test_loader, (bundle.train_flat.shape[1],), "peptide", bundle

    noisy = tforms.Lambda(_dequant)
    clean = tforms.Lambda(lambda x: x)

    def T(sz, noise=True):
        return tforms.Compose([tforms.Resize(sz), tforms.ToTensor(), noisy if noise else clean])

    if args.data == "mnist":
        C, H = 1, args.imagesize or 28
        train_set = dset.MNIST (args.data_root, train=True,  transform=T(H),        download=True)
        test_set  = dset.MNIST (args.data_root, train=False, transform=T(H, False), download=True)
    elif args.data == "svhn":
        C, H = 3, args.imagesize or 32
        train_set = dset.SVHN  (args.data_root, split="train", transform=T(H),        download=True)
        test_set  = dset.SVHN  (args.data_root, split="test",  transform=T(H, False), download=True)
    elif args.data == "cifar10":
        C, H = 3, args.imagesize or 32
        train_set = dset.CIFAR10(args.data_root, train=True, download=True,
            transform=tforms.Compose([tforms.Resize(H), tforms.RandomHorizontalFlip(),
                                      tforms.ToTensor(), noisy]))
        test_set  = dset.CIFAR10(args.data_root, train=False, transform=T(H, False), download=True)
    else:
        raise ValueError(f"Unknown dataset: {args.data}")

    data_shape: Tuple[int, ...] = (C, H, H)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True, pin_memory=True)
    return train_set, val_loader, test_loader, data_shape, "image", None


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction test (check invertibility)
# ─────────────────────────────────────────────────────────────────────────────
def reconstruction_test(model: nn.Module, x: torch.Tensor, logger: logging.Logger):
    """Test whether model(x, reverse=False) followed by reverse gives back x."""
    model.eval()
    with torch.no_grad():
        z, _ = model(x, torch.zeros(x.shape[0], 1, device=x.device), reverse=False)
        x_recon, _ = model(z, torch.zeros(x.shape[0], 1, device=x.device), reverse=True)
        mse = F.mse_loss(x_recon, x)
        logger.info(f"Reconstruction test MSE: {mse.item():.6f}")
    return mse.item()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser("FORT: Forward-Only Regression Training of Normalizing Flows")

# Flow
parser.add_argument("--flow", choices=["realnvp", "nsf", "glow"], default="realnvp")
# Data
parser.add_argument("--data", choices=sorted(IMAGE_DATASETS | TABULAR_DATASETS | PEPTIDE_DATASETS), type=str.lower, default="mnist")
parser.add_argument("--data-root",       type=str,   default="./data")
parser.add_argument("--imagesize",        type=int,   default=None)
parser.add_argument("--alpha",            type=float, default=1e-6,
                    help="Logit-transform alpha.")
parser.add_argument("--max_train_samples", type=int, default=None,
                    help="Optional cap on the number of training samples for vector datasets.")
# Architecture
parser.add_argument("--mid_channels",     type=int,   default=64)
parser.add_argument("--coupling_blocks",  type=int,   default=2)
parser.add_argument("--spline_bins",      type=int,   default=8)
parser.add_argument("--spline_tail",      type=float, default=5.0)
parser.add_argument("--glow_levels",      type=int,   default=2)
parser.add_argument("--glow_steps",       type=int,   default=4)
parser.add_argument("--tabular-depth",    type=int,   default=10,
                    help="Number of coupling layers for tabular FORT students.")
parser.add_argument("--tabular-hidden-dims", type=str, default="100-100",
                    help="Hidden layer sizes for tabular coupling nets, e.g. '256-256'.")
parser.add_argument("--tabular-batch-norm", action="store_true",
                    help="Enable MovingBatchNorm1d between tabular coupling layers.")
parser.add_argument("--tabular-bn-lag",   type=float, default=0.0,
                    help="Batch norm lag for tabular MovingBatchNorm1d.")
parser.add_argument("--tabular-glow",     action="store_true",
                    help="Insert BruteForceLayer permutations between tabular coupling layers.")

# FORT
parser.add_argument("--target", choices=["ot", "reflow"], default="reflow",
                    help="Target coupling strategy. (reflow recommended)")
parser.add_argument("--lambda_r",         type=float, default=1e-6,  # FIX: reduced from 1e-5
                    help="Log-det regularisation weight λr.")
parser.add_argument("--lambda_r_warmup",  type=int,   default=20,
                    help="Ramp λr linearly from 0 over this many epochs.  "
                         "Set 0 to disable.")
parser.add_argument("--noise_std",        type=float, default=0.0,   # FIX: default 0.0
                    help="Gaussian noise added to targets.  Set 0 to disable.")
parser.add_argument("--reflow_model",     type=str,   default=None,
                    help="Path to pretrained FFJORD checkpoint for reflow targets.")
parser.add_argument("--teacher_ckpt",     type=str,   default=None,
                    help="CNF checkpoint for teacher reference visualisation.")
parser.add_argument("--metric-samples",   type=int,   default=250000,
                    help="Number of generated peptide samples used for ESS/E-W1/T-W2.")
parser.add_argument("--tw2-subsample",    type=int,   default=4096,
                    help="Subsample size used for peptide torus-W2 estimation.")
parser.add_argument("--openmm-platform",  type=str,   default="auto",
                    choices=["auto", "cpu", "cuda", "opencl", "reference"],
                    help="OpenMM platform used for peptide energies.")
# OT options
parser.add_argument("--ot_eps",           type=float, default=0.1,
                    help="Sinkhorn entropic regularisation ε.")
parser.add_argument("--ot_iters",         type=int,   default=50,
                    help="Sinkhorn iterations.")

# Optimiser  — paper Appendix C.2: lr=5e-4, weight_decay=0.01
parser.add_argument("--num_epochs",       type=int,   default=500)
parser.add_argument("--batch_size",       type=int,   default=256)
parser.add_argument("--test_batch_size",  type=int,   default=200)
parser.add_argument("--lr",               type=float, default=5e-4,
                    help="Peak learning rate after warm-up.")
parser.add_argument("--weight_decay",     type=float, default=0.01,
                    help="AdamW weight decay.")
parser.add_argument("--max_grad_norm",    type=float, default=1.0)
parser.add_argument("--warmup_iters",     type=int,   default=1000,
                    help="Linear LR warmup in iterations.")
parser.add_argument("--lr_min",           type=float, default=1e-6,
                    help="Minimum LR for cosine schedule.")
parser.add_argument("--ema_decay",        type=float, default=0.999,
                    help="EMA decay.")
parser.add_argument("--lr_schedule",
                    choices=["cosine", "plateau", "none"], default="cosine",
                    help="LR schedule after warm-up.")
parser.add_argument("--plateau_patience", type=int,   default=20,
                    help="Patience for ReduceLROnPlateau (epochs).")

# Checkpointing / logging
parser.add_argument("--begin_epoch",     type=int,  default=1)
parser.add_argument("--resume",          type=str,  default=None)
parser.add_argument("--save",            type=str,  default="experiments/fort")
parser.add_argument("--val_freq",        type=int,  default=5)
parser.add_argument("--log_freq",        type=int,  default=50)
parser.add_argument("--early_stop",      type=int,  default=60,
                    help="Stop if the validation metric has not improved for this many epochs.  "
                         "Set 0 to disable.")

args = parser.parse_args()
args.data = canonicalize_dataset_name(args.data)


# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────────────────────

utils.makedirs(args.save)
_ts      = time.strftime("%Y%m%d_%H%M%S")
logger   = build_logger(os.path.join(args.save, f"logs_{_ts}.txt"))
logger.info("Args: %s", vars(args))


# ─────────────────────────────────────────────────────────────────────────────
# Teacher CNF adapter  (loaded from FFJORD checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class TeacherCNF(nn.Module):
    """
    Wraps a pretrained FFJORD model.  Exposes two methods:
      encode(x) → z   forward pass  (data → latent)
      decode(z) → x   reverse pass  (latent → data, for reflow targets)

    For reflow target generation we need decode().  If the student NF has
    SqueezeLayer(s), its latent_shape differs from data_shape.  We handle
    this by accepting optional squeeze_layers: before calling decode() the
    latent z is passed through Squeeze.reverse() to map back to data_shape,
    where the teacher operates.  This is valid because Squeeze is a spatial
    permutation: N(0,I) in latent_shape IS N(0,I) in data_shape after
    Squeeze.reverse (same elements, different layout).
    """

    def __init__(self, model: nn.Module,
                 squeeze_layers: Optional[List[nn.Module]] = None):
        super().__init__()
        self.model   = model
        self._squeeze = nn.ModuleList(squeeze_layers or [])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        zero = x.new_zeros(x.shape[0], 1)
        return self.model(x, zero)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode z (in student latent_shape) → image (data_shape).

        Steps:
          1. Unsqueeze z back to data_shape via Squeeze.reverse chain.
             (SqueezeLayer is a spatial permutation; N(0,I) is invariant.)
          2. Run teacher CNF in reverse (latent → data).
        """
        for sq in reversed(self._squeeze):
            z = sq(z, reverse=True)   # (C*f², H/f, W/f) → (C, H, W)
        zero = z.new_zeros(z.shape[0], 1)
        out  = self.model(z, zero, reverse=True)
        # ODENVP / SequentialFlow both return (x, logpx) when logpx is provided
        return out[0] if isinstance(out, (tuple, list)) else out


def _build_teacher(ckpt_path: str, data_shape: Tuple[int, ...],
                   device: torch.device,
                   squeeze_layers: Optional[List[nn.Module]] = None,
                   data_kind: str = "image") -> TeacherCNF:
    """Reconstruct a FFJORD model from a train_cnf.py checkpoint."""
    import lib.layers as _L
    import lib.odenvp as _odenvp
    import lib.multiscale_parallel as _msp

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt:
        raise ValueError(f"Checkpoint '{ckpt_path}' has no 'args' key.")

    a              = ckpt["args"]
    if data_kind in {"tabular", "peptide"}:
        reg_fns, _ = create_regularization_fns(a)
        cnf = build_model_tabular(a, data_shape[0], reg_fns)
        filtered_state_dict = {}
        for key, value in ckpt["state_dict"].items():
            if "diffeq.diffeq" not in key:
                filtered_state_dict[key.replace("module.", "")] = value
        cnf.load_state_dict(filtered_state_dict, strict=False)
        set_cnf_options(a, cnf)
        cnf.to(device).eval()
        for p in cnf.parameters():
            p.requires_grad_(False)
        logger.info(
            "Teacher CNF: arch=TabularSequentialFlow solver=%s atol=%s rtol=%s path=%s",
            getattr(a, "solver", "dopri5"),
            getattr(a, "atol", 1e-5),
            getattr(a, "rtol", 1e-5),
            ckpt_path,
        )
        return TeacherCNF(cnf, squeeze_layers=squeeze_layers)

    hidden_dims    = tuple(map(int, a.dims.split(",")))
    reg_fns, _     = create_regularization_fns(a)

    if getattr(a, "multiscale", False):
        cnf = _odenvp.ODENVP(
            (a.batch_size, *data_shape),
            n_blocks            = a.num_blocks,
            intermediate_dims   = hidden_dims,
            nonlinearity        = getattr(a, "nonlinearity", "softplus"),
            alpha               = getattr(a, "alpha", 1e-6),
            cnf_kwargs=dict(
                T                   = getattr(a, "time_length", 1.0),
                train_T             = getattr(a, "train_T",     True),
                regularization_fns  = reg_fns,
            ),
        )
    elif getattr(a, "parallel", False):
        cnf = _msp.MultiscaleParallelCNF(
            (a.batch_size, *data_shape),
            n_blocks          = a.num_blocks,
            intermediate_dims = hidden_dims,
            alpha             = getattr(a, "alpha", 1e-6),
            time_length       = getattr(a, "time_length", 1.0),
        )
    else:
        strides = tuple(map(int, a.strides.split(",")))

        def _cnf():
            diffeq  = _L.ODEnet(
                hidden_dims   = hidden_dims,
                input_shape   = data_shape,
                strides       = strides,
                conv          = getattr(a, "conv",         True),
                layer_type    = getattr(a, "layer_type",   "ignore"),
                nonlinearity  = getattr(a, "nonlinearity", "softplus"),
            )
            odefunc = _L.ODEfunc(
                diffeq         = diffeq,
                divergence_fn  = getattr(a, "divergence_fn", "approximate"),
                residual       = getattr(a, "residual",       False),
                rademacher     = getattr(a, "rademacher",     True),
            )
            return _L.CNF(
                odefunc            = odefunc,
                T                  = getattr(a, "time_length", 1.0),
                train_T            = getattr(a, "train_T",     True),
                regularization_fns = reg_fns,
                solver             = getattr(a, "solver",      "dopri5"),
            )

        alpha = getattr(a, "alpha", 1e-6)
        chain  = ([_L.LogitTransform(alpha)] if alpha > 0 else [_L.ZeroMeanTransform()])
        chain += [_cnf() for _ in range(a.num_blocks)]
        if getattr(a, "batch_norm", False):
            chain.append(_L.MovingBatchNorm2d(data_shape[0]))
        cnf = _L.SequentialFlow(chain)

    cnf.load_state_dict(ckpt["state_dict"])
    set_cnf_options(a, cnf)
    cnf.to(device).eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    arch = ("ODENVP"            if getattr(a, "multiscale", False)
            else "MultiscaleParallel" if getattr(a, "parallel",   False)
            else "SequentialFlow")
    logger.info("Teacher CNF: arch=%s  solver=%s  atol=%s  rtol=%s  path=%s",
                arch, getattr(a,"solver","dopri5"),
                getattr(a,"atol",1e-5), getattr(a,"rtol",1e-5), ckpt_path)

    adapter = TeacherCNF(cnf, squeeze_layers=squeeze_layers)
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if args.target == "reflow" and args.reflow_model is None:
        sys.exit("ERROR: --target reflow requires --reflow_model <path/to/cnf.pth>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvt    = lambda t: t.float().to(device, non_blocking=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_set, val_loader, test_loader, data_shape, data_kind, peptide_bundle = get_dataset(args)
    metric_name = "NLL" if data_kind in {"tabular", "peptide"} else "BPD"
    logger.info("Dataset: %s  data_kind: %s  data_shape: %s", args.data, data_kind, data_shape)

    # ── Student NF ────────────────────────────────────────────────────────────
    model = build_student_flow(data_shape, data_kind, args).to(device)
    logger.info("Flow: %s | params: %d", args.flow, count_parameters(model))

    # Determine latent_shape (SqueezeLayer changes spatial dims)
    with torch.no_grad():
        _z = model(torch.full((1, *data_shape), 0.5, device=device))
        latent_shape: Tuple[int, ...] = tuple(_z.shape[1:])
    logger.info("Latent shape: %s", latent_shape)

    # Extract SqueezeLayer(s) from student for teacher latent alignment
    student_squeeze: List[nn.Module] = [
        l for l in model.chain if isinstance(l, SqueezeLayer)
    ]
    if student_squeeze:
        logger.info("Found %d SqueezeLayer(s) - will unsqueeze z before teacher decode",
                    len(student_squeeze))

    # ── EMA  (Appendix C.2: decay=0.999) ─────────────────────────────────────
    ema = EMA(model, decay=args.ema_decay)

    # ── Optimiser  (Appendix C.2: Adam lr=5e-4, wd=0.01) ────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ── LR schedule ───────────────────────────────────────────────────────────
    # Phase 1: linear warmup (per iteration)
    # Phase 2: cosine / plateau (per epoch, after warmup)
    _warmup_done = [False]

    def update_lr(itr: int) -> None:
        if _warmup_done[0]:
            return
        frac = min((itr + 1) / max(args.warmup_iters, 1), 1.0)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * frac
        if itr + 1 >= args.warmup_iters:
            _warmup_done[0] = True

    # Cosine scheduler will be created after first epoch (needs actual iters/epoch)
    post_scheduler = None
    _cosine_pending = [args.lr_schedule == "cosine"]

    if args.lr_schedule == "plateau":
        post_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=args.plateau_patience, min_lr=args.lr_min)

    # ── Teacher CNF for reflow targets ────────────────────────────────────────
    teacher: Optional[TeacherCNF] = None
    if args.target == "reflow":
        teacher = _build_teacher(args.reflow_model, data_shape, device,
                                 squeeze_layers=student_squeeze, data_kind=data_kind)

    # Separate teacher for visualisation figures (may reuse same checkpoint)
    vis_teacher: Optional[TeacherCNF] = None
    _vis_path = args.teacher_ckpt or (args.reflow_model if args.target == "reflow" else None)
    if _vis_path:
        if teacher is not None and _vis_path == args.reflow_model:
            vis_teacher = teacher      # reuse
        else:
            vis_teacher = _build_teacher(_vis_path, data_shape, device, data_kind=data_kind)

    # ── Resume ────────────────────────────────────────────────────────────────
    best_metric = float("inf")
    best_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optim_state_dict"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        best_metric = float(ckpt.get("best_loss",  float("inf")))
        best_epoch = int  (ckpt.get("best_epoch", 0))
        if args.begin_epoch <= 1:
            args.begin_epoch = int(ckpt.get("epoch", 0)) + 1
        if post_scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            post_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("Resumed: begin_epoch=%d  best_%s=%.4f", args.begin_epoch, metric_name.lower(), best_metric)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    def _unwrap(m: nn.Module) -> SequentialFlow:
        return m.module if isinstance(m, nn.DataParallel) else m

    # ── Teacher reference figure (once, before training) ─────────────────────
    fig_dir = os.path.join(args.save, "figs")
    utils.makedirs(fig_dir)
    if vis_teacher is not None and data_kind == "image":
        # Teacher operates in data_shape directly — no Squeeze involved.
        # decode() is only for training (latent_shape → data_shape via unsqueeze).
        # Here z is already in data_shape so we call the underlying model directly.
        _z_teacher = cvt(torch.randn(100, *data_shape))
        with torch.no_grad():
            _x_teacher = vis_teacher.model(_z_teacher, reverse=True).clamp(0., 1.)
        save_image(_x_teacher.view(-1, *data_shape),
                   os.path.join(fig_dir, "teacher.jpg"), nrow=10)
        logger.info("Teacher figures saved.")

    # Fixed latent for student vis (latent_shape = post-squeeze shape)
    fixed_z = cvt(torch.randn(100, *latent_shape))

    # ── Reconstruction test (check invertibility) ────────────────────────────
    with torch.no_grad():
        test_batch = next(iter(test_loader))[0][:4].to(device)
        recon_err = reconstruction_test(_unwrap(model), test_batch, logger)

    # ── Meters ────────────────────────────────────────────────────────────────
    t_meter   = utils.RunningAverageMeter(0.97)
    L_meter   = utils.RunningAverageMeter(0.97)
    mse_meter = utils.RunningAverageMeter(0.97)
    reg_meter = utils.RunningAverageMeter(0.97)

    # ── Training loop ─────────────────────────────────────────────────────────
    itr          = 0
    no_improve   = 0   # consecutive val epochs without improvement (early stop)

    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size,
            shuffle=True, drop_last=True, pin_memory=(data_kind == "image"))
        logger.info("Epoch %04d | %d iters", epoch, len(train_loader))

        # ── Deferred cosine scheduler init (needs actual iters/epoch) ─────────
        if _cosine_pending[0] and _warmup_done[0]:
            _ipe    = len(train_loader)
            _we     = max(1, args.warmup_iters // _ipe)
            _t_max  = max(1, args.num_epochs - (args.begin_epoch - 1) - _we)
            post_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=_t_max, eta_min=args.lr_min)
            _cosine_pending[0] = False
            logger.info("CosineAnnealingLR: iters/epoch=%d  warmup_epochs=%d  T_max=%d",
                        _ipe, _we, _t_max)

        # ── Progressive λr warm-up ─────────────────────────────────────────────
        if args.lambda_r_warmup > 0:
            _frac = min(1.0, (epoch - args.begin_epoch + 1) / args.lambda_r_warmup)
            _lambda_r = args.lambda_r * _frac
        else:
            _lambda_r = args.lambda_r

        # ── Iterate ───────────────────────────────────────────────────────────
        for x_real, _ in train_loader:
            t0 = time.time()
            update_lr(itr)
            optimizer.zero_grad()

            x_real = cvt(x_real)          # (B, C, H, W) ∈ [0,1]
            B      = x_real.shape[0]

            # ── GENERATIVE DIRECTION (Algorithm 1) ───────────────────────────
            # Sample z ~ N(0,I) in latent_shape
            z = cvt(torch.randn(B, *latent_shape))

            if args.target == "reflow":
                # Teacher decodes z to data space.
                with torch.no_grad():
                    x_target = teacher.decode(z)
                    if data_kind == "image":
                        x_target = x_target.clamp(0.0, 1.0)
            else:  # OT (experimental)
                x0_data = cvt(torch.randn(B, *data_shape))
                _, x_target = ot_match(x0_data, x_real,
                                       eps=args.ot_eps, n_iter=args.ot_iters)
                with torch.no_grad():
                    z, _ = _unwrap(model)(x_target, x_target.new_zeros(B, 1))
                    z = z.detach()

            # Add stability noise to targets (if enabled)
            if args.noise_std > 0:
                x_target = x_target + args.noise_std * torch.randn_like(x_target)
                if data_kind == "image":
                    x_target = x_target.clamp(0.0, 1.0)
            x_target = x_target.detach()

            # ── FORT generative loss ──────────────────────────────────────────
            total, mse, reg = fort_loss_generation(
                z, x_target, _unwrap(model), _lambda_r)

            if not torch.isfinite(total):
                logger.warning("Non-finite loss (%.4g) at iter %d — skipping batch", total.item(), itr)
                optimizer.zero_grad()
                itr += 1
                continue

            total.backward()
            # Second safety net: check for NaN/Inf gradients before writing to weights.
            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if not grad_ok:
                logger.warning("Non-finite gradient at iter %d — skipping batch", itr)
                optimizer.zero_grad()
                itr += 1
                continue
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # EMA update after every optimizer step
            ema.update(_unwrap(model))

            t_meter.update(time.time() - t0)
            L_meter.update(total.item())
            mse_meter.update(mse.item())
            reg_meter.update(reg.item())

            if itr % args.log_freq == 0:
                logger.info(
                    "Iter %06d | t %.3f(%.3f) | Loss %.4f(%.4f) | "
                    "MSE %.4f(%.4f) | Reg %.2e(%.2e) | λr %.2e | LR %.2e | GN %.3f",
                    itr,
                    t_meter.val,   t_meter.avg,
                    L_meter.val,   L_meter.avg,
                    mse_meter.val, mse_meter.avg,
                    reg_meter.val, reg_meter.avg,
                    _lambda_r,
                    optimizer.param_groups[0]["lr"],
                    gnorm,
                )
            itr += 1

        # ── Step epoch-level scheduler (if any) ───────────────────────────────
        # Cosine scheduler steps after each epoch (once warmup done)
        if post_scheduler and _warmup_done[0]:
            if isinstance(post_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Will be stepped inside validation with val loss
                pass
            else:
                post_scheduler.step()

        # ── Validation (using EMA weights) ────────────────────────────────────
        if epoch % args.val_freq == 0:
            ema.store(_unwrap(model))
            ema.apply(_unwrap(model))
            _unwrap(model).eval()

            val_metric = evaluate_loader_metric(val_loader, _unwrap(model), data_kind, cvt)
            logger.info("Epoch %04d | Val %s %.4f  (EMA weights)", epoch, metric_name, val_metric)

            # Save checkpoint (EMA weights)
            _ckpt = {
                "epoch":               epoch,
                "best_loss":           best_metric,
                "best_epoch":          best_epoch,
                "state_dict":          _unwrap(model).state_dict(),   # EMA weights
                "ema_state_dict":      ema.state_dict(),
                "optim_state_dict":    optimizer.state_dict(),
                "scheduler_state_dict": post_scheduler.state_dict() if post_scheduler else None,
                "args":                args,
            }
            if val_metric < best_metric:
                best_metric = val_metric
                best_epoch = epoch
                no_improve = 0
                _ckpt["best_loss"]  = best_metric
                _ckpt["best_epoch"] = best_epoch
                best_path = os.path.join(args.save, "checkpt.pth")
                torch.save(_ckpt, best_path)
                logger.info("===> BEST  epoch=%04d  %s=%.4f  %s",
                            best_epoch, metric_name, best_metric, best_path)
            else:
                no_improve += args.val_freq

            if isinstance(post_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                post_scheduler.step(val_metric)

            ema.restore(_unwrap(model))

        # ── Early stopping ────────────────────────────────────────────────────
        if args.early_stop > 0 and no_improve >= args.early_stop:
            logger.info("Early stopping: no %s improvement for %d epochs "
                        "(best=%.4f @ epoch %d).", metric_name, no_improve, best_metric, best_epoch)
            break

        # ── Visualisation (EMA weights) ────────────────────────────────────────
        if data_kind == "image":
            ema.store(_unwrap(model))
            ema.apply(_unwrap(model))
            _unwrap(model).eval()
            with torch.no_grad():
                try:
                    samples = _unwrap(model)(fixed_z, reverse=True).clamp(0., 1.)
                    save_image(samples.view(-1, *data_shape),
                               os.path.join(fig_dir, f"{epoch:04d}.jpg"), nrow=10)
                except Exception as e:
                    logger.warning("Visualisation failed: %s", e)
            ema.restore(_unwrap(model))

        # ── Save latest ────────────────────────────────────────────────────────
        torch.save({
            "epoch":                epoch,
            "best_loss":            best_metric,
            "best_epoch":           best_epoch,
            "state_dict":           _unwrap(model).state_dict(),
            "ema_state_dict":       ema.state_dict(),
            "optim_state_dict":     optimizer.state_dict(),
            "scheduler_state_dict": post_scheduler.state_dict() if post_scheduler else None,
            "args":                 args,
        }, os.path.join(args.save, "latest.pth"))

        logger.info("Epoch %04d done | best=%04d (%s=%.4f) | lr=%.2e",
                    epoch, best_epoch, metric_name, best_metric, optimizer.param_groups[0]["lr"])

    best_path = os.path.join(args.save, "checkpt.pth")
    latest_path = os.path.join(args.save, "latest.pth")
    eval_path = best_path if os.path.exists(best_path) else latest_path

    if os.path.exists(eval_path):
        eval_ckpt = torch.load(eval_path, map_location="cpu", weights_only=False)
        _unwrap(model).load_state_dict(eval_ckpt["state_dict"])

    _unwrap(model).eval()
    test_metric = evaluate_loader_metric(test_loader, _unwrap(model), data_kind, cvt)
    logger.info("Final test %s %.4f | checkpoint=%s", metric_name, test_metric, eval_path)

    metrics_payload = {
        "dataset": args.data,
        "data_kind": data_kind,
        "metric_name": metric_name,
        "test_metric": test_metric,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "checkpoint": eval_path,
    }
    if data_kind == "peptide":
        peptide_metrics = evaluate_peptide_student(
            _unwrap(model),
            bundle=peptide_bundle,
            latent_shape=latent_shape,
            device=device,
            metric_samples=args.metric_samples,
            tw2_subsample=args.tw2_subsample,
            openmm_platform=args.openmm_platform,
            artifact_prefix=os.path.join(fig_dir, "student_final"),
            logger=logger,
        )
        metrics_payload.update(peptide_metrics)

    metrics_path = os.path.join(args.save, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    logger.info("Saved test metrics to %s", metrics_path)
