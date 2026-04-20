#!/usr/bin/env python3
"""
MNIST Rectified Flow Training Pipeline

This script handles:
1. Train/Load RF (Rectified Flow)
2. Train/Load RF Student (CNF-Reflow distillation)
3. Evaluate Student
4. Summary

Prerequisites:
- CNF model trained using train_cnf.py (or standalone CNF training)
- MNIST dataset

Usage:
    # Train everything from scratch (requires CNF model)
    python train_rf_pipeline.py --cnf-path experiments/cnf/checkpt.pth

    # Skip RF training, load existing
    python train_rf_pipeline.py --cnf-path experiments/cnf/checkpt.pth --skip-rf

    # Skip Student training, load existing
    python train_rf_pipeline.py --cnf-path experiments/cnf/checkpt.pth --skip-student

    # Evaluate all models (load all existing)
    python train_rf_pipeline.py --cnf-path experiments/cnf/checkpt.pth --eval-only

    # Resume RF or Student training
    python train_rf_pipeline.py --cnf-path experiments/cnf/checkpt.pth --resume-rf --resume-student
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
import json
import math
import time
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from tabular_benchmarks import (
    TABULAR_DATASETS,
    TabularBundle,
    build_tabular_loaders,
    load_tabular_bundle,
)

try:
    from torchdiffeq import odeint
except ImportError:
    print("Please install torchdiffeq: pip install torchdiffeq")
    raise

from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance

# ==================== Configuration ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def ensure_parent_dir(path):
    """Create the parent directory for a file path if needed."""
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ==================== LogitTransform ====================

class LogitTransform:
    """
    Reversible transformation: [0, 1] <-> Ã¢â€žÂ
    
    Forward (for training): x Ã¢Ë†Ë† [0,1] Ã¢â€ â€™ y Ã¢Ë†Ë† Ã¢â€žÂ
    Inverse (for sampling): y Ã¢Ë†Ë† Ã¢â€žÂ Ã¢â€ â€™ x Ã¢Ë†Ë† [0,1]
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def forward(self, x):
        """Forward transform: [0, 1] Ã¢â€ â€™ Ã¢â€žÂ"""
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        return y
    
    def inverse(self, y):
        """Inverse transform: Ã¢â€žÂ Ã¢â€ â€™ [0, 1]"""
        s = torch.sigmoid(y)
        x = (s - self.alpha) / (1 - 2 * self.alpha)
        return x
    
    def logdet(self, x):
        """Compute log|det(dy/dx)|"""
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdet = torch.sum(
            np.log(1 - 2 * self.alpha) - torch.log(s) - torch.log(1 - s),
            dim=[1, 2, 3]
        )
        return logdet


# Global LogitTransform instance
LOGIT_TRANSFORM = LogitTransform(alpha=0.05)


# Global data shape (C,H,W). Will be overwritten in main() based on args/checkpoint.
DATA_SHAPE = (1, 28, 28)
DATA_SPEC = None
# Detect Windows for num_workers setting
IS_WINDOWS = os.name == 'nt'
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 4

IMAGE_DATASETS = {"mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "stl10", "imagefolder", "folder"}
PEPTIDE_CANONICAL_DATASETS = {"aldp", "al3", "al4"}
PEPTIDE_DATASET_ALIASES = {"tetra": "al4"}
PEPTIDE_DATASETS = PEPTIDE_CANONICAL_DATASETS | set(PEPTIDE_DATASET_ALIASES)
PEPTIDE_DATASET_EXPECTATIONS = {
    "aldp": {"num_atoms": 22, "torsion_count": 2},
    "al3": {"num_atoms": 33, "torsion_count": 4},
    "al4": {"num_atoms": 42, "torsion_count": 6},
}


@dataclass
class PeptideBundle:
    data_name: str
    root_dir: Path
    train_coords: np.ndarray
    val_coords: np.ndarray
    test_coords: np.ndarray
    train_flat: np.ndarray
    val_flat: np.ndarray
    test_flat: np.ndarray
    num_atoms: int
    coord_shape: Tuple[int, int]
    masses: np.ndarray
    std_scale: float
    torsion_atom_indices: List[Tuple[int, int, int, int]]
    prmtop_path: Optional[Path]
    inpcrd_path: Optional[Path]
    topology_pdb_path: Optional[Path]
    forcefield_files: Tuple[str, ...]
    nonbonded_method: str
    nonbonded_cutoff_nm: Optional[float]
    temperature_kelvin: float
    coordinate_unit: str
    cache_dir: Path

    def flatten_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        return coords.reshape(coords.shape[0], -1)

    def normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        centered = remove_center_of_mass(coords, self.masses)
        return self.flatten_coords(centered) / self.std_scale

    def denormalize_flat(self, flat: np.ndarray) -> np.ndarray:
        flat = np.asarray(flat, dtype=np.float32)
        return flat.reshape(flat.shape[0], *self.coord_shape) * self.std_scale

    def empty_labels(self, count: int) -> torch.Tensor:
        return torch.zeros(count, dtype=torch.long)


@dataclass
class DataSpec:
    data_name: str
    data_type: str
    data_shape: Tuple[int, ...]
    flat_dim: int
    data_root: Path
    peptide_bundle: Optional[PeptideBundle] = None
    tabular_bundle: Optional[TabularBundle] = None

    @property
    def is_image(self) -> bool:
        return self.data_type == "image"

    @property
    def is_peptide(self) -> bool:
        return self.data_type == "peptide"

    @property
    def is_tabular(self) -> bool:
        return self.data_type == "tabular"

    @property
    def uses_vector_backbone(self) -> bool:
        return self.data_type in {"peptide", "tabular"}

    @property
    def coord_shape(self) -> Optional[Tuple[int, int]]:
        if self.peptide_bundle is None:
            return None
        return self.peptide_bundle.coord_shape


def canonicalize_dataset_name(data_name: Optional[str]) -> Optional[str]:
    if data_name is None:
        return None
    normalized = str(data_name).lower()
    if normalized in PEPTIDE_DATASETS:
        return PEPTIDE_DATASET_ALIASES.get(normalized, normalized)
    return normalized


def _validate_peptide_metadata(data_name: str, metadata: Dict[str, Any]) -> None:
    expected = PEPTIDE_DATASET_EXPECTATIONS.get(data_name)
    if expected is None:
        return

    actual_num_atoms = int(metadata.get("num_atoms", -1))
    if actual_num_atoms != expected["num_atoms"]:
        raise ValueError(
            f"Peptide dataset '{data_name}' expects num_atoms={expected['num_atoms']}, "
            f"but metadata.json declares {actual_num_atoms}. Please download the correct benchmark bundle."
        )

    torsions = metadata.get("torsion_atom_indices")
    if torsions is not None and len(torsions) != expected["torsion_count"]:
        raise ValueError(
            f"Peptide dataset '{data_name}' expects {expected['torsion_count']} torsion definitions, "
            f"but metadata.json contains {len(torsions)}. Please download the correct benchmark bundle."
        )


def _shape_numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _expand_time_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], *([1] * (x.dim() - 1)))


def _safe_model_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _default_artifact_prefix(data_name: str, model_key: str) -> str:
    return f"{data_name}_{model_key}"


def _resolve_output_path(path: Optional[str], default_filename: str) -> str:
    return path if path else default_filename


def _cache_key(parts: Sequence[Any]) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _coerce_coord_array(array: np.ndarray, num_atoms: int, field_name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[1:] == (num_atoms, 3):
        return array
    if array.ndim == 2 and array.shape[1] == num_atoms * 3:
        return array.reshape(array.shape[0], num_atoms, 3)
    raise ValueError(
        f"{field_name} must have shape [N,{num_atoms},3] or [N,{num_atoms * 3}], got {tuple(array.shape)}"
    )


def remove_center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)
    weights = masses.reshape(1, -1, 1)
    com = (coords * weights).sum(axis=1, keepdims=True) / weights.sum(axis=1, keepdims=True)
    return coords - com


def _load_peptide_arrays(root_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits_path = root_dir / "splits.npz"
    if splits_path.exists():
        with np.load(splits_path) as split_data:
            try:
                return split_data["train"], split_data["val"], split_data["test"]
            except KeyError as exc:
                raise ValueError("splits.npz must contain train, val, and test arrays") from exc

    arrays = []
    for split_name in ("train", "val", "test"):
        split_path = root_dir / f"{split_name}.npy"
        if not split_path.exists():
            raise ValueError(
                f"Expected either {splits_path} or individual train/val/test .npy files in {root_dir}"
            )
        arrays.append(np.load(split_path))
    return tuple(arrays)


def _resolve_peptide_root(data_name: str, data_root: str) -> Path:
    root_dir = Path(data_root)
    metadata_here = root_dir / "metadata.json"
    if metadata_here.exists():
        return root_dir

    canonical_name = canonicalize_dataset_name(data_name)
    candidate_names = []
    for candidate_name in (canonical_name, str(data_name).lower()):
        if candidate_name and candidate_name not in candidate_names:
            candidate_names.append(candidate_name)
    if canonical_name is not None:
        for alias_name, alias_target in PEPTIDE_DATASET_ALIASES.items():
            if alias_target == canonical_name and alias_name not in candidate_names:
                candidate_names.append(alias_name)

    for candidate_name in candidate_names:
        nested_root = root_dir / candidate_name
        if (nested_root / "metadata.json").exists():
            return nested_root

    raise ValueError(
        f"Could not locate peptide bundle metadata.json under {root_dir} "
        f"or any of {[str(root_dir / name) for name in candidate_names]}"
    )


def _load_masses_from_prmtop(prmtop_path: Path) -> np.ndarray:
    try:
        from openmm import app
    except ImportError:
        try:
            from simtk.openmm import app  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenMM is required to infer atom masses from amber_prmtop. "
                "Install openmm or provide metadata.json['atom_masses']."
            ) from exc

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    masses = []
    for atom in prmtop.topology.atoms():
        mass = atom.element.mass.value_in_unit(atom.element.mass.unit)
        masses.append(float(mass))
    return np.asarray(masses, dtype=np.float32)


def _load_masses_from_pdb(pdb_path: Path) -> np.ndarray:
    try:
        from openmm import app
    except ImportError:
        try:
            from simtk.openmm import app  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenMM is required to infer atom masses from topology_pdb. "
                "Install openmm or provide metadata.json['atom_masses']."
            ) from exc

    pdb = app.PDBFile(str(pdb_path))
    masses = []
    for atom in pdb.topology.atoms():
        if atom.element is None:
            raise ValueError(f"Atom {atom.name} in {pdb_path} does not define an element")
        mass = atom.element.mass.value_in_unit(atom.element.mass.unit)
        masses.append(float(mass))
    return np.asarray(masses, dtype=np.float32)


def load_peptide_bundle(data_name: str, data_root: str) -> PeptideBundle:
    canonical_name = canonicalize_dataset_name(data_name) or str(data_name).lower()
    root_dir = _resolve_peptide_root(data_name, data_root)
    metadata_path = root_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    missing_keys = [key for key in ("num_atoms", "torsion_atom_indices") if key not in metadata]
    if missing_keys:
        raise ValueError(f"metadata.json is missing required keys: {', '.join(missing_keys)}")
    if "amber_prmtop" not in metadata and "topology_pdb" not in metadata:
        raise ValueError("metadata.json must include either 'amber_prmtop' or 'topology_pdb'")
    _validate_peptide_metadata(canonical_name, metadata)

    num_atoms = int(metadata["num_atoms"])
    train_raw, val_raw, test_raw = _load_peptide_arrays(root_dir)
    train_coords = _coerce_coord_array(train_raw, num_atoms, "train")
    val_coords = _coerce_coord_array(val_raw, num_atoms, "val")
    test_coords = _coerce_coord_array(test_raw, num_atoms, "test")

    atom_masses = metadata.get("atom_masses")
    if atom_masses is not None:
        masses = np.asarray(atom_masses, dtype=np.float32)
    else:
        if "amber_prmtop" in metadata:
            prmtop_candidate = Path(metadata["amber_prmtop"])
            if not prmtop_candidate.is_absolute():
                prmtop_candidate = root_dir / prmtop_candidate
            masses = _load_masses_from_prmtop(prmtop_candidate)
        else:
            pdb_candidate = Path(metadata["topology_pdb"])
            if not pdb_candidate.is_absolute():
                pdb_candidate = root_dir / pdb_candidate
            masses = _load_masses_from_pdb(pdb_candidate)

    if masses.shape != (num_atoms,):
        raise ValueError(f"Expected {num_atoms} atom masses, got shape {tuple(masses.shape)}")

    train_centered = remove_center_of_mass(train_coords, masses)
    val_centered = remove_center_of_mass(val_coords, masses)
    test_centered = remove_center_of_mass(test_coords, masses)
    std_scale = float(np.std(train_centered.reshape(train_centered.shape[0], -1)))
    std_scale = max(std_scale, 1e-6)

    torsion_atom_indices = [tuple(map(int, item)) for item in metadata["torsion_atom_indices"]]
    if not torsion_atom_indices:
        raise ValueError("metadata.json['torsion_atom_indices'] must contain at least one torsion definition")

    prmtop_value = metadata.get("amber_prmtop")
    prmtop_path = None
    if prmtop_value:
        prmtop_path = Path(prmtop_value)
        if not prmtop_path.is_absolute():
            prmtop_path = root_dir / prmtop_path

    topology_pdb_value = metadata.get("topology_pdb")
    topology_pdb_path = None
    if topology_pdb_value:
        topology_pdb_path = Path(topology_pdb_value)
        if not topology_pdb_path.is_absolute():
            topology_pdb_path = root_dir / topology_pdb_path

    inpcrd_value = metadata.get("amber_inpcrd")
    inpcrd_path = None
    if inpcrd_value:
        inpcrd_path = Path(inpcrd_value)
        if not inpcrd_path.is_absolute():
            inpcrd_path = root_dir / inpcrd_path

    forcefield_files = tuple(metadata.get("forcefield_files", ()))
    nonbonded_method = str(metadata.get("nonbonded_method", "NoCutoff"))
    nonbonded_cutoff_nm = metadata.get("nonbonded_cutoff_nm")
    if nonbonded_cutoff_nm is not None:
        nonbonded_cutoff_nm = float(nonbonded_cutoff_nm)

    cache_dir = root_dir / "metrics_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return PeptideBundle(
        data_name=canonical_name,
        root_dir=root_dir,
        train_coords=train_centered,
        val_coords=val_centered,
        test_coords=test_centered,
        train_flat=train_centered.reshape(train_centered.shape[0], -1) / std_scale,
        val_flat=val_centered.reshape(val_centered.shape[0], -1) / std_scale,
        test_flat=test_centered.reshape(test_centered.shape[0], -1) / std_scale,
        num_atoms=num_atoms,
        coord_shape=(num_atoms, 3),
        masses=masses,
        std_scale=std_scale,
        torsion_atom_indices=torsion_atom_indices,
        prmtop_path=prmtop_path,
        inpcrd_path=inpcrd_path,
        topology_pdb_path=topology_pdb_path,
        forcefield_files=forcefield_files,
        nonbonded_method=nonbonded_method,
        nonbonded_cutoff_nm=nonbonded_cutoff_nm,
        temperature_kelvin=float(metadata.get("temperature_kelvin", 300.0)),
        coordinate_unit=str(metadata.get("coordinate_unit", "angstrom")).lower(),
        cache_dir=cache_dir,
    )


def build_peptide_loaders(
    bundle: PeptideBundle,
    batch_size: int,
    max_train_samples: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    train_tensor = torch.from_numpy(bundle.train_flat.astype(np.float32))
    test_tensor = torch.from_numpy(bundle.test_flat.astype(np.float32))

    train_dataset = TensorDataset(train_tensor, bundle.empty_labels(train_tensor.shape[0]))
    if max_train_samples is not None and max_train_samples > 0 and len(train_dataset) > max_train_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(train_dataset), size=max_train_samples, replace=False)
        train_dataset = Subset(train_dataset, indices.tolist())

    test_dataset = TensorDataset(test_tensor, bundle.empty_labels(test_tensor.shape[0]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0))
    return train_loader, test_loader


def build_data_spec(args) -> DataSpec:
    ckpt_data, _ = _infer_data_from_cnf_checkpoint(args.cnf_path)
    data_name = canonicalize_dataset_name(args.data or ckpt_data or "mnist")
    data_type = getattr(args, "data_type", "auto")
    data_root = getattr(args, "data_root", "./data")
    if data_type == "auto":
        if data_name in PEPTIDE_DATASETS:
            data_type = "peptide"
        elif data_name in TABULAR_DATASETS:
            data_type = "tabular"
        else:
            data_type = "image"

    if data_type == "peptide":
        if data_name not in PEPTIDE_DATASETS:
            raise ValueError(
                f"Unsupported peptide dataset '{data_name}'. "
                f"Only {sorted(PEPTIDE_CANONICAL_DATASETS)} are officially supported in this pipeline "
                f"(legacy alias: tetra -> al4)."
            )
        bundle = load_peptide_bundle(data_name, data_root)
        return DataSpec(
            data_name=data_name,
            data_type="peptide",
            data_shape=(bundle.train_flat.shape[1],),
            flat_dim=bundle.train_flat.shape[1],
            data_root=bundle.root_dir,
            peptide_bundle=bundle,
            tabular_bundle=None,
        )

    if data_type == "tabular":
        if data_name not in TABULAR_DATASETS:
            raise ValueError(
                f"Unsupported tabular dataset '{data_name}'. "
                f"Expected one of: {sorted(TABULAR_DATASETS)}"
            )
        bundle = load_tabular_bundle(data_name, data_root=data_root)
        return DataSpec(
            data_name=data_name,
            data_type="tabular",
            data_shape=(bundle.input_dim,),
            flat_dim=bundle.input_dim,
            data_root=bundle.data_root,
            peptide_bundle=None,
            tabular_bundle=bundle,
        )

    data_shape = _parse_shape_arg(getattr(args, "data_shape", None))
    if data_shape is None:
        _, ckpt_shape = _infer_data_from_cnf_checkpoint(args.cnf_path)
        data_shape = ckpt_shape
    mapping = {
        "mnist": (1, 28, 28),
        "fashionmnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "svhn": (3, 32, 32),
        "stl10": (3, 96, 96),
    }
    if data_shape is None and data_name in mapping:
        data_shape = mapping[data_name]
    if data_shape is None:
        data_shape = (1, 28, 28)
    root_dir = Path(getattr(args, "data_root", "./data"))
    return DataSpec(
        data_name=data_name,
        data_type="image",
        data_shape=tuple(data_shape),
        flat_dim=_shape_numel(tuple(data_shape)),
        data_root=root_dir,
        peptide_bundle=None,
        tabular_bundle=None,
    )


# ==================== Data Loading ====================

class MNISTPreprocessTransform:
    """
    Picklable transform for MNIST preprocessing.
    Applies dequantization and LogitTransform.
    
    This is defined as a class (not a nested function) to support
    Windows multiprocessing which requires picklable transforms.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def __call__(self, x):
        # Dequantization: add uniform noise to avoid discrete values
        x = (x * 255.0 + torch.rand_like(x)) / 256.0
        # LogitTransform: [0, 1] -> R
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        return y


def get_mnist_loaders(batch_size=64, num_workers=None):
    """Load MNIST with dequantization and LogitTransform"""
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        MNISTPreprocessTransform(alpha=LOGIT_TRANSFORM.alpha),
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )

    return train_loader, test_loader


def get_mnist_loaders_raw(batch_size=64, num_workers=None):
    """Load raw MNIST in [0, 1] range (for FID calculation)"""
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )

    return test_loader




# ==================== Generalized Dataset Utilities ====================

def _parse_shape_arg(s: str):
    """Parse shape string like '3,32,32' -> (3,32,32)."""
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 0:
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"--data-shape must be comma-separated ints like '3,32,32', got: {s}") from e


def _infer_data_from_cnf_checkpoint(cnf_path: str):
    """Try to infer dataset name and shape from a FFJORD train_cnf.py checkpoint."""
    try:
        ckpt = torch.load(cnf_path, map_location='cpu', weights_only=False)
    except Exception:
        return None, None

    saved_args = ckpt.get("args", None)
    if saved_args is None:
        return None, None

    data = getattr(saved_args, "data", None)
    # Best-effort shape inference
    # 1) explicit fields if they exist
    for (c_key, s_key) in [("num_channels", "image_size"), ("channels", "image_size"), ("nc", "img_size"), ("n_channels", "img_size")]:
        if hasattr(saved_args, c_key) and hasattr(saved_args, s_key):
            try:
                c = int(getattr(saved_args, c_key))
                sz = int(getattr(saved_args, s_key))
                return canonicalize_dataset_name(data), (c, sz, sz)
            except Exception:
                pass

    # 2) common dataset mapping
    mapping = {
        "mnist": (1, 28, 28),
        "fashionmnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "svhn": (3, 32, 32),
        "stl10": (3, 96, 96),
    }
    if isinstance(data, str) and data.lower() in mapping:
        return canonicalize_dataset_name(data.lower()), mapping[data.lower()]

    return canonicalize_dataset_name(data), None


def resolve_data_config(args):
    """Resolve args.data + args.data_shape (C,H,W).
    Priority:
      1) CLI --data-shape
      2) CNF checkpoint metadata (if available)
      3) dataset-name mapping (CLI --data)
      4) fallback MNIST
    """
    # CLI overrides
    data_shape = _parse_shape_arg(getattr(args, "data_shape", None))
    data_name = canonicalize_dataset_name(getattr(args, "data", None))

    ckpt_data, ckpt_shape = _infer_data_from_cnf_checkpoint(args.cnf_path)
    if data_name is None and isinstance(ckpt_data, str):
        data_name = canonicalize_dataset_name(ckpt_data)

    if data_shape is None and ckpt_shape is not None:
        data_shape = ckpt_shape

    mapping = {
        "mnist": (1, 28, 28),
        "fashionmnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "svhn": (3, 32, 32),
        "stl10": (3, 96, 96),
    }
    if data_shape is None and isinstance(data_name, str) and data_name.lower() in mapping:
        data_shape = mapping[data_name.lower()]

    if data_name is None:
        data_name = "mnist"
    if data_shape is None:
        data_shape = (1, 28, 28)

    return data_name.lower(), tuple(data_shape)


def get_raw_test_loader(data_name: str, data_root: str, batch_size=64, num_workers=None, download=True):
    """Raw loader in [0,1] for FID/statistics."""
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    transform = transforms.Compose([transforms.ToTensor()])
    name = (data_name or "mnist").lower()

    if name == "mnist":
        ds = datasets.MNIST(root=data_root, train=False, download=download, transform=transform)
    elif name == "fashionmnist":
        ds = datasets.FashionMNIST(root=data_root, train=False, download=download, transform=transform)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=False, download=download, transform=transform)
    elif name == "svhn":
        ds = datasets.SVHN(root=data_root, split="test", download=download, transform=transform)
    elif name == "stl10":
        ds = datasets.STL10(root=data_root, split="test", download=download, transform=transform)
    elif name in ["imagefolder", "folder"]:
        ds = datasets.ImageFolder(root=data_root, transform=transform)
    else:
        raise ValueError(f"Unsupported --data '{data_name}'. Provide --data-shape and use --data imagefolder, or extend get_raw_test_loader().")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0))


def get_train_loaders_logit(data_name: str, data_root: str, batch_size=64, num_workers=None, alpha=0.05, download=True):
    """Train/test loaders in logit space (with dequantization + logit), for RF/student training."""
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    transform = transforms.Compose([
        transforms.ToTensor(),
        MNISTPreprocessTransform(alpha=alpha),  # works for any 8-bit image tensor in [0,1]
    ])
    name = (data_name or "mnist").lower()

    if name == "mnist":
        train_ds = datasets.MNIST(root=data_root, train=True, download=download, transform=transform)
        test_ds  = datasets.MNIST(root=data_root, train=False, download=download, transform=transform)
    elif name == "fashionmnist":
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=download, transform=transform)
        test_ds  = datasets.FashionMNIST(root=data_root, train=False, download=download, transform=transform)
    elif name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform)
        test_ds  = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform)
    elif name == "cifar100":
        train_ds = datasets.CIFAR100(root=data_root, train=True, download=download, transform=transform)
        test_ds  = datasets.CIFAR100(root=data_root, train=False, download=download, transform=transform)
    elif name == "svhn":
        train_ds = datasets.SVHN(root=data_root, split="train", download=download, transform=transform)
        test_ds  = datasets.SVHN(root=data_root, split="test", download=download, transform=transform)
    elif name == "stl10":
        train_ds = datasets.STL10(root=data_root, split="train", download=download, transform=transform)
        test_ds  = datasets.STL10(root=data_root, split="test", download=download, transform=transform)
    elif name in ["imagefolder", "folder"]:
        # Expect ImageFolder structure in data_root/{train,test}/...
        train_dir = os.path.join(data_root, "train")
        test_dir  = os.path.join(data_root, "test")
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            raise ValueError("For --data imagefolder, expect folders: <data_root>/train and <data_root>/test")
        train_ds = datasets.ImageFolder(root=train_dir, transform=transform)
        test_ds  = datasets.ImageFolder(root=test_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported --data '{data_name}'. Extend get_train_loaders_logit().")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=(num_workers > 0))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0))
    return train_loader, test_loader

# ==================== Network Architectures ====================

class ResidualBlock(nn.Module):
    """Residual block with GroupNorm"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class CNFNet(nn.Module):
    """
    Continuous Normalizing Flow network for MNIST
    Compatible with train_cnf.py architecture expectations
    """
    def __init__(self, img_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.img_channels = img_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.conv_in = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Down blocks
        self.down1 = nn.ModuleList([
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ])
        self.down_sample1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        # Middle
        self.mid1 = ResidualBlock(base_channels * 2)
        self.mid2 = ResidualBlock(base_channels * 2)

        # Time projection layers
        self.time_proj1 = nn.Linear(time_dim, base_channels)
        self.time_proj2 = nn.Linear(time_dim, base_channels * 2)

        # Up blocks
        self.up_sample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ])

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )

    def forward(self, t, x):
        # Handle scalar time
        if len(t.shape) == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif len(t.shape) == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])

        # Get time embedding
        t_emb = self.time_embed(t)

        # Encoder
        h = self.conv_in(x)

        # Down 1
        for block in self.down1:
            h = block(h)
        h = h + self.time_proj1(t_emb)[:, :, None, None]
        h = self.down_sample1(h)

        # Middle
        h = self.mid1(h)
        h = h + self.time_proj2(t_emb)[:, :, None, None]
        h = self.mid2(h)

        # Up 1
        h = self.up_sample1(h)
        for block in self.up1:
            h = block(h)

        # Output
        return self.conv_out(h)


class RFNet(nn.Module):
    """Rectified Flow network (larger architecture)"""
    def __init__(self, img_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        self.conv_in = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.blocks = nn.ModuleList([
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ])

        self.time_proj = nn.Linear(time_dim, base_channels)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )

    def forward(self, t, x):
        if len(t.shape) == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif len(t.shape) == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)

        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]

        return self.conv_out(h)


class RFStudent(nn.Module):
    """
    Student model for Rectified Flow distillation
    
    Now uses same architecture as RFNet for better learning capacity.
    """
    def __init__(self, img_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)  # Added extra layer
        )

        self.conv_in = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Increased from 3 to 5 residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ])

        self.time_proj = nn.Linear(time_dim, base_channels)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )

    def forward(self, t, x):
        if len(t.shape) == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif len(t.shape) == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)

        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]

        return self.conv_out(h)


class VectorResidualBlock(nn.Module):
    """Residual MLP block with time conditioning for vector-valued peptide data."""
    def __init__(self, dim: int, hidden_dim: int, time_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) + self.time_proj(t_emb)
        return x + self.net(h)


class VectorRFNet(nn.Module):
    """Time-conditioned residual MLP used for peptide RF models."""
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_blocks: int = 6, time_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            VectorResidualBlock(hidden_dim, hidden_dim, time_dim)
            for _ in range(num_blocks)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif len(t.shape) == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h)


class VectorRFStudent(VectorRFNet):
    """Student variant reusing the same vector RF backbone."""
    pass


def build_rf_model(data_spec: DataSpec) -> nn.Module:
    if data_spec.uses_vector_backbone:
        return VectorRFNet(input_dim=data_spec.flat_dim).to(device)
    return RFNet(img_channels=data_spec.data_shape[0], base_channels=64).to(device)


def build_student_model(data_spec: DataSpec, args=None) -> nn.Module:
    if data_spec.uses_vector_backbone:
        # Added: allow the vector student size to be configured from CLI args.
        hidden_dim = 512 if args is None else args.student_hidden_dim
        num_blocks = 6 if args is None else args.student_num_blocks
        time_dim = 128 if args is None else args.student_time_dim
        return VectorRFStudent(
            input_dim=data_spec.flat_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            time_dim=time_dim,
        ).to(device)
    return RFStudent(img_channels=data_spec.data_shape[0]).to(device)


# ==================== Training Functions ====================

def train_rf(
    model,
    dataloader,
    epochs=50,
    lr=1e-3,
    save_interval=100,
    optimizer=None,
    start_epoch=0,
    ckpt_path=None,
    model_path=None,
    snapshot_prefix=None,
):
    """
    Train Rectified Flow model
    
    RF objective: Learn velocity v(t, x) such that x_t = (1-t)*z + t*x_1
    where v_target = x_1 - z
    
    Args:
        model: RFNet model
        dataloader: DataLoader with logit-space MNIST
        epochs: Number of training epochs
        lr: Learning rate
        save_interval: Save checkpoint every N epochs
        optimizer: Optional pre-initialized optimizer (for resume)
        start_epoch: Starting epoch number (for resume)
        ckpt_path: Path to save checkpoints
        model_path: Path to save final model weights
    
    Returns:
        model, optimizer, total_epochs
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    loss_fn = nn.MSELoss()
    model.train()

    for local_epoch in range(epochs):
        global_epoch = start_epoch + local_epoch
        total_loss = 0
        num_batches = 0

        for batch_idx, (x1, _) in enumerate(dataloader):
            # x1 is target in logit space
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)  # Sample from standard normal

            # Random time
            t = torch.rand(x0.size(0), device=device)

            # Linear interpolation in either image or vector space
            t_view = _expand_time_like(t, x0)
            xt = (1 - t_view) * x0 + t_view * x1
            
            # Target velocity
            v_target = x1 - x0

            # Predict velocity
            v_pred = model(t, xt)
            loss = loss_fn(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            # Removed aggressive gradient clipping (was 1.0) to match mnist_cnf_reflow_tiny.py
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"[RF] Epoch {global_epoch+1} Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, num_batches)
        print(f"[RF] Epoch {global_epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save periodic checkpoint
        if save_interval and ((global_epoch + 1) % save_interval == 0):
            prefix = snapshot_prefix or "rf"
            snapshot_path = f"{prefix}_epoch_{global_epoch+1}.pth"
            ensure_parent_dir(snapshot_path)
            torch.save(model.state_dict(), snapshot_path)
            print(f"[RF] Snapshot saved: {snapshot_path}")

        # Save resume checkpoint
        if ckpt_path is not None:
            ensure_parent_dir(ckpt_path)
            torch.save({
                "epoch": global_epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    if model_path is not None:
        ensure_parent_dir(model_path)
        torch.save(model.state_dict(), model_path)
        print(f"[RF] Final model saved: {model_path}")

    return model, optimizer, start_epoch + epochs

def train_rf_student(
    cnf_teacher,
    dataloader,
    epochs=50,
    batch_size=256,
    num_steps=20,
    lr=1e-3,
    save_interval=100,
    optimizer=None,
    start_epoch=0,
    ckpt_path=None,
    model_path=None,
    student_model=None,
    data_spec=None,
    snapshot_prefix=None,
    model_args=None,
):
    """
    Train RF Student model via CNF-Reflow distillation
    
    The student learns to predict the straight-line velocity from noise to data.
    For FFJORD models, we generate (z, x) pairs and train on v = x - z.
    
    Args:
        cnf_teacher: Trained CNF model (frozen) - can be FFJORD or standalone
        dataloader: DataLoader (used for batch counting)
        epochs: Number of training epochs
        batch_size: Batch size for trajectory collection
        num_steps: Number of ODE steps for trajectory (for standalone models)
        lr: Learning rate
        save_interval: Save checkpoint every N epochs
        optimizer: Optional pre-initialized optimizer
        start_epoch: Starting epoch number
        ckpt_path: Path to save checkpoints
        model_path: Path to save final model weights
        student_model: Optional pre-initialized student model
    
    Returns:
        student_model, optimizer, total_epochs
    """
    cnf_teacher.eval()

    print("\n" + "=" * 50)
    print("Collecting training data from CNF teacher...")
    print("=" * 50)

    # Check if teacher is FFJORD model
    use_ffjord = is_ffjord_model(cnf_teacher)
    sample_shape = getattr(cnf_teacher, "_data_shape", DATA_SHAPE)

    if use_ffjord:
        print("Using FFJORD model for distillation (generating z->x pairs)")
        # For FFJORD: generate (noise, sample) pairs and train on straight-line velocity
        z_samples = []
        x_samples = []
        
        # num_collection_batches = 100 original
        num_collection_batches = 200  # Increased from 30 to 100 for more training data
        
        with torch.no_grad():
            for batch_idx in range(num_collection_batches):
                z = torch.randn(batch_size, *sample_shape).to(device)
                # FFJORD: reverse=True maps noise to data
                x = cnf_teacher(z, reverse=True)
                
                z_samples.append(z.cpu())
                x_samples.append(x.cpu())
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"Collected {batch_idx + 1}/{num_collection_batches} batches")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        z_train = torch.cat(z_samples, dim=0)
        x_train = torch.cat(x_samples, dim=0)

        if data_spec is not None and data_spec.is_image:
            # FFJORD image models emit [0,1] samples, but RF students train in logit space.
            x_train_processed = LOGIT_TRANSFORM.forward(torch.clamp(x_train, 0.001, 0.999))
        else:
            x_train_processed = x_train

        print(f"Collected {z_train.shape[0]} (z, x) pairs")
        print(f"z range: [{z_train.min():.3f}, {z_train.max():.3f}]")
        print(f"x range: [{x_train.min():.3f}, {x_train.max():.3f}]")
        print(f"x_processed range: [{x_train_processed.min():.3f}, {x_train_processed.max():.3f}]")
        
        # For FFJORD distillation, we train on (t, x_t, v) where:
        # x_t = (1-t)*z + t*x_logit
        # v = x_logit - z (constant for straight-line flow)
        
    else:
        print("Using standalone CNF model (collecting trajectory data)")
        # For standalone models: collect (t, x, v) along trajectories
        t_vals = torch.linspace(0.0, 1.0, num_steps).to(device)

        t_samples = []
        x_samples = []
        v_samples = []

        num_collection_batches = 50  # Increased from 20 to 50

        with torch.no_grad():
            for batch_idx in range(num_collection_batches):
                z0 = torch.randn(batch_size, *sample_shape).to(device)

                trajectory = odeint(
                    cnf_teacher, z0, t_vals,
                    method='rk4',
                    options=dict(step_size=1.0/(num_steps-1))
                )

                for i in range(num_steps):
                    t_i = t_vals[i]
                    x_i = trajectory[i]
                    v_i = cnf_teacher(t_i, x_i)

                    t_samples.append(t_i.expand(x_i.shape[0]).cpu())
                    x_samples.append(x_i.cpu())
                    v_samples.append(v_i.cpu())

                if (batch_idx + 1) % 5 == 0:
                    print(f"Collected {batch_idx + 1}/{num_collection_batches} batches")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        t_train = torch.cat(t_samples, dim=0)
        x_train = torch.cat(x_samples, dim=0)
        v_train = torch.cat(v_samples, dim=0)

        print(f"Collected {t_train.shape[0]} training samples")

    # Initialize student model
    if student_model is None:
        if data_spec is None:
            raise ValueError("data_spec is required to initialize a student model")
        # Added: forward CLI-configured student architecture when the student is created here.
        student_model = build_student_model(data_spec, model_args)
        student_model._data_shape = data_spec.data_shape

    if optimizer is None:
        optimizer = optim.Adam(student_model.parameters(), lr=lr)
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    print(f"\nStudent model parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    if use_ffjord:
        # FFJORD distillation: train on random t with interpolated x_t
        num_data = z_train.shape[0]
        num_batches = (num_data + batch_size - 1) // batch_size
        
        print(f"\nTraining student for {epochs} epochs (FFJORD distillation)...")
        print("=" * 50)
        
        for local_epoch in range(epochs):
            global_epoch = start_epoch + local_epoch
            perm = torch.randperm(num_data)
            total_loss = 0
            
            for i in range(num_batches):
                idx = perm[i * batch_size: (i + 1) * batch_size]
                z_batch = z_train[idx].to(device)
                x_batch = x_train_processed[idx].to(device)

                # Random time
                t = torch.rand(z_batch.shape[0], device=device)

                # Interpolate: x_t = (1-t)*z + t*x
                t_expanded = _expand_time_like(t, z_batch)
                x_t = (1 - t_expanded) * z_batch + t_expanded * x_batch
                
                # Target velocity: v = x - z (constant for rectified flow)
                v_target = x_batch - z_batch
                
                # Predict velocity
                pred_v = student_model(t, x_t)
                loss = ((pred_v - v_target) ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                # Removed gradient clipping to match mnist_cnf_reflow_tiny.py
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if i % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / max(1, num_batches)
            
            if (global_epoch + 1) % 10 == 0 or global_epoch == 0:
                print(f"[RF-Student] Epoch {global_epoch+1} | Loss: {avg_loss:.4f}")
            
            # Save periodic checkpoint
            if save_interval and ((global_epoch + 1) % save_interval == 0):
                prefix = snapshot_prefix or "student"
                snapshot_path = f"{prefix}_epoch_{global_epoch+1}.pth"
                ensure_parent_dir(snapshot_path)
                torch.save(student_model.state_dict(), snapshot_path)
            
            # Save resume checkpoint
            if ckpt_path is not None:
                torch.save({
                    "epoch": global_epoch + 1,
                    "model": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)
    else:
        # Standalone CNF: train on collected (t, x, v) data
        num_data = t_train.shape[0]
        num_batches = (num_data + batch_size - 1) // batch_size

        print(f"\nTraining student for {epochs} epochs...")
        print("=" * 50)

        for local_epoch in range(epochs):
            global_epoch = start_epoch + local_epoch
            perm = torch.randperm(num_data)
            total_loss = 0

            for i in range(num_batches):
                idx = perm[i * batch_size: (i + 1) * batch_size]
                t_batch = t_train[idx].to(device)
                x_batch = x_train[idx].to(device)
                v_batch = v_train[idx].to(device)

                pred_v = student_model(t_batch, x_batch)
                loss = ((pred_v - v_batch) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                # Removed gradient clipping to match mnist_cnf_reflow_tiny.py
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                if i % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / max(1, num_batches)
            
            if (global_epoch + 1) % 10 == 0 or global_epoch == 0:
                print(f"[RF-Student] Epoch {global_epoch+1} | Loss: {avg_loss:.4f}")

            # Save periodic checkpoint
            if save_interval and ((global_epoch + 1) % save_interval == 0):
                prefix = snapshot_prefix or "student"
                snapshot_path = f"{prefix}_epoch_{global_epoch+1}.pth"
                ensure_parent_dir(snapshot_path)
                torch.save(student_model.state_dict(), snapshot_path)

            # Save resume checkpoint
            if ckpt_path is not None:
                ensure_parent_dir(ckpt_path)
                torch.save({
                    "epoch": global_epoch + 1,
                    "model": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)

    # Save final model
    if model_path is not None:
        ensure_parent_dir(model_path)
        torch.save(student_model.state_dict(), model_path)
        print(f"[RF-Student] Final model saved: {model_path}")

    return student_model, optimizer, start_epoch + epochs


# ==================== Sampling Functions ====================

def is_ffjord_model(model):
    """Check if model is a FFJORD model (has SequentialFlow structure)"""
    model_type = type(model).__name__
    return model_type in ['SequentialFlow', 'MultiscaleParallelCNF', 'ODENVP']


def sample_from_ffjord(model, n_samples=64, data_shape=None):
    """
    Sample from FFJORD model using its native reverse method.
    
    FFJORD includes LogitTransform in its chain, so output is already in [0, 1].
    """
    model.eval()
    if data_shape is None:
        data_shape = getattr(model, "_data_shape", DATA_SHAPE)
    with torch.no_grad():
        z = torch.randn(n_samples, *data_shape).to(device)
        # FFJORD: reverse=True goes from noise to data
        x = model(z, reverse=True)
        # FFJORD's LogitTransform already converts to [0, 1]
        x = torch.clamp(x, 0, 1)
    return x


def sample_from_model(model, n_samples=64, method="rk4", steps=25):
    """
    Sample images from a trained flow model
    
    Handles both FFJORD models and standalone models.
    """
    # Check if this is a FFJORD model
    if is_ffjord_model(model):
        return sample_from_ffjord(model, n_samples)
    
    # Standalone model: use odeint
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(n_samples, *getattr(model, '_data_shape', DATA_SHAPE)).to(device)
        t_vals = torch.linspace(0.0, 1.0, steps).to(device)

        trajectory = odeint(
            model, z0, t_vals,
            method=method,
            options=dict(step_size=1.0/(steps-1))
        )
        x_logit = trajectory[-1]

        # Inverse LogitTransform
        x_image = LOGIT_TRANSFORM.inverse(x_logit)
        x_image = torch.clamp(x_image, 0, 1)

    return x_image


def sample_from_model_raw(model, n_samples=64, method="rk4", steps=25):
    """Return raw logit-space samples (for diagnostics)"""
    # For FFJORD models, we can't easily get logit-space output
    if is_ffjord_model(model):
        # Return the [0,1] samples with a note
        samples = sample_from_ffjord(model, n_samples)
        print("Note: FFJORD model returns [0,1] space directly (includes LogitTransform)")
        return samples
    
    model.eval()
    with torch.no_grad():
        z0 = torch.randn(n_samples, *getattr(model, '_data_shape', DATA_SHAPE)).to(device)
        t_vals = torch.linspace(0.0, 1.0, steps).to(device)

        trajectory = odeint(
            model, z0, t_vals,
            method=method,
            options=dict(step_size=1.0/(steps-1))
        )
        x_logit = trajectory[-1]

    return x_logit


# ==================== Evaluation Functions ====================

def calculate_fid_simplified(real_images, generated_images):
    """
    Simplified FID calculation using image statistics
    Note: For proper FID, use a pretrained InceptionV3 model
    
    This version uses PCA to reduce dimensionality and avoid ill-conditioned matrices.
    """
    real_flat = real_images.reshape(real_images.shape[0], -1).cpu().numpy()
    gen_flat = generated_images.reshape(generated_images.shape[0], -1).cpu().numpy()
    
    # Use fewer dimensions to avoid ill-conditioned covariance matrices
    # Rule of thumb: need at least 10x more samples than dimensions
    n_samples = min(real_flat.shape[0], gen_flat.shape[0])
    max_dims = min(n_samples // 10, 64)  # Use at most 64 PCA components
    
    if max_dims < 2:
        print("Warning: Not enough samples for reliable FID calculation")
        # Fall back to simple L2 distance between means
        mu_real = np.mean(real_flat, axis=0)
        mu_gen = np.mean(gen_flat, axis=0)
        return np.sum((mu_real - mu_gen) ** 2)
    
    try:
        from sklearn.decomposition import PCA
        
        # Fit PCA on real data and transform both
        pca = PCA(n_components=max_dims)
        real_pca = pca.fit_transform(real_flat)
        gen_pca = pca.transform(gen_flat)
        
        mu_real = np.mean(real_pca, axis=0)
        mu_gen = np.mean(gen_pca, axis=0)
        
        sigma_real = np.cov(real_pca, rowvar=False)
        sigma_gen = np.cov(gen_pca, rowvar=False)
        
        # Add small regularization for numerical stability
        eps = 1e-6
        sigma_real += eps * np.eye(max_dims)
        sigma_gen += eps * np.eye(max_dims)
        
    except ImportError:
        # Fall back to original method with regularization
        print("sklearn not available, using regularized full-dimensional FID")
        mu_real = np.mean(real_flat, axis=0)
        mu_gen = np.mean(gen_flat, axis=0)
        
        sigma_real = np.cov(real_flat, rowvar=False)
        sigma_gen = np.cov(gen_flat, rowvar=False)
        
        # Add regularization
        eps = 1e-4
        sigma_real += eps * np.eye(sigma_real.shape[0])
        sigma_gen += eps * np.eye(sigma_gen.shape[0])

    diff = mu_real - mu_gen
    
    try:
        covmean = sqrtm(sigma_real @ sigma_gen)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    except Exception as e:
        print(f"Warning: sqrtm failed ({e}), using simplified metric")
        fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen)
    
    return fid


def visualize_samples(model, n_samples=64, save_path='samples.png', method="rk4", steps=25):
    """Generate and visualize samples"""
    
    if is_ffjord_model(model):
        # FFJORD model: output is already in [0, 1]
        samples = sample_from_ffjord(model, n_samples)
        print(f"[{save_path}] FFJORD output (image space): "
              f"min={samples.min().item():.3f}, max={samples.max().item():.3f}, "
              f"mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")
    else:
        # Standalone model: get logit space then convert
        x_logit = sample_from_model_raw(model, n_samples, method=method, steps=steps)

        print(f"[{save_path}] logit space: "
              f"min={x_logit.min().item():.3f}, max={x_logit.max().item():.3f}, "
              f"mean={x_logit.mean().item():.3f}, std={x_logit.std().item():.3f}")

        samples = LOGIT_TRANSFORM.inverse(x_logit)
        samples = torch.clamp(samples, 0, 1)

        print(f"[{save_path}] image space: "
              f"min={samples.min().item():.3f}, max={samples.max().item():.3f}, "
              f"mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")

    grid = make_grid(samples, nrow=8, padding=2)
    save_image(grid, save_path)

    plt.figure(figsize=(10, 10))
    img_np = grid.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[2] == 1:
        plt.imshow(img_np[:, :, 0], cmap='gray')
    else:
        plt.imshow(img_np)
    plt.axis('off')
    plt.title(f'Generated Samples ({n_samples} images)')
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return samples


# ==================== Likelihood / NLL Utilities ====================

def standard_normal_logprob(z):
    """Log p(z) under a standard Gaussian, summed over non-batch dims."""
    log_z = -0.5 * np.log(2.0 * np.pi)
    return (log_z - 0.5 * z.pow(2)).reshape(z.shape[0], -1).sum(dim=1)


def dequantize_batch(x):
    """Uniform dequantization matching the training-side image preprocessing."""
    return (x * 255.0 + torch.rand_like(x)) / 256.0


def divergence_hutchinson(dx, x, n_probe=1):
    """Estimate div f(x) with Hutchinson's trace estimator."""
    div = 0.0
    for i in range(n_probe):
        e = torch.randn_like(x)
        grad = torch.autograd.grad(
            outputs=(dx * e).reshape(dx.shape[0], -1).sum(),
            inputs=x,
            create_graph=False,
            retain_graph=(i < n_probe - 1),
            only_inputs=True,
        )[0]
        div = div + (grad * e).reshape(x.shape[0], -1).sum(dim=1)
    return div / float(n_probe)


def _call_ffjord_with_logp(model, x, logpx0):
    """Best-effort FFJORD forward call returning (z, delta_logp)."""
    attempts = [
        lambda: model(x, logpx0),
        lambda: model(x, logpx0, reverse=False),
        lambda: model(x, logpx=logpx0),
        lambda: model(x, logpx=logpx0, reverse=False),
    ]
    last_err = None
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return out[0], out[1]
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not call FFJORD model with log-density output. Last error: {last_err}")


class FlowWithLogp(nn.Module):
    """
    Wrap a velocity-field model for joint state/log-density integration:
        dx/dt    = f(t, x)
        dlogp/dt = -div_x f(t, x)
    """
    def __init__(self, model, n_probe=1):
        super().__init__()
        self.model = model
        self.n_probe = n_probe

    def forward(self, t, states):
        x, logp = states

        with torch.enable_grad():
            x = x.requires_grad_(True)

            if not torch.is_tensor(t):
                t_batch = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
            elif t.ndim == 0:
                t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            elif t.ndim == 1 and t.shape[0] == 1:
                t_batch = t.expand(x.shape[0]).to(device=x.device, dtype=x.dtype)
            else:
                t_batch = t.to(device=x.device, dtype=x.dtype)

            dx = self.model(t_batch, x)
            div = divergence_hutchinson(dx, x, n_probe=self.n_probe)

        return dx, -div


def compute_ffjord_nll(model, test_loader_raw, max_batches=None):
    """Exact NLL for FFJORD teacher checkpoints."""
    model.eval()

    nll_list = []
    bpd_list = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader_raw):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device)
            x = dequantize_batch(x)

            logpx0 = torch.zeros(x.shape[0], 1, device=device)
            z, delta_logp = _call_ffjord_with_logp(model, x, logpx0)

            logpz = standard_normal_logprob(z)
            delta_logp = delta_logp.reshape(x.shape[0], -1).sum(dim=1)
            logpx = logpz - delta_logp

            nll = -logpx
            dims = float(np.prod(x.shape[1:]))
            bpd = nll / (dims * np.log(2.0))

            nll_list.append(nll.detach().cpu())
            bpd_list.append(bpd.detach().cpu())

    nll_all = torch.cat(nll_list)
    bpd_all = torch.cat(bpd_list)

    return {
        "nll": nll_all.mean().item(),
        "nll_std": nll_all.std(unbiased=False).item(),
        "bpd": bpd_all.mean().item(),
        "bpd_std": bpd_all.std(unbiased=False).item(),
        "num_examples": int(nll_all.numel()),
        "mode": "exact",
    }


def compute_flow_nll(model, test_loader_raw, steps=25, method="rk4", n_probe=1, max_batches=None):
    """Approximate NLL for RF / Student / standalone CNF-style velocity fields."""
    model.eval()
    flow_with_logp = FlowWithLogp(model, n_probe=n_probe).to(device)

    nll_list = []
    bpd_list = []

    t_span = torch.tensor([1.0, 0.0], device=device)
    ode_kwargs = {"method": method}
    if method in ["euler", "midpoint", "rk4"]:
        ode_kwargs["options"] = dict(step_size=1.0 / max(steps - 1, 1))

    for batch_idx, (x, _) in enumerate(test_loader_raw):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        x = dequantize_batch(x)
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)

        y = LOGIT_TRANSFORM.forward(x)
        logdet = LOGIT_TRANSFORM.logdet(x)

        logp_terminal = torch.zeros(x.shape[0], device=device)

        y_traj, logp_traj = odeint(
            flow_with_logp,
            (y, logp_terminal),
            t_span,
            **ode_kwargs,
        )

        z0 = y_traj[-1]
        delta_back = logp_traj[-1]

        logpz = standard_normal_logprob(z0)
        logpy = logpz - delta_back
        logpx = logpy + logdet

        nll = -logpx
        dims = float(np.prod(x.shape[1:]))
        bpd = nll / (dims * np.log(2.0))

        nll_list.append(nll.detach().cpu())
        bpd_list.append(bpd.detach().cpu())

        if torch.cuda.is_available() and (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

    nll_all = torch.cat(nll_list)
    bpd_all = torch.cat(bpd_list)

    return {
        "nll": nll_all.mean().item(),
        "nll_std": nll_all.std(unbiased=False).item(),
        "bpd": bpd_all.mean().item(),
        "bpd_std": bpd_all.std(unbiased=False).item(),
        "num_examples": int(nll_all.numel()),
        "mode": "approx_hutchinson",
    }


def compute_model_nll(model, test_loader_raw, steps=25, method="rk4", n_probe=1, max_batches=None):
    """Dispatch NLL computation based on model type."""
    if is_ffjord_model(model):
        return compute_ffjord_nll(model, test_loader_raw=test_loader_raw, max_batches=max_batches)
    return compute_flow_nll(
        model,
        test_loader_raw=test_loader_raw,
        steps=steps,
        method=method,
        n_probe=n_probe,
        max_batches=max_batches,
    )


def evaluate_model(
    model,
    test_loader_raw,
    model_name="Model",
    n_samples=1000,
    steps=25,
    eval_nll=False,
    nll_steps=25,
    nll_method="rk4",
    nll_hutchinson_samples=1,
    nll_max_batches=None,
):
    """Evaluate sampling quality, runtime, and optional NLL/BPD."""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")

    is_ffjord = is_ffjord_model(model)
    if is_ffjord:
        print("(FFJORD model - using native sampling)")

    real_images = []
    for _, (imgs, _) in enumerate(test_loader_raw):
        real_images.append(imgs)
        if len(real_images) * imgs.shape[0] >= n_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:n_samples].to(device)

    print(f"\nReal data statistics (image space [0,1]):")
    print(f"  min: {real_images.min().item():.3f}, max: {real_images.max().item():.3f}")
    print(f"  mean: {real_images.mean().item():.3f}, std: {real_images.std().item():.3f}")

    print(f"\nGenerating {n_samples} samples...")
    start_time = time.time()

    if is_ffjord:
        generated_samples = sample_from_ffjord(model, n_samples=n_samples)
        actual_steps = "N/A (FFJORD)"
    else:
        generated_samples = sample_from_model(model, n_samples=n_samples, steps=steps)
        actual_steps = steps

    sampling_time = time.time() - start_time
    print(f"Sampling time: {sampling_time:.2f}s ({sampling_time/n_samples*1000:.2f}ms per image)")

    print(f"\nGenerated data statistics (image space [0,1]):")
    print(f"  min: {generated_samples.min().item():.3f}, max: {generated_samples.max().item():.3f}")
    print(f"  mean: {generated_samples.mean().item():.3f}, std: {generated_samples.std().item():.3f}")

    mean_diff = abs(generated_samples.mean().item() - real_images.mean().item())
    std_diff = abs(generated_samples.std().item() - real_images.std().item())
    print(f"\nDistribution comparison:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std difference: {std_diff:.4f}")

    fid_score = calculate_fid_simplified(real_images, generated_samples)
    print(f"FID Score (simplified): {fid_score:.2f}")

    nll_results = None
    if eval_nll:
        print(f"\nComputing NLL for {model_name}...")
        nll_results = compute_model_nll(
            model,
            test_loader_raw=test_loader_raw,
            steps=nll_steps,
            method=nll_method,
            n_probe=nll_hutchinson_samples,
            max_batches=nll_max_batches,
        )
        print(f"  NLL ({nll_results['mode']}): {nll_results['nll']:.4f} ± {nll_results['nll_std']:.4f} nats/image")
        print(f"  BPD: {nll_results['bpd']:.4f} ± {nll_results['bpd_std']:.4f}")
        print(f"  Examples used: {nll_results['num_examples']}")

    vis_path = f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_samples.png"
    visualize_samples(model, n_samples=64, save_path=vis_path, steps=steps)
    print(f"Samples saved to {vis_path}")

    return {
        'fid': fid_score,
        'sampling_time': sampling_time,
        'time_per_image': sampling_time / n_samples,
        'steps': actual_steps,
        'nll': None if nll_results is None else nll_results['nll'],
        'nll_std': None if nll_results is None else nll_results['nll_std'],
        'bpd': None if nll_results is None else nll_results['bpd'],
        'bpd_std': None if nll_results is None else nll_results['bpd_std'],
        'nll_mode': None if nll_results is None else nll_results['mode'],
        'nll_examples': None if nll_results is None else nll_results['num_examples'],
    }


def _build_ode_kwargs(method: str, steps: int) -> Dict[str, Any]:
    ode_kwargs: Dict[str, Any] = {"method": method}
    if method in ["euler", "midpoint", "rk4"]:
        ode_kwargs["options"] = dict(step_size=1.0 / max(steps - 1, 1))
    return ode_kwargs


def sample_from_ffjord(model, n_samples=64, data_shape=None, data_spec: Optional[DataSpec] = None):
    """Override: support both image and peptide FFJORD sampling."""
    model.eval()
    if data_shape is None:
        data_shape = getattr(model, "_data_shape", DATA_SHAPE)
    with torch.no_grad():
        z = torch.randn(n_samples, *data_shape).to(device)
        x = model(z, reverse=True)
        if data_spec is not None and data_spec.is_image:
            x = torch.clamp(x, 0, 1)
    return x


def sample_from_model_state(model, n_samples=64, data_spec: Optional[DataSpec] = None, method="rk4", steps=25):
    """Sample model state-space outputs."""
    if is_ffjord_model(model):
        return sample_from_ffjord(model, n_samples, data_spec=data_spec)

    model.eval()
    with torch.no_grad():
        z0 = torch.randn(n_samples, *getattr(model, '_data_shape', DATA_SHAPE)).to(device)
        t_vals = torch.linspace(0.0, 1.0, steps).to(device)
        trajectory = odeint(model, z0, t_vals, **_build_ode_kwargs(method, steps))
        return trajectory[-1]


def sample_from_model(model, n_samples=64, data_spec: Optional[DataSpec] = None, method="rk4", steps=25):
    """Override: sample in observation space for images and peptides."""
    states = sample_from_model_state(model, n_samples=n_samples, data_spec=data_spec, method=method, steps=steps)
    if data_spec is not None and data_spec.is_image and not is_ffjord_model(model):
        states = LOGIT_TRANSFORM.inverse(states)
        states = torch.clamp(states, 0, 1)
    return states


def sample_from_model_raw(model, n_samples=64, data_spec: Optional[DataSpec] = None, method="rk4", steps=25):
    return sample_from_model_state(model, n_samples=n_samples, data_spec=data_spec, method=method, steps=steps)


def _torsion_angles(coords: np.ndarray, torsion_atom_indices: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    torsions = []
    for a, b, c, d in torsion_atom_indices:
        p0 = coords[:, a]
        p1 = coords[:, b]
        p2 = coords[:, c]
        p3 = coords[:, d]

        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / np.linalg.norm(b1, axis=1, keepdims=True).clip(min=1e-8)
        v = b0 - (b0 * b1_norm).sum(axis=1, keepdims=True) * b1_norm
        w = b2 - (b2 * b1_norm).sum(axis=1, keepdims=True) * b1_norm

        x = (v * w).sum(axis=1)
        y = (np.cross(b1_norm, v) * w).sum(axis=1)
        torsions.append(np.arctan2(y, x))

    return np.stack(torsions, axis=1).astype(np.float32)


def wrap_to_pi(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def torus_squared_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = wrap_to_pi(x[:, None, :] - y[None, :, :])
    return np.sum(diff ** 2, axis=-1, dtype=np.float64)


def normalize_log_weights(log_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    shift = log_weights.max()
    weights = np.exp(log_weights - shift)
    weights = weights / weights.sum()
    return weights, log_weights - shift


def compute_kish_ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return 1.0 / np.sum(np.square(weights))


def _weighted_choice(weights: np.ndarray, size: int, replace: bool = True) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(42)
    return rng.choice(len(weights), size=size, replace=replace, p=weights / weights.sum())


def sinkhorn2_uniform(cost_matrix: np.ndarray, reg: float = 0.05, n_iters: int = 200) -> float:
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    n, m = cost_matrix.shape
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)
    K = np.exp(-cost_matrix / max(reg, 1e-6))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iters):
        Kv = np.clip(K @ v, 1e-12, None)
        u = a / Kv
        KTu = np.clip(K.T @ u, 1e-12, None)
        v = b / KTu
    transport = (u[:, None] * K) * v[None, :]
    return float(np.sum(transport * cost_matrix))


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
    }


class OpenMMPeptideMetricBackend:
    """Compute reduced energies from Amber topologies and torsions from coordinates."""
    GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324

    def __init__(self, bundle: PeptideBundle, platform: str = "auto"):
        self.bundle = bundle
        try:
            import openmm
            from openmm import app, unit
        except ImportError:
            try:
                import simtk.openmm as openmm  # type: ignore
                from simtk.openmm import app, unit  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Peptide metrics require OpenMM. Install openmm to enable ESS/E-W1/T-W2 evaluation."
                ) from exc

        self.openmm = openmm
        self.app = app
        self.unit = unit
        self.platform = self._select_platform(platform)
        self.prmtop = None
        self.pdb = None
        if bundle.prmtop_path is not None:
            self.prmtop = app.AmberPrmtopFile(str(bundle.prmtop_path))
            self.system = self.prmtop.createSystem(nonbondedMethod=app.NoCutoff)
        elif bundle.topology_pdb_path is not None:
            if not bundle.forcefield_files:
                raise ValueError(
                    "metadata.json must include 'forcefield_files' when using topology_pdb for peptide metrics."
                )
            self.pdb = app.PDBFile(str(bundle.topology_pdb_path))
            forcefield = app.ForceField(*bundle.forcefield_files)
            nonbonded_method = getattr(app, bundle.nonbonded_method)
            create_kwargs = {
                "nonbondedMethod": nonbonded_method,
                "constraints": None,
            }
            if bundle.nonbonded_cutoff_nm is not None:
                create_kwargs["nonbondedCutoff"] = bundle.nonbonded_cutoff_nm * unit.nanometer
            self.system = forcefield.createSystem(self.pdb.topology, **create_kwargs)
        else:
            raise ValueError("Peptide bundle must define either amber_prmtop or topology_pdb")
        self.integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        if self.platform is None:
            self.context = openmm.Context(self.system, self.integrator)
        else:
            self.context = openmm.Context(self.system, self.integrator, self.platform)
        unit_name = bundle.coordinate_unit.lower()
        self.coord_unit = unit.angstrom if unit_name.startswith("ang") else unit.nanometer
        self.beta = 1.0 / (self.GAS_CONSTANT_KJ_PER_MOL_K * bundle.temperature_kelvin)

    def _select_platform(self, platform_name: str):
        openmm = self.openmm
        if platform_name == "auto":
            for candidate in ("CUDA", "OpenCL", "CPU", "Reference"):
                try:
                    return openmm.Platform.getPlatformByName(candidate)
                except Exception:
                    continue
            return None
        mapping = {"cuda": "CUDA", "cpu": "CPU", "reference": "Reference", "opencl": "OpenCL"}
        return openmm.Platform.getPlatformByName(mapping.get(platform_name.lower(), platform_name))

    def compute_reduced_energies(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        energies = []
        for frame in coords:
            positions = self.unit.Quantity(frame, self.coord_unit)
            self.context.setPositions(positions)
            state = self.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole)
            energies.append(float(energy) * self.beta)
        return np.asarray(energies, dtype=np.float64)

    def compute_torsions(self, coords: np.ndarray) -> np.ndarray:
        return _torsion_angles(coords, self.bundle.torsion_atom_indices)


def _load_or_compute_npz(cache_path: Path, compute_fn):
    if cache_path.exists():
        with np.load(cache_path) as data:
            return {key: data[key] for key in data.files}
    values = compute_fn()
    ensure_parent_dir(str(cache_path))
    np.savez(cache_path, **values)
    return values


def _reference_observables(bundle: PeptideBundle, backend: OpenMMPeptideMetricBackend) -> Dict[str, np.ndarray]:
    cache_path = bundle.cache_dir / f"{bundle.data_name}_reference_metrics.npz"
    return _load_or_compute_npz(
        cache_path,
        lambda: {
            "reduced_energies": backend.compute_reduced_energies(bundle.test_coords),
            "torsions": backend.compute_torsions(bundle.test_coords),
        },
    )


def _proposal_observables(
    coords: np.ndarray,
    backend: OpenMMPeptideMetricBackend,
    cache_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    def compute():
        return {
            "reduced_energies": backend.compute_reduced_energies(coords),
            "torsions": backend.compute_torsions(coords),
        }

    if cache_path is None:
        return compute()
    return _load_or_compute_npz(cache_path, compute)


def estimate_torus_w2(
    proposal_torsions: np.ndarray,
    reference_torsions: np.ndarray,
    proposal_weights: np.ndarray,
    subsample_size: int,
) -> float:
    subsample_size = min(subsample_size, len(reference_torsions), len(proposal_torsions))
    proposal_idx = _weighted_choice(proposal_weights, size=subsample_size, replace=True)
    rng = np.random.default_rng(42)
    reference_idx = rng.choice(len(reference_torsions), size=subsample_size, replace=False)
    proposal_sub = proposal_torsions[proposal_idx]
    reference_sub = reference_torsions[reference_idx]
    cost_matrix = torus_squared_cost_matrix(proposal_sub, reference_sub)
    try:
        import ot  # type: ignore

        a = np.full(subsample_size, 1.0 / subsample_size, dtype=np.float64)
        b = np.full(subsample_size, 1.0 / subsample_size, dtype=np.float64)
        sinkhorn_val = ot.sinkhorn2(a, b, cost_matrix, reg=0.05)
        if isinstance(sinkhorn_val, tuple):
            sinkhorn_val = sinkhorn_val[0]
        return float(sinkhorn_val)
    except Exception as exc:
        warnings.warn(f"POT not available for T-W2; using internal Sinkhorn fallback ({exc})")
        return sinkhorn2_uniform(cost_matrix, reg=0.05, n_iters=200)


def visualize_peptide_metrics(
    artifact_prefix: str,
    reference_energies: np.ndarray,
    proposal_energies: np.ndarray,
    proposal_weights: np.ndarray,
    reference_torsions: np.ndarray,
    proposal_torsions: np.ndarray,
):
    energy_path = f"{artifact_prefix}_energy_hist.png"
    plt.figure(figsize=(10, 6))
    plt.hist(reference_energies, bins=60, alpha=0.45, density=True, label="Reference")
    plt.hist(proposal_energies, bins=60, alpha=0.35, density=True, label="Proposal")
    plt.hist(proposal_energies, bins=60, weights=proposal_weights, alpha=0.45, density=True, label="Reweighted")
    plt.xlabel("Reduced Energy")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(energy_path, dpi=150)
    plt.close()

    torsion_pairs = max(1, math.ceil(reference_torsions.shape[1] / 2))
    fig, axes = plt.subplots(1, torsion_pairs, figsize=(6 * torsion_pairs, 5), squeeze=False)
    for pair_idx in range(torsion_pairs):
        ax = axes[0, pair_idx]
        x_idx = pair_idx * 2
        y_idx = min(x_idx + 1, reference_torsions.shape[1] - 1)
        ax.scatter(reference_torsions[:, x_idx], reference_torsions[:, y_idx], s=3, alpha=0.12, label="Reference")
        proposal_idx = _weighted_choice(proposal_weights, size=min(5000, len(proposal_torsions)), replace=True)
        ax.scatter(
            proposal_torsions[proposal_idx, x_idx],
            proposal_torsions[proposal_idx, y_idx],
            s=3,
            alpha=0.12,
            label="Reweighted",
        )
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(f"Torsion {x_idx}")
        ax.set_ylabel(f"Torsion {y_idx}")
        ax.set_title(f"Ramachandran Pair {pair_idx + 1}")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    rama_path = f"{artifact_prefix}_ramachandran.png"
    fig.savefig(rama_path, dpi=150)
    plt.close(fig)
    return {"energy_hist": energy_path, "ramachandran": rama_path}


def compute_ffjord_nll(model, test_loader_raw, data_spec: DataSpec, max_batches=None):
    """Override: exact NLL for FFJORD on either images or vectors."""
    model.eval()
    nll_list = []
    bpd_list = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader_raw):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device)
            if data_spec.is_image:
                x = dequantize_batch(x)

            logpx0 = torch.zeros(x.shape[0], 1, device=device)
            z, delta_logp = _call_ffjord_with_logp(model, x, logpx0)
            logpz = standard_normal_logprob(z)
            delta_logp = delta_logp.reshape(x.shape[0], -1).sum(dim=1)
            logpx = logpz - delta_logp
            nll = -logpx
            nll_list.append(nll.detach().cpu())

            if data_spec.is_image:
                dims = float(np.prod(x.shape[1:]))
                bpd_list.append((nll / (dims * np.log(2.0))).detach().cpu())

    nll_all = torch.cat(nll_list)
    return {
        "nll": nll_all.mean().item(),
        "nll_std": nll_all.std(unbiased=False).item(),
        "bpd": None if not bpd_list else torch.cat(bpd_list).mean().item(),
        "bpd_std": None if not bpd_list else torch.cat(bpd_list).std(unbiased=False).item(),
        "num_examples": int(nll_all.numel()),
        "mode": "exact",
    }


def compute_flow_nll(model, test_loader_raw, data_spec: DataSpec, steps=25, method="rk4", n_probe=1, max_batches=None):
    """Override: approximate NLL for RF / Student on either images or vectors."""
    model.eval()
    flow_with_logp = FlowWithLogp(model, n_probe=n_probe).to(device)
    nll_list = []
    bpd_list = []
    t_span = torch.tensor([1.0, 0.0], device=device)
    ode_kwargs = _build_ode_kwargs(method, steps)

    for batch_idx, (x, _) in enumerate(test_loader_raw):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        if data_spec.is_image:
            x = dequantize_batch(x)
            x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
            y = LOGIT_TRANSFORM.forward(x)
            logdet = LOGIT_TRANSFORM.logdet(x)
        else:
            y = x
            logdet = torch.zeros(x.shape[0], device=device)

        logp_terminal = torch.zeros(x.shape[0], device=device)
        y_traj, logp_traj = odeint(flow_with_logp, (y, logp_terminal), t_span, **ode_kwargs)
        z0 = y_traj[-1]
        delta_back = logp_traj[-1]

        logpz = standard_normal_logprob(z0)
        logpx = (logpz - delta_back) + logdet
        nll = -logpx
        nll_list.append(nll.detach().cpu())

        if data_spec.is_image:
            dims = float(np.prod(x.shape[1:]))
            bpd_list.append((nll / (dims * np.log(2.0))).detach().cpu())

        if torch.cuda.is_available() and (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

    nll_all = torch.cat(nll_list)
    return {
        "nll": nll_all.mean().item(),
        "nll_std": nll_all.std(unbiased=False).item(),
        "bpd": None if not bpd_list else torch.cat(bpd_list).mean().item(),
        "bpd_std": None if not bpd_list else torch.cat(bpd_list).std(unbiased=False).item(),
        "num_examples": int(nll_all.numel()),
        "mode": "approx_hutchinson",
    }


def compute_model_nll(model, test_loader_raw, data_spec: DataSpec, steps=25, method="rk4", n_probe=1, max_batches=None):
    if is_ffjord_model(model):
        return compute_ffjord_nll(model, test_loader_raw=test_loader_raw, data_spec=data_spec, max_batches=max_batches)
    return compute_flow_nll(
        model,
        test_loader_raw=test_loader_raw,
        data_spec=data_spec,
        steps=steps,
        method=method,
        n_probe=n_probe,
        max_batches=max_batches,
    )


def sample_with_logq(
    model,
    n_samples: int,
    data_spec: DataSpec,
    steps: int = 25,
    method: str = "rk4",
    n_probe: int = 1,
    batch_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample states together with log q(x) in state space."""
    states: List[torch.Tensor] = []
    logq_values: List[torch.Tensor] = []

    if is_ffjord_model(model):
        for start in range(0, n_samples, batch_size):
            current = min(batch_size, n_samples - start)
            x = sample_from_ffjord(model, current, data_spec=data_spec).to(device)
            logpx0 = torch.zeros(x.shape[0], 1, device=device)
            z, delta_logp = _call_ffjord_with_logp(model, x, logpx0)
            logq = standard_normal_logprob(z) - delta_logp.reshape(x.shape[0], -1).sum(dim=1)
            states.append(x.detach().cpu())
            logq_values.append(logq.detach().cpu())
        return torch.cat(states, dim=0), torch.cat(logq_values, dim=0)

    flow_with_logp = FlowWithLogp(model, n_probe=n_probe).to(device)
    t_span = torch.tensor([0.0, 1.0], device=device)
    ode_kwargs = _build_ode_kwargs(method, steps)
    for start in range(0, n_samples, batch_size):
        current = min(batch_size, n_samples - start)
        z0 = torch.randn(current, *data_spec.data_shape, device=device)
        logp0 = standard_normal_logprob(z0)
        x_traj, logp_traj = odeint(flow_with_logp, (z0, logp0), t_span, **ode_kwargs)
        states.append(x_traj[-1].detach().cpu())
        logq_values.append(logp_traj[-1].detach().cpu())
    return torch.cat(states, dim=0), torch.cat(logq_values, dim=0)


def evaluate_image_model(
    model,
    data_spec: DataSpec,
    test_loader_raw,
    model_name="Model",
    n_samples=1000,
    steps=25,
    eval_nll=False,
    nll_steps=25,
    nll_method="rk4",
    nll_hutchinson_samples=1,
    nll_max_batches=None,
    artifact_prefix: Optional[str] = None,
):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")

    is_ffjord = is_ffjord_model(model)
    if is_ffjord:
        print("(FFJORD model - using native sampling)")

    real_images = []
    for _, (imgs, _) in enumerate(test_loader_raw):
        real_images.append(imgs)
        if len(real_images) * imgs.shape[0] >= n_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:n_samples].to(device)

    real_stats = _tensor_stats(real_images)
    print(f"\nReal data statistics (image space [0,1]):")
    print(f"  min: {real_stats['min']:.3f}, max: {real_stats['max']:.3f}")
    print(f"  mean: {real_stats['mean']:.3f}, std: {real_stats['std']:.3f}")

    print(f"\nGenerating {n_samples} samples...")
    start_time = time.time()
    generated_samples = sample_from_model(model, n_samples=n_samples, data_spec=data_spec, steps=steps)
    actual_steps = "N/A (FFJORD)" if is_ffjord else steps
    sampling_time = time.time() - start_time
    print(f"Sampling time: {sampling_time:.2f}s ({sampling_time/n_samples*1000:.2f}ms per image)")

    gen_stats = _tensor_stats(generated_samples)
    print(f"\nGenerated data statistics (image space [0,1]):")
    print(f"  min: {gen_stats['min']:.3f}, max: {gen_stats['max']:.3f}")
    print(f"  mean: {gen_stats['mean']:.3f}, std: {gen_stats['std']:.3f}")

    mean_diff = abs(gen_stats['mean'] - real_stats['mean'])
    std_diff = abs(gen_stats['std'] - real_stats['std'])
    print(f"\nDistribution comparison:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std difference: {std_diff:.4f}")

    fid_score = calculate_fid_simplified(real_images, generated_samples)
    print(f"FID Score (simplified): {fid_score:.2f}")

    nll_results = None
    if eval_nll:
        print(f"\nComputing NLL for {model_name}...")
        nll_results = compute_model_nll(
            model,
            test_loader_raw=test_loader_raw,
            data_spec=data_spec,
            steps=nll_steps,
            method=nll_method,
            n_probe=nll_hutchinson_samples,
            max_batches=nll_max_batches,
        )
        print(f"  NLL ({nll_results['mode']}): {nll_results['nll']:.4f} ± {nll_results['nll_std']:.4f} nats/image")
        if nll_results["bpd"] is not None:
            print(f"  BPD: {nll_results['bpd']:.4f} ± {nll_results['bpd_std']:.4f}")
        print(f"  Examples used: {nll_results['num_examples']}")

    vis_path = f"{artifact_prefix or _safe_model_name(model_name)}_samples.png"
    visualize_image_samples(model, data_spec, n_samples=64, save_path=vis_path, steps=steps)
    print(f"Samples saved to {vis_path}")

    return {
        'fid': fid_score,
        'sampling_time': sampling_time,
        'time_per_image': sampling_time / n_samples,
        'steps': actual_steps,
        'nll': None if nll_results is None else nll_results['nll'],
        'nll_std': None if nll_results is None else nll_results['nll_std'],
        'bpd': None if nll_results is None else nll_results['bpd'],
        'bpd_std': None if nll_results is None else nll_results['bpd_std'],
        'nll_mode': None if nll_results is None else nll_results['mode'],
        'nll_examples': None if nll_results is None else nll_results['num_examples'],
        'artifacts': {'samples': vis_path},
    }


def evaluate_tabular_model(
    model,
    data_spec: DataSpec,
    test_loader_raw,
    model_name="Model",
    n_samples=5000,
    steps=25,
    eval_nll=False,
    nll_steps=25,
    nll_method="rk4",
    nll_hutchinson_samples=1,
    nll_max_batches=None,
):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} (tabular)")
    print(f"{'='*50}")

    real_batches = []
    total = 0
    for _, (batch, _) in enumerate(test_loader_raw):
        real_batches.append(batch)
        total += batch.shape[0]
        if total >= n_samples:
            break
    real_data = torch.cat(real_batches, dim=0)[:n_samples].to(device)

    print(f"Using {real_data.shape[0]} held-out examples with input_dim={real_data.shape[1]}")
    print(f"Generating {real_data.shape[0]} samples...")
    start_time = time.time()
    generated_samples = sample_from_model(
        model,
        n_samples=real_data.shape[0],
        data_spec=data_spec,
        steps=steps,
    )
    sampling_time = time.time() - start_time
    actual_steps = "N/A (FFJORD)" if is_ffjord_model(model) else steps

    real_mean = real_data.mean(dim=0)
    real_std = real_data.std(dim=0, unbiased=False)
    gen_mean = generated_samples.mean(dim=0)
    gen_std = generated_samples.std(dim=0, unbiased=False)
    mean_l1 = (gen_mean - real_mean).abs().mean().item()
    std_l1 = (gen_std - real_std).abs().mean().item()

    print(f"Sampling time: {sampling_time:.2f}s ({sampling_time/real_data.shape[0]*1000:.2f}ms per sample)")
    print("Marginal moments:")
    print(f"  mean(real): {real_mean.mean().item():.4f} | mean(gen): {gen_mean.mean().item():.4f}")
    print(f"  std(real):  {real_std.mean().item():.4f} | std(gen):  {gen_std.mean().item():.4f}")
    print(f"  mean L1 gap across features: {mean_l1:.4f}")
    print(f"  std  L1 gap across features: {std_l1:.4f}")

    nll_results = None
    if eval_nll:
        print(f"\nComputing NLL for {model_name}...")
        nll_results = compute_model_nll(
            model,
            test_loader_raw=test_loader_raw,
            data_spec=data_spec,
            steps=nll_steps,
            method=nll_method,
            n_probe=nll_hutchinson_samples,
            max_batches=nll_max_batches,
        )
        print(f"  NLL ({nll_results['mode']}): {nll_results['nll']:.4f} ± {nll_results['nll_std']:.4f} nats/sample")
        print(f"  Examples used: {nll_results['num_examples']}")

    return {
        'fid': None,
        'sampling_time': sampling_time,
        'time_per_image': sampling_time / real_data.shape[0],
        'steps': actual_steps,
        'nll': None if nll_results is None else nll_results['nll'],
        'nll_std': None if nll_results is None else nll_results['nll_std'],
        'bpd': None,
        'bpd_std': None,
        'nll_mode': None if nll_results is None else nll_results['mode'],
        'nll_examples': None if nll_results is None else nll_results['num_examples'],
        'mean_l1': mean_l1,
        'std_l1': std_l1,
    }


def evaluate_peptide_model(
    model,
    data_spec: DataSpec,
    test_loader_raw,
    model_name="Model",
    steps=25,
    eval_nll=False,
    nll_steps=25,
    nll_method="rk4",
    nll_hutchinson_samples=1,
    nll_max_batches=None,
    metric_samples: int = 250000,
    tw2_subsample: int = 4096,
    openmm_platform: str = "auto",
    artifact_prefix: Optional[str] = None,
    peptide_metric_backend: Optional[OpenMMPeptideMetricBackend] = None,
):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} (peptide)")
    print(f"{'='*50}")
    bundle = data_spec.peptide_bundle
    if bundle is None:
        raise ValueError("Peptide evaluation requires a PeptideBundle")

    backend = peptide_metric_backend or OpenMMPeptideMetricBackend(bundle, platform=openmm_platform)
    artifact_prefix = artifact_prefix or _safe_model_name(model_name)

    reference = _reference_observables(bundle, backend)
    print(f"Loaded reference observables for {len(reference['reduced_energies'])} test samples")

    print(f"Generating {metric_samples} peptide proposal samples...")
    start_time = time.time()
    proposal_states, proposal_logq = sample_with_logq(
        model,
        n_samples=metric_samples,
        data_spec=data_spec,
        steps=steps,
        method=nll_method,
        n_probe=nll_hutchinson_samples,
        batch_size=min(2048, metric_samples),
    )
    sampling_time = time.time() - start_time
    print(f"Sampling time: {sampling_time:.2f}s ({sampling_time / metric_samples * 1000:.3f}ms per sample)")

    proposal_coords = bundle.denormalize_flat(proposal_states.numpy())
    proposal_cache = bundle.cache_dir / f"{artifact_prefix}_{_cache_key([metric_samples, tw2_subsample])}.npz"
    proposal = _proposal_observables(proposal_coords, backend, cache_path=proposal_cache)

    log_weights = -proposal["reduced_energies"] - proposal_logq.numpy()
    proposal_weights, _ = normalize_log_weights(log_weights)
    ess = compute_kish_ess(proposal_weights)
    e_w1 = wasserstein_distance(
        reference["reduced_energies"],
        proposal["reduced_energies"],
        u_weights=np.full(len(reference["reduced_energies"]), 1.0 / len(reference["reduced_energies"]), dtype=np.float64),
        v_weights=proposal_weights,
    )
    t_w2 = estimate_torus_w2(
        proposal_torsions=proposal["torsions"],
        reference_torsions=reference["torsions"],
        proposal_weights=proposal_weights,
        subsample_size=tw2_subsample,
    )

    nll_results = None
    if eval_nll:
        print(f"\nComputing NLL for {model_name}...")
        nll_results = compute_model_nll(
            model,
            test_loader_raw=test_loader_raw,
            data_spec=data_spec,
            steps=nll_steps,
            method=nll_method,
            n_probe=nll_hutchinson_samples,
            max_batches=nll_max_batches,
        )
        print(f"  NLL ({nll_results['mode']}): {nll_results['nll']:.4f} ± {nll_results['nll_std']:.4f} nats/sample")
        print(f"  Examples used: {nll_results['num_examples']}")

    artifact_paths = visualize_peptide_metrics(
        artifact_prefix=artifact_prefix,
        reference_energies=reference["reduced_energies"],
        proposal_energies=proposal["reduced_energies"],
        proposal_weights=proposal_weights,
        reference_torsions=reference["torsions"],
        proposal_torsions=proposal["torsions"],
    )
    print(f"Saved peptide visualizations: {artifact_paths}")

    return {
        'fid': None,
        'sampling_time': sampling_time,
        'time_per_image': sampling_time / metric_samples,
        'steps': "N/A (FFJORD)" if is_ffjord_model(model) else steps,
        'nll': None if nll_results is None else nll_results['nll'],
        'nll_std': None if nll_results is None else nll_results['nll_std'],
        'bpd': None,
        'bpd_std': None,
        'nll_mode': None if nll_results is None else nll_results['mode'],
        'nll_examples': None if nll_results is None else nll_results['num_examples'],
        'ess': ess,
        'ess_ratio': ess / float(metric_samples),
        'e_w1': float(e_w1),
        't_w2': float(t_w2),
        'metric_samples': metric_samples,
        'artifacts': artifact_paths,
    }


def evaluate_model(
    model,
    data_spec: DataSpec,
    test_loader_raw,
    model_name="Model",
    n_samples=1000,
    steps=25,
    eval_nll=False,
    nll_steps=25,
    nll_method="rk4",
    nll_hutchinson_samples=1,
    nll_max_batches=None,
    metric_samples: int = 250000,
    tw2_subsample: int = 4096,
    openmm_platform: str = "auto",
    artifact_prefix: Optional[str] = None,
    peptide_metric_backend: Optional[OpenMMPeptideMetricBackend] = None,
):
    """Override: evaluate models with modality-specific metrics."""
    if data_spec.is_peptide:
        return evaluate_peptide_model(
            model,
            data_spec=data_spec,
            test_loader_raw=test_loader_raw,
            model_name=model_name,
            steps=steps,
            eval_nll=eval_nll,
            nll_steps=nll_steps,
            nll_method=nll_method,
            nll_hutchinson_samples=nll_hutchinson_samples,
            nll_max_batches=nll_max_batches,
            metric_samples=metric_samples,
            tw2_subsample=tw2_subsample,
            openmm_platform=openmm_platform,
            artifact_prefix=artifact_prefix,
            peptide_metric_backend=peptide_metric_backend,
        )
    if data_spec.is_tabular:
        return evaluate_tabular_model(
            model,
            data_spec=data_spec,
            test_loader_raw=test_loader_raw,
            model_name=model_name,
            n_samples=n_samples,
            steps=steps,
            eval_nll=eval_nll,
            nll_steps=nll_steps,
            nll_method=nll_method,
            nll_hutchinson_samples=nll_hutchinson_samples,
            nll_max_batches=nll_max_batches,
        )

    return evaluate_image_model(
        model,
        data_spec=data_spec,
        test_loader_raw=test_loader_raw,
        model_name=model_name,
        n_samples=n_samples,
        steps=steps,
        eval_nll=eval_nll,
        nll_steps=nll_steps,
        nll_method=nll_method,
        nll_hutchinson_samples=nll_hutchinson_samples,
        nll_max_batches=nll_max_batches,
        artifact_prefix=artifact_prefix,
    )


# ==================== CNF Loading Functions ====================

class FFJORDModelWrapper(nn.Module):
    """
    Wrapper to make FFJORD model compatible with torchdiffeq odeint interface.
    
    FFJORD models use: model(x, reverse=True) for sampling
    torchdiffeq expects: model(t, x) -> dx/dt
    
    This wrapper extracts the velocity field from FFJORD's internal structure.
    """
    def __init__(self, ffjord_model):
        super().__init__()
        self.ffjord_model = ffjord_model
        self._extract_odefunc()
    
    def _extract_odefunc(self):
        """Extract the ODE function from FFJORD model structure"""
        # FFJORD uses SequentialFlow with chain of transforms
        # We need to find the CNF blocks and their odefuncs
        self.cnf_blocks = []
        
        def find_cnf_blocks(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                # Check if this is a CNF block
                if hasattr(child, 'odefunc'):
                    self.cnf_blocks.append(child)
                else:
                    find_cnf_blocks(child, full_name)
        
        find_cnf_blocks(self.ffjord_model)
        
        if self.cnf_blocks:
            print(f"Found {len(self.cnf_blocks)} CNF blocks in FFJORD model")
        else:
            print("Warning: No CNF blocks found, using direct model call")
    
    def forward(self, t, x):
        """
        Compute velocity field at time t and state x.
        
        For FFJORD, we use the first CNF block's odefunc.
        """
        if self.cnf_blocks:
            # Use the first CNF block's odefunc
            cnf = self.cnf_blocks[0]
            odefunc = cnf.odefunc
            
            # Set time for odefunc
            if hasattr(odefunc, '_e'):
                # Some implementations store noise vector
                if odefunc._e is None or odefunc._e.shape[0] != x.shape[0]:
                    odefunc._e = torch.randn_like(x)
            
            # Call the diffeq network
            if hasattr(odefunc, 'odefunc') and hasattr(odefunc.odefunc, 'diffeq'):
                # Standard FFJORD structure
                diffeq = odefunc.odefunc.diffeq
                # ConcatSquash layers need (t, x) format
                return diffeq(t, x)
            elif hasattr(odefunc, 'diffeq'):
                return odefunc.diffeq(t, x)
            else:
                # Fallback: try direct call
                return odefunc(t, x, torch.zeros(x.shape[0], device=x.device))[0]
        else:
            # Fallback for non-standard models
            raise RuntimeError("Cannot extract velocity field from FFJORD model")
    
    def sample(self, n_samples, img_shape=(1, 28, 28), steps=25):
        """
        Sample from FFJORD model using its native sampling method.
        """
        self.ffjord_model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, *img_shape).to(device)
            # FFJORD sampling: reverse=True goes from z to x
            x = self.ffjord_model(z, reverse=True)
            return x


def load_ffjord_model(checkpoint_path: str, args_override: dict = None):
    """
    Load FFJORD model from train_cnf.py checkpoint.
    
    Uses train_cnf.py's create_model function directly by importing it.
    
    Args:
        checkpoint_path: Path to checkpoint from train_cnf.py
        args_override: Optional dict to override saved args
    
    Returns:
        Loaded FFJORD model, saved_args
    """
    print(f"\nLoading FFJORD model from: {checkpoint_path}")
    
    # Load checkpoint first to get args
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "args" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'args'. Not a train_cnf.py checkpoint?")
    
    saved_args = checkpoint["args"]
    state_dict = checkpoint["state_dict"]
    
    print(f"Loaded args: data={saved_args.data}, dims={saved_args.dims}")
    print(f"  multiscale={saved_args.multiscale}, parallel={getattr(saved_args, 'parallel', False)}")
    print(f"  num_blocks={saved_args.num_blocks}, layer_type={saved_args.layer_type}")
    
    # Import required modules
    try:
        import lib.layers as layers
        import lib.odenvp as odenvp
        import lib.multiscale_parallel as multiscale_parallel
        from train_misc import set_cnf_options, create_regularization_fns
        print("FFJORD lib modules imported successfully")
    except ImportError as e:
        print(f"Warning: Could not import FFJORD lib modules: {e}")
        raise ImportError(
            "FFJORD lib modules required. Run from ffjord-master directory."
        )
    
    # Determine data shape
    if saved_args.data == "mnist":
        data_shape = (1, 28, 28)
    elif saved_args.data in ["svhn", "cifar10"]:
        data_shape = (3, 32, 32)
    else:
        data_shape = (1, 28, 28)
    
    # Create regularization functions
    regularization_fns, _ = create_regularization_fns(saved_args)
    
    # === Replicate create_model from train_cnf.py EXACTLY ===
    # This is copied from train_cnf.py lines 204-310
    
    hidden_dims = tuple(map(int, saved_args.dims.split(",")))
    strides = tuple(map(int, saved_args.strides.split(",")))
    
    if saved_args.multiscale:
        # ODENVP - multiscale uses ODENVP in train_cnf.py
        model = odenvp.ODENVP(
            (saved_args.batch_size, *data_shape),
            n_blocks=saved_args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=saved_args.nonlinearity,
            alpha=saved_args.alpha,
            cnf_kwargs={"T": saved_args.time_length, "train_T": saved_args.train_T, "regularization_fns": regularization_fns},
        )
    elif saved_args.parallel:
        # MultiscaleParallelCNF - parallel uses this in train_cnf.py
        model = multiscale_parallel.MultiscaleParallelCNF(
            (saved_args.batch_size, *data_shape),
            n_blocks=saved_args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=saved_args.alpha,
            time_length=saved_args.time_length,
        )
    elif saved_args.autoencode:
        def build_cnf():
            autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                hidden_dims=hidden_dims,
                input_shape=data_shape,
                strides=strides,
                conv=saved_args.conv,
                layer_type=saved_args.layer_type,
                nonlinearity=saved_args.nonlinearity,
            )
            odefunc = layers.AutoencoderODEfunc(
                autoencoder_diffeq=autoencoder_diffeq,
                divergence_fn=saved_args.divergence_fn,
                residual=saved_args.residual,
                rademacher=saved_args.rademacher,
            )
            cnf = layers.CNF(
                odefunc=odefunc,
                T=saved_args.time_length,
                regularization_fns=regularization_fns,
                solver=saved_args.solver,
            )
            return cnf
        
        chain = [layers.LogitTransform(alpha=saved_args.alpha)] if saved_args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(saved_args.num_blocks)]
        if saved_args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    else:
        def build_cnf():
            diffeq = layers.ODEnet(
                hidden_dims=hidden_dims,
                input_shape=data_shape,
                strides=strides,
                conv=saved_args.conv,
                layer_type=saved_args.layer_type,
                nonlinearity=saved_args.nonlinearity,
            )
            odefunc = layers.ODEfunc(
                diffeq=diffeq,
                divergence_fn=saved_args.divergence_fn,
                residual=saved_args.residual,
                rademacher=saved_args.rademacher,
            )
            cnf = layers.CNF(
                odefunc=odefunc,
                T=saved_args.time_length,
                train_T=saved_args.train_T,
                regularization_fns=regularization_fns,
                solver=saved_args.solver,
            )
            return cnf
        
        chain = [layers.LogitTransform(alpha=saved_args.alpha)] if saved_args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(saved_args.num_blocks)]
        if saved_args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    
    # Debug: print model structure vs state_dict structure
    print("\nModel state_dict keys (first 5):")
    model_keys = list(model.state_dict().keys())[:5]
    for k in model_keys:
        print(f"  {k}")
    
    print("\nCheckpoint state_dict keys (first 5):")
    ckpt_keys = list(state_dict.keys())[:5]
    for k in ckpt_keys:
        print(f"  {k}")
    
    # Check if there's a structural mismatch
    if model_keys and ckpt_keys and model_keys[0].split('.')[0] != ckpt_keys[0].split('.')[0]:
        print("\nWarning: Possible architecture mismatch detected!")
        print("This may be due to library version differences.")
    
    # Try to load state dict
    try:
        model.load_state_dict(state_dict)
        print("State dict loaded successfully!")
    except RuntimeError as e:
        print(f"\nError loading state dict: {e}")
        print("\nThis usually means the FFJORD library version differs from when the model was trained.")
        print("Options:")
        print("  1. Retrain the CNF with the current library version")
        print("  2. Use a CNF model trained with mnist_cnf_reflow_tiny.py instead")
        raise
    
    model = model.to(device)
    model.eval()

    # Attach shape for generalized sampling/eval
    model._data_shape = data_shape
    
    # Apply CNF options
    set_cnf_options(saved_args, model)
    
    print(f"\nFFJORD model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, saved_args


def load_cnf_standalone(model_path: str, base_channels: int = 48):
    """
    Load CNF model - tries FFJORD format first, then standalone CNFNet.
    
    Args:
        model_path: Path to model weights (state_dict or checkpoint)
        base_channels: Base channel count for standalone CNFNet
    
    Returns:
        Loaded model (either FFJORD or CNFNet)
    """
    print(f"\nLoading CNF from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if this is a FFJORD checkpoint (has 'args' key)
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        print("Detected FFJORD checkpoint format (from train_cnf.py)")
        model, saved_args = load_ffjord_model(model_path)
        return model
    
    # Check state dict keys to determine format
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Check for FFJORD-style keys
    sample_key = list(state_dict.keys())[0] if state_dict else ""
    if "transforms" in sample_key or "odefunc" in sample_key:
        print("Detected FFJORD state_dict format")
        raise ValueError(
            "This appears to be a FFJORD model but without 'args'. "
            "Please use the full checkpoint from train_cnf.py."
        )
    
    # Load as standalone CNFNet
    print("Loading as standalone CNFNet")
    cnf_model = CNFNet(base_channels=base_channels).to(device)
    
    try:
        cnf_model.load_state_dict(state_dict, strict=True)
        print("CNFNet weights loaded successfully")
    except RuntimeError as e:
        print(f"Error loading CNFNet: {e}")
        raise
    
    return cnf_model


def load_tabular_ffjord_model(checkpoint_path: str, input_dim: int):
    """Load a vector FFJORD checkpoint with saved args for tabular or peptide data."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "args" not in checkpoint:
        raise ValueError("Vector FFJORD checkpoint must include saved args")

    saved_args = checkpoint["args"]
    state_dict = checkpoint["state_dict"]

    from train_misc import build_model_tabular, create_regularization_fns, set_cnf_options

    regularization_fns, _ = create_regularization_fns(saved_args)
    model = build_model_tabular(saved_args, input_dim, regularization_fns).to(device)
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'diffeq.diffeq' not in key:
            filtered_state_dict[key.replace('module.', '')] = value
    model.load_state_dict(filtered_state_dict, strict=False)
    set_cnf_options(saved_args, model)
    model._data_shape = (input_dim,)
    model.eval()
    return model, saved_args


def load_teacher_model(model_path: str, data_spec: DataSpec, base_channels: int = 48):
    """Unified teacher loader for image and peptide pipelines."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        saved_args = checkpoint["args"]
        is_tabular = getattr(saved_args, "conv", None) is False or data_spec.uses_vector_backbone
        if is_tabular:
            model, _ = load_tabular_ffjord_model(model_path, data_spec.flat_dim)
        else:
            model, _ = load_ffjord_model(model_path)
        model._data_shape = data_spec.data_shape
        return model

    if data_spec.uses_vector_backbone:
        raise ValueError(
            "Vector teacher checkpoints must be FFJORD checkpoints with saved args, such as train_cnf.py runs with --conv False."
        )

    model = load_cnf_standalone(model_path, base_channels=base_channels)
    model._data_shape = data_spec.data_shape
    return model


# ==================== Main Pipeline ====================

def main(args):
    """Main training and evaluation pipeline"""
    global DATA_SHAPE, LOGIT_TRANSFORM
    
    print("=" * 70)
    print("MNIST Rectified Flow Training Pipeline")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"LogitTransform alpha: {LOGIT_TRANSFORM.alpha}")
    print()
    
    # Configuration summary
    print("Configuration:")
    print(f"  CNF path: {args.cnf_path}")
    print(f"  RF model path: {args.rf_model_path}")
    print(f"  RF checkpoint path: {args.rf_ckpt_path}")
    print(f"  Student model path: {args.student_model_path}")
    print(f"  Student checkpoint path: {args.student_ckpt_path}")
    print(f"  Skip RF: {args.skip_rf}")
    print(f"  Skip Student: {args.skip_student}")
    print(f"  Resume RF: {args.resume_rf}")
    print(f"  Resume Student: {args.resume_student}")
    print(f"  RF epochs: {args.rf_epochs}")
    print(f"  Student epochs: {args.student_epochs}")
    print()
    

    # ========== Resolve Dataset & Load Data ==========
    data_name, data_shape = resolve_data_config(args)
    DATA_SHAPE = data_shape
    LOGIT_TRANSFORM = LogitTransform(alpha=getattr(args, "alpha", 0.05))

    print(f"Dataset: {data_name} | DATA_SHAPE={DATA_SHAPE}")
    print(f"Data root: {getattr(args, 'data_root', './data')}")
    print(f"Using num_workers={DEFAULT_NUM_WORKERS} {'(Windows detected)' if IS_WINDOWS else ''}")

    need_train_data = (not args.eval_only) and (not args.skip_rf or not args.skip_student)
    if need_train_data:
        print(f"Loading {data_name} dataset for training (logit space)...")
        train_loader, test_loader = get_train_loaders_logit(
            data_name=data_name,
            data_root=getattr(args, 'data_root', './data'),
            batch_size=32,
            alpha=LOGIT_TRANSFORM.alpha,
            download=(not getattr(args, 'no_download', False)),
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")

        # Print data stats (logit space - what RF trains on)
        sample_batch, _ = next(iter(train_loader))
        print(f"\nTraining data in logit space:")
        print(f"  min: {sample_batch.min().item():.3f}")
        print(f"  max: {sample_batch.max().item():.3f}")
        print(f"  mean: {sample_batch.mean().item():.3f}")
        print(f"  std: {sample_batch.std().item():.3f}")

        # Also print what it looks like in image space for comparison
        sample_image = LOGIT_TRANSFORM.inverse(sample_batch)
        sample_image = torch.clamp(sample_image, 0, 1)
        print(f"\nTraining data in image space [0,1] (after inverse LogitTransform):")
        print(f"  min: {sample_image.min().item():.3f}")
        print(f"  max: {sample_image.max().item():.3f}")
        print(f"  mean: {sample_image.mean().item():.3f}")
        print(f"  std: {sample_image.std().item():.3f}")
    else:
        train_loader, test_loader = None, None

    print(f"Loading {data_name} raw test data for FID/statistics...")
    test_loader_raw = get_raw_test_loader(
        data_name=data_name,
        data_root=getattr(args, 'data_root', './data'),
        batch_size=64,
        download=(not getattr(args, 'no_download', False)),
    )

    # Print raw test data stats for comparison (what FID will compare against)
    raw_sample, _ = next(iter(test_loader_raw))
    print(f"\nRaw test data in image space [0,1] (ground truth for FID):")
    print(f"  min: {raw_sample.min().item():.3f}")
    print(f"  max: {raw_sample.max().item():.3f}")
    print(f"  mean: {raw_sample.mean().item():.3f}")
    print(f"  std: {raw_sample.std().item():.3f}")
    
    # ========== Load CNF ==========
    print("\n" + "=" * 70)
    print("Loading CNF Model (Teacher)")
    print("=" * 70)
    
    if not os.path.exists(args.cnf_path):
        print(f"Error: CNF model not found at {args.cnf_path}")
        print("Please train CNF first using train_cnf.py or provide valid path")
        return
    
    cnf_model = load_cnf_standalone(args.cnf_path, base_channels=48)
    # Ensure sampling uses the resolved DATA_SHAPE (overrides checkpoint defaults if needed)
    cnf_model._data_shape = DATA_SHAPE
    cnf_model.eval()
    print(f"CNF parameters: {sum(p.numel() for p in cnf_model.parameters()):,}")
    
    # Optionally evaluate CNF
    if args.eval_cnf:
        cnf_results = evaluate_model(
            cnf_model,
            test_loader_raw,
            model_name="CNF_Teacher",
            n_samples=args.n_samples,
            eval_nll=args.eval_nll,
            nll_steps=args.nll_steps,
            nll_method=args.nll_method,
            nll_hutchinson_samples=args.nll_hutchinson_samples,
            nll_max_batches=args.nll_max_batches,
        )
    else:
        cnf_results = None
    
    # ========== Train/Load RF ==========
    print("\n" + "=" * 70)
    print("Stage 1: Rectified Flow (RF)")
    print("=" * 70)
    
    rf_model = RFNet(img_channels=DATA_SHAPE[0], base_channels=64).to(device)
    rf_model._data_shape = DATA_SHAPE
    print(f"RF parameters: {sum(p.numel() for p in rf_model.parameters()):,}")
    
    rf_optimizer = optim.Adam(rf_model.parameters(), lr=args.rf_lr)
    rf_start_epoch = 0
    
    rf_ckpt_exists = os.path.exists(args.rf_ckpt_path)
    rf_model_exists = os.path.exists(args.rf_model_path)
    
    if args.skip_rf:
        # Load existing model
        if rf_model_exists:
            print(f"Loading RF model from {args.rf_model_path}")
            rf_model.load_state_dict(torch.load(args.rf_model_path, map_location=device, weights_only=False))
            print("RF model loaded successfully!")
        elif rf_ckpt_exists:
            print(f"Loading RF from checkpoint {args.rf_ckpt_path}")
            ckpt = torch.load(args.rf_ckpt_path, map_location=device, weights_only=False)
            rf_model.load_state_dict(ckpt["model"])
            print(f"RF loaded (epoch {ckpt.get('epoch', 'unknown')})")
        else:
            print("Warning: skip_rf=True but no RF model found. Training from scratch...")
            args.skip_rf = False
    
    if not args.skip_rf:
        # Resume or train from scratch
        if args.resume_rf and rf_ckpt_exists:
            print(f"Resuming RF from checkpoint: {args.rf_ckpt_path}")
            ckpt = torch.load(args.rf_ckpt_path, map_location=device, weights_only=False)
            rf_model.load_state_dict(ckpt["model"])
            rf_optimizer.load_state_dict(ckpt["optimizer"])
            rf_start_epoch = ckpt["epoch"]
            print(f"Resumed from epoch {rf_start_epoch}")
        elif args.resume_rf and rf_model_exists:
            print(f"Loading RF weights from {args.rf_model_path}")
            rf_model.load_state_dict(torch.load(args.rf_model_path, map_location=device, weights_only=False))
        
        print(f"\nTraining RF for {args.rf_epochs} epochs...")
        rf_model, rf_optimizer, rf_total_epochs = train_rf(
            rf_model,
            train_loader,
            epochs=args.rf_epochs,
            lr=args.rf_lr,
            save_interval=100,
            optimizer=rf_optimizer,
            start_epoch=rf_start_epoch,
            ckpt_path=args.rf_ckpt_path,
            model_path=args.rf_model_path,
        )
        print(f"RF training complete. Total epochs: {rf_total_epochs}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Evaluate RF
    rf_results = evaluate_model(
        rf_model,
        test_loader_raw,
        model_name="RF_Logit",
        n_samples=args.n_samples,
        eval_nll=args.eval_nll,
        nll_steps=args.nll_steps,
        nll_method=args.nll_method,
        nll_hutchinson_samples=args.nll_hutchinson_samples,
        nll_max_batches=args.nll_max_batches,
    )
    
    # ========== Train/Load RF Student (CNF-Reflow) ==========
    print("\n" + "=" * 70)
    print("Stage 2: RF Student via CNF-Reflow Distillation")
    print("=" * 70)
    
    student_model = RFStudent(img_channels=DATA_SHAPE[0]).to(device)
    student_model._data_shape = DATA_SHAPE
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    student_optimizer = optim.Adam(student_model.parameters(), lr=args.student_lr)
    student_start_epoch = 0
    
    student_ckpt_exists = os.path.exists(args.student_ckpt_path)
    student_model_exists = os.path.exists(args.student_model_path)
    
    if args.skip_student:
        # Load existing model
        if student_model_exists:
            print(f"Loading Student model from {args.student_model_path}")
            student_model.load_state_dict(torch.load(args.student_model_path, map_location=device, weights_only=False))
            print("Student model loaded successfully!")
        elif student_ckpt_exists:
            print(f"Loading Student from checkpoint {args.student_ckpt_path}")
            ckpt = torch.load(args.student_ckpt_path, map_location=device, weights_only=False)
            student_model.load_state_dict(ckpt["model"])
            print(f"Student loaded (epoch {ckpt.get('epoch', 'unknown')})")
        else:
            print("Warning: skip_student=True but no Student model found. Training from scratch...")
            args.skip_student = False
    
    if not args.skip_student:
        # Resume or train from scratch
        if args.resume_student and student_ckpt_exists:
            print(f"Resuming Student from checkpoint: {args.student_ckpt_path}")
            ckpt = torch.load(args.student_ckpt_path, map_location=device, weights_only=False)
            student_model.load_state_dict(ckpt["model"])
            student_optimizer.load_state_dict(ckpt["optimizer"])
            student_start_epoch = ckpt["epoch"]
            print(f"Resumed from epoch {student_start_epoch}")
        elif args.resume_student and student_model_exists:
            print(f"Loading Student weights from {args.student_model_path}")
            student_model.load_state_dict(torch.load(args.student_model_path, map_location=device, weights_only=False))
        
        print(f"\nTraining Student for {args.student_epochs} epochs...")
        student_model, student_optimizer, student_total_epochs = train_rf_student(
            cnf_model,  # Use CNF as teacher
            train_loader,
            epochs=args.student_epochs,
            batch_size=args.student_batch_size,
            num_steps=args.student_num_steps,
            lr=args.student_lr,
            save_interval=100,
            optimizer=student_optimizer,
            start_epoch=student_start_epoch,
            ckpt_path=args.student_ckpt_path,
            model_path=args.student_model_path,
            student_model=student_model,
        )
        print(f"Student training complete. Total epochs: {student_total_epochs}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Evaluate Student
    student_results = evaluate_model(
        student_model,
        test_loader_raw,
        model_name="RF_Student_Logit",
        n_samples=args.n_samples,
        eval_nll=args.eval_nll,
        nll_steps=args.nll_steps,
        nll_method=args.nll_method,
        nll_hutchinson_samples=args.nll_hutchinson_samples,
        nll_max_batches=args.nll_max_batches,
    )
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    def _fmt_metric(x, digits=2):
        if x is None:
            return "N/A"
        return f"{x:.{digits}f}"

    results_table = [
        ["Model", "FID Score", "NLL", "BPD", "Time/Image (ms)", "ODE Steps"],
    ]

    if cnf_results:
        results_table.append([
            "CNF (Teacher)",
            _fmt_metric(cnf_results['fid'], 2),
            _fmt_metric(cnf_results['nll'], 4),
            _fmt_metric(cnf_results['bpd'], 4),
            _fmt_metric(cnf_results['time_per_image'] * 1000, 2),
            f"{cnf_results['steps']}",
        ])

    results_table.append([
        "RF (LogitTransform)",
        _fmt_metric(rf_results['fid'], 2),
        _fmt_metric(rf_results['nll'], 4),
        _fmt_metric(rf_results['bpd'], 4),
        _fmt_metric(rf_results['time_per_image'] * 1000, 2),
        f"{rf_results['steps']}",
    ])

    results_table.append([
        "RF Student (CNF-Reflow)",
        _fmt_metric(student_results['fid'], 2),
        _fmt_metric(student_results['nll'], 4),
        _fmt_metric(student_results['bpd'], 4),
        _fmt_metric(student_results['time_per_image'] * 1000, 2),
        f"{student_results['steps']}",
    ])

    # Print table
    col_widths = [max(len(str(row[i])) for row in results_table) for i in range(6)]
    for i, row in enumerate(results_table):
        print(" | ".join(str(item).ljust(col_widths[j]) for j, item in enumerate(row)))
        if i == 0:
            print("-" * (sum(col_widths) + 9))
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    
    print("\nSaved models:")
    print(f"  - {args.rf_model_path}")
    print(f"  - {args.student_model_path}")
    
    print("\nSaved checkpoints:")
    print(f"  - {args.rf_ckpt_path}")
    print(f"  - {args.student_ckpt_path}")
    
    print("\nSaved visualizations:")
    print("  - rf_logit_samples.png")
    print("  - rf_student_logit_samples.png")
    
    return {
        'cnf': cnf_results,
        'rf': rf_results,
        'student': student_results,
    }


def _resolve_default_paths(args, data_spec: DataSpec) -> None:
    defaults = {
        "rf_model_path": "rf_model_final.pth",
        "rf_ckpt_path": "rf_ckpt.pth",
        "student_model_path": "student_model_final.pth",
        "student_ckpt_path": "student_ckpt.pth",
    }
    legacy_defaults = {
        "rf_model_path": "mnist_rf_logit_model_final.pth",
        "rf_ckpt_path": "mnist_rf_logit_ckpt.pth",
        "student_model_path": "mnist_student_logit_model_final.pth",
        "student_ckpt_path": "mnist_student_logit_ckpt.pth",
    }
    replacements = {
        "rf_model_path": f"{data_spec.data_name}_rf_model_final.pth",
        "rf_ckpt_path": f"{data_spec.data_name}_rf_ckpt.pth",
        "student_model_path": f"{data_spec.data_name}_student_model_final.pth",
        "student_ckpt_path": f"{data_spec.data_name}_student_ckpt.pth",
    }
    for attr, default_value in defaults.items():
        if getattr(args, attr) in {default_value, legacy_defaults[attr]}:
            setattr(args, attr, replacements[attr])


def _artifact_prefix_from_path(path: str) -> str:
    return str(Path(path).with_suffix(""))


def _print_data_overview(data_spec: DataSpec, train_loader, eval_loader) -> None:
    print(f"Dataset: {data_spec.data_name} | data_type={data_spec.data_type} | DATA_SHAPE={data_spec.data_shape}")
    print(f"Data root: {data_spec.data_root}")
    print(f"Using num_workers={DEFAULT_NUM_WORKERS} {'(Windows detected)' if IS_WINDOWS else ''}")
    if train_loader is None:
        return

    sample_batch, _ = next(iter(train_loader))
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Eval samples: {len(eval_loader.dataset)}")
    print(f"Sample batch stats:")
    print(f"  min: {sample_batch.min().item():.3f}")
    print(f"  max: {sample_batch.max().item():.3f}")
    print(f"  mean: {sample_batch.mean().item():.3f}")
    print(f"  std: {sample_batch.std().item():.3f}")

    if data_spec.is_image:
        sample_image = LOGIT_TRANSFORM.inverse(sample_batch)
        sample_image = torch.clamp(sample_image, 0, 1)
        print(f"Image-space stats after inverse logit:")
        print(f"  min: {sample_image.min().item():.3f}")
        print(f"  max: {sample_image.max().item():.3f}")
        print(f"  mean: {sample_image.mean().item():.3f}")
        print(f"  std: {sample_image.std().item():.3f}")
    elif data_spec.is_peptide:
        bundle = data_spec.peptide_bundle
        assert bundle is not None
        coords = bundle.denormalize_flat(sample_batch[: min(8, sample_batch.shape[0])].numpy())
        print(f"Peptide coord stats after denormalization:")
        print(f"  min: {coords.min():.3f}")
        print(f"  max: {coords.max():.3f}")
        print(f"  mean: {coords.mean():.3f}")
        print(f"  std: {coords.std():.3f}")
    else:
        print("Tabular per-feature summary:")
        print(f"  mean(feature means): {sample_batch.mean(dim=0).mean().item():.3f}")
        print(f"  mean(feature stds): {sample_batch.std(dim=0, unbiased=False).mean().item():.3f}")


def _print_results_summary(data_spec: DataSpec, cnf_results, rf_results, student_results):
    def _fmt(x, digits=4):
        if x is None:
            return "N/A"
        return f"{x:.{digits}f}"

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    if data_spec.is_peptide:
        results_table = [["Model", "NLL", "ESS", "E-W1", "T-W2", "Time/Sample (ms)"]]
        for label, result in [("CNF (Teacher)", cnf_results), ("RF", rf_results), ("RF Student", student_results)]:
            if result is None:
                continue
            results_table.append([
                label,
                _fmt(result.get("nll")),
                _fmt(result.get("ess"), 2),
                _fmt(result.get("e_w1")),
                _fmt(result.get("t_w2")),
                _fmt(result.get("time_per_image", 0.0) * 1000, 2),
            ])
    elif data_spec.is_tabular:
        results_table = [["Model", "NLL", "Mean L1", "Std L1", "Time/Sample (ms)", "ODE Steps"]]
        for label, result in [("CNF (Teacher)", cnf_results), ("RF", rf_results), ("RF Student", student_results)]:
            if result is None:
                continue
            results_table.append([
                label,
                _fmt(result.get("nll")),
                _fmt(result.get("mean_l1")),
                _fmt(result.get("std_l1")),
                _fmt(result.get("time_per_image", 0.0) * 1000, 2),
                str(result.get("steps")),
            ])
    else:
        results_table = [["Model", "FID Score", "NLL", "BPD", "Time/Image (ms)", "ODE Steps"]]
        for label, result in [("CNF (Teacher)", cnf_results), ("RF", rf_results), ("RF Student", student_results)]:
            if result is None:
                continue
            results_table.append([
                label,
                _fmt(result.get("fid"), 2),
                _fmt(result.get("nll")),
                _fmt(result.get("bpd")),
                _fmt(result.get("time_per_image", 0.0) * 1000, 2),
                str(result.get("steps")),
            ])

    col_widths = [max(len(str(row[i])) for row in results_table) for i in range(len(results_table[0]))]
    for row_idx, row in enumerate(results_table):
        print(" | ".join(str(item).ljust(col_widths[col_idx]) for col_idx, item in enumerate(row)))
        if row_idx == 0:
            print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))


def main(args):
    """Override: modality-aware main pipeline for image, tabular, and peptide experiments."""
    global DATA_SHAPE, DATA_SPEC, LOGIT_TRANSFORM

    data_spec = build_data_spec(args)
    DATA_SPEC = data_spec
    DATA_SHAPE = data_spec.data_shape
    if data_spec.is_image:
        LOGIT_TRANSFORM = LogitTransform(alpha=getattr(args, "alpha", 0.05))

    _resolve_default_paths(args, data_spec)

    print("=" * 70)
    print("Rectified Flow Training Pipeline")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"CNF path: {args.cnf_path}")
    print(f"RF model path: {args.rf_model_path}")
    print(f"Student model path: {args.student_model_path}")
    print(f"RF epochs: {args.rf_epochs} | Student epochs: {args.student_epochs}")
    if data_spec.is_image:
        print(f"LogitTransform alpha: {LOGIT_TRANSFORM.alpha}")
    print()

    need_train_data = (not args.eval_only) and (not args.skip_rf or not args.skip_student)
    train_loader = None

    if data_spec.is_image:
        if need_train_data:
            train_loader, _ = get_train_loaders_logit(
                data_name=data_spec.data_name,
                data_root=str(data_spec.data_root),
                batch_size=max(32, args.student_batch_size),
                alpha=LOGIT_TRANSFORM.alpha,
                download=(not getattr(args, 'no_download', False)),
            )
        eval_loader = get_raw_test_loader(
            data_name=data_spec.data_name,
            data_root=str(data_spec.data_root),
            batch_size=64,
            download=(not getattr(args, 'no_download', False)),
        )
    elif data_spec.is_peptide:
        train_loader, peptide_eval_loader = build_peptide_loaders(
            data_spec.peptide_bundle,
            batch_size=max(32, args.student_batch_size),
            max_train_samples=args.max_train_samples,
        )
        eval_loader = peptide_eval_loader
        if not need_train_data:
            train_loader = None
    else:
        _, tabular_train_loader, _, tabular_test_loader = build_tabular_loaders(
            data_spec.data_name,
            train_batch_size=max(32, args.student_batch_size),
            eval_batch_size=max(64, args.student_batch_size),
            num_workers=DEFAULT_NUM_WORKERS,
            data_root=str(data_spec.data_root),
        )
        train_loader = tabular_train_loader if need_train_data else None
        eval_loader = tabular_test_loader

    _print_data_overview(data_spec, train_loader, eval_loader)

    print("\n" + "=" * 70)
    print("Loading CNF Model (Teacher)")
    print("=" * 70)
    if not os.path.exists(args.cnf_path):
        raise FileNotFoundError(f"CNF model not found at {args.cnf_path}")

    cnf_model = load_teacher_model(args.cnf_path, data_spec, base_channels=48)
    cnf_model._data_shape = data_spec.data_shape
    cnf_model.eval()
    print(f"CNF parameters: {sum(p.numel() for p in cnf_model.parameters()):,}")

    cnf_results = None
    if args.eval_cnf:
        cnf_results = evaluate_model(
            cnf_model,
            data_spec=data_spec,
            test_loader_raw=eval_loader,
            model_name="CNF_Teacher",
            n_samples=args.n_samples,
            eval_nll=args.eval_nll,
            nll_steps=args.nll_steps,
            nll_method=args.nll_method,
            nll_hutchinson_samples=args.nll_hutchinson_samples,
            nll_max_batches=args.nll_max_batches,
            metric_samples=args.metric_samples,
            tw2_subsample=args.tw2_subsample,
            openmm_platform=args.openmm_platform,
            artifact_prefix=_default_artifact_prefix(data_spec.data_name, "cnf_teacher"),
        )

    print("\n" + "=" * 70)
    print("Stage 1: Rectified Flow (RF)")
    print("=" * 70)
    rf_model = build_rf_model(data_spec)
    rf_model._data_shape = data_spec.data_shape
    print(f"RF parameters: {sum(p.numel() for p in rf_model.parameters()):,}")

    rf_optimizer = optim.Adam(rf_model.parameters(), lr=args.rf_lr)
    rf_start_epoch = 0
    if args.skip_rf:
        if os.path.exists(args.rf_model_path):
            rf_model.load_state_dict(torch.load(args.rf_model_path, map_location=device, weights_only=False))
        elif os.path.exists(args.rf_ckpt_path):
            ckpt = torch.load(args.rf_ckpt_path, map_location=device, weights_only=False)
            rf_model.load_state_dict(ckpt["model"])
        else:
            args.skip_rf = False

    if not args.skip_rf:
        if train_loader is None:
            raise ValueError("Training requested but no training loader is available")
        if args.resume_rf and os.path.exists(args.rf_ckpt_path):
            ckpt = torch.load(args.rf_ckpt_path, map_location=device, weights_only=False)
            rf_model.load_state_dict(ckpt["model"])
            rf_optimizer.load_state_dict(ckpt["optimizer"])
            rf_start_epoch = ckpt["epoch"]
        elif args.resume_rf and os.path.exists(args.rf_model_path):
            rf_model.load_state_dict(torch.load(args.rf_model_path, map_location=device, weights_only=False))

        rf_model, rf_optimizer, _ = train_rf(
            rf_model,
            train_loader,
            epochs=args.rf_epochs,
            lr=args.rf_lr,
            save_interval=100,
            optimizer=rf_optimizer,
            start_epoch=rf_start_epoch,
            ckpt_path=args.rf_ckpt_path,
            model_path=args.rf_model_path,
            snapshot_prefix=_artifact_prefix_from_path(args.rf_model_path),
        )

    rf_results = evaluate_model(
        rf_model,
        data_spec=data_spec,
        test_loader_raw=eval_loader,
        model_name="RF",
        n_samples=args.n_samples,
        eval_nll=args.eval_nll,
        nll_steps=args.nll_steps,
        nll_method=args.nll_method,
        nll_hutchinson_samples=args.nll_hutchinson_samples,
        nll_max_batches=args.nll_max_batches,
        metric_samples=args.metric_samples,
        tw2_subsample=args.tw2_subsample,
        openmm_platform=args.openmm_platform,
        artifact_prefix=_default_artifact_prefix(data_spec.data_name, "rf"),
    )

    print("\n" + "=" * 70)
    print("Stage 2: RF Student via CNF-Reflow Distillation")
    print("=" * 70)
    # Added: build the student using CLI-configured vector dimensions.
    student_model = build_student_model(data_spec, args)
    student_model._data_shape = data_spec.data_shape
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    student_optimizer = optim.Adam(student_model.parameters(), lr=args.student_lr)
    student_start_epoch = 0
    if args.skip_student:
        if os.path.exists(args.student_model_path):
            student_model.load_state_dict(torch.load(args.student_model_path, map_location=device, weights_only=False))
        elif os.path.exists(args.student_ckpt_path):
            ckpt = torch.load(args.student_ckpt_path, map_location=device, weights_only=False)
            student_model.load_state_dict(ckpt["model"])
        else:
            args.skip_student = False

    if not args.skip_student:
        if train_loader is None:
            raise ValueError("Training requested but no training loader is available")
        if args.resume_student and os.path.exists(args.student_ckpt_path):
            ckpt = torch.load(args.student_ckpt_path, map_location=device, weights_only=False)
            student_model.load_state_dict(ckpt["model"])
            student_optimizer.load_state_dict(ckpt["optimizer"])
            student_start_epoch = ckpt["epoch"]
        elif args.resume_student and os.path.exists(args.student_model_path):
            student_model.load_state_dict(torch.load(args.student_model_path, map_location=device, weights_only=False))

        student_model, student_optimizer, _ = train_rf_student(
            cnf_model,
            train_loader,
            epochs=args.student_epochs,
            batch_size=args.student_batch_size,
            num_steps=args.student_num_steps,
            lr=args.student_lr,
            save_interval=100,
            optimizer=student_optimizer,
            start_epoch=student_start_epoch,
            ckpt_path=args.student_ckpt_path,
            model_path=args.student_model_path,
            student_model=student_model,
            data_spec=data_spec,
            snapshot_prefix=_artifact_prefix_from_path(args.student_model_path),
            # Added: pass CLI-configured student architecture into the training helper.
            model_args=args,
        )

    student_results = evaluate_model(
        student_model,
        data_spec=data_spec,
        test_loader_raw=eval_loader,
        model_name="RF_Student",
        n_samples=args.n_samples,
        eval_nll=args.eval_nll,
        nll_steps=args.nll_steps,
        nll_method=args.nll_method,
        nll_hutchinson_samples=args.nll_hutchinson_samples,
        nll_max_batches=args.nll_max_batches,
        metric_samples=args.metric_samples,
        tw2_subsample=args.tw2_subsample,
        openmm_platform=args.openmm_platform,
        artifact_prefix=_default_artifact_prefix(data_spec.data_name, "rf_student"),
    )

    _print_results_summary(data_spec, cnf_results, rf_results, student_results)

    print("\nSaved models:")
    print(f"  - {args.rf_model_path}")
    print(f"  - {args.student_model_path}")
    print("\nSaved checkpoints:")
    print(f"  - {args.rf_ckpt_path}")
    print(f"  - {args.student_ckpt_path}")

    return {'cnf': cnf_results, 'rf': rf_results, 'student': student_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rectified Flow training and evaluation pipeline for image, tabular, and peptide datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train/evaluate an image pipeline
  python train_rf_pipeline.py --cnf-path mnist_cnf_logit_model_final.pth --data mnist

  # Evaluate a tabular teacher checkpoint on Miniboone
  python train_rf_pipeline.py --cnf-path experiments/miniboone_tabular_ffjord.pth --data miniboone --data-type tabular --eval-only --eval-nll

  # Evaluate an existing peptide teacher checkpoint on ALDP
  python train_rf_pipeline.py --cnf-path experiments/aldp_vector_ffjord.pth --data aldp --data-type peptide --eval-only

  # Evaluate FORT-style peptide checkpoints on AL3 / AL4
  python train_rf_pipeline.py --cnf-path experiments/al3_vector_ffjord.pth --data al3 --data-type peptide --eval-only
  python train_rf_pipeline.py --cnf-path experiments/al4_vector_ffjord.pth --data al4 --data-type peptide --eval-only

  # Resume RF training with dataset-aware output names
  python train_rf_pipeline.py --cnf-path mnist_cnf_logit_model_final.pth --data cifar10 --resume-rf --rf-epochs 20
        """
    )
    
    # CNF model path
    parser.add_argument(
        "--cnf-path", type=str, required=True,
        help="Path to trained CNF model (checkpoint or weights)"
    )
    parser.add_argument(
        "--eval-cnf", action="store_true",
        help="Also evaluate the CNF model"
    )


    # Dataset options (generalized)
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name (e.g., mnist, cifar10, svhn, stl10, fashionmnist, imagefolder, "
             "power, gas, hepmass, miniboone, bsds300, aldp, al3, al4). "
             "Legacy alias 'tetra' maps to 'al4'. If omitted, tries to infer from CNF checkpoint."
    )
    parser.add_argument(
        "--data-type", type=str, default="auto", choices=["auto", "image", "tabular", "peptide"],
        help="Force image, tabular, or peptide pipeline selection. Default: auto"
    )
    parser.add_argument(
        "--data-root", type=str, default="./data",
        help="Root directory for datasets (default: ./data)"
    )
    parser.add_argument(
        "--data-shape", type=str, default=None,
        help="Override image data shape as 'C,H,W' (e.g., '3,32,32'). Ignored for vector datasets."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="LogitTransform alpha used in preprocessing/inversion (default: 0.05)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Disable torchvision dataset downloading"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=100000,
        help="Optional cap for peptide training examples to match FORT-scale runs (default: 100000)"
    )
    
    # RF options
    parser.add_argument(
        "--skip-rf", action="store_true",
        help="Skip RF training, load existing model"
    )
    parser.add_argument(
        "--resume-rf", action="store_true",
        help="Resume RF training from checkpoint"
    )
    parser.add_argument(
        "--rf-epochs", type=int, default=30,
        help="Number of RF training epochs (default: 30)"
    )
    parser.add_argument(
        "--rf-lr", type=float, default=1e-3,
        help="RF learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--rf-model-path", type=str, default="rf_model_final.pth",
        help="Path to save/load RF model"
    )
    parser.add_argument(
        "--rf-ckpt-path", type=str, default="rf_ckpt.pth",
        help="Path to RF checkpoint"
    )
    
    # Student options
    parser.add_argument(
        "--skip-student", action="store_true",
        help="Skip Student training, load existing model"
    )
    parser.add_argument(
        "--resume-student", action="store_true",
        help="Resume Student training from checkpoint"
    )
    parser.add_argument(
        "--student-epochs", type=int, default=100,
        help="Number of Student training epochs (default: 100)"
    )
    parser.add_argument(
        "--student-lr", type=float, default=1e-3,
        help="Student learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--student-batch-size", type=int, default=256,
        help="Student training batch size (default: 256)"
    )
    parser.add_argument(
        "--student-num-steps", type=int, default=20,
        help="ODE steps for trajectory collection (default: 20)"
    )
    parser.add_argument(
        "--student-hidden-dim", type=int, default=512,
        help="Hidden dimension for vector-valued student backbones (default: 512)"
    )
    parser.add_argument(
        "--student-num-blocks", type=int, default=6,
        help="Number of residual blocks for vector-valued student backbones (default: 6)"
    )
    parser.add_argument(
        "--student-time-dim", type=int, default=128,
        help="Time embedding dimension for vector-valued student backbones (default: 128)"
    )
    parser.add_argument(
        "--student-model-path", type=str, default="student_model_final.pth",
        help="Path to save/load Student model"
    )
    parser.add_argument(
        "--student-ckpt-path", type=str, default="student_ckpt.pth",
        help="Path to Student checkpoint"
    )
    
    # Evaluation options
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate, skip all training"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of samples for FID evaluation (default: 1000)"
    )
    parser.add_argument(
        "--metric-samples", type=int, default=250000,
        help="Number of peptide proposal samples used for ESS / E-W1 / T-W2 (default: 250000)"
    )
    parser.add_argument(
        "--tw2-subsample", type=int, default=4096,
        help="Subsample size used for peptide T-W2 Sinkhorn estimation (default: 4096)"
    )
    parser.add_argument(
        "--openmm-platform", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="OpenMM platform to use for peptide metrics (default: auto)"
    )
    parser.add_argument(
        "--eval-nll", action="store_true",
        help="Also compute test NLL / BPD during evaluation"
    )
    parser.add_argument(
        "--nll-max-batches", type=int, default=None,
        help="Limit the number of test batches used for NLL (useful for fast CIFAR debugging)"
    )
    parser.add_argument(
        "--nll-steps", type=int, default=25,
        help="ODE steps for RF/Student NLL integration (default: 25)"
    )
    parser.add_argument(
        "--nll-method", type=str, default="rk4",
        help="ODE solver for RF/Student NLL integration (default: rk4)"
    )
    parser.add_argument(
        "--nll-hutchinson-samples", type=int, default=1,
        help="Number of Hutchinson probe vectors for RF/Student NLL estimation (default: 1)"
    )
    
    args = parser.parse_args()
    args.data = canonicalize_dataset_name(args.data)
    
    # Handle eval-only mode
    if args.eval_only:
        print("=" * 70)
        print("EVALUATION ONLY MODE")
        print("=" * 70)
        args.skip_rf = True
        args.skip_student = True
    
    main(args)
