import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.utils import save_image

import lib.layers as layers
import lib.multiscale_parallel as multiscale_parallel
import lib.odenvp as odenvp
import lib.utils as utils
from tabular_benchmarks import TABULAR_DATASETS, load_tabular_bundle
from train_misc import (
    add_spectral_norm,
    append_regularization_to_log,
    build_model_tabular,
    count_nfe,
    count_parameters,
    count_total_time,
    create_regularization_fns,
    get_regularization,
    set_cnf_options,
    spectral_norm_power_iteration,
    standard_normal_logprob,
)

# go fast boi!!
torch.backends.cudnn.benchmark = True

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]
IMAGE_DATASETS = {"mnist", "svhn", "cifar10", "lsun_church"}
PEPTIDE_CANONICAL_DATASETS = {"aldp", "al3", "al4"}
PEPTIDE_DATASET_ALIASES = {"tetra": "al4"}
PEPTIDE_DATASETS = PEPTIDE_CANONICAL_DATASETS | set(PEPTIDE_DATASET_ALIASES)
PEPTIDE_DATASET_EXPECTATIONS = {
    "aldp": 22,
    "al3": 33,
    "al4": 42,
}
ALL_DATASETS = IMAGE_DATASETS | TABULAR_DATASETS | PEPTIDE_DATASETS


@dataclass
class PeptideTrainBundle:
    data_name: str
    root_dir: Path
    train_flat: np.ndarray
    val_flat: np.ndarray
    test_flat: np.ndarray
    coord_shape: Tuple[int, int]


@dataclass
class CNFDataSpec:
    data_name: str
    data_type: str
    data_shape: Tuple[int, ...]
    train_set: Dataset
    eval_loader: DataLoader
    test_loader: Optional[DataLoader]
    metric_name: str
    visual_shape: Optional[Tuple[int, ...]] = None

    @property
    def is_image(self) -> bool:
        return self.data_type == "image"

    @property
    def uses_vector_backbone(self) -> bool:
        return self.data_type in {"tabular", "peptide"}


def canonicalize_dataset_name(data_name: Optional[str]) -> Optional[str]:
    if data_name is None:
        return None
    normalized = str(data_name).lower()
    if normalized in PEPTIDE_DATASETS:
        return PEPTIDE_DATASET_ALIASES.get(normalized, normalized)
    return normalized


def _validate_peptide_num_atoms(data_name: str, metadata: dict) -> None:
    expected_num_atoms = PEPTIDE_DATASET_EXPECTATIONS.get(data_name)
    if expected_num_atoms is None:
        return
    actual_num_atoms = int(metadata.get("num_atoms", -1))
    if actual_num_atoms != expected_num_atoms:
        raise ValueError(
            f"Peptide dataset '{data_name}' expects num_atoms={expected_num_atoms}, "
            f"but metadata.json declares {actual_num_atoms}. Please download the correct benchmark bundle."
        )


def build_logger(logpath: str) -> logging.Logger:
    """Create a logger that writes to stdout and a file without dumping source code."""
    logger = logging.getLogger("cnf")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.handlers = []

    fh = logging.FileHandler(logpath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_model_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """Return a non-DataParallel state_dict."""
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_checkpoint(
    path: str,
    *,
    args: argparse.Namespace,
    epoch: int,
    best_loss: float,
    best_epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    utils.makedirs(os.path.dirname(path) or ".")
    torch.save(
        {
            "args": args,
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "state_dict": get_model_state_dict(model),
            "optim_state_dict": optimizer.state_dict(),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument("--data", choices=sorted(ALL_DATASETS), type=str.lower, default="mnist")
    parser.add_argument("--data-type", choices=["auto", "image", "tabular", "peptide"], default="auto")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--dims", type=str, default="8,32,32,8")
    parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
    parser.add_argument("--num_blocks", type=int, default=1, help="Number of stacked CNFs.")

    parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--layer_type",
        type=str,
        default="ignore",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"],
    )
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    parser.add_argument("--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"])
    parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

    parser.add_argument("--test_solver", type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument("--test_atol", type=float, default=None)
    parser.add_argument("--test_rtol", type=float, default=None)

    parser.add_argument("--imagesize", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--time_length", type=float, default=1.0)
    parser.add_argument("--train_T", type=eval, default=True)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--batch_size_schedule", type=str, default="", help="Increase batchsize at given epochs, dash separated.")
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=float, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--spectral_norm_niter", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for vector-dataset training samples.")

    parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
    parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--bn_lag", type=float, default=0.0)
    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--autoencode", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=True, choices=[True, False])
    parser.add_argument("--spectral_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--multiscale", type=eval, default=False, choices=[True, False])
    parser.add_argument("--parallel", type=eval, default=False, choices=[True, False])
    parser.add_argument("--no-download", action="store_true", help="Disable torchvision dataset downloading.")

    parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
    parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
    parser.add_argument("--dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2")
    parser.add_argument("--JFrobint", type=float, default=None, help="int_t ||df/dx||_F")
    parser.add_argument("--JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F")
    parser.add_argument("--JoffdiagFrobint", type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

    parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
    parser.add_argument("--max_grad_norm", type=float, default=1e10, help="Max norm of gradients.")
    parser.add_argument("--begin_epoch", type=int, default=1, help="First epoch index for this run.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint (best or latest).")
    parser.add_argument("--save", type=str, default="experiments/cnf", help="Experiment directory.")
    parser.add_argument("--val_freq", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--log_freq", type=int, default=10, help="Log every N iterations.")
    return parser


def _resolve_data_type(args_: argparse.Namespace) -> str:
    if args_.data_type != "auto":
        return args_.data_type
    if args_.data in TABULAR_DATASETS:
        return "tabular"
    if args_.data in PEPTIDE_DATASETS:
        return "peptide"
    return "image"


def _configure_args_for_data_type(args_: argparse.Namespace, data_type: str, logger: logging.Logger) -> None:
    if data_type == "image":
        return

    if args_.conv:
        logger.info("Vector dataset detected; forcing --conv False for %s.", args_.data)
        args_.conv = False
    if args_.multiscale:
        raise ValueError("Vector datasets do not support --multiscale in train_cnf.py.")
    if args_.parallel:
        raise ValueError("Vector datasets do not support --parallel in train_cnf.py.")
    if args_.autoencode:
        raise ValueError("Vector datasets do not support --autoencode in train_cnf.py.")


def build_add_noise_fn(use_noise: bool):
    def add_noise(x: torch.Tensor) -> torch.Tensor:
        if use_noise:
            noise = x.new().resize_as_(x).uniform_()
            x = x * 255 + noise
            x = x / 256
        return x

    return add_noise


def update_lr(optimizer: optim.Optimizer, itr: int, args_: argparse.Namespace) -> None:
    iter_frac = min(float(itr + 1) / max(args_.warmup_iters, 1), 1.0)
    lr = args_.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_loader(train_set: Dataset, epoch: int, args_: argparse.Namespace, logger: logging.Logger) -> DataLoader:
    if args_.batch_size_schedule != "":
        epochs = [0] + list(map(int, args_.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args_.batch_size * n_passed)
    else:
        current_batch_size = args_.batch_size

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=current_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    logger.info("===> Using batch size %d. Total %d iterations/epoch.", current_batch_size, len(train_loader))
    return train_loader


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


def load_peptide_training_bundle(data_name: str, data_root: str) -> PeptideTrainBundle:
    canonical_name = canonicalize_dataset_name(data_name) or str(data_name).lower()
    root_dir = _resolve_peptide_root(data_name, data_root)
    metadata_path = root_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if "num_atoms" not in metadata:
        raise ValueError("metadata.json must include 'num_atoms' for peptide training")
    _validate_peptide_num_atoms(canonical_name, metadata)

    num_atoms = int(metadata["num_atoms"])
    train_raw, val_raw, test_raw = _load_peptide_arrays(root_dir)
    train_coords = _coerce_coord_array(train_raw, num_atoms, "train")
    val_coords = _coerce_coord_array(val_raw, num_atoms, "val")
    test_coords = _coerce_coord_array(test_raw, num_atoms, "test")

    atom_masses = metadata.get("atom_masses")
    if atom_masses is not None:
        masses = np.asarray(atom_masses, dtype=np.float32)
    elif "amber_prmtop" in metadata:
        prmtop_path = Path(metadata["amber_prmtop"])
        if not prmtop_path.is_absolute():
            prmtop_path = root_dir / prmtop_path
        masses = _load_masses_from_prmtop(prmtop_path)
    elif "topology_pdb" in metadata:
        pdb_path = Path(metadata["topology_pdb"])
        if not pdb_path.is_absolute():
            pdb_path = root_dir / pdb_path
        masses = _load_masses_from_pdb(pdb_path)
    else:
        raise ValueError(
            "metadata.json must include 'atom_masses', 'amber_prmtop', or 'topology_pdb' for peptide training"
        )

    if masses.shape != (num_atoms,):
        raise ValueError(f"Expected {num_atoms} atom masses, got shape {tuple(masses.shape)}")

    train_centered = remove_center_of_mass(train_coords, masses)
    val_centered = remove_center_of_mass(val_coords, masses)
    test_centered = remove_center_of_mass(test_coords, masses)
    std_scale = float(np.std(train_centered.reshape(train_centered.shape[0], -1)))
    std_scale = max(std_scale, 1e-6)

    return PeptideTrainBundle(
        data_name=canonical_name,
        root_dir=root_dir,
        train_flat=train_centered.reshape(train_centered.shape[0], -1) / std_scale,
        val_flat=val_centered.reshape(val_centered.shape[0], -1) / std_scale,
        test_flat=test_centered.reshape(test_centered.shape[0], -1) / std_scale,
        coord_shape=(num_atoms, 3),
    )


def _make_vector_dataset(x: torch.Tensor) -> TensorDataset:
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    return TensorDataset(x, labels)


def _limit_train_dataset(train_set: Dataset, max_train_samples: Optional[int]) -> Dataset:
    if max_train_samples is None or max_train_samples <= 0 or len(train_set) <= max_train_samples:
        return train_set
    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_set), size=max_train_samples, replace=False)
    return Subset(train_set, indices.tolist())


def _make_eval_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def get_image_data_spec(args_: argparse.Namespace) -> CNFDataSpec:
    add_noise = build_add_noise_fn(args_.add_noise)
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args_.data == "mnist":
        im_dim = 1
        im_size = 28 if args_.imagesize is None else args_.imagesize
        train_set = dset.MNIST(root=args_.data_root, train=True, transform=trans(im_size), download=not args_.no_download)
        test_set = dset.MNIST(root=args_.data_root, train=False, transform=trans(im_size), download=not args_.no_download)
    elif args_.data == "svhn":
        im_dim = 3
        im_size = 32 if args_.imagesize is None else args_.imagesize
        train_set = dset.SVHN(root=args_.data_root, split="train", transform=trans(im_size), download=not args_.no_download)
        test_set = dset.SVHN(root=args_.data_root, split="test", transform=trans(im_size), download=not args_.no_download)
    elif args_.data == "cifar10":
        im_dim = 3
        im_size = 32 if args_.imagesize is None else args_.imagesize
        train_set = dset.CIFAR10(
            root=args_.data_root,
            train=True,
            transform=tforms.Compose(
                [
                    tforms.Resize(im_size),
                    tforms.RandomHorizontalFlip(),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
            download=not args_.no_download,
        )
        test_set = dset.CIFAR10(root=args_.data_root, train=False, transform=trans(im_size), download=not args_.no_download)
    elif args_.data == "lsun_church":
        im_dim = 3
        im_size = 64 if args_.imagesize is None else args_.imagesize
        train_set = dset.LSUN(
            args_.data_root,
            ["church_outdoor_train"],
            transform=tforms.Compose(
                [
                    tforms.Resize(96),
                    tforms.RandomCrop(64),
                    tforms.Resize(im_size),
                    tforms.ToTensor(),
                    add_noise,
                ]
            ),
        )
        test_set = dset.LSUN(
            args_.data_root,
            ["church_outdoor_val"],
            transform=tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise]),
        )
    else:
        raise ValueError(f"Unsupported image dataset: {args_.data}")

    visual_shape = (im_dim, im_size, im_size)
    model_shape = visual_shape if args_.conv else (im_dim * im_size * im_size,)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args_.test_batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    return CNFDataSpec(
        data_name=args_.data,
        data_type="image",
        data_shape=model_shape,
        train_set=train_set,
        eval_loader=test_loader,
        test_loader=test_loader,
        metric_name="Bit/dim",
        visual_shape=visual_shape,
    )


def get_tabular_data_spec(args_: argparse.Namespace) -> CNFDataSpec:
    bundle = load_tabular_bundle(args_.data, data_root=args_.data_root)
    train_set = _limit_train_dataset(_make_vector_dataset(bundle.train_x), args_.max_train_samples)
    val_loader = _make_eval_loader(_make_vector_dataset(bundle.val_x), args_.test_batch_size)
    test_loader = _make_eval_loader(_make_vector_dataset(bundle.test_x), args_.test_batch_size)
    return CNFDataSpec(
        data_name=args_.data,
        data_type="tabular",
        data_shape=(bundle.input_dim,),
        train_set=train_set,
        eval_loader=val_loader,
        test_loader=test_loader,
        metric_name="NLL",
        visual_shape=None,
    )


def get_peptide_data_spec(args_: argparse.Namespace) -> CNFDataSpec:
    bundle = load_peptide_training_bundle(args_.data, args_.data_root)
    train_tensor = torch.from_numpy(bundle.train_flat.astype(np.float32))
    val_tensor = torch.from_numpy(bundle.val_flat.astype(np.float32))
    test_tensor = torch.from_numpy(bundle.test_flat.astype(np.float32))
    train_set = _limit_train_dataset(_make_vector_dataset(train_tensor), args_.max_train_samples)
    val_loader = _make_eval_loader(_make_vector_dataset(val_tensor), args_.test_batch_size)
    test_loader = _make_eval_loader(_make_vector_dataset(test_tensor), args_.test_batch_size)
    return CNFDataSpec(
        data_name=args_.data,
        data_type="peptide",
        data_shape=(train_tensor.shape[1],),
        train_set=train_set,
        eval_loader=val_loader,
        test_loader=test_loader,
        metric_name="NLL",
        visual_shape=None,
    )


def get_data_spec(args_: argparse.Namespace) -> CNFDataSpec:
    data_type = _resolve_data_type(args_)
    if data_type == "image":
        return get_image_data_spec(args_)
    if data_type == "tabular":
        return get_tabular_data_spec(args_)
    if data_type == "peptide":
        return get_peptide_data_spec(args_)
    raise ValueError(f"Unsupported data type: {data_type}")


def compute_bits_per_dim(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    zero = torch.zeros(x.shape[0], 1).to(x)
    z, delta_logp = model(x, zero)

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
    return bits_per_dim


def compute_vector_nll(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    zero = torch.zeros(x.shape[0], 1).to(x)
    z, delta_logp = model(x, zero)
    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)
    logpx = logpz - delta_logp
    return -torch.mean(logpx)


def compute_loss(x: torch.Tensor, model: torch.nn.Module, data_spec: CNFDataSpec) -> torch.Tensor:
    if data_spec.is_image:
        return compute_bits_per_dim(x, model)
    return compute_vector_nll(x, model)


def create_image_model(args_: argparse.Namespace, data_shape: Tuple[int, ...], regularization_fns):
    hidden_dims = tuple(map(int, args_.dims.split(",")))
    strides = tuple(map(int, args_.strides.split(",")))

    if args_.multiscale:
        model = odenvp.ODENVP(
            (args_.batch_size, *data_shape),
            n_blocks=args_.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args_.nonlinearity,
            alpha=args_.alpha,
            cnf_kwargs={"T": args_.time_length, "train_T": args_.train_T, "regularization_fns": regularization_fns},
        )
    elif args_.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args_.batch_size, *data_shape),
            n_blocks=args_.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args_.alpha,
            time_length=args_.time_length,
        )
    else:
        if args_.autoencode:

            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args_.conv,
                    layer_type=args_.layer_type,
                    nonlinearity=args_.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args_.divergence_fn,
                    residual=args_.residual,
                    rademacher=args_.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args_.time_length,
                    regularization_fns=regularization_fns,
                    solver=args_.solver,
                )
                return cnf

        else:

            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args_.conv,
                    layer_type=args_.layer_type,
                    nonlinearity=args_.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args_.divergence_fn,
                    residual=args_.residual,
                    rademacher=args_.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args_.time_length,
                    train_T=args_.train_T,
                    regularization_fns=regularization_fns,
                    solver=args_.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args_.alpha)] if args_.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args_.num_blocks)]
        if args_.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)

    return model


def create_model(args_: argparse.Namespace, data_spec: CNFDataSpec, regularization_fns):
    if data_spec.uses_vector_backbone:
        return build_model_tabular(args_, data_spec.data_shape[0], regularization_fns)
    return create_image_model(args_, data_spec.data_shape, regularization_fns)


def _prepare_batch(x: torch.Tensor, data_spec: CNFDataSpec, cvt) -> torch.Tensor:
    if data_spec.is_image and data_spec.visual_shape is not None and data_spec.data_shape != data_spec.visual_shape:
        x = x.view(x.shape[0], -1)
    return cvt(x)


def main(parsed_args: Optional[argparse.Namespace] = None) -> int:
    parser = build_parser()
    args = parsed_args or parser.parse_args()
    args.data = canonicalize_dataset_name(args.data)

    utils.makedirs(args.save)
    job_id = os.environ.get("JOB_ID", "nojid")
    ts = time.strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(args.save, f"logs_{job_id}_{ts}.txt")
    logger = build_logger(logpath)

    if args.layer_type == "blend":
        logger.info("Setting time_length to 1.0 due to use of Blend layers.")
        args.time_length = 1.0

    data_type = _resolve_data_type(args)
    _configure_args_for_data_type(args, data_type, logger)

    logger.info("Args: %s", vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    data_spec = get_data_spec(args)
    logger.info(
        "Dataset: %s | data_type=%s | data_shape=%s | train=%d | eval=%d",
        data_spec.data_name,
        data_spec.data_type,
        data_spec.data_shape,
        len(data_spec.train_set),
        len(data_spec.eval_loader.dataset),
    )
    if data_spec.test_loader is not None and data_spec.test_loader is not data_spec.eval_loader:
        logger.info("Held-out test samples available for downstream evaluation: %d", len(data_spec.test_loader.dataset))

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_spec, regularization_fns)

    if args.spectral_norm:
        add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: %d", count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_epoch = 0

    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage, weights_only=False)
        model.load_state_dict(checkpt["state_dict"])

        if "optim_state_dict" in checkpt:
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = cvt(value)

        if "best_loss" in checkpt:
            best_loss = float(checkpt["best_loss"])
        if "best_epoch" in checkpt:
            best_epoch = int(checkpt["best_epoch"])

        if args.begin_epoch <= 0 and "epoch" in checkpt:
            args.begin_epoch = int(checkpt["epoch"]) + 1

        logger.info(
            "Resumed from %s | ckpt_epoch=%s | begin_epoch=%d | best_epoch=%d | best_loss=%.4f",
            args.resume,
            str(checkpt.get("epoch", "N/A")),
            args.begin_epoch,
            best_epoch,
            best_loss,
        )

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    fixed_z = None
    if data_spec.is_image:
        fixed_z = cvt(torch.randn(100, *data_spec.data_shape))

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and args.resume is None:
        spectral_norm_power_iteration(model, 500)

    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(data_spec.train_set, epoch, args, logger)

        for _, (x, _y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr, args)
            optimizer.zero_grad()

            x = _prepare_batch(x, data_spec, cvt)
            loss = compute_loss(x, model, data_spec)

            if regularization_coeffs:
                reg_states = get_regularization(model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff
                    for reg_state, coeff in zip(reg_states, regularization_coeffs)
                    if coeff != 0
                )
                loss = loss + reg_loss

            total_time = count_total_time(model)
            loss = loss + total_time * args.time_penalty

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if args.spectral_norm:
                spectral_norm_power_iteration(model, args.spectral_norm_niter)

            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(model))
            grad_meter.update(float(grad_norm))
            tt_meter.update(total_time)

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | {} {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f})".format(
                        itr,
                        time_meter.val,
                        time_meter.avg,
                        data_spec.metric_name,
                        loss_meter.val,
                        loss_meter.avg,
                        steps_meter.val,
                        steps_meter.avg,
                        grad_meter.val,
                        grad_meter.avg,
                        tt_meter.val,
                        tt_meter.avg,
                    )
                )
                if regularization_coeffs:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                logger.info(log_message)

            itr += 1

        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                for (x, _y) in data_spec.eval_loader:
                    x = _prepare_batch(x, data_spec, cvt)
                    losses.append(compute_loss(x, model, data_spec).cpu().item())
                val_loss = float(np.mean(losses))

                logger.info("Epoch %04d | Time %.4f, %s %.4f", epoch, time.time() - start, data_spec.metric_name, val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_path = os.path.join(args.save, "checkpt.pth")
                    save_checkpoint(
                        best_path,
                        args=args,
                        epoch=epoch,
                        best_loss=best_loss,
                        best_epoch=best_epoch,
                        model=model,
                        optimizer=optimizer,
                    )
                    logger.info(
                        "===> BEST updated: epoch=%04d | %s=%.4f | %s",
                        best_epoch,
                        data_spec.metric_name,
                        best_loss,
                        best_path,
                    )

        if data_spec.is_image and fixed_z is not None and data_spec.visual_shape is not None:
            with torch.no_grad():
                fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
                utils.makedirs(os.path.dirname(fig_filename))
                generated_samples = model(fixed_z, reverse=True).view(-1, *data_spec.visual_shape)
                save_image(generated_samples, fig_filename, nrow=10)

        latest_path = os.path.join(args.save, "latest.pth")
        save_checkpoint(
            latest_path,
            args=args,
            epoch=epoch,
            best_loss=best_loss,
            best_epoch=best_epoch,
            model=model,
            optimizer=optimizer,
        )
        logger.info(
            "Saved latest: epoch=%04d | %s | current best epoch=%04d (%s=%.4f)",
            epoch,
            latest_path,
            best_epoch,
            data_spec.metric_name,
            best_loss,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
