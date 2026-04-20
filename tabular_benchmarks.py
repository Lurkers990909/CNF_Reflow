from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

TABULAR_DATASETS = {"power", "gas", "hepmass", "miniboone", "bsds300"}


@dataclass
class TabularBundle:
    data_name: str
    train_x: torch.Tensor
    val_x: torch.Tensor
    test_x: torch.Tensor
    input_dim: int
    data_root: Path


def _dataset_factory(data_name: str):
    tabular_datasets = _get_tabular_datasets_module()
    mapping = {
        "power": tabular_datasets.POWER,
        "gas": tabular_datasets.GAS,
        "hepmass": tabular_datasets.HEPMASS,
        "miniboone": tabular_datasets.MINIBOONE,
        "bsds300": tabular_datasets.BSDS300,
    }
    if data_name not in mapping:
        raise ValueError(f"Unsupported tabular dataset '{data_name}'. Expected one of: {sorted(TABULAR_DATASETS)}")
    return mapping[data_name]


def _get_tabular_datasets_module():
    try:
        import datasets as tabular_datasets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Tabular benchmarks require the optional dependencies used by datasets/ "
            "(for example pandas and h5py). Install them before using power/gas/hepmass/miniboone/bsds300."
        ) from exc
    return tabular_datasets


def _resolve_data_root(data_root: str | Path | None) -> Path:
    tabular_datasets = _get_tabular_datasets_module()
    if data_root is None:
        return Path(tabular_datasets.root).resolve()
    return Path(data_root).resolve()


def load_tabular_bundle(data_name: str, data_root: str | Path | None = None) -> TabularBundle:
    tabular_datasets = _get_tabular_datasets_module()
    resolved_root = _resolve_data_root(data_root)
    previous_root = tabular_datasets.root
    tabular_datasets.root = str(resolved_root).replace("\\", "/").rstrip("/") + "/"
    try:
        dataset = _dataset_factory(data_name)()
    finally:
        tabular_datasets.root = previous_root

    train_x = torch.from_numpy(dataset.trn.x).float()
    val_x = torch.from_numpy(dataset.val.x).float()
    test_x = torch.from_numpy(dataset.tst.x).float()
    return TabularBundle(
        data_name=data_name,
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        input_dim=int(dataset.n_dims),
        data_root=resolved_root,
    )


def _make_dataset(x: torch.Tensor) -> TensorDataset:
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    return TensorDataset(x, labels)


def build_tabular_loaders(
    data_name: str,
    train_batch_size: int,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    data_root: str | Path | None = None,
) -> Tuple[TabularBundle, DataLoader, DataLoader, DataLoader]:
    bundle = load_tabular_bundle(data_name, data_root=data_root)
    if eval_batch_size is None:
        eval_batch_size = train_batch_size

    train_loader = DataLoader(
        _make_dataset(bundle.train_x),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )
    val_loader = DataLoader(
        _make_dataset(bundle.val_x),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )
    test_loader = DataLoader(
        _make_dataset(bundle.test_x),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )
    return bundle, train_loader, val_loader, test_loader
