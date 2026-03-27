from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.transforms import NormalizeFeatures


@dataclass
class LoadedGraphDataset:
    data: Data
    num_features: int
    num_classes: int
    canonical_name: str


def _canonicalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "ogbn-arxiv": "ogbn-arxiv",
        "arxiv": "ogbn-arxiv",
        "cora": "cora",
        "pubmed": "pubmed",
        "amazon-photo": "amazon-photo",
        "amazonphoto": "amazon-photo",
        "photo": "amazon-photo",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported dataset '{name}'. Expected one of: "
            "ogbn-arxiv, cora, pubmed, amazon-photo."
        )
    return aliases[normalized]


def _to_mask(indices: torch.Tensor, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask


def _build_random_split(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    generator = torch.Generator().manual_seed(seed)
    labels = labels.view(-1).cpu()
    num_nodes = labels.numel()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for class_id in torch.unique(labels):
        class_indices = torch.where(labels == class_id)[0]
        class_indices = class_indices[torch.randperm(class_indices.numel(), generator=generator)]

        train_count = max(1, int(class_indices.numel() * train_ratio))
        val_count = max(1, int(class_indices.numel() * val_ratio))
        remaining = class_indices.numel() - train_count - val_count
        if remaining <= 0:
            val_count = max(1, min(val_count, class_indices.numel() - train_count - 1))
            remaining = class_indices.numel() - train_count - val_count
        if remaining <= 0:
            raise ValueError(
                "Random split produced an empty test subset for at least one class. "
                "Use smaller train/val ratios."
            )

        train_indices = class_indices[:train_count]
        val_indices = class_indices[train_count : train_count + val_count]
        test_indices = class_indices[train_count + val_count :]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def load_dataset(
    name: str,
    root: str | Path,
    seed: int,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
) -> LoadedGraphDataset:
    canonical_name = _canonicalize_name(name)
    root = Path(root)
    transform = NormalizeFeatures()

    if canonical_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "ogb"), transform=transform)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        num_nodes = data.num_nodes
        data.train_mask = _to_mask(split_idx["train"], num_nodes)
        data.val_mask = _to_mask(split_idx["valid"], num_nodes)
        data.test_mask = _to_mask(split_idx["test"], num_nodes)
        data.y = data.y.view(-1)
        return LoadedGraphDataset(
            data=data,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            canonical_name=canonical_name,
        )

    if canonical_name == "cora":
        dataset = Planetoid(root=str(root / "planetoid"), name="Cora", transform=transform)
    elif canonical_name == "pubmed":
        dataset = Planetoid(root=str(root / "planetoid"), name="PubMed", transform=transform)
    else:
        dataset = Amazon(root=str(root / "amazon"), name="Photo", transform=transform)

    data = dataset[0]
    data.y = data.y.view(-1)

    if canonical_name == "amazon-photo":
        train_mask, val_mask, test_mask = _build_random_split(
            labels=data.y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    return LoadedGraphDataset(
        data=data,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        canonical_name=canonical_name,
    )
