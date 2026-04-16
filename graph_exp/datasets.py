from __future__ import annotations

import json
import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    feature_type: str


PUBLIC_DATASET_ALIASES = {
    "ogbn-arxiv": "ogbn-arxiv",
    "arxiv": "ogbn-arxiv",
    "cora": "cora",
    "pubmed": "pubmed",
    "amazon-photo": "amazon-photo",
    "amazonphoto": "amazon-photo",
    "photo": "photo",
}

LOCAL_DATASET_ALIASES = {
    "children": "children",
    "history": "history",
    "photo": "photo",
}

LOCAL_FEATURE_ALIASES = {
    "plm": "roberta-base-512-cls",
    "roberta": "roberta-base-512-cls",
    "roberta-base": "roberta-base-512-cls",
    "roberta-base-512-cls": "roberta-base-512-cls",
    "qwen": "qwen2.5-7b-256-mean",
    "qwen2.5": "qwen2.5-7b-256-mean",
    "qwen2.5-7b-256-mean": "qwen2.5-7b-256-mean",
}


def _canonicalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in LOCAL_DATASET_ALIASES:
        return LOCAL_DATASET_ALIASES[normalized]
    if normalized in PUBLIC_DATASET_ALIASES:
        return PUBLIC_DATASET_ALIASES[normalized]
    raise ValueError(
        f"Unsupported dataset '{name}'. Expected one of: "
        "ogbn-arxiv, cora, pubmed, amazon-photo, children, history, photo."
    )


def _normalize_feature_type(feature_type: str | None) -> str:
    if feature_type is None:
        return "raw"
    normalized = feature_type.strip().lower().replace("_", "-")
    return normalized or "raw"


def _to_mask(indices: torch.Tensor, size: int) -> torch.Tensor:
    indices = indices.view(-1).long()
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


def _load_public_dataset(
    canonical_name: str,
    root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> LoadedGraphDataset:
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
            feature_type="raw",
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
        feature_type="raw",
    )


def _load_tensor_file(path: Path) -> torch.Tensor | dict | list:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npy":
        return torch.from_numpy(np.load(path, allow_pickle=True))
    if suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        return {key: torch.from_numpy(loaded[key]) for key in loaded.files}
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file format: {path}")


def _resolve_existing_file(base_dir: Path, relative_candidates: list[str]) -> Path:
    for relative_path in relative_candidates:
        candidate = base_dir / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find any of the expected files under {base_dir}: "
        + ", ".join(relative_candidates)
    )


def _resolve_case_insensitive_dir(root: Path, target_name: str) -> Path:
    direct_path = root / target_name
    if direct_path.exists():
        return direct_path

    target_lower = target_name.lower()
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == target_lower:
            return child

    raise FileNotFoundError(f"Dataset directory does not exist under {root}: {target_name}")


def _coerce_feature_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError("Feature file must resolve to a tensor-like value.")
    if value.dim() != 2:
        raise ValueError(f"Feature tensor must be 2D, got shape {tuple(value.shape)}.")
    return value.to(torch.float)


def _coerce_edge_index(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError("Edge file must resolve to a tensor-like value.")
    if value.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got shape {tuple(value.shape)}.")
    if value.size(0) != 2 and value.size(1) == 2:
        value = value.t()
    if value.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges], got {tuple(value.shape)}.")
    return value.long().contiguous()


def _coerce_labels(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError("Label file must resolve to a tensor-like value.")
    if value.dim() > 1:
        value = value.view(-1)
    value = value.long()
    unique_labels = torch.unique(value)
    if unique_labels.numel() == 0:
        raise ValueError("Label tensor is empty.")
    if unique_labels[0].item() != 0 or unique_labels[-1].item() != unique_labels.numel() - 1:
        remapped = torch.empty_like(value)
        for new_id, old_id in enumerate(unique_labels.tolist()):
            remapped[value == old_id] = new_id
        value = remapped
    return value


def _dataset_stem(dataset_dir: Path) -> str:
    return dataset_dir.name


def _load_local_dataframe(dataset_dir: Path) -> pd.DataFrame:
    stem = _dataset_stem(dataset_dir)
    csv_path = _resolve_existing_file(
        dataset_dir,
        [
            f"{stem}.csv",
            f"{stem.lower()}.csv",
            "data.csv",
        ],
    )
    return pd.read_csv(csv_path)


def _parse_neighbour_list(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
    raise ValueError(f"Unsupported neighbour field: {value!r}")


def _build_edge_index_from_dataframe(df: pd.DataFrame) -> torch.Tensor:
    if "node_id" not in df.columns or "neighbour" not in df.columns:
        raise ValueError("Local dataset CSV must contain 'node_id' and 'neighbour' columns.")

    sources: list[int] = []
    targets: list[int] = []
    for row in df.itertuples(index=False):
        node_id = int(getattr(row, "node_id"))
        neighbours = _parse_neighbour_list(getattr(row, "neighbour"))
        for neighbour in neighbours:
            sources.append(node_id)
            targets.append(neighbour)

    if not sources:
        raise ValueError("No edges were parsed from the 'neighbour' column.")

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    reversed_edge_index = edge_index[[1, 0], :]
    return torch.cat([edge_index, reversed_edge_index], dim=1).contiguous()


def _load_feature_tensor(dataset_dir: Path, feature_type: str) -> torch.Tensor:
    normalized_feature_type = LOCAL_FEATURE_ALIASES.get(feature_type, feature_type)
    stem = _dataset_stem(dataset_dir)
    candidates = [
        f"{feature_type}.pt",
        f"{feature_type}.pth",
        f"{feature_type}.npy",
        f"x_{feature_type}.pt",
        f"x_{feature_type}.npy",
        f"{feature_type}_x.pt",
        f"{feature_type}_x.npy",
        f"features_{feature_type}.pt",
        f"features_{feature_type}.npy",
        f"{feature_type}_features.pt",
        f"{feature_type}_features.npy",
        f"feature_{feature_type}.pt",
        f"feature_{feature_type}.npy",
        f"features/{feature_type}.pt",
        f"features/{feature_type}.npy",
        f"features/{feature_type}/x.pt",
        f"features/{feature_type}/x.npy",
        f"Feature/{stem}_{normalized_feature_type.replace('-', '_')}.npy",
        f"Feature/{stem}_{normalized_feature_type.replace('-', '_')}.pt",
        f"Feature/{stem}_{normalized_feature_type.replace('-', '.').replace('.', '_')}.npy",
    ]
    if feature_type == "raw":
        candidates.extend(["x.pt", "x.npy", "features.pt", "features.npy"])
    if normalized_feature_type != feature_type:
        candidates.extend(
            [
                f"{normalized_feature_type}.pt",
                f"{normalized_feature_type}.npy",
                f"Feature/{stem}_{normalized_feature_type.replace('-', '_')}.npy",
            ]
        )
    feature_path = _resolve_existing_file(dataset_dir, candidates)
    return _coerce_feature_tensor(_load_tensor_file(feature_path))


def _load_labels(dataset_dir: Path) -> torch.Tensor:
    label_path = _resolve_existing_file(
        dataset_dir,
        [
            "y.pt",
            "y.npy",
            "labels.pt",
            "labels.npy",
            "label.pt",
            "label.npy",
            "targets.pt",
            "targets.npy",
        ],
    )
    return _coerce_labels(_load_tensor_file(label_path))


def _load_edge_index(dataset_dir: Path) -> torch.Tensor:
    edge_path = _resolve_existing_file(
        dataset_dir,
        [
            "edge_index.pt",
            "edge_index.npy",
            "edges.pt",
            "edges.npy",
            "graph.pt",
            "graph.npy",
            "adjacency.pt",
            "adjacency.npy",
        ],
    )
    loaded = _load_tensor_file(edge_path)
    if isinstance(loaded, dict):
        for key in ["edge_index", "edges", "graph"]:
            if key in loaded:
                loaded = loaded[key]
                break
    return _coerce_edge_index(loaded)


def _extract_split_indices(value: object, num_nodes: int) -> torch.Tensor:
    if isinstance(value, dict):
        for key in ["index", "idx", "indices"]:
            if key in value:
                value = value[key]
                break
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            if value.numel() != num_nodes:
                raise ValueError("Boolean split mask does not match number of nodes.")
            return value.nonzero(as_tuple=False).view(-1).long()
        return value.view(-1).long()
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            if value.size != num_nodes:
                raise ValueError("Boolean split mask does not match number of nodes.")
            return torch.from_numpy(np.flatnonzero(value)).long()
        return torch.from_numpy(value.reshape(-1)).long()
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.long).view(-1)
    raise TypeError(f"Unsupported split value type: {type(value)!r}")


def _load_split_indices(dataset_dir: Path, split_name: str, num_nodes: int) -> torch.Tensor:
    canonical_names = {
        "train": ["train"],
        "valid": ["valid", "val"],
        "test": ["test"],
    }[split_name]

    bundled_candidates = [
        "split_idx.pt",
        "split_idx.pth",
        "split_idx.npz",
        "split_idx.json",
        "splits.pt",
        "splits.pth",
        "splits.npz",
        "splits.json",
        "split.pt",
        "split.pth",
        "split.npz",
        "split.json",
    ]
    for candidate_name in bundled_candidates:
        candidate_path = dataset_dir / candidate_name
        if not candidate_path.exists():
            continue
        loaded = _load_tensor_file(candidate_path)
        if isinstance(loaded, dict):
            for key in canonical_names:
                if key in loaded:
                    return _extract_split_indices(loaded[key], num_nodes)

    for key in canonical_names:
        for relative_path in [
            f"{key}_idx.pt",
            f"{key}_idx.pth",
            f"{key}_idx.npy",
            f"{key}_index.pt",
            f"{key}_index.pth",
            f"{key}_index.npy",
            f"{key}_mask.pt",
            f"{key}_mask.pth",
            f"{key}_mask.npy",
        ]:
            file_path = dataset_dir / relative_path
            if file_path.exists():
                return _extract_split_indices(_load_tensor_file(file_path), num_nodes)

    raise FileNotFoundError(f"Could not resolve {split_name} split files in {dataset_dir}.")


def _load_local_dataset(
    canonical_name: str,
    root: Path,
    feature_type: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> LoadedGraphDataset:
    dataset_dir = _resolve_case_insensitive_dir(root, canonical_name)
    dataframe = _load_local_dataframe(dataset_dir)
    if "label" not in dataframe.columns:
        raise ValueError(f"Local dataset CSV must contain a 'label' column: {dataset_dir}")
    if "node_id" not in dataframe.columns:
        raise ValueError(f"Local dataset CSV must contain a 'node_id' column: {dataset_dir}")

    dataframe = dataframe.sort_values("node_id").reset_index(drop=True)
    expected_node_ids = np.arange(len(dataframe))
    actual_node_ids = dataframe["node_id"].to_numpy()
    if not np.array_equal(actual_node_ids, expected_node_ids):
        raise ValueError(
            f"Expected node_id to be contiguous from 0..N-1 in {dataset_dir}, "
            f"got first ids {actual_node_ids[:5].tolist()}."
        )

    x = _load_feature_tensor(dataset_dir, feature_type)
    y = _coerce_labels(torch.tensor(dataframe["label"].to_numpy(), dtype=torch.long))
    edge_index = _build_edge_index_from_dataframe(dataframe)

    if x.size(0) != y.numel():
        raise ValueError(
            f"Feature rows ({x.size(0)}) and label count ({y.numel()}) do not match for {dataset_dir}."
        )

    train_mask, val_mask, test_mask = _build_random_split(
        labels=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return LoadedGraphDataset(
        data=data,
        num_features=x.size(-1),
        num_classes=int(y.max().item()) + 1,
        canonical_name=canonical_name,
        feature_type=feature_type,
    )


def load_dataset(
    name: str,
    root: str | Path,
    seed: int,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    feature_type: str | None = None,
) -> LoadedGraphDataset:
    canonical_name = _canonicalize_name(name)
    normalized_feature_type = _normalize_feature_type(feature_type)
    root = Path(root)

    if canonical_name in {"children", "history", "photo"}:
        return _load_local_dataset(
            canonical_name=canonical_name,
            root=root,
            feature_type=normalized_feature_type,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    return _load_public_dataset(
        canonical_name=canonical_name,
        root=root,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
