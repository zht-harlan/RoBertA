from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from graph_exp.metrics import ClassificationMetrics, compute_classification_metrics


SUPPORTED_DATASETS = ["children", "history", "photo"]


@dataclass
class SplitResult:
    accuracy: float
    f1_macro: float


@dataclass
class RunResult:
    seed: int
    best_epoch: int
    train: SplitResult
    val: SplitResult
    test: SplitResult


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict[str, list[int] | torch.Tensor], labels: torch.Tensor) -> None:
        self.encodings = encodings
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {}
        for key, value in self.encodings.items():
            item[key] = value[idx] if isinstance(value, torch.Tensor) else torch.tensor(value[idx])
        item["labels"] = self.labels[idx]
        return item


def select_encoding_subset(
    encodings: dict[str, list[list[int]]],
    indices: torch.Tensor,
) -> dict[str, list[list[int]]]:
    index_list = indices.tolist()
    return {key: [value[i] for i in index_list] for key, value in encodings.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoBERTa text classification runner.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--root", required=True, help="Root directory containing dataset folders.")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results/roberta_text")
    args = parser.parse_args()

    if not args.dataset and not args.datasets:
        parser.error("Provide --dataset or --datasets.")
    args.datasets = [_normalize_dataset_name(name) for name in (args.datasets or [args.dataset])]
    return args


def _normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise SystemExit(f"Unsupported dataset '{name}'. Choose from: {', '.join(SUPPORTED_DATASETS)}")
    return normalized


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_dataset_dir(root: Path, dataset_name: str) -> Path:
    direct_path = root / dataset_name
    if direct_path.exists():
        return direct_path
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == dataset_name.lower():
            return child
    raise FileNotFoundError(f"Dataset directory does not exist under {root}: {dataset_name}")


def load_dataframe(dataset_dir: Path, dataset_name: str) -> pd.DataFrame:
    stem = dataset_dir.name
    for candidate in [f"{stem}.csv", f"{dataset_name}.csv", f"{stem.lower()}.csv", "data.csv"]:
        path = dataset_dir / candidate
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find CSV file in {dataset_dir}.")


def remap_labels(labels: np.ndarray) -> tuple[torch.Tensor, int]:
    unique_labels = sorted(np.unique(labels).tolist())
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped = np.array([label_map[int(label)] for label in labels], dtype=np.int64)
    return torch.tensor(remapped, dtype=torch.long), len(unique_labels)


def build_stratified_split(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    generator = torch.Generator().manual_seed(seed)
    labels = labels.view(-1).cpu()
    train_indices: list[torch.Tensor] = []
    val_indices: list[torch.Tensor] = []
    test_indices: list[torch.Tensor] = []

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
            raise ValueError("Random split produced an empty test subset for at least one class.")

        train_indices.append(class_indices[:train_count])
        val_indices.append(class_indices[train_count : train_count + val_count])
        test_indices.append(class_indices[train_count + val_count :])

    return (
        torch.cat(train_indices, dim=0),
        torch.cat(val_indices, dim=0),
        torch.cat(test_indices, dim=0),
    )


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> dict[str, list[list[int]]]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, tokenizer):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )


def evaluate(model, dataloader, device: torch.device) -> ClassificationMetrics:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits_list.append(outputs.logits.detach().cpu())
            labels_list.append(batch["labels"].detach().cpu())
    return compute_classification_metrics(torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0))


def train_one_run(args: argparse.Namespace, run_seed: int, device: torch.device) -> RunResult:
    set_seed(run_seed)
    dataset_dir = resolve_dataset_dir(Path(args.root), args.dataset)
    dataframe = load_dataframe(dataset_dir, args.dataset)
    if args.text_column not in dataframe.columns:
        raise ValueError(f"Missing text column '{args.text_column}' in {dataset_dir}.")
    if args.label_column not in dataframe.columns:
        raise ValueError(f"Missing label column '{args.label_column}' in {dataset_dir}.")

    texts = dataframe[args.text_column].fillna("").astype(str).tolist()
    labels, num_labels = remap_labels(dataframe[args.label_column].to_numpy())
    train_idx, val_idx, test_idx = build_stratified_split(
        labels=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=run_seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encodings = tokenize_texts(tokenizer, texts, args.max_length)

    train_dataset = TextClassificationDataset(select_encoding_subset(encodings, train_idx), labels[train_idx])
    val_dataset = TextClassificationDataset(select_encoding_subset(encodings, val_idx), labels[val_idx])
    test_dataset = TextClassificationDataset(select_encoding_subset(encodings, test_idx), labels[test_idx])

    train_loader = make_dataloader(train_dataset, args.batch_size, shuffle=True, tokenizer=tokenizer)
    val_loader = make_dataloader(val_dataset, args.batch_size, shuffle=False, tokenizer=tokenizer)
    test_loader = make_dataloader(test_dataset, args.batch_size, shuffle=False, tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        improved = (val_metrics.f1_macro > best_val_f1) or (
            np.isclose(val_metrics.f1_macro, best_val_f1) and val_metrics.accuracy > best_val_acc
        )
        if improved:
            best_val_f1 = val_metrics.f1_macro
            best_val_acc = val_metrics.accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    return RunResult(
        seed=run_seed,
        best_epoch=best_epoch,
        train=SplitResult(train_metrics.accuracy, train_metrics.f1_macro),
        val=SplitResult(val_metrics.accuracy, val_metrics.f1_macro),
        test=SplitResult(test_metrics.accuracy, test_metrics.f1_macro),
    )


def summarise_runs(results: list[RunResult]) -> dict[str, float]:
    test_acc = [result.test.accuracy for result in results]
    test_f1 = [result.test.f1_macro for result in results]
    return {
        "test_accuracy_mean": mean(test_acc),
        "test_accuracy_std": stdev(test_acc) if len(test_acc) > 1 else 0.0,
        "test_f1_macro_mean": mean(test_f1),
        "test_f1_macro_std": stdev(test_f1) if len(test_f1) > 1 else 0.0,
    }


def save_results(args: argparse.Namespace, results: list[RunResult], summary: dict[str, float]) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_roberta_text.json"
    payload = {
        "config": vars(args),
        "runs": [asdict(result) for result in results],
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_experiment(args: argparse.Namespace, device: torch.device) -> Path:
    results: list[RunResult] = []
    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        result = train_one_run(args, run_seed, device)
        results.append(result)
        print(
            f"[run {run_idx + 1}/{args.runs}] "
            f"dataset={args.dataset} model={args.model_name} "
            f"seed={run_seed} best_epoch={result.best_epoch} "
            f"test_acc={result.test.accuracy:.4f} "
            f"test_f1_macro={result.test.f1_macro:.4f}"
        )

    summary = summarise_runs(results)
    output_path = save_results(args, results, summary)
    print(
        f"[summary] dataset={args.dataset} model={args.model_name} "
        f"acc={summary['test_accuracy_mean']:.4f}+/-{summary['test_accuracy_std']:.4f} "
        f"f1_macro={summary['test_f1_macro_mean']:.4f}+/-{summary['test_f1_macro_std']:.4f}"
    )
    print(f"[saved] {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    saved_paths: list[Path] = []

    for dataset_name in args.datasets:
        run_args = copy.deepcopy(args)
        run_args.dataset = dataset_name
        saved_paths.append(run_experiment(run_args, device))

    print(f"[done] saved {len(saved_paths)} result files to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
