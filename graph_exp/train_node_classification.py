from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F

from graph_exp.datasets import load_dataset
from graph_exp.metrics import ClassificationMetrics, compute_classification_metrics
from graph_exp.models import build_model


SUPPORTED_DATASETS = ["ogbn-arxiv", "cora", "pubmed", "amazon-photo", "children", "history", "photo"]
SUPPORTED_MODELS = ["mlp", "gcn", "sage", "gat", "sgc", "jknet", "appnp"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Node classification benchmark runner.")
    parser.add_argument("--dataset")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--feature-type", dest="feature_type")
    parser.add_argument("--feature-types", nargs="+", dest="feature_types")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results/graph_exp")
    args = parser.parse_args()

    if not args.dataset and not args.datasets:
        parser.error("Provide --dataset or --datasets.")
    if not args.model and not args.models:
        parser.error("Provide --model or --models.")

    args.datasets = [_normalize_dataset_name(name) for name in (args.datasets or [args.dataset])]
    args.models = [_normalize_model_name(name) for name in (args.models or [args.model])]
    args.feature_types = [_normalize_feature_name(name) for name in (args.feature_types or ([args.feature_type] if args.feature_type else ["raw"]))]
    return args


def _normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    if normalized not in SUPPORTED_DATASETS:
        raise SystemExit(
            f"Unsupported dataset '{name}'. Choose from: {', '.join(SUPPORTED_DATASETS)}"
        )
    return normalized


def _normalize_model_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_MODELS:
        raise SystemExit(
            f"Unsupported model '{name}'. Choose from: {', '.join(SUPPORTED_MODELS)}"
        )
    return normalized


def _normalize_feature_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


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


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    mask: torch.Tensor,
) -> ClassificationMetrics:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[mask]
        labels = data.y[mask]
    return compute_classification_metrics(logits, labels)


def train_one_run(args: argparse.Namespace, run_seed: int, device: torch.device) -> RunResult:
    set_seed(run_seed)
    loaded = load_dataset(
        name=args.dataset,
        root=args.root,
        seed=run_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_type=args.feature_type,
    )
    data = loaded.data.to(device)

    model = build_model(
        model_name=args.model,
        in_channels=loaded.num_features,
        hidden_channels=args.hidden_dim,
        out_channels=loaded.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, data, data.val_mask)
        improved = (val_metrics.f1_macro > best_val_f1) or (
            np.isclose(val_metrics.f1_macro, best_val_f1) and val_metrics.accuracy > best_val_acc
        )
        if improved:
            best_val_f1 = val_metrics.f1_macro
            best_val_acc = val_metrics.accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    train_metrics = evaluate(model, data, data.train_mask)
    val_metrics = evaluate(model, data, data.val_mask)
    test_metrics = evaluate(model, data, data.test_mask)

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
    feature_suffix = f"_{args.feature_type}" if args.feature_type else ""
    output_path = output_dir / f"{args.dataset}{feature_suffix}_{args.model}.json"
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
            f"dataset={args.dataset} feature={args.feature_type} model={args.model} "
            f"seed={run_seed} best_epoch={result.best_epoch} "
            f"test_acc={result.test.accuracy:.4f} "
            f"test_f1_macro={result.test.f1_macro:.4f}"
        )

    summary = summarise_runs(results)
    output_path = save_results(args, results, summary)
    print(
        f"[summary] dataset={args.dataset} feature={args.feature_type} model={args.model} "
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
        for feature_type in args.feature_types:
            for model_name in args.models:
                run_args = copy.deepcopy(args)
                run_args.dataset = dataset_name
                run_args.feature_type = feature_type
                run_args.model = model_name
                saved_paths.append(run_experiment(run_args, device))

    print(f"[done] saved {len(saved_paths)} result files to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
