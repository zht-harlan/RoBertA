# Graph Benchmark Runner

This directory adds a minimal node-classification pipeline for:

- `ogbn-arxiv`
- `cora`
- `pubmed`
- `amazon-photo`

Metrics:

- `accuracy`
- `f1_macro`

## Environment

Use the existing server conda environment, then install this repo and the graph dependencies:

```bash
conda activate <your-env>
cd /path/to/fairseq
pip install --no-build-isolation -e .
pip install ogb==1.3.6 torch_geometric==2.6.1 scikit_learn==1.4.1.post1
```

If `torch==2.5.1+cu121` is already present in that environment, keep it as-is.

## Run

Single run on `ogbn-arxiv`:

```bash
python -m graph_exp.train_node_classification \
  --dataset ogbn-arxiv \
  --model gcn \
  --device cuda \
  --epochs 500 \
  --patience 100
```

Five runs on `cora`:

```bash
python -m graph_exp.train_node_classification \
  --dataset cora \
  --model gcn \
  --device cuda \
  --runs 5 \
  --seed 42
```

`amazon-photo` uses a stratified random split because the dataset does not ship with official train/val/test masks in PyG:

```bash
python -m graph_exp.train_node_classification \
  --dataset amazon-photo \
  --model sage \
  --device cuda \
  --train-ratio 0.1 \
  --val-ratio 0.1 \
  --runs 5
```

Results are written to `results/graph_exp/<dataset>_<model>.json`.

## Local dataset batches

The runner also supports local datasets stored under `--root/<dataset>/` with
separate graph, label, split, and feature files. The local dataset names
currently supported are:

- `children`
- `history`
- `photo`

For local datasets, feature tensors can be selected with `--feature-type` or
`--feature-types`. Common filenames such as `plm.pt`, `x_plm.pt`,
`features/plm.pt`, `labels.pt`, `edge_index.pt`, `split_idx.pt`,
`train_idx.pt`, and `val_mask.pt` are detected automatically.

Batch run example:

```bash
python -m graph_exp.train_node_classification \
  --root ../dataset \
  --datasets children history photo \
  --feature-types plm \
  --models mlp gcn sage gat sgc jknet appnp \
  --num-layers 1 \
  --hidden-dim 256 \
  --dropout 0.2 \
  --lr 0.01 \
  --runs 3 \
  --output-dir outputs_plm
```

Outputs are written as `outputs_plm/<dataset>_<feature_type>_<model>.json`.
