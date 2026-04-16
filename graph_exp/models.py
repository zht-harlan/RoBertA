from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import APPNP, GATConv, GCNConv, JumpingKnowledge, SAGEConv, SGConv


def _require_layers(num_layers: int) -> None:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)

        self.dropout = dropout
        if num_layers == 1:
            self.layers = nn.ModuleList()
            self.output = nn.Linear(in_channels, out_channels)
            return

        layers = [nn.Linear(in_channels, hidden_channels)]
        layers.extend(nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 2))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)

        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GCNConv(in_channels, out_channels))
            return

        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)

        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(SAGEConv(in_channels, out_channels))
            return

        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        heads: int = 4,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)

        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
            return

        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        self.layers.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index)


class SGC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)
        self.conv = SGConv(in_channels, out_channels, K=num_layers, cached=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class JKNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.jump = JumpingKnowledge(mode="cat")
        self.output = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        x = self.jump(layer_outputs)
        return self.output(x)


class APPNPNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_layers(num_layers)

        self.dropout = dropout
        self.input = nn.Linear(in_channels, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels) for _ in range(max(0, num_layers - 1))
        )
        self.output = nn.Linear(hidden_channels, out_channels)
        self.propagation = APPNP(K=max(1, num_layers), alpha=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        return self.propagation(x, edge_index)


def build_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    dropout: float,
    gat_heads: int,
) -> nn.Module:
    normalized = model_name.strip().lower()
    if normalized == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if normalized == "sage":
        return GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if normalized == "gat":
        return GAT(in_channels, hidden_channels, out_channels, num_layers, dropout, heads=gat_heads)
    if normalized == "mlp":
        return MLP(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if normalized == "sgc":
        return SGC(in_channels, out_channels, num_layers)
    if normalized == "jknet":
        return JKNet(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if normalized == "appnp":
        return APPNPNet(in_channels, hidden_channels, out_channels, num_layers, dropout)
    raise ValueError("Unsupported model. Expected one of: mlp, gcn, sage, gat, sgc, jknet, appnp.")
