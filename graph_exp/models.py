from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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
        if num_layers < 2:
            raise ValueError("MLP requires num_layers >= 2.")

        self.dropout = dropout
        layers = [nn.Linear(in_channels, hidden_channels)]
        layers.extend(
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 2)
        )
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
        if num_layers < 2:
            raise ValueError("GCN requires num_layers >= 2.")

        self.dropout = dropout
        self.layers = nn.ModuleList()
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
        if num_layers < 2:
            raise ValueError("GraphSAGE requires num_layers >= 2.")

        self.dropout = dropout
        self.layers = nn.ModuleList()
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
        if num_layers < 2:
            raise ValueError("GAT requires num_layers >= 2.")

        self.dropout = dropout
        self.layers = nn.ModuleList()
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
        return GAT(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            dropout,
            heads=gat_heads,
        )
    if normalized == "mlp":
        return MLP(in_channels, hidden_channels, out_channels, num_layers, dropout)
    raise ValueError("Unsupported model. Expected one of: gcn, sage, gat, mlp.")
