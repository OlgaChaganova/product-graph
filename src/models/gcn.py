import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv
from tqdm import tqdm


class GraphConvNetwork(nn.Module):
    def __init__(
        self,
        num_features: tp.List[int],
        num_classes: int,
        dropout: float
    ):
        super().__init__()
        self.layers = []
        num_layers = len(num_features)
        for i in range(num_layers - 1):
            self.layers.append(
                GCNConv(in_channels=num_features[i], out_channels=num_features[i + 1]),
            )
        self.layers.append(
            GCNConv(in_channels=num_features[-1], out_channels=num_classes),
        )
        self.dropout = dropout

    def forward(self, x: torch.tensor, adj_t: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all: torch.tensor, subgraph_loader: NeighborSampler) -> torch.tensor:
        pbar = tqdm(total=x_all.size(0) * len(self.layers))
        pbar.set_description('Inference')
        for i, conv in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                x = x_all[n_id]
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.layers) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
