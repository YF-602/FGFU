import torch
import torch.nn as nn
from torch_scatter import scatter
from models.mlp import MLP


class FGFUConv(nn.Module):
    """
    FGFU Convolutional Neural
    This work is co-opted from "Molecular Hypergraph Neural Networks",
    but is used in a different way.

    Args:
        config: the config about this model
    """
    def __init__(self, config):
        super().__init__()
        
        self.hid_dim = config["hidden_dim"]
        self.mlp1_layers = config["mlp1_layers"]
        self.mlp2_layers = config["mlp2_layers"]
        self.mlp3_layers = config["mlp3_layers"]
        self.mlp4_layers = config["mlp4_layers"]
        self.aggr = config["aggr"]
        self.dropout = config["dropout"]
        self.normalization = config["normalization"]
        self.input_norm = config["input_norm"]

        if self.mlp1_layers > 0:
            self.W1 = MLP(self.hid_dim*2, self.hid_dim, self.hid_dim, self.mlp1_layers,
                dropout=self.dropout, Normalization=self.normalization, InputNorm=self.input_norm)
        else:
            self.W1 = lambda X: X[..., self.hid_dim:]

        if self.mlp2_layers > 0:
            self.W2 = MLP(self.hid_dim*2, self.hid_dim, self.hid_dim, self.mlp2_layers,
                dropout=self.dropout, Normalization=self.normalization, InputNorm=self.input_norm)
        else:
            self.W2 = lambda X: X[..., self.hid_dim:]

        if self.mlp3_layers > 0:
            self.W3 = MLP(self.hid_dim*2, self.hid_dim, self.hid_dim, self.mlp3_layers,
                dropout=self.dropout, Normalization=self.normalization, InputNorm=self.input_norm)
        else:
            self.W3 = lambda X: X[..., self.hid_dim:]

        if self.mlp4_layers > 0:
            self.W4 = MLP(self.hid_dim*2, self.hid_dim, self.hid_dim, self.mlp4_layers,
                dropout=self.dropout, Normalization=self.normalization, InputNorm=self.input_norm)
        else:
            self.W4 = lambda X: X[..., self.hid_dim:]

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W3, MLP):
            self.W3.reset_parameters()
        if isinstance(self.W4, MLP):
            self.W4.reset_parameters()

    def forward(self, X, E, vertex, edges):
        N = X.shape[-2]

        Mve = self.W1(torch.cat((X[..., vertex, :], E[..., edges, :]), -1))
        Me = scatter(Mve, edges, dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        E = self.W2(torch.cat((E, Me), -1))
        # E = E*0.5 + e_in*0.5  # Residual connection.
        Mev = self.W3(torch.cat((X[..., vertex, :], E[..., edges, :]), -1))
        Mv = scatter(Mev, vertex, dim=-2, reduce=self.aggr, dim_size=N)
        X = self.W4(torch.cat((X, Mv), -1))
        # X = X*0.5 + X0*0.5  # Residual connection.

        return X, E
