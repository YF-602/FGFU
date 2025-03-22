import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from utils.mol_encoder import AtomEncoder, HbondEncoder

from models.conv import FGFUConv
from models.mlp import MLP


class FGFU(nn.Module):
    def __init__(self,config):
        """Functional group fusion Neural Network (FGFU)
        (Shared parameters between all message passing layers)
        This work is co-opted from "Molecular Hypergraph Neural Networks",
        but is used in a different way.
        
        Args:
            num_target (int): number of output targets
            args (NamedTuple): global args
        """
        super().__init__()
        
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}

        self.nlayer = config["All_num_layers"]
        self.emb_dim = config["emb_dim"]

        self.conv_config = config["conv_config"]
        self.mlp_config = config["mlp_config"]
        
        self.act = act[config["activation"]]
        self.dropout = nn.Dropout(config["dropout"])
        
        self.atom_encoder = AtomEncoder(emb_dim=self.emb_dim)
        self.hbond_encoder = HbondEncoder(emb_dim=self.emb_dim)
        
        self.conv = FGFUConv(self.conv_config)
        
        self.mlp_out = MLP(in_channels=self.mlp_config["in_channels"],
                          hidden_channels=self.mlp_config["hidden_channels"],
                          out_channels=self.mlp_config["out_channels"],
                          num_layers=self.mlp_config["num_layers"],
                          dropout=self.mlp_config["dropout"],
                          Normalization=self.mlp_config["normalization"],
                          InputNorm=False)

    def forward(self, data):
        V, E = data.edge_index0, data.edge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x.device)

        x = self.atom_encoder(data.x)
        e = self.hbond_encoder(data.edge_attr.squeeze())

        for i in range(self.nlayer):
            x, e = self.conv(x, e, V, E)
            if i == self.nlayer - 1:
                # remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_add_pool(x, data.batch)
        e = global_add_pool(e, e_batch)
        # print(f"x shape: {x.shape}, e shape: {e.shape}")
        # min_size = min(x.shape[0], e.shape[0])
        # x, e = x[:min_size], e[:min_size]
        out = self.mlp_out(torch.cat((x, e), -1))
        return out.view(-1)
            
                
