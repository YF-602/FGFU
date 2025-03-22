import torch
from utils.features import get_atom_feature_dims, get_bond_feature_dims

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, optional_full_atom_features_dims=None):
        super(AtomEncoder, self).__init__()


        self.atom_embedding_list = torch.nn.ModuleList()

        if optional_full_atom_features_dims is not None:
            full_atom_feature_dims = optional_full_atom_features_dims
        else:
            full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding
    
class HbondEncoder(torch.nn.Module):
    def __init__(self,emb_dim):
        super(HbondEncoder,self).__init__()
        
        normal_bondtype_num=5 #single,double,triple,aromatic,anony(匿名的，即，如果不是前面四个，则为anony)
        conjugated_bondtype_num=1 #conjugated
        fgroup_bondtype_num=8 #C,C&O,H,N,O,P,S,X
        total_bondtype_num=normal_bondtype_num+conjugated_bondtype_num+fgroup_bondtype_num
        
        self.emb = torch.nn.Embedding(total_bondtype_num,emb_dim)
        torch.nn.init.xavier_uniform_(self.emb.weight.data)
        
    def forward(self, hedge_attr): #注意：hedge_attr只有键类型
        hbond_embedding = self.emb(hedge_attr)
        return hbond_embedding
