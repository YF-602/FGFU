import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from utils.dataset_load import *
from utils.utils import smi2hgraph


class HData(Data):
    """ PyG data class for molecular hypergraphs
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None,
                 edge_index0=None, edge_index1=None, n_e=None, smi=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.edge_index0 = edge_index0
        self.edge_index1 = edge_index1
        self.n_e = n_e #边的总数
        self.smi = smi

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index0':
            return self.x.size(0)
        if key == 'edge_index1':
            return self.n_e
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
        
class RawMoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        seed,
        mode="train",
        dataset="bace",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):

        self.dataset = dataset
        self.root = root
        self.seed = seed
        self.mode = mode
        self.weight = 0
        super(RawMoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.transform, self.pre_transform, self.pre_filter = (
            transform,
            pre_transform,
            pre_filter,
        )

        if self.mode == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == "valid":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ["{}.csv".format(self.dataset)]

    @property
    def processed_file_names(self):
        return [
            "{}_train_{}.pt".format(self.dataset, self.seed),
            "{}_val_{}.pt".format(self.dataset, self.seed),
            "{}_test_{}.pt".format(self.dataset, self.seed),
        ]

    def download(self):
        pass

    def process(self):
        path_prefix_name = ["train", "valid", "test"]

        if self.dataset == "bace":
            trn_df, valid_df, test_df = load_bace_dataset(
                self.raw_paths[0], self.seed, random=False
            )
            tasks = ["Class"]
        elif self.dataset == "bbbp":
            trn_df, valid_df, test_df = load_bbbp_dataset(
                self.raw_paths[0], self.seed, random=False
            )
            tasks = ["p_np"]
        elif self.dataset == "hiv":
            trn_df, valid_df, test_df = load_hiv_dataset(
                self.raw_paths[0], self.seed, random=False
            )
            tasks = ["HIV_active"]
        elif self.dataset == "clintox":
            trn_df, valid_df, test_df = load_clintox_dataset(
                self.raw_paths[0], self.seed
            )
            tasks = ["FDA_APPROVED", "CT_TOX"]
        elif self.dataset == "tox21":
            trn_df, valid_df, test_df = load_tox21_dataset(
                self.raw_paths[0], self.seed)
            tasks = [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ]
        elif self.dataset == "sider":
            trn_df, valid_df, test_df = load_sider_dataset(
                self.raw_paths[0], self.seed)
            tasks = list(trn_df.columns)[2:]
            # print(tasks)
        elif self.dataset == "toxcast":
            trn_df, valid_df, test_df = load_toxcast_dataset(
                self.raw_paths[0], self.seed
            )
            tasks = list(trn_df.columns)[2:]
        elif self.dataset == "muv":
            trn_df, valid_df, test_df = load_muv_dataset(
                self.raw_paths[0], self.seed)
            tasks = [
                "MUV-466",
                "MUV-548",
                "MUV-600",
                "MUV-644",
                "MUV-652",
                "MUV-689",
                "MUV-692",
                "MUV-712",
                "MUV-713",
                "MUV-733",
                "MUV-737",
                "MUV-810",
                "MUV-832",
                "MUV-846",
                "MUV-852",
                "MUV-858",
                "MUV-859",
            ]

        else:
            raise ValueError("Not Supported Dataset")
        
        
        for i, dataframe in enumerate([trn_df, valid_df, test_df]):
            data_list = []
            label_list = dataframe[tasks].to_numpy()
            label_list = torch.tensor(label_list, dtype=torch.float)
            smiles_list = dataframe["smiles"].to_numpy()
            
            for j in tqdm(range(len(label_list))):
                smiles = smiles_list[j]
                # Then, we need to judge whether the SMILES is valid
                # But, when you "load_{}_dataset", the step has been done
                # So, we don't need to consider it
                
                # Then, we need to get the attr of the molecule

                mol = Chem.MolFromSmiles(smiles)
                if(len(mol.GetBonds()) <= 0):
                    continue

                atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smiles)
                x = torch.tensor(atom_fvs, dtype=torch.long)
                edge_index0 = torch.tensor(n_idx, dtype=torch.long)
                edge_index1 = torch.tensor(e_idx, dtype=torch.long)
                edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
                y = label_list[j].unsqueeze(0)
                n_e = len(edge_index1.unique())
                
                data = HData(x=x, y=y, n_e=n_e, smi=smiles,
                            edge_index0=edge_index0,
                            edge_index1=edge_index1,
                            edge_attr=edge_attr)
                
                data_list.append(data)
                
            data, slices = self.collate(data_list)
            
            torch.save((data, slices), self.processed_paths[i])
                
                
                
                
