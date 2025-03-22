import numpy as np
import bisect
from rdkit import Chem
from utils.features import atom_to_feature_vector, bond_to_feature_vector, fg_sindex, DAY_LIGHT_FG_SMARTS_LIST


def canonical_smiles(smiles_list):
    if not isinstance(smiles_list,list):
        smiles_list=[smiles_list]
    ans=[]
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            ans.append("None") #use None to flag the fault smiles
        else:
            ans.append(Chem.MolToSmiles(mol))
    return ans


def he_conj(mol):
    """ get node index and hyperedge index of conjugated structure in a molecule

    Args:
        mol (RDKit MOL): input molecule

    Returns:
        tuple: node index and hyperedge index
    """
    num_atom = mol.GetNumAtoms()
    reso = Chem.ResonanceMolSupplier(mol)
    num_he = reso.GetNumConjGrps()
    # assert num_he != 0
    n_idx, e_idx = [], []
    for i in range(num_atom):
        _conj = reso.GetAtomConjGrpIdx(i)
        if _conj > -1 and _conj < num_he:
            n_idx.append(i)
            e_idx.append(_conj)
    return n_idx, e_idx


def he_fg(mol):
    """ get node index and hyperedge index of functional_group structure in a molecule

    Args:
        mol (RDKit MOL): input molecule

    Returns:
        tuple: node index, hyperedge index and fg_type
    """
    n_idx, e_idx, fg_type = [], [], []
    e_cnt=0
    # 逐一匹配DAY_LIGHT_FG_SMARTS_LIST中的每个官能团
    for i, smarts in enumerate(DAY_LIGHT_FG_SMARTS_LIST):
        idx = bisect.bisect_right(fg_sindex, i) - 1
        # 将SMARTS转为RDKit模式
        pattern = Chem.MolFromSmarts(smarts)
        # 使用GetSubstructMatches方法查找匹配
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            matches_list = [list(match) for match in matches]
            for match in matches_list:
                for atom_id in match:
                    n_idx.append(atom_id)
                    e_idx.append(e_cnt)
                    fg_type.append(idx)
                e_cnt+=1
    return n_idx, e_idx, fg_type


def smi2hgraph(smiles_string):
    """
    Converts a SMILES string to hypergraph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_fvs = []
    for atom in mol.GetAtoms():
        atom_fvs.append(atom_to_feature_vector(atom))

    # bonds
    num_bond_features = 1  # bond type
    if len(mol.GetBonds()) > 0: # mol has bonds
        n_idx, e_idx, bond_fvs = [], [], []
        for i, bond in enumerate(mol.GetBonds()):
            n_idx.append(bond.GetBeginAtomIdx())
            n_idx.append(bond.GetEndAtomIdx())
            e_idx.append(i)
            e_idx.append(i)
            bond_type = bond_to_feature_vector(bond)[0]
            bond_fvs.append([bond_type])

    else:   # mol has no bonds
        print('Invalid SMILES: {}'.format(smiles_string))
        n_idx, e_idx= [], []
        bond_fvs = np.empty((0, num_bond_features), dtype=np.int64)
        return (atom_fvs, n_idx, e_idx, bond_fvs)
    
    # hyperedges for conjugated bonds
    he_n, he_e = he_conj(mol)
    num_bond = mol.GetNumBonds()
    CONJUGATED_BOND_TYPE = 5  # 定义共轭键的索引
    if len(he_n) != 0:
        he_e = [_id + num_bond for _id in he_e]
        n_idx += he_n
        e_idx += he_e
        
        num_conj_bond = len(np.unique(he_e))
        bond_fvs += num_conj_bond * [num_bond_features * [CONJUGATED_BOND_TYPE]]
    else:
        num_conj_bond = 0
    
    # hyperedges for functional_group bonds
    he_n, he_e, fg_type = he_fg(mol)
    delt = 6 #because there are 6 bond_type before
    if len(he_n) != 0:
        he_e = [_id + num_bond + num_conj_bond for _id in he_e]
        n_idx += he_n
        e_idx += he_e
        
        unique_edges = np.unique(he_e)
        edge_type_dict = {}
        for edge, etype in zip(he_e, fg_type):
            edge_type_dict[edge] = etype  # 只记录最后一次出现的类型
        bond_fvs += [[edge_type_dict[e]+delt] for e in unique_edges]
    
    return (atom_fvs, n_idx, e_idx, bond_fvs)


if __name__ == "__main__":
    smiles = 'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5'
    atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smiles)
    print(len(atom_fvs))
    print(len(n_idx))
    print(len(e_idx))
    print(bond_fvs)
    print(np.unique(e_idx))

