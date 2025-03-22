# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}


# functional groups from https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
DAY_LIGHT_FG_SMARTS_LIST = [
    # C
    "[CX4]",
    "[$([CX2](=C)=C)]",
    "[$([CX3]=[CX3])]",
    "[$([CX2]#C)]",
    # C & O
    "[CX3]=[OX1]",
    "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
    "[CX3](=[OX1])C",
    "[OX1]=CN",
    "[CX3](=[OX1])O",
    "[CX3](=[OX1])[F,Cl,Br,I]",
    "[CX3H1](=O)[#6]",
    "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "[NX3][CX3](=[OX1])[#6]",
    "[NX3][CX3]=[NX3+]",
    "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
    "[NX3][CX3](=[OX1])[OX2H0]",
    "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
    "[CX3](=O)[O-]",
    "[CX3](=[OX1])(O)O",
    "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
    "C[OX2][CX3](=[OX1])[OX2]C",
    "[CX3](=O)[OX2H1]",
    "[CX3](=O)[OX1H0-,OX2H1]",
    "[NX3][CX2]#[NX1]",
    "[#6][CX3](=O)[OX2H0][#6]",
    "[#6][CX3](=O)[#6]",
    "[OD2]([#6])[#6]",
    # H
    "[H]",
    "[!#1]",
    "[H+]",
    "[+H]",
    "[!H]",
    # N
    "[NX3;H2,H1;!$(NC=O)]",
    "[NX3][CX3]=[CX3]",
    "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
    "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
    "[NX3][$(C=C),$(cc)]",
    "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
    "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
    "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
    "[CH3X4]",
    "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
    "[CH2X4][CX3](=[OX1])[NX3H2]",
    "[CH2X4][CX3](=[OX1])[OH0-,OH]",
    "[CH2X4][SX2H,SX1H0-]",
    "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
    "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
    "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
    "[CHX4]([CH3X4])[CH2X4][CH3X4]",
    "[CH2X4][CHX4]([CH3X4])[CH3X4]",
    "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
    "[CH2X4][CH2X4][SX2][CH3X4]",
    "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
    "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
    "[CH2X4][OX2H]",
    "[NX3][CX3]=[SX1]",
    "[CHX4]([CH3X4])[OX2H]",
    "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
    "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
    "[CHX4]([CH3X4])[CH3X4]",
    "N[CX4H2][CX3](=[OX1])[O,N]",
    "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
    "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
    "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
    "[#7]",
    "[NX2]=N",
    "[NX2]=[NX2]",
    "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
    "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
    "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
    "[NX3][NX3]",
    "[NX3][NX2]=[*]",
    "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
    "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "[NX3+]=[CX3]",
    "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
    "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
    "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
    "[NX1]#[CX2]",
    "[CX1-]#[NX2+]",
    "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "[NX2]=[OX1]",
    "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
    # O
    "[OX2H]",
    "[#6][OX2H]",
    "[OX2H][CX3]=[OX1]",
    "[OX2H]P",
    "[OX2H][#6X3]=[#6]",
    "[OX2H][cX3]:[c]",
    "[OX2H][$(C=C),$(cc)]",
    "[$([OH]-*=[!#6])]",
    "[OX2,OX1-][OX2,OX1-]",
    # P
    "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
    "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
    # S
    "[S-][CX3](=S)[#6]",
    "[#6X3](=[SX1])([!N])[!N]",
    "[SX2]",
    "[#16X2H]",
    "[#16!H0]",
    "[#16X2H0]",
    "[#16X2H0][!#16]",
    "[#16X2H0][#16X2H0]",
    "[#16X2H0][!#16].[#16X2H0][!#16]",
    "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
    "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
    "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
    "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
    "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
    "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
    "[SX4](C)(C)(=O)=N",
    "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
    "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
    "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
    "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
    "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
    "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
    "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
    "[#16X2][OX2H,OX1H0-]",
    "[#16X2][OX2H0]",
    # X
    "[#6][F,Cl,Br,I]",
    "[F,Cl,Br,I]",
    "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
]

fg_sindex=[0,4,27,32,86,95,97,124] #共127个官能团，所列是8种官能团的开始编号：C,C&O,H,N,O,P,S,X

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature
# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx, 
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict
# # uses same atom_feature as atom_to_feature_vector test
# atom_feature_dict = atom_feature_vector_to_dict(atom_feature)
# assert atom_feature_dict['atomic_num'] == 6
# assert atom_feature_dict['chirality'] == 'CHI_TETRAHEDRAL_CCW'
# assert atom_feature_dict['degree'] == 4
# assert atom_feature_dict['formal_charge'] == 0
# assert atom_feature_dict['num_h'] == 1
# assert atom_feature_dict['num_rad_e'] == 0
# assert atom_feature_dict['hybridization'] == 'SP3'
# assert atom_feature_dict['is_aromatic'] == False
# assert atom_feature_dict['is_in_ring'] == False

def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx, 
    bond_stereo_idx,
    is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict
# # uses same bond as bond_to_feature_vector test
# bond_feature_dict = bond_feature_vector_to_dict(bond_feature)
# assert bond_feature_dict['bond_type'] == 'DOUBLE'
# assert bond_feature_dict['bond_stereo'] == 'STEREOE'
# assert bond_feature_dict['is_conjugated'] == False
