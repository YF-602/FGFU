import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import pandas as pd
from utils.splitter import random_split, scaffold_split, scaffold_randomized_spliting

def load_tox21_dataset(input_path, seed):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
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

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()
    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)

    # trn_df, val_df, test_df = random_split(new_df, seed=seed)
    train_df, valid_df, test_df = scaffold_split(new_df, seed=seed)

    return train_df, valid_df, test_df


def load_hiv_dataset(input_path, seed, random=False):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    tasks = ["HIV_active"]

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()

    # change label value
    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)

    if random:
        trn_df, val_df, test_df = scaffold_randomized_spliting(
            new_df, tasks=tasks, random_seed=seed
        )
    else:
        trn_df, val_df, test_df = scaffold_split(new_df, seed=seed)

    return trn_df, val_df, test_df


def load_bace_dataset(input_path, seed, random=False):
    input_df = pd.read_csv(input_path, sep=",").rename(columns={"mol": "smiles"})
    smiles_list = input_df["smiles"]
    tasks = ["Class"]

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()

    # for the bace dataset, the prediction target is `Class`

    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)

    if random:
        trn_df, val_df, test_df = scaffold_randomized_spliting(
            new_df, tasks=tasks, random_seed=seed
        )
    else:
        trn_df, val_df, test_df = scaffold_split(new_df, seed=seed)
    return trn_df, val_df, test_df


def load_bbbp_dataset(input_path, seed, random=False):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    tasks = ["p_np"]

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()

    # change the label value
    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)

    if random:
        trn_df, val_df, test_df = scaffold_randomized_spliting(
            new_df, tasks=tasks, random_seed=seed
        )
    else:
        trn_df, val_df, test_df = scaffold_split(new_df, seed=seed)

    return trn_df, val_df, test_df


def load_clintox_dataset(input_path, seed):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = list(input_df["smiles"])
    tasks = ["FDA_APPROVED", "CT_TOX"]
    # remained_list = []
    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
            # remained_list.append(smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()
    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)
    # train_df, valid_df, test_df = random_split(new_df, seed=seed)
    train_df, valid_df, test_df = scaffold_split(new_df, seed=seed)
    return train_df, valid_df, test_df


def load_muv_dataset(input_path, seed):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
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

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
            # remained_list.append(smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()
    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)
    train_df, valid_df, test_df = random_split(new_df, seed=seed)
    return train_df, valid_df, test_df


def load_sider_dataset(input_path, seed):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    tasks = list(input_df.columns)[1:]

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
            # remained_list.append(smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()
    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)
    # train_df, valid_df, test_df = random_split(new_df, seed=seed)
    train_df, valid_df, test_df = scaffold_split(new_df, seed=seed)
    return train_df, valid_df, test_df


def load_toxcast_dataset(input_path, seed):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    tasks = list(input_df.columns)[1:]

    canonical_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(
                mol, isomericSmiles=False, canonical=True
            )
            input_df["smiles"] = input_df["smiles"].replace(smiles, canonical_smiles)
            canonical_smiles_list.append(canonical_smiles)
        else:
            print("not successfully processed smiles: ", smiles)

    # create new dataframe
    new_df = input_df[input_df["smiles"].isin(canonical_smiles_list)].reset_index()

    new_df[tasks] = new_df[tasks].replace(0, -1)
    new_df[tasks] = new_df[tasks].fillna(0)
    # train_df, valid_df, test_df = random_split(new_df, seed=seed)
    train_df, valid_df, test_df = scaffold_split(new_df, seed=seed)
    return train_df, valid_df, test_df
