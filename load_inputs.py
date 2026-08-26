import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def load_anchor(anchor_file):
    check_file_exists(anchor_file)
    supplier = Chem.SDMolSupplier(anchor_file)
    if supplier is None or len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Invalid anchor file: {anchor_file}")
    return supplier[0]

def load_reaction(rxn_file):
    check_file_exists(rxn_file)
    reaction = rdChemReactions.ReactionFromRxnFile(rxn_file)
    if reaction is None:
        raise ValueError(f"Invalid reaction file: {rxn_file}")
    return reaction

def load_reagents(reagent_files):
    reagents = []
    for file_path in reagent_files:
        check_file_exists(file_path)
        df = pd.read_csv(file_path, names=["SMILES"], delimiter="\t")
        reagents.append(df)
    return reagents

def load_rxn_inputs(anchor_file, rxn_file, reagent_files):
    anchor_mol = load_anchor(anchor_file)
    reaction = load_reaction(rxn_file)
    reagents = load_reagents(reagent_files)
    return anchor_mol, reaction, reagents
