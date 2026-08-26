import datamol as dm # type: ignore
from rdkit import Chem # type: ignore


def to_aromatic_smiles(mol):
    if mol is None:
        return None
    try:
        return dm.to_smiles(mol, kekulize=False)
    except Exception:
        return None
    
def save_mol_to_sdf(mol, filename):
    writer = Chem.SDWriter(filename)
    writer.write(mol)
    writer.close()

