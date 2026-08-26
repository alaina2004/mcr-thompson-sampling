import random
import datamol as dm  # type: ignore
from apply_reaction import apply_multi_reactant_reaction_rdkit
from utils import to_aromatic_smiles

def trial_phase(reaction, reagents, use_random=False, n=1):
    """Run trial phase: generate n products from selected reagents."""
    
    if any(r.empty for r in reagents):
        raise ValueError("One of the reagent lists is empty.")

    # Prepare list of all combinations of reagents
    all_reagent_sets = list(zip(*[r["SMILES"].tolist() for r in reagents]))
    
    if use_random:
        reagent_sets = random.sample(all_reagent_sets, min(n, len(all_reagent_sets)))
    else:
        reagent_sets = all_reagent_sets[:n]

    results = []

    for smiles_set in reagent_sets:
        selected_mols = []
        for smiles in smiles_set:
            mol = dm.to_mol(smiles)
            if mol is None:
                print(f"Invalid SMILES skipped: {smiles}")
                break
            selected_mols.append(mol)
        else:  # Only run reaction if all mols are valid
            try:
                selected_smiles = [to_aromatic_smiles(m) for m in selected_mols]
                print(f"Selected Reagents SMILES: {selected_smiles}")
                product_smiles = apply_multi_reactant_reaction_rdkit(reaction, selected_mols)
                if product_smiles:
                    results.append({
                        "product": product_smiles,
                        "reagents": smiles_set
                    })
            except Exception as e:
                print(f"Reaction failed for reagents {smiles_set}: {e}")

    return results
