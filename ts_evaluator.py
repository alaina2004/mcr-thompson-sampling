import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from trial_phase import trial_phase
from docking import calculate_center_and_size_from_mol
from align_core_to_ligand import (
    find_mcs_between_core_and_ligand,
    align_full_ligand_to_core,
    generate_3d_from_smiles,
    save_mol,
    save_mol_as_pdbqt
)
from MolecularConformerOptimizer import genetic_docking



# Simple cache to avoid redoing the entire expensive pipeline for identical inputs
_evaluation_cache = {}

def evaluate_arm(
    arm,
    reaction,
    core,
    receptor_pdbqt,
    ga_kwargs,
    failure_score=-1e6,
    use_cache=True
):
    """
    arm: tuple (r1, r2, r3)
    core: pre-aligned core RDKit Mol (with 3D coords)
    ga_kwargs: dict of arguments to pass to genetic_docking (generations, pop_size, etc.)
    Returns: (score: float, product_smiles: str, best_mol: RDKit Mol)
    """
    key = (arm, )  # don't include mutable objects like core in cache key
    if use_cache and key in _evaluation_cache:
        return _evaluation_cache[key]

    r1, r2, r3 = arm
    reagents = [
        pd.DataFrame({"SMILES": [r1]}),
        pd.DataFrame({"SMILES": [r2]}),
        pd.DataFrame({"SMILES": [r3]}),
    ]

    # 1. Reaction product
    try:
        results = trial_phase(reaction, reagents, use_random=False, n=1)
    except Exception as e:
        print(f"[evaluate_arm] reaction failed for {arm}: {e}")
        return failure_score, None, None

    if not results:
        return failure_score, None, None

    product_smiles = results[0].get("product")
    if not product_smiles:
        return failure_score, None, None

    # 2. Generate 3D
    try:
        full_ligand = generate_3d_from_smiles(product_smiles)
    except Exception as e:
        print(f"[evaluate_arm] 3D generation failed for {arm}: {e}")
        return failure_score, product_smiles, None

    # 2b. Cheap drug-likeness screen before expensive alignment/GA work
    from MolecularConformerOptimizer import check_lipinski_veber
    if not check_lipinski_veber(full_ligand):
        print(f"[evaluate_arm] {arm} skipped: not drug-like")
        return failure_score, product_smiles, None

    # 3. MCS between core and ligand
    try:
        mcs_mol = find_mcs_between_core_and_ligand(core, full_ligand)
    except Exception as e:
        print(f"[evaluate_arm] MCS detection failed for {arm}: {e}")
        return failure_score, product_smiles, None

    if mcs_mol is None:
        print(f"[evaluate_arm] No MCS found for {arm}")
        return failure_score, product_smiles, None

    # 4. Align ligand to core
    try:
        aligned_ligand, frozen_indices = align_full_ligand_to_core(full_ligand, core, mcs_mol)
        os.makedirs("aligned_ligands", exist_ok=True)
        ligand_index = len(os.listdir("aligned_ligands")) + 1
        sdf_path = f"aligned_ligands/aligned_{ligand_index}.sdf"

        save_mol(aligned_ligand, sdf_path)
    except Exception as e:
        print(f"[evaluate_arm] Alignment failed for {arm}: {e}")
        return failure_score, product_smiles, None

    if not frozen_indices:
        print(f"[evaluate_arm] No frozen indices for {arm}")
        return failure_score, product_smiles, None

    # Copy exact coordinates for MCS atoms
    try:
        conf_ligand = aligned_ligand.GetConformer()
        conf_core = core.GetConformer()
        match_ligand = aligned_ligand.GetSubstructMatch(mcs_mol)
        match_core = core.GetSubstructMatch(mcs_mol)
        for lig_idx, core_idx in zip(match_ligand, match_core):
            pos = conf_core.GetAtomPosition(core_idx)
            conf_ligand.SetAtomPosition(lig_idx, pos)
    except Exception as e:
        print(f"[evaluate_arm] Coordinate copy failed for {arm}: {e}")
        # continue; alignment may still be usable

    # Build coord_map and box
    coord_map = {
        idx: aligned_ligand.GetConformer().GetAtomPosition(idx)
        for idx in frozen_indices
    }

    try:
        cx, cy, cz, sx, sy, sz = calculate_center_and_size_from_mol(aligned_ligand)
        center = (cx, cy, cz)
        size = (sx, sy, sz)
    except Exception as e:
        print(f"[evaluate_arm] Box calculation failed for {arm}: {e}")
        return failure_score, product_smiles, None

    # 5. Run genetic docking refinement
    try:
        best_mol, best_score = genetic_docking(
            seed_mol=aligned_ligand,
            frozen_atom_indices=list(frozen_indices),
            coord_map=coord_map,
            receptor_pdbqt=receptor_pdbqt,
            center=center,
            size=size,
            **ga_kwargs
        )
        if best_mol is None or best_score is None:
            print(f"[evaluate_arm] genetic_docking returned no pose for {arm}")
            return failure_score, product_smiles, None
        os.makedirs("best_conformers", exist_ok=True)
        sdf_path = f"best_conformers/best_{ligand_index}.sdf"
        save_mol(best_mol, sdf_path)
        
    except Exception as e:
        print(f"[evaluate_arm] genetic_docking failed for {arm}: {e}")
        return failure_score, product_smiles, None

    if best_mol is None or best_score is None:
        return failure_score, product_smiles, None

    # Optional: canonicalize SMILES of best conformer (strip coordinates)
    try:
        best_smiles = Chem.MolToSmiles(Chem.Mol(best_mol), canonical=True)
    except Exception:
        best_smiles = product_smiles

    result = (best_score, best_smiles, best_mol)
    if use_cache:
        _evaluation_cache[key] = result
    return result
