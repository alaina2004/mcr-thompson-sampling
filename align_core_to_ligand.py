from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS, Draw
import os
from typing import Optional
from rdkit.Chem import rdMolTransforms
import numpy as np
from rdkit.Chem.rdMolAlign import GetAlignmentTransform

def align_core_to_anchor_by_atom_map(
    core_smiles: str,
    anchor_sdf_path: str,
    anchor_smiles: str,
    output_core_path: str,
    output_anchor_path: str = None
):
    def extract_atom_map(mol):
        return {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}

    anchor_3d = Chem.SDMolSupplier(anchor_sdf_path, removeHs=False)[0]
    if anchor_3d is None:
        raise ValueError("Failed to load anchor from SDF.")

    anchor_mapped = Chem.MolFromSmiles(anchor_smiles, sanitize=False)
    Chem.SanitizeMol(anchor_mapped)

    match = anchor_3d.GetSubstructMatch(anchor_mapped)
    if not match:
        raise ValueError("Anchor SMILES does not match anchor SDF structure.")

    for sdf_idx, map_atom in zip(match, anchor_mapped.GetAtoms()):
        anchor_3d.GetAtomWithIdx(sdf_idx).SetAtomMapNum(map_atom.GetAtomMapNum())

    core = Chem.MolFromSmiles(core_smiles, sanitize=False)
    Chem.SanitizeMol(core)
    core = Chem.AddHs(core)
    AllChem.EmbedMolecule(core, AllChem.ETKDGv3())

    core_map = extract_atom_map(core)
    anchor_map = extract_atom_map(anchor_3d)
    common_keys = set(core_map) & set(anchor_map)
    if not common_keys:
        raise ValueError("No overlapping atom map numbers found.")

    atom_map = [(core_map[k], anchor_map[k]) for k in sorted(common_keys)]
    print(f"Using atom map: {atom_map}")

    rmsd = rdMolAlign.AlignMol(core, anchor_3d, atomMap=atom_map)
    print(f"Core aligned to anchor. RMSD = {rmsd:.3f} Å")

    Chem.SDWriter(output_core_path).write(core)
    print(f"Aligned core saved to: {output_core_path}")

    if output_anchor_path:
        Chem.SDWriter(output_anchor_path).write(anchor_3d)
        print(f"Anchor saved to: {output_anchor_path}")

    return core, anchor_3d

def load_mol(file_path):
    mol = Chem.SDMolSupplier(file_path, removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Failed to load molecule from {file_path}.")
    return mol

def generate_3d_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"[ERROR] Could not parse SMILES: {smiles}")

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"[ERROR] Sanitization failed for SMILES: {smiles}\nReason: {e}")

    mol = Chem.AddHs(mol)

    try:
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            raise ValueError(f"[ERROR] 3D embedding failed for SMILES: {smiles}")
    except Exception as e:
        raise ValueError(f"[ERROR] 3D generation crashed for SMILES: {smiles}\nReason: {e}")

    return mol

def find_mcs_between_core_and_ligand(aligned_core, full_ligand):
    try:
        Chem.Kekulize(full_ligand, clearAromaticFlags=True)
    except Exception as e:
        print(f"[WARNING] Kekulization failed for ligand: {e}")

    try:
        Chem.Kekulize(aligned_core, clearAromaticFlags=True)
    except Exception as e:
        print(f"[WARNING] Kekulization failed for core: {e}")

    attempts = []

    ring_params = rdFMCS.MCSParameters()
    ring_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    ring_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    ring_params.RingMatchesRingOnly = True
    ring_params.CompleteRingsOnly = False
    ring_params.MatchValences = False
    ring_params.MatchChiralTag = True

    relaxed_params = rdFMCS.MCSParameters()
    relaxed_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    relaxed_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    relaxed_params.RingMatchesRingOnly = False
    relaxed_params.CompleteRingsOnly = False
    relaxed_params.MatchValences = False
    relaxed_params.MatchChiralTag = False

    attempts.append(("Ring-focused", ring_params))
    attempts.append(("Relaxed", relaxed_params))

    for name, params in attempts:
        print(f"[INFO] Trying MCS: {name} mode")
        mcs_result = rdFMCS.FindMCS((aligned_core, full_ligand), params)
        if mcs_result.smartsString:
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            if mcs_mol is None:
                print(f"[ERROR] Failed to parse SMARTS: {mcs_result.smartsString}")
                continue
            print(f"[INFO] MCS found using {name} mode.")
            print("Found MCS SMARTS:", mcs_result.smartsString)
            print("Found MCS SMILES:", Chem.MolToSmiles(mcs_mol))
            return mcs_mol

    raise ValueError("No valid MCS SMARTS found in any mode.")

def get_best_substruct_match(mol, mcs_mol, ref_mol=None):
    if mcs_mol is None:
        raise ValueError("Cannot find substructure match: MCS molecule is None.")

    matches = mol.GetSubstructMatches(mcs_mol)
    if not matches:
        raise ValueError("No substructure match found.")

    if len(matches) == 1 or ref_mol is None:
        return matches[0]

    best_match = None
    best_rmsd = float("inf")
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    ref_coords = [ref_mol.GetConformer().GetAtomPosition(i) for i in ref_match]

    for match in matches:
        coords = [mol.GetConformer().GetAtomPosition(i) for i in match]
        rmsd = sum((a - b).LengthSq() for a, b in zip(coords, ref_coords)) / len(coords)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_match = match

    return best_match

def visualize_mcs(full_ligand, mcs_mol, output_image_path, core_mol=None):
    if mcs_mol is None:
        raise ValueError("Cannot visualize MCS: MCS molecule is None.")

    match = get_best_substruct_match(full_ligand, mcs_mol, ref_mol=core_mol)
    img = Draw.MolToImage(full_ligand, highlightAtoms=list(match), size=(600, 600))
    img.save(output_image_path)
    print(f"Saved MCS highlight image to {output_image_path}")


def align_full_ligand_to_core(full_ligand, core_mol, mcs_mol):
    """
    Aligns the ligand to the core based on the MCS.
    Returns:
        - Aligned ligand
        - Atom indices in ligand that matched the MCS (frozen atoms)
    """
    match_ligand = full_ligand.GetSubstructMatch(mcs_mol)
    match_core = core_mol.GetSubstructMatch(mcs_mol)

    if not match_ligand or not match_core:
        raise ValueError("Failed to find MCS match between ligand and core.")

    if len(match_ligand) != len(match_core):
        raise ValueError("Mismatch in number of atoms between ligand and core MCS matches.")

    atom_map = list(zip(match_ligand, match_core))

    print(f" Using atom map: {atom_map}")
    
    # Step 1: AlignMol using MCS atom map
    rmsd1 = rdMolAlign.AlignMol(full_ligand, core_mol, atomMap=atom_map)
    print(f" AlignMol RMSD: {rmsd1:.3f} Å")

    # Step 2: Refine alignment with O3A
    try:
        probe_props = AllChem.MMFFGetMoleculeProperties(full_ligand)
        ref_props = AllChem.MMFFGetMoleculeProperties(core_mol)

        o3a = rdMolAlign.GetO3A(full_ligand, core_mol, probe_props, ref_props)
        rmsd2 = o3a.Align()
        print(f" O3A refinement RMSD: {rmsd2:.3f} Å")
    except Exception as e:
        print(f" O3A refinement failed: {e}")

    
    # Step 3: Use O3A's rigid transform to re-place full_ligand in 3D space
    

    conf = full_ligand.GetConformer()

    transform = o3a.Trans()
    print("Raw transform output:", transform)
    print(f"Transform type: {type(transform)}")

    # Case 1: (RMSD, 4x4 matrix) tuple — legacy RDKit behavior
    if isinstance(transform, tuple) and len(transform) == 2:
        rmsd_val, matrix = transform
        if isinstance(matrix, np.ndarray) and matrix.shape == (4, 4):
            print("✔ Extracted 4x4 matrix from (rmsd, matrix) tuple.")
            for i in range(full_ligand.GetNumAtoms()):
                pos = np.array(conf.GetAtomPosition(i))
                pos_homogeneous = np.append(pos, 1.0)
                new_pos = np.dot(matrix, pos_homogeneous)[:3]
                conf.SetAtomPosition(i, new_pos.tolist())
            print("✔ Full ligand aligned using 4x4 matrix (from tuple).")
        else:
            raise ValueError(f"Tuple second element is not a valid 4x4 matrix. Got shape: {getattr(matrix, 'shape', None)}")

    # Case 2: Raw matrix — new RDKit behavior
    elif isinstance(transform, np.ndarray) and transform.shape == (4, 4):
        print("✔ Applying 4x4 matrix directly from o3a.Trans().")
        for i in range(full_ligand.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(i))
            pos_homogeneous = np.append(pos, 1.0)
            new_pos = np.dot(transform, pos_homogeneous)[:3]
            conf.SetAtomPosition(i, new_pos.tolist())
        print("✔ Full ligand aligned using direct 4x4 matrix.")

    # Case 3: Unexpected format
    else:
        raise ValueError(f"Unexpected transform format from o3a.Trans(): {type(transform)}, value: {transform}")

    print(" Full ligand aligned to core based on MCS.")
    return full_ligand, match_ligand

def save_mol(mol, path):
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
    print(f" Molecule saved to {path}")

def save_mol_as_pdbqt(sdf_file_path, output_pdbqt_path):
    """Convert an SDF to PDBQT without re-embedding, via Open Babel."""
    from openbabel import pybel
    print(f"Converting {sdf_file_path} to PDBQT without reembedding...")
    mols = list(pybel.readfile("sdf", sdf_file_path))
    if not mols:
        print("Error: No molecules found in SDF file.")
        return
    with open(output_pdbqt_path, "w") as f:
        for mol in mols:
            mol.calccharges(model="gasteiger")
            f.write(mol.write("pdbqt"))
    print(f"Final PDBQT with {len(mols)} conformers saved to {output_pdbqt_path}")
