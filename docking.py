import subprocess
from config import resolve_smina
import os
import sys
import warnings
from pathlib import Path
import pandas as pd
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import tempfile
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------------------
# Protein Preparation
# -------------------------------

def prepare_receptor(input_pdb, output_pdbqt, ph=7.0):
    """
    Fix a PDB protein structure and convert to PDBQT format for docking.
    """
    print(f"Loading and fixing {input_pdb}...")
    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)

    fixed_pdb_path = input_pdb.replace(".pdb", "_fixed.pdb")

    print(f"Saving fixed PDB to {fixed_pdb_path}...")
    with open(fixed_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    print(f"Converting fixed PDB to PDBQT...")
    result = subprocess.run(
        ["obabel", "-xr", "-ipdb", fixed_pdb_path, "-opdbqt", "-O", output_pdbqt],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error during Open Babel conversion:")
        print(result.stderr)
        raise RuntimeError("Failed to convert PDB to PDBQT.")
    else:
        print(f"Receptor prepared successfully: {output_pdbqt}")


# -------------------------------
# Ligand Preparation
# -------------------------------

def prepare_ligand_for_docking(input_sdf, output_pdbqt):
    """
    Prepare a ligand SDF file into a docking-ready PDBQT using Meeko.
    """
    mol = Chem.MolFromMolFile(input_sdf, removeHs=False)
    if mol is None:
        raise ValueError("Failed to load molecule for preparation.")

    mol = Chem.AddHs(mol)
    if not mol.GetConformer().Is3D():
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    preparator._addCharges = True
    preparator._addHydrogens = True  # Already added with RDKit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        preparator.prepare(mol)
        preparator.write_pdbqt_file(output_pdbqt)

    print(f"Ligand prepared and saved to {output_pdbqt}")


# -------------------------------
# Docking Functions
# -------------------------------

def calculate_center_and_size_from_mol(mol):
    """
    Calculate docking box center and size from an RDKit Mol object.
    """
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

    xs = [pos.x for pos in coords]
    ys = [pos.y for pos in coords]
    zs = [pos.z for pos in coords]

    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    center_z = (max(zs) + min(zs)) / 2

    size_x = (max(xs) - min(xs)) + 6
    size_y = (max(ys) - min(ys)) + 6
    size_z = (max(zs) - min(zs)) + 6

    return (round(center_x, 2), round(center_y, 2), round(center_z, 2),
            round(size_x, 1), round(size_y, 1), round(size_z, 1))


def dock_with_smina(receptor_pdbqt, ligand_pdbqt, output_pdbqt=None,
                    center_x=None, center_y=None, center_z=None,
                    size_x=None, size_y=None, size_z=None,
                    exhaustiveness=8, num_modes=20, cpu=4,
                    score_only=False, scoring_function="vinardo",
                    autobox_ligand=None, autobox_add=4.0):
    """
    Run smina. Supports either explicit box (center/size) or autobox based on a ligand.

    Returns: best_affinity (float or None), list of affinities parsed.
    """
    smina_executable = resolve_smina()

    cmd = [smina_executable, "-r", receptor_pdbqt, "-l", ligand_pdbqt,
           "--scoring", scoring_function,
           "--cpu", str(cpu),
           "--seed", "42",
           "--exhaustiveness", str(exhaustiveness)]

    if autobox_ligand is not None:
        cmd += ["--autobox_ligand", autobox_ligand, "--autobox_add", str(autobox_add)]
    else:
        # require center/size if not using autobox
        if None in (center_x, center_y, center_z, size_x, size_y, size_z):
            raise ValueError("Center and size must be provided when not using autobox.")
        cmd += [
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
        ]

    if score_only:
        cmd.append("--score_only")
        cmd.append("--minimize")
        cmd += [
            "--minimize_iters", "2000",
            "--force_cap", "10",
            "--accurate_line",
            "--approximation", "exact"
        ]
    else:
        if output_pdbqt is not None:
            cmd += ["-o", output_pdbqt, "--num_modes", str(num_modes)]

    # Run
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout

    # parse affinities
    affinities = []
    if score_only:
        for line in out.splitlines():
            if line.strip().startswith("Affinity:"):
                try:
                    affinities.append(float(line.split()[1]))
                except (ValueError, IndexError):
                    pass
    else:
        parsing = False
        for line in out.splitlines():
            if "mode |   affinity" in line:
                parsing = True
                continue
            if parsing:
                if not line.strip() or line.startswith("-----"):
                    continue
                parts = line.split()
                try:
                    affinities.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass

    best_affinity = min(affinities) if affinities else None
    return best_affinity, affinities




def smina_score(mol, receptor_pdbqt, center, size, scoring_function="vinardo"):
    """Calculate SMINA score for a molecule using score-only docking with optional autobox fallback."""
    with tempfile.NamedTemporaryFile(suffix=".sdf") as sdf_file, \
         tempfile.NamedTemporaryFile(suffix=".pdbqt") as ligand_pdbqt_file:

        Chem.MolToMolFile(mol, sdf_file.name)
        prepare_ligand_for_docking(sdf_file.name, ligand_pdbqt_file.name)

        # Use autobox by default here if center/size are invalid? keep old behavior
        best_affinity, _ = dock_with_smina(
            receptor_pdbqt,
            ligand_pdbqt_file.name,
            None,
            center[0], center[1], center[2],
            size[0], size[1], size[2],
            score_only=True,
            scoring_function=scoring_function
        )

        return best_affinity if best_affinity is not None else 0.0


def hybrid_score(mol, receptor_pdbqt, center, size, scoring_function="vinardo"):
    """Calculate and return both SMINA and RF-Score with combined fitness."""
    smina_val = smina_score(mol, receptor_pdbqt, center, size, scoring_function=scoring_function)
    #rf_val = rf_score(mol, receptor_pdbqt)

    # Combined fitness (adjust weights as needed)
    #fitness = 0.7 * smina_val + 0.3 * -rf_val
    fitness =  smina_val
    return {
        'smina': smina_val,
        #'rf_score': rf_val,
        'fitness': fitness
    }
