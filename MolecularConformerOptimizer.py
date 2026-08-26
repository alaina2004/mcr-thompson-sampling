#!/usr/bin/env python
# Core-Frozen GA Docking with Hybrid Scoring (SMINA + RF-Score)

import random
import tempfile
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdMolDescriptors
from docking import prepare_ligand_for_docking, dock_with_smina, calculate_center_and_size_from_mol
from align_core_to_ligand import find_mcs_between_core_and_ligand, align_full_ligand_to_core
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.Geometry import Point3D
from scipy.spatial import ConvexHull, Delaunay
from rdkit.Chem import SanitizeMol
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import itertools
from pathlib import Path


_RUN_COUNTER = itertools.count()


def _out_path(filename: str) -> Path:
    """Write GA diagnostics into results/ga_runs/<n>/ instead of the CWD."""
    from config import RESULTS_DIR
    run_dir = RESULTS_DIR / "ga_runs" / f"run_{next(_RUN_COUNTER):05d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / filename




# --- Utilities ---

def detect_atom_clashes(mol, conf_id=0, threshold=1.0):
    conf = mol.GetConformer(conf_id)
    positions = conf.GetPositions()
    clash_count = 0
    clash_pairs = []

    n = mol.GetNumAtoms()
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    close = [(int(i), int(j)) for i, j in zip(*iu) if dists[i, j] < threshold]
    for i, j in close:
        if mol.GetBondBetweenAtoms(i, j):
            continue
        clash_count += 1
        clash_pairs.append((i, j))

    return clash_count, clash_pairs


def detect_ring_penetration(mol, conf_id=0, tolerance=1.5, vdw_scale=0.8):
    ring_info = mol.GetRingInfo()
    conf = mol.GetConformer(conf_id)
    rings = ring_info.AtomRings()

    penetrations = 0
    for ring in rings:
        if len(ring) < 5:
            continue

        # 1) Gather ring 3D points & center
        pts3 = np.array([conf.GetAtomPosition(i) for i in ring])
        center = pts3.mean(axis=0)

        # 2) PCA/SVD → get ring normal + two in‑plane axes
        U, S, Vt = np.linalg.svd(pts3 - center)
        normal = Vt[2]      # small singular value direction
        axis_u, axis_v = Vt[0], Vt[1]

        # 3) Project ring atoms into 2D local coords
        coords2d = np.dot(pts3 - center, np.vstack([axis_u, axis_v]).T)

        # 4) Build 2D convex hull & Delaunay for inclusion test
        hull = ConvexHull(coords2d)
        polygon = coords2d[hull.vertices]
        delaunay = Delaunay(polygon)

        # 5) Check every non‑ring bond for plane crossing
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in ring or a2 in ring:
                continue

            p1 = np.array(conf.GetAtomPosition(a1))
            p2 = np.array(conf.GetAtomPosition(a2))
            d1, d2 = np.dot(p1-center, normal), np.dot(p2-center, normal)

            # must strictly straddle plane
            if d1 * d2 >= 0:
                continue

            # find intersection point
            t = -d1 / (d2 - d1)
            X = p1 + t * (p2 - p1)

            # project intersection into ring plane
            X2d = np.dot(X - center, np.vstack([axis_u, axis_v]).T)

            # 6) robust 2D inside‐polygon test
            if delaunay.find_simplex(X2d) >= 0:
                # finally, check it’s not just grazing (within tolerance)
                if abs(np.dot(X - center, normal)) < tolerance:
                    penetrations += 1

    return penetrations


def hybrid_score(mol, receptor_pdbqt, center, size,scoring_function="vinardo"):
    """Calculate and return both SMINA and RF-Score with combined fitness."""
    smina_val = smina_score(mol, receptor_pdbqt, center, size, scoring_function=scoring_function)
    #rf_val = rf_score(mol, receptor_pdbqt)
    
    # Combined fitness (adjust weights as needed)
    #fitness = 0.7 * smina_val + -0.3 * rf_val  
    fitness = smina_val
    return {
        'smina': smina_val,
        #'rf_score': rf_val,
        'fitness': fitness
    }

def check_lipinski_veber(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    fails = 0
    if mw > 500: fails += 1
    if logp > 5: fails += 1
    if hbd > 5: fails += 1
    if hba > 10: fails += 1
    if rot_bonds > 10: fails += 1
    if tpsa > 140: fails += 1

    if fails >= 2:
        print(" Failed Drug-Likeness Check (>=2 violations):")
        print(f"  MW: {mw:.2f}, LogP: {logp:.2f}, HBD: {hbd}, HBA: {hba}")
        print(f"  RotBonds: {rot_bonds}, TPSA: {tpsa:.2f}")
        return False
    return True

def find_torsion_atoms(mol, atom1, atom2):
    for path in mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-[*]-[*]-[*]')):
        if atom1 in path and atom2 in path:
            i, j, k, l = path
            if (j == atom1 and k == atom2) or (j == atom2 and k == atom1):
                return (i, j, k, l)
    return None

def extract_torsions(mol, rot_bonds):
    conf = mol.GetConformer()
    torsions = []
    ring_info = mol.GetRingInfo()
    ring_atoms = set([atom_idx for ring in ring_info.AtomRings() for atom_idx in ring])

    for b in rot_bonds:
        tpl = find_torsion_atoms(mol, *b)
        if tpl:
            if any(i in ring_atoms for i in tpl):
                continue
            ang = rdMolTransforms.GetDihedralDeg(conf, *tpl)
            torsions.append((tpl, ang))
    return torsions

def apply_torsions(mol, torsions):
    conf = mol.GetConformer()
    for tpl, ang in torsions:
        rdMolTransforms.SetDihedralDeg(conf, *tpl, ang)

def mutate_torsions(torsions, rate=0.2, max_angle=10):
    new = []
    for tpl, ang in torsions:
        if random.random() < rate:
            ang += random.uniform(-max_angle, max_angle)
        new.append((tpl, ang))
    return new

def crossover_torsions(t1, t2):
    if len(t1) < 2 or len(t2) < 2:
        return t1
    k = random.randint(1, len(t1) - 1)
    return t1[:k] + t2[k:]

def minimize(mol, frozen_atom_indices):
    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        if ff is None:
            print("MMFF failed, trying UFF...")
            ff = AllChem.UFFGetMoleculeForceField(mol)
        
        if ff is not None:
            for idx in frozen_atom_indices:
                ff.AddFixedPoint(idx)
            ff.Minimize(maxIts=200)
            return True
        else:
            print("Both MMFF and UFF failed!")
            return False
    except Exception as e:
        print(f"Minimization failed: {e}")
        return False

def smina_score(mol, receptor_pdbqt, center, size,scoring_function="vinardo"):
    # Ensure center and size are numpy arrays with correct dimensions
    center = np.asarray(center).reshape(3)
    size = np.asarray(size).reshape(3)
    
    with tempfile.NamedTemporaryFile(suffix=".sdf") as sdf_f, \
         tempfile.NamedTemporaryFile(suffix=".pdbqt") as pqt_f:
        Chem.MolToMolFile(mol, sdf_f.name)
        prepare_ligand_for_docking(sdf_f.name, pqt_f.name)
        score, _ = dock_with_smina(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=pqt_f.name,
            output_pdbqt=None,
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=size[0], size_y=size[1], size_z=size[2],
            exhaustiveness=16, cpu=4, score_only=True, scoring_function= scoring_function
        )
    return score

def restore_core_positions(mol, core_pos_map):
    conf = mol.GetConformer()
    for idx, pos in core_pos_map.items():
        if isinstance(pos, Point3D):
            conf.SetAtomPosition(idx, pos)
        else:
            conf.SetAtomPosition(idx, Point3D(*pos[:3]))  # Ensure 3D coordinates




# --- Genetic Algorithm ---
def genetic_docking(seed_mol, frozen_atom_indices, coord_map, receptor_pdbqt, center, size,
                   generations=10, pop_size=50, keep_top=10, patience=3, seed=42):
    if not check_lipinski_veber(seed_mol):
        print(" Molecule failed Lipinski + Veber filter. Skipping docking.")
        return None, None
    
    # Ensure center and size are properly shaped numpy arrays
    center = np.asarray(center).reshape(3)
    size = np.asarray(size).reshape(3)
    
    core_pos_map = {
        idx: seed_mol.GetConformer().GetAtomPosition(idx)
        for idx in frozen_atom_indices
    }
    random.seed(seed)

    mol = Chem.AddHs(seed_mol)
    params = AllChem.ETKDGv3()
    params.SetCoordMap(coord_map)
    params.pruneRmsThresh = 0.05
    params.numThreads = 0
    params.enforceChirality = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True  
    params.useSmallRingTorsions = True
    params.useRandomCoords = False  
    params.randomSeed = seed
    
    # Generate initial population
    AllChem.EmbedMultipleConfs(mol, numConfs=pop_size, params=params)
    
    # Write initial conformers
    initial_writer = Chem.SDWriter(str(_out_path("initial_rdkit_conformers.sdf")))
    for cid in range(mol.GetNumConformers()):
        conf_mol = Chem.Mol(mol)
        conf_mol.RemoveAllConformers()
        conf_mol.AddConformer(mol.GetConformer(cid), assignId=True)
        conf_mol.SetProp("ConformerID", f"initial_{cid}")
        initial_writer.write(conf_mol)
    initial_writer.close()
    
    rot_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[!$(*#*)&!D1]-!@[!$(*#*)&!D1]'))
    population = []

    # Evaluate initial population
    for cid in range(mol.GetNumConformers()):
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        m.AddConformer(mol.GetConformer(cid), assignId=True)
        restore_core_positions(m, core_pos_map)

        try:
            Chem.SanitizeMol(m)
        except Exception as e:
            print(f"Skipping conformer {cid}: sanitization failed ({e})")
            continue
        
        # Skip bad conformers
        clash_count, _ = detect_atom_clashes(m)
        #penetration_count = detect_ring_penetration(m)
        if clash_count > 0 :
            print(f"Skipping conformer {cid}: {clash_count} clashes")
            continue
        
        if minimize(m, frozen_atom_indices):
            scores = hybrid_score(m, receptor_pdbqt, center, size, scoring_function="vinardo")
            if scores is None:
                continue
            #print(f"Conformer {cid} Scores: SMINA={scores['smina']:.2f}, RF-Score={scores['rf_score']:.2f}, Fitness={scores['fitness']:.2f}")
            print(f"Conformer {cid} Scores: SMINA={scores['smina']:.2f}, Fitness={scores['fitness']:.2f}")

            population.append((scores['fitness'], scores, m))
        else:
            print(f"Skipping conformer {cid} due to minimization failure")

    if not population:
        print("No valid conformers generated!")
        return None, None

    best_fitness = float('inf')
    no_improve = 0

    # Evolution loop
    for gen in range(1, generations + 1):
        population.sort(key=lambda x: x[0])  # Sort by fitness
        current_best = population[0][0]
        print(f"\nGeneration {gen}: Best fitness = {current_best:.2f}")
        #print(f"SMINA={population[0][1]['smina']:.2f}, RF-Score={population[0][1]['rf_score']:.2f}")
        print(f"SMINA={population[0][1]['smina']:.2f}")


        if current_best < best_fitness:
            best_fitness = current_best
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop after {patience} gens without improvement.")
                break

        parents = population[:keep_top]
        new_pop = parents.copy()

        while len(new_pop) < pop_size:
            if random.random() < 0.5:  # Mutation
                _, _, parent = random.choice(parents)
                child = Chem.Mol(parent)
                tors = extract_torsions(child, rot_bonds)
                mutated = mutate_torsions(tors)
                apply_torsions(child, mutated)
            else:  # Crossover
                _, _, p1 = random.choice(parents)
                _, _, p2 = random.choice(parents)
                child = Chem.Mol(p1)
                t1 = extract_torsions(p1, rot_bonds)
                t2 = extract_torsions(p2, rot_bonds)
                crossed = crossover_torsions(t1, t2)
                apply_torsions(child, crossed)

            
            restore_core_positions(child, core_pos_map)
            Chem.SanitizeMol(child) 
            child = Chem.AddHs(child, addCoords=True)
            
            # Strict quality control
            penetration_count = detect_ring_penetration(child)
            clash_count, _ = detect_atom_clashes(child)
            if penetration_count > 0 or clash_count > 0:
                continue
                
            if minimize(child, frozen_atom_indices):
                scores = hybrid_score(child, receptor_pdbqt, center, size)
                if scores is None:
                    continue
                new_pop.append((scores['fitness'], scores, child))
            else:
                print("Skipping child due to minimization failure")

        population = sorted(new_pop, key=lambda x: x[0])

    # Save results
    writer = Chem.SDWriter(str(_out_path("final_generation.sdf")))
    results = []
    for fitness, scores, mol in population:
        mol.SetProp("DockingScore", f"{fitness:.4f}")
        mol.SetProp("SMINA_Score", f"{scores['smina']:.4f}")
        #mol.SetProp("RF_Score", f"{scores['rf_score']:.4f}")
        writer.write(mol)
        results.append({
            'SMINA': scores['smina'],
            'Fitness': fitness
        })
    writer.close()

    pd.DataFrame(results).to_csv(_out_path("final_scores_detailed.csv"), index=False)

    best_fitness, best_scores, best_mol = min(population, key=lambda x: x[0])
    print("\nBest final scores:")
    #print(f"Fitness: {best_fitness:.2f} (SMINA={best_scores['smina']:.2f}, RF-Score={best_scores['rf_score']:.2f})")
    print(f"Fitness: {best_fitness:.2f} (SMINA={best_scores['smina']:.2f})")

    return best_mol, best_fitness