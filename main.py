import os
import csv
import random
import warnings
from rdkit import Chem
import datetime
from collections import Counter
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)

import signal

def evaluate_arm_with_real_timeout(arm, reaction, core, receptor_pdbqt, ga_kwargs, failure_score=1000000, timeout_minutes=10):
    """Actually interrupts hanging evaluate_arm calls"""
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Arm evaluation timed out after {timeout_minutes} minutes")
    
    # Set the alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_minutes * 60)  # Convert to seconds
    
    try:
        raw_score, best_smiles, best_mol = evaluate_arm(
            arm, reaction, core, receptor_pdbqt, ga_kwargs, failure_score, use_cache=False
        )
        signal.alarm(0)  # Cancel alarm
        return raw_score, best_smiles, best_mol
        
    except TimeoutError:
        print(f"⏰ REAL TIMEOUT: Arm interrupted after {timeout_minutes} minutes")
        return 1000000, None, None
        
    except Exception as e:
        print(f" Arm failed: {e}")
        return 1000000, None, None
        
    finally:
        signal.alarm(0)  # Always cancel alarm
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)  # Restore handler

# Log setup
os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join("logs", f"run_log_{timestamp}.txt")
log_file = open(log_path, "w")

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = sys.stderr = Tee(sys.__stdout__, log_file)

from load_inputs import load_rxn_inputs
from align_core_to_ligand import align_core_to_anchor_by_atom_map
from docking import (
    calculate_center_and_size_from_mol,
    prepare_receptor,
    prepare_ligand_for_docking,
    dock_with_smina
)
from ts_driver import ThompsonSamplingCombinatorial
from ts_evaluator import evaluate_arm  # GA+docking evaluator
import time
start_time = time.time()


def run_active_learning():
    # === Paths / config ===
    anchor_sdf = "input_data/anchor.sdf"
    rxn_file = "input_data/ggb.rxn"
    reagent_files = [
        "input_data/reagent_r1.smi",
        "input_data/reagent_r2.smi",
        "input_data/reagent_r3.smi"
    ]
    protein_folder = "protein"
    receptor_pdb = os.path.join(protein_folder, "prepared_protein.pdb")
    receptor_pdbqt = os.path.join(protein_folder, "prepared_protein.pdbqt")
    out_dir = "generated_ligands"
    os.makedirs(out_dir, exist_ok=True)

    # === Prepare receptor ===
    print("Preparing receptor...")
    prepare_receptor(receptor_pdb, receptor_pdbqt, ph=7.0)

    # === Load reaction and reagents ===
    print("Loading reaction and reagents...")
    _, reaction, reagents = load_rxn_inputs(anchor_sdf, rxn_file, reagent_files)
    r1_list = reagents[0]["SMILES"].tolist()
    r2_list = reagents[1]["SMILES"].tolist()
    r3_list = reagents[2]["SMILES"].tolist()

    # === Align core to anchor ===
    print("Aligning core to anchor...")
    core_smiles = "[H]NC1=C[N:1]=[CH:4][NH:5]1"
    anchor_smiles = "ClC1=C[C:4]2=[C:5](C=C[NH:1]2)C=C1"
    aligned_core_path = os.path.join(out_dir, "aligned_core.sdf")
    aligned_anchor_path = os.path.join(out_dir, "aligned_anchor.sdf")
    core, _ = align_core_to_anchor_by_atom_map(
        core_smiles=core_smiles,
        anchor_sdf_path=anchor_sdf,
        anchor_smiles=anchor_smiles,
        output_core_path=aligned_core_path,
        output_anchor_path=aligned_anchor_path
    )

    # Optional: get initial box from core if needed downstream
    cx, cy, cz, sx, sy, sz = calculate_center_and_size_from_mol(core)
    print(f"Core-derived docking box: center=({cx},{cy},{cz}), size=({sx},{sy},{sz})")

    # === GA + docking hyperparams ===
    ga_kwargs = {
        "generations": 10,
        "pop_size": 50,
        "keep_top": 10,
        "patience": 3
    }

    # === Full arm space ===
    all_arms = list((a, b, c) for a in r1_list for b in r2_list for c in r3_list)

    # === Seed initial random arms ===
    print("Seeding initial arms...")
    random.seed(42)
    initial_arms = random.sample(all_arms, min(100, len(all_arms)))
    
    print("=== SEEDING PHASE ===")
    print("\nInitial arms selected (all at once):")
    for i, (r1, r2, r3) in enumerate(initial_arms, 1):
        print(f"{i}: R1={r1} | R2={r2} | R3={r3}")
    initial_history = []   
    failure_raw = 1e6  # for lower-is-better

    for i, arm in enumerate(initial_arms):
        print(f"\n=== ARM {i+1}/{len(initial_arms)} ===")
        
        raw_score, best_smiles, best_mol = evaluate_arm_with_real_timeout(
            arm, reaction, core, receptor_pdbqt, ga_kwargs, 
            failure_score=failure_raw, timeout_minutes=10
    )
    
        print(f"Seeded arm {arm} -> raw score {raw_score}")
        initial_history.append((arm, raw_score, best_smiles, best_mol))
    # === Initialize TS with those seeds, no warmup ===
    ts = ThompsonSamplingCombinatorial(
        r1_list=r1_list,
        r2_list=r2_list,
        r3_list=r3_list,
        reaction=reaction,
        core=core,
        receptor_pdbqt=receptor_pdbqt,
        ga_kwargs=ga_kwargs,
        prior_mean=0.0,
        prior_variance=1.0,
        observation_std=1.0,
        warmup_k=50,
        max_arms=None,
        seed=43,
        epsilon=0.1,
        do_warmup=False,
        lower_is_better=True,
    )
    ts.seed_initial(initial_history)

    # === Active learning loop ===
    best_overall = (None, float("inf"), None, None)  # arm, raw_score, smiles, mol
    no_improve = 0
    patience = 10
    min_iters = 100
    max_iters = 1000

    for it in range(max_iters):
        arm, raw_score, reward, best_smiles, best_mol = ts.select_and_update()
        print(f"[TS iter {it+1}] Selected {arm} -> raw score {raw_score:.2f}, reward {reward:.2f}")
        if raw_score < best_overall[1]:
            best_overall = (arm, raw_score, best_smiles, best_mol)
            no_improve = 0
        else:
            no_improve += 1
        if it + 1 >= min_iters and no_improve >= patience:
            print(f"No improvement for {patience} iterations after {min_iters} runs; stopping.")
            break
    # === Reagent frequency analysis ===
    if not ts.history:
        print("No arms were evaluated - skipping frequency analysis")
    else:
        print("\n=== REAGENT FREQUENCY ANALYSIS ===")
        
        
        r1_counter = Counter()
        r2_counter = Counter()
        r3_counter = Counter()
        
        # Count from all evaluated arms in TS history
        for arm, raw_score, reward, smiles, mol in ts.history:
            r1, r2, r3 = arm
            r1_counter[r1] += 1
            r2_counter[r2] += 1
            r3_counter[r3] += 1
        
        # Print most common reagents
        print(f"\nMost frequent R1 reagents (top 10):")
        for reagent, count in r1_counter.most_common(10):
            print(f"  {reagent}: {count} times")
        
        print(f"\nMost frequent R2 reagents (top 10):")
        for reagent, count in r2_counter.most_common(10):
            print(f"  {reagent}: {count} times")
        
        print(f"\nMost frequent R3 reagents (top 10):")
        for reagent, count in r3_counter.most_common(10):
            print(f"  {reagent}: {count} times")
        
        # Save reagent frequency to CSV
        freq_path = os.path.join(out_dir, "reagent_frequencies.csv")
        with open(freq_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reagent_type", "reagent_smiles", "frequency"])
            
            for reagent, count in r1_counter.most_common():
                writer.writerow(["R1", reagent, count])
            for reagent, count in r2_counter.most_common():
                writer.writerow(["R2", reagent, count])
            for reagent, count in r3_counter.most_common():
                writer.writerow(["R3", reagent, count])
        
        print(f"\nReagent frequencies saved to {freq_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total arms evaluated: {len(ts.history)}")
        print(f"Unique R1 reagents used: {len(r1_counter)}")
        print(f"Unique R2 reagents used: {len(r2_counter)}")
        print(f"Unique R3 reagents used: {len(r3_counter)}")
        
        # Most common overall reagent
        all_reagents = Counter()
        for reagent, count in r1_counter.items():
            all_reagents[f"R1: {reagent}"] = count
        for reagent, count in r2_counter.items():
            all_reagents[f"R2: {reagent}"] = count
        for reagent, count in r3_counter.items():
            all_reagents[f"R3: {reagent}"] = count
        
        print(f"\nMost frequently used reagent overall:")
        most_common_reagent, most_common_count = all_reagents.most_common(1)[0]
        print(f"  {most_common_reagent}: {most_common_count} times")
        
        print(f"\nReagent diversity:")
        print(f"R1 diversity: {len(r1_counter)}/{len(r1_list)} reagents used ({len(r1_counter)/len(r1_list)*100:.1f}%)")
        print(f"R2 diversity: {len(r2_counter)}/{len(r2_list)} reagents used ({len(r2_counter)/len(r2_list)*100:.1f}%)")
        print(f"R3 diversity: {len(r3_counter)}/{len(r3_list)} reagents used ({len(r3_counter)/len(r3_list)*100:.1f}%)")
        
    # === Select top 5 unique arms by lowest raw_score ===
    # Build best per arm
    best_per_arm = {}
    for arm, raw_score, reward, smiles, mol in ts.history:
        if mol is None:
            continue
        if arm not in best_per_arm or raw_score < best_per_arm[arm][0]:
            best_per_arm[arm] = (raw_score, reward, smiles, mol)

    sorted_best = sorted(
        [(arm, *vals) for arm, vals in best_per_arm.items()],
        key=lambda x: x[1]  # raw_score ascending
    )
    topk = 20
    top_arms = sorted_best[:topk]

    # === Save top-20 conformers and do autobox docking on each ===
    summary_path = os.path.join(out_dir, "top20_active_learning_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "r1", "r2", "r3",
            "raw_score", "reward", "product_smiles",
            "autobox_affinity", "docked_pose_pdbqt"
        ])
        for rank, entry in enumerate(top_arms, start=1):
            arm, raw_score, reward, smiles, mol = entry
            r1, r2, r3 = arm
            # save conformer
            sdf_path = os.path.join(out_dir, f"best_active_conformer_rank_{rank}.sdf")
            Chem.MolToMolFile(mol, sdf_path)

            # prepare ligand pdbqt
            ligand_pdbqt = os.path.join(out_dir, f"best_active_conformer_rank_{rank}.pdbqt")
            try:
                prepare_ligand_for_docking(sdf_path, ligand_pdbqt)
            except Exception as e:
                print(f"[Top{rank}] Ligand prep failed: {e}")
                writer.writerow([rank, r1, r2, r3, raw_score, reward, smiles, "", "prep_failed"])
                continue

            # autobox docking
            output_pose = os.path.join(out_dir, f"best_docked_autobox_rank_{rank}.pdbqt")
            affinity, affinities = dock_with_smina(
                receptor_pdbqt,
                ligand_pdbqt,
                output_pose,
                exhaustiveness=8,
                num_modes=20,
                score_only=False,
                scoring_function="vinardo",
                autobox_ligand=ligand_pdbqt,
                autobox_add=4.0
            )
            print(f"[Top{rank}] Arm {arm} → fitness (GA): {raw_score:.2f}, autobox affinity: {affinity}")
            writer.writerow([
                rank, r1, r2, r3,
                raw_score, reward, smiles,
                affinity if affinity is not None else "",
                output_pose if affinity is not None else ""
            ])

    print(f"Top-{topk} summary written to {summary_path}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total runtime: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    run_active_learning()
