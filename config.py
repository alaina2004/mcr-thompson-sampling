"""Central configuration: paths, external executables, run parameters."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input_data"
PROTEIN_DIR = PROJECT_ROOT / "protein"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"

ANCHOR_SDF = INPUT_DIR / "anchor.sdf"
RXN_FILE = INPUT_DIR / "ggb.rxn"
REAGENT_FILES = [
    INPUT_DIR / "reagent_r1.smi",
    INPUT_DIR / "reagent_r2.smi",
    INPUT_DIR / "reagent_r3.smi",
]

RECEPTOR_PDB = PROTEIN_DIR / "prepared_protein.pdb"
RECEPTOR_PDBQT = PROTEIN_DIR / "prepared_protein.pdbqt"

CORE_SMILES = "[H]NC1=C[N:1]=[CH:4][NH:5]1"
ANCHOR_SMILES = "ClC1=C[C:4]2=[C:5](C=C[NH:1]2)C=C1"

FAILURE_SCORE = 1e6


def resolve_smina() -> str:
    """Locate smina: $SMINA_PATH, then PATH, then ./bin/smina."""
    env_path = os.environ.get("SMINA_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    on_path = shutil.which("smina")
    if on_path:
        return on_path

    local = PROJECT_ROOT / "bin" / "smina"
    if local.is_file():
        return str(local)

    raise FileNotFoundError(
        "smina not found. Install it (conda install -c conda-forge smina) "
        "or set SMINA_PATH, e.g. export SMINA_PATH=/path/to/smina"
    )


def resolve_obabel() -> str:
    on_path = shutil.which("obabel")
    if not on_path:
        raise FileNotFoundError(
            "obabel not found. Install Open Babel "
            "(conda install -c conda-forge openbabel)."
        )
    return on_path
