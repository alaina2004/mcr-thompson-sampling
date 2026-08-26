#!/usr/bin/env bash
set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

TARGETS=(
  "__pycache__" "logs" "aligned_ligands" "best_conformers" "generated_ligands"
  "smina" ".DS_Store" "input_data/.DS_Store" "gnina_cpu"
  "RFScore_v1_pdbbind2016.pickle" "temp_out.pdbqt" "test.pdb" "test.pdbqt"
  "cleaned_protein.pdb" "initial_rdkit_conformers.sdf" "final_generation.sdf"
  "final_generation_scores.csv" "final_scores_detailed.csv"
  "docking_summary.csv" "ts_sampling.py"
  "protein/prepared_protein.pdbqt" "protein/prepared_protein_fixed.pdb"
)

echo "=== Files/directories to remove ==="
TOTAL=0
for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then
    SIZE=$(du -sh "$t" 2>/dev/null | cut -f1)
    echo "  $t  ($SIZE)"
    TOTAL=$((TOTAL + 1))
  fi
done
echo "=== $TOTAL entries found ==="

if [[ $APPLY -eq 0 ]]; then
  echo ""
  echo "Dry run only. Re-run with --apply to delete."
  exit 0
fi

echo ""
echo "Removing..."
for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then
    git rm -r --cached --quiet "$t" 2>/dev/null || true
    rm -rf "$t"
    echo "  removed $t"
  fi
done

find . -name ".DS_Store" -not -path "./.git/*" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "Done."
