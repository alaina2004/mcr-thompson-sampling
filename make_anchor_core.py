from rdkit import Chem

# Define small anchor core as SMILES
core_smiles = "[H]N1C([H])=NC(C)=C1NC"  # your small anchor
core_mol = Chem.MolFromSmiles(core_smiles)

# Save as SDF
w = Chem.SDWriter('input_data/anchor_core.sdf')
w.write(core_mol)
w.close()

print("Anchor core saved to input_data/anchor_core.sdf")
