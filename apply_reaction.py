from rdkit.Chem import rdChemReactions
from utils import to_aromatic_smiles

def apply_multi_reactant_reaction_rdkit(reaction, reactants):
    """Apply a multi-reactant reaction using RDKit and return product SMILES."""
    try:
        # Reactants must be passed as a list of tuple of molecules
        reactant_tuple = tuple(reactants)
        
        # Apply the reaction
        products = reaction.RunReactants(reactant_tuple)
        
        if not products:
            print("No products generated.")
            return None

        # Take first product set and first molecule
        product = products[0][0]

        product_smiles = to_aromatic_smiles(product)
        return product_smiles

    except Exception as e:
        print(f"RDKit reaction failed: {e}")
        return None
