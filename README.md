# Thompson Sampling for MCR-Based Drug Discovery

Structure-guided exploration of a multi-component reaction (MCR) product space using
Thompson sampling. Instead of enumerating and docking all ~1.5 million possible
products, the sampler learns which reagents contribute to good binders and
concentrates its evaluation budget there.

**Status:** research code from an MSc thesis project. It runs end to end but is not
a packaged tool — expect to edit `config.py` for your own target.

## How it works

1. Thompson sampling picks one (r1, r2, r3) reagent combination
2. RDKit applies the `.rxn` template to give a product SMILES
3. The product is embedded in 3D and aligned onto a pre-placed core via MCS;
   the core atoms are frozen thereafter
4. A genetic algorithm perturbs the free torsions, scoring each conformer with
   smina (Vinardo) against the receptor
5. The best score updates the sampler's posterior, and the loop repeats

The core-freezing step is what distinguishes this from plain virtual screening:
every product is scored in a pose that keeps a known binding motif in place, so the
score reflects the decorations rather than the pose search.

## Installation

```bash
git clone https://github.com/alaina2004/Thompson-sampling-for-mcr-reaction-based-drug-discovery.git
cd Thompson-sampling-for-mcr-reaction-based-drug-discovery
conda env create -f environment.yml
conda activate mcr-ts
```

### External dependency: smina

`smina` is not bundled (it is GPL-licensed and platform-specific). Install it with
`conda install -c conda-forge smina`, or download a build and point the code at it:

```bash
export SMINA_PATH=/full/path/to/smina
```

`config.resolve_smina()` checks `$SMINA_PATH`, then `PATH`, then `./bin/smina`.

## Running

```bash
python main.py
```

Key parameters live in `config.py` and near the top of `run_active_learning()` in
`main.py`.

**Runtime warning.** One arm costs roughly 2-3 minutes at default settings
(`pop_size=50`, `generations=10`, ~450 smina calls). A 500-iteration run is on the
order of a day. Reduce `max_iters` and the GA parameters to test the plumbing first.

## Repository layout

## Input format

* `ggb.rxn` — MDL reaction file with three reactant templates
* `reagent_r*.smi` — one SMILES per line, tab-delimited, no header
* `anchor.sdf` — reference ligand whose pose defines the frozen core
* `CORE_SMILES` / `ANCHOR_SMILES` in `config.py` are atom-mapped SMILES; shared map
  numbers define which core atoms land on which anchor atoms

To retarget the workflow, swap the reaction file, reagent lists, anchor, and those
two mapped SMILES.

## Known limitations

* Docking scores come from a rigid receptor and a single scoring function; they rank
  candidates, they do not predict affinity.
* The RF-Score branch of the hybrid scoring function is present but disabled.
* The GA only samples acyclic torsions, so rigid products get little conformer search.
* The sampler maintains an independent posterior per product rather than per reagent,
  which limits how much it generalises across the combinatorial space.

## License

MIT — see `LICENSE`. smina, RDKit, and Open Babel carry their own licenses.

## Citation

Thompson sampling for reaction-based enumeration follows Klarich et al.,
*Thompson Sampling — An Efficient Method for Searching Ultra-Large Synthesis on
Demand Databases*, J. Chem. Inf. Model. (2024).
