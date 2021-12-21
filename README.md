# PanDDA 2

## Installation

Just git pull and pip install!

```bash
conda create -n pandda2 python=3.9
conda activate pandda2
conda install -c conda-forge fire numpy scipy pandas gemmi
git clone https://github.com/ConorFWild/pandda_2_gemmi.git
cd /path/to/cloned/repository
pip install .

```

## Example usage

from_aligned_dataset_c requires the "custom.cpp" snippet of c++ code to be in the local gemmi install.

```python
from flexible_alignment import Dataset, Reference, Alignment, Grid, Xmap,
    StructureFactors

structure_factors = StructureFactors('FWT', 'PHWT')

datasets = {}
for dtag, pdb_file, mtz_file in files:
    datasets[dtag] = Dataset.from_files(pdb_file, mtz_file)

reference = Reference.from_datasets(datasets)

alignments = {}
for dtag, dataset in datasets.items():
    alignments[dtag] = Alignment.from_dataset(reference, datasets[dtag])

grid = Grid.from_reference(
    reference,
    outer_mask_radius,
    inner_mask_symmetry,
    sample_rate=sample_rate,
)

flexibly_aligned_xmaps = {}
for dtag, dataset in datasets.items():
    flexibly_aligned_xmaps[dtag] = Xmap.from_unaligned_dataset_c(
        dataset,
        alignments[dtag],
        grid=grid,
        structure_factors=structure_factors,
    )
    
    # or 
    # flexibly_aligned_xmaps[dtag] = Xmap.from_unaligned_dataset(
    #     dataset,
    #     alignments[dtag],
    #     grid=grid,
    #     structure_factors=structure_factors,
    # )
    



```