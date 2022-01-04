from pathlib import Path

import fire

from flexible_alignment import Dataset, Reference, Alignment, Grid, Xmap, StructureFactors, Dtag


def test(
        reference_dtag,
        reference_pdb_path,
        reference_mtz_path,
        moving_dtag,
        moving_pdb_path,
        moving_mtz_path,
        output_map_path,
        outer_mask_radius=6.0,
        sample_rate=3.0,
        f="FWT",
        phi="PHWT",
):
    print('Converting input paths to python path')
    reference_pdb_path = Path(reference_pdb_path)
    reference_mtz_path = Path(reference_mtz_path)
    moving_pdb_path = Path(moving_pdb_path)
    moving_mtz_path = Path(moving_mtz_path)
    output_map_path = Path(output_map_path)

    print('Converting input structure factors to StructureFactors')
    structure_factors = StructureFactors(f, phi)

    print('Getting the reference and moving data as Datasets')
    datasets = {
        Dtag(reference_dtag): Dataset.from_files(
            reference_pdb_path,
            reference_mtz_path,
        ),
        Dtag(moving_dtag): Dataset.from_files(
            moving_pdb_path,
            moving_mtz_path,
        ),
    }

    print('Determining the reference from resolution')
    reference = Reference(
        Dtag(reference_dtag),
        datasets[Dtag(reference_dtag)],
        )

    print('Aligning datasets to reference')
    alignments = {}
    for dtag, dataset in datasets.items():
        alignments[dtag] = Alignment.from_dataset(reference, datasets[dtag])

    print('Getting grid')
    grid = Grid.from_reference(
        reference,
        outer_mask_radius,
        sample_rate=sample_rate,
    )

    print('Flexibly aligning datasets')
    flexibly_aligned_xmaps = {}
    for dtag, dataset in datasets.items():
        flexibly_aligned_xmaps[dtag] = Xmap.from_unaligned_dataset(
            dataset,
            alignments[dtag],
            grid=grid,
            structure_factors=structure_factors,
        )

    flexibly_aligned_xmaps[Dtag(moving_dtag)].save(output_map_path)


if __name__ == "__main__":
    fire.Fire(test)
