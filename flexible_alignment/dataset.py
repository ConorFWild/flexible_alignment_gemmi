from __future__ import annotations

import typing
from typing import *
import dataclasses
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import scipy
from sklearn import neighbors

from flexible_alignment.constants import RESIDUE_NAMES
from flexible_alignment.common import Dtag


@dataclasses.dataclass()
class Resolution:
    resolution: float

    @staticmethod
    def from_float(res: float):
        return Resolution(res)

    def to_float(self) -> float:
        return self.resolution


@dataclasses.dataclass()
class ResidueID:
    model: str
    chain: str
    insertion: str

    @staticmethod
    def from_residue_chain(model: gemmi.Model, chain: gemmi.Chain, res: gemmi.Residue):
        return ResidueID(model.name,
                         chain.name,
                         str(res.seqid.num),
                         )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return ((self.model, self.chain, self.insertion) ==
                    (other.model, other.chain, other.insertion))
        return NotImplemented

    def __hash__(self):
        return hash((self.model, self.chain, self.insertion))


@dataclasses.dataclass()
class RFree:
    rfree: float

    @staticmethod
    def from_structure(structure: Structure):
        rfree = structure.structure.make_mmcif_document()[0].find_loop("_refine.ls_R_factor_R_free")[0]

        return RFree(float(rfree))

    def to_float(self):
        return self.rfree


@dataclasses.dataclass()
class Structure:
    structure: gemmi.Structure
    path: typing.Union[Path, None] = None

    @staticmethod
    def from_file(file: Path) -> Structure:
        try:
            structure = gemmi.read_structure(str(file))
        except Exception as e:
            raise Exception(f'Error trying to open file: {file}: {e}')
        structure.setup_entities()
        return Structure(structure, file)

    def rfree(self):
        return RFree.from_structure(self)

    def __getitem__(self, item: ResidueID):
        return self.structure[item.model][item.chain][item.insertion]

    def protein_residue_ids(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    if residue.name.upper() not in RESIDUE_NAMES:
                        continue

                    try:
                        has_ca = residue["CA"][0]
                    except Exception as e:
                        continue

                    resid = ResidueID.from_residue_chain(model, chain, residue)
                    yield resid

    def protein_atoms(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():

                    if residue.name.upper() not in RESIDUE_NAMES:
                        continue

                    for atom in residue:
                        yield atom

    def all_atoms(self, exclude_waters=False):
        if exclude_waters:

            for model in self.structure:
                for chain in model:
                    for residue in chain:
                        if residue.is_water():
                            continue

                        for atom in residue:
                            yield atom

        else:
            for model in self.structure:
                for chain in model:
                    for residue in chain:

                        for atom in residue:
                            yield atom


@dataclasses.dataclass()
class StructureFactors:
    f: str
    phi: str

    @staticmethod
    def from_string(string: str):
        factors = string.split(",")
        assert len(factors) == 2
        return StructureFactors(f=factors[0],
                                phi=factors[1],
                                )


@dataclasses.dataclass()
class Symops:
    symops: typing.List[gemmi.Op]

    @staticmethod
    def from_grid(grid: gemmi.FloatGrid):
        spacegroup = grid.spacegroup
        operations = list(spacegroup.operations())
        return Symops(operations)

    def __iter__(self):
        for symop in self.symops:
            yield symop


@dataclasses.dataclass()
class Reflections:
    reflections: gemmi.Mtz
    path: typing.Union[Path, None] = None

    @staticmethod
    def from_file(file: Path) -> Reflections:

        try:
            reflections = gemmi.read_mtz_file(str(file))
        except Exception as e:
            raise Exception(f'Error trying to open file: {file}: {e}')
        return Reflections(reflections, file)

    def resolution(self) -> Resolution:
        return Resolution.from_float(self.reflections.resolution_high())

    def truncate_resolution(self, resolution: Resolution) -> Reflections:
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        # Add columns
        for column in self.reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Get data
        data_array = np.array(self.reflections, copy=True)
        data = pd.DataFrame(data_array,
                            columns=self.reflections.column_labels(),
                            )
        data.set_index(["H", "K", "L"], inplace=True)

        # add resolutions
        data["res"] = self.reflections.make_d_array()

        # Truncate by resolution
        data_truncated = data[data["res"] >= resolution.resolution]

        # Rem,ove res colum
        data_dropped = data_truncated.drop("res", "columns")

        # To numpy
        data_dropped_array = data_dropped.to_numpy()

        # new data
        new_data = np.hstack([data_dropped.index.to_frame().to_numpy(),
                              data_dropped_array,
                              ]
                             )

        # Update
        new_reflections.set_data(new_data)

        # Update resolution
        new_reflections.update_reso()

        return Reflections(new_reflections)

    def truncate_reflections(self, index=None) -> Reflections:
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        # Add columns
        for column in self.reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Get data
        data_array = np.array(self.reflections, copy=True)
        data = pd.DataFrame(data_array,
                            columns=self.reflections.column_labels(),
                            )
        data.set_index(["H", "K", "L"], inplace=True)

        # Truncate by index
        data_indexed = data.loc[index]

        # To numpy
        data_dropped_array = data_indexed.to_numpy()

        # new data
        new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
                              data_dropped_array,
                              ]
                             )

        # Update
        new_reflections.set_data(new_data)

        # Update resolution
        new_reflections.update_reso()

        return Reflections(new_reflections)

    def drop_columns(self, structure_factors: StructureFactors):
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        free_flag = None

        for column in self.reflections.columns:
            if column.label == "FREE":
                free_flag = "FREE"
                break
            if column.label == "FreeR_flag":
                free_flag = "FreeR_flag"
                break

        if not free_flag:
            raise Exception("No RFree Flag found!")

        # Add columns
        for column in self.reflections.columns:
            if column.label in ["H", "K", "L", free_flag, structure_factors.f, structure_factors.phi]:
                new_reflections.add_column(column.label, column.type)

        # Get data
        data_array = np.array(self.reflections, copy=True)
        data = pd.DataFrame(data_array,
                            columns=self.reflections.column_labels(),
                            )
        data.set_index(["H", "K", "L"], inplace=True)

        # Truncate by columns
        data_indexed = data[[free_flag, structure_factors.f, structure_factors.phi]]

        # To numpy
        data_dropped_array = data_indexed.to_numpy()

        # new data
        new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
                              data_dropped_array,
                              ]
                             )

        # Update
        new_reflections.set_data(new_data)

        # Update resolution
        new_reflections.update_reso()

        return Reflections(new_reflections)

    def spacegroup(self):
        return self.reflections.spacegroup

    def columns(self):
        return self.reflections.column_labels()

    def missing(self, structure_factors: StructureFactors, resolution: Resolution) -> pd.DataFrame:
        all_data = np.array(self.reflections, copy=True)
        resolution_array = self.reflections.make_d_array()

        table = pd.DataFrame(data=all_data, columns=self.reflections.column_labels())

        reflections_in_resolution = table[resolution_array >= resolution.to_float()]

        amplitudes = reflections_in_resolution[structure_factors.f]

        missing = reflections_in_resolution[amplitudes == 0]

        return missing

    def common_set(self, other_reflections: Reflections):
        # Index own reflections
        reflections_array = np.array(self.reflections, copy=False, )
        hkl_dict = {}
        f_index = self.reflections.column_labels().index("F")
        for i, row in enumerate(reflections_array):
            hkl = (row[0], row[1], row[2])
            if not np.isnan(row[f_index]):
                hkl_dict[hkl] = i

        # Index the other array
        other_reflections_array = np.array(other_reflections.reflections, copy=False, )
        other_hkl_dict = {}
        f_other_index = other_reflections.reflections.column_labels().index("F")
        for i, row in enumerate(other_reflections_array):
            hkl = (row[0], row[1], row[2])
            if not np.isnan(row[f_other_index]):
                other_hkl_dict[hkl] = i

        # Allocate the masks
        self_mask = np.zeros(reflections_array.shape[0],
                             dtype=np.bool,
                             )

        other_mask = np.zeros(other_reflections_array.shape[0],
                              dtype=np.bool,
                              )

        # Fill the masks
        for hkl, index in hkl_dict.items():
            try:
                other_index = other_hkl_dict[hkl]
                self_mask[index] = True
                other_mask[other_index] = True

            except:
                continue

        return self_mask, other_mask

    # TODO: Make this work reasonably?
    def scale_reflections(self, other: Reflections, cut: float = 99.6):

        data_table = pd.DataFrame(data=np.array(self.reflections),
                                  columns=self.reflections.column_labels(),
                                  index=["H", "K", "L"],
                                  )
        data_other_table = pd.DataFrame(data=np.array(other.reflections),
                                        columns=other.reflections.column_labels(),
                                        index=["H", "K", "L"],
                                        )

        # Set resolutions
        data_table["1_d2"] = self.reflections.make_1_d2_array()
        data_other_table["1_d2"] = other.reflections.make_1_d2_array()

        # Get common indexes
        data_index = data_table[~data_table["F"].isna()].index.to_flat_index()
        data_other_index = data_other_table[~data_other_table["F"].isna()].to_flat_index()
        intersection_index = data_index.intersection(data_other_index)
        intersection_list = intersection_index.to_list()

        # Select common data
        data_common_table = data_table[intersection_list]
        data_other_common_table = data_other_table[intersection_list]

        # Select common amplitudes
        f_array = data_common_table["F"].to_numpy()
        f_other_array = data_other_common_table["F"].to_numpy()

        # Select common resolutions
        res_array = data_common_table["1_d2"].to_numpy()
        res_other_array = data_other_common_table["1_d2"].to_numpy()

        min_scale_list = []
        for i in range(6):

            # Trunate outliers
            diff_array = np.abs(f_array - f_other_array)
            high_diff = np.percentile(diff_array, cut)

            x_truncated = f_array[diff_array < high_diff]
            y_truncated = f_other_array[diff_array < high_diff]

            x_r_truncated = res_array[diff_array < high_diff]
            y_r_truncated = res_other_array[diff_array < high_diff]

            # Interpolate
            knn_y = neighbors.RadiusNeighborsRegressor(0.01)
            knn_y.fit(y_r_truncated.reshape(-1, 1),
                      y_truncated.reshape(-1, 1),
                      )

            knn_x = neighbors.RadiusNeighborsRegressor(0.01)
            knn_x.fit(x_r_truncated.reshape(-1, 1),
                      x_truncated.reshape(-1, 1),
                      )

            sample_grid = np.linspace(min(y_r_truncated), max(y_r_truncated), 100)

            x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)
            y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

            # optimise scale
            scales = []
            rmsds = []

            for scale in np.linspace(-4, 4, 280):
                x = x_f
                y_s = y_f * np.exp(scale * sample_grid)

                rmsd = np.sum(np.square(x - y_s))

                scales.append(scale)
                rmsds.append(rmsd)

            min_scale = scales[np.argmin(np.log(rmsds))] / -0.5
            min_scale_list.append(min_scale)

            x_all = x_truncated
            y_all = y_truncated

            x_r_all = x_r_truncated
            y_r_all = y_r_truncated


@dataclasses.dataclass()
class Reference:
    dtag: Dtag
    dataset: Dataset

    @staticmethod
    def from_datasets(datasets: Dict[Dtag, Dataset]):
        # Reference.assert_from_datasets(datasets)

        resolutions: typing.Dict[Dtag, Resolution] = {}
        for dtag in datasets:
            resolutions[dtag] = datasets[dtag].reflections.resolution()

        min_resolution_dtag = min(
            resolutions,
            key=lambda dtag: resolutions[dtag].to_float(),
        )

        min_resolution_structure = datasets[min_resolution_dtag].structure
        min_resolution_reflections = datasets[min_resolution_dtag].reflections

        return Reference(min_resolution_dtag,
                         datasets[min_resolution_dtag]
                         )


@dataclasses.dataclass()
class Dataset:
    structure: Structure
    reflections: Reflections
    smoothing_factor: float = 0.0

    @staticmethod
    def from_files(pdb_file: Path, mtz_file: Path, ):
        strucure: Structure = Structure.from_file(pdb_file)
        reflections: Reflections = Reflections.from_file(mtz_file)

        return Dataset(structure=strucure,
                       reflections=reflections,
                       )

    def truncate_resolution(self, resolution: Resolution) -> Dataset:
        return Dataset(self.structure,
                       self.reflections.truncate_resolution(resolution,
                                                            )
                       )

    def truncate_reflections(self, index=None) -> Dataset:
        return Dataset(self.structure,
                       self.reflections.truncate_reflections(index,
                                                             )
                       )

    def scale_reflections(self, reflections: Reflections):
        new_reflections = self.reflections.scale_reflections(reflections)
        return Dataset(self.structure,
                       new_reflections,
                       )

    def drop_columns(self, structure_factors: StructureFactors):
        new_reflections = self.reflections.drop_columns(structure_factors)

        return Dataset(self.structure,
                       new_reflections)

    def common_reflections(self,
                           reference_ref: Reflections,
                           structure_factors: StructureFactors,
                           ):
        # Get own reflections
        dtag_reflections = self.reflections.reflections
        dtag_reflections_array = np.array(dtag_reflections, copy=True)
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                              columns=dtag_reflections.column_labels(),
                                              )
        dtag_reflections_table.set_index(["H", "K", "L"], inplace=True)
        dtag_flattened_index = dtag_reflections_table[
            ~dtag_reflections_table[structure_factors.f].isna()].index.to_flat_index()

        # Get reference
        reference_reflections = reference_ref.reflections
        reference_reflections_array = np.array(reference_reflections, copy=True)
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_reflections_table.set_index(["H", "K", "L"], inplace=True)
        reference_flattened_index = reference_reflections_table[
            ~reference_reflections_table[structure_factors.f].isna()].index.to_flat_index()

        running_index = dtag_flattened_index.intersection(reference_flattened_index)

        return running_index.to_list()
