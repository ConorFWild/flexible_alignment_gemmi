from __future__ import annotations

from typing import *
import dataclasses
from pathlib import Path

import gemmi
import numpy as np

from flexible_alignment.dataset import StructureFactors, Reflections, Dataset
from flexible_alignment.edalignment.alignments import Alignment
from flexible_alignment.edalignment.grid import Grid, Partitioning


@dataclasses.dataclass()
class Xmap:
    xmap: gemmi.FloatGrid

    @staticmethod
    def from_reflections(reflections: Reflections):
        pass

    @staticmethod
    def from_file(file):
        ccp4 = gemmi.read_ccp4_map(str(file))
        ccp4.setup()
        return Xmap(ccp4.grid)

    @staticmethod
    def interpolate_grid(grid: gemmi.FloatGrid,
                         positions: Dict[Tuple[int],
                                         gemmi.Position]) -> Dict[Tuple[int], float]:
        return {coord: grid.interpolate_value(pos) for coord, pos in positions.items()}

    @staticmethod
    def from_unaligned_dataset(dataset: Dataset, alignment: Alignment, grid: Grid, structure_factors: StructureFactors,
                               sample_rate: float = 3.0):

        unaligned_xmap: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                                 structure_factors.phi,
                                                                                                 sample_rate=sample_rate,
                                                                                                 )
        unaligned_xmap_array = np.array(unaligned_xmap, copy=False)
        std = np.std(unaligned_xmap_array)

        # unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

        interpolated_values_tuple = ([], [], [], [])

        for residue_id in alignment:
            alignment_positions: Dict[Tuple[int], gemmi.Position] = grid.partitioning[residue_id]

            transformed_positions: Dict[Tuple[int],
                                        gemmi.Position] = alignment[residue_id].apply_reference_to_moving(
                alignment_positions)

            transformed_positions_fractional: Dict[Tuple[int], gemmi.Fractional] = {
                point: unaligned_xmap.unit_cell.fractionalize(pos) for point, pos in transformed_positions.items()}

            interpolated_values: Dict[Tuple[int], float] = Xmap.interpolate_grid(
                unaligned_xmap,
                transformed_positions_fractional)

            interpolated_values_tuple = (interpolated_values_tuple[0] + [index[0] for index in interpolated_values],
                                         interpolated_values_tuple[1] + [index[1] for index in interpolated_values],
                                         interpolated_values_tuple[2] + [index[2] for index in interpolated_values],
                                         interpolated_values_tuple[3] + [interpolated_values[index] for index in
                                                                         interpolated_values],
                                         )

        new_grid = grid.new_grid()

        grid_array = np.array(new_grid, copy=False)

        grid_array[interpolated_values_tuple[0:3]] = interpolated_values_tuple[3]

        return Xmap(new_grid)

    @staticmethod
    def from_unaligned_dataset_c(dataset: Dataset,
                                 alignment: Alignment,
                                 grid: Grid,
                                 structure_factors: StructureFactors,
                                 sample_rate: float = 3.0,
                                 ):

        unaligned_xmap: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                                 structure_factors.phi,
                                                                                                 sample_rate=sample_rate,
                                                                                                 )
        unaligned_xmap_array = np.array(unaligned_xmap, copy=False)

        std = np.std(unaligned_xmap_array)
        unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

        new_grid = grid.new_grid()
        # Unpack the points, poitions and transforms
        point_list: List[Tuple[int, int, int]] = []
        position_list: List[Tuple[float, float, float]] = []
        transform_list: List[gemmi.transform] = []
        com_moving_list: List[np.array] = []
        com_reference_list: List[np.array] = []

        for residue_id, point_position_dict in grid.partitioning.partitioning.items():

            al = alignment[residue_id]
            transform = al.transform.inverse()
            com_moving = al.com_moving
            com_reference = al.com_reference

            for point, position in point_position_dict.items():
                point_list.append(point)
                position_list.append(position)
                transform_list.append(transform)
                com_moving_list.append(com_moving)
                com_reference_list.append(com_reference)

        # Interpolate values
        interpolated_grid = gemmi.interpolate_points(unaligned_xmap,
                                                     new_grid,
                                                     point_list,
                                                     position_list,
                                                     transform_list,
                                                     com_moving_list,
                                                     com_reference_list,
                                                     )

        return Xmap(interpolated_grid)

    def new_grid(self):
        spacing = [self.xmap.nu, self.xmap.nv, self.xmap.nw]
        unit_cell = self.xmap.unit_cell
        grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
        grid.set_unit_cell(unit_cell)
        grid.spacegroup = self.xmap.spacegroup
        return grid

    def to_array(self, copy=True):
        return np.array(self.xmap, copy=copy)

    def save(self, path: Path, p1: bool = True):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.xmap
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))

    @staticmethod
    def from_grid_array(grid: Grid, array_flat):
        new_grid = grid.new_grid()

        mask = grid.partitioning.protein_mask
        mask_array = np.array(mask, copy=False, dtype=np.int8)

        array = np.zeros(mask_array.shape, dtype=np.float32)
        array[np.nonzero(mask_array)] = array_flat

        for point in new_grid:
            u = point.u
            v = point.v
            w = point.w
            new_grid.set_value(u, v, w, float(array[u, v, w]))

        return Xmap(new_grid)
