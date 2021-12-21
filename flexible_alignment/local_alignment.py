import typing
import dataclasses

import gemmi
import numpy as np
import scipy
from scipy import spatial

from flexible_alignment.dataset import Dataset, ResidueID, Reference, Datasets


class AlignmentUnmatchedAtomsError(Exception):
    def __init__(self, reference_array, other_array):
        message = f"Reference array has size {reference_array.size} while other array has size f{other_array.size}"

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class ExceptionUnmatchedAlignmentMarker(Exception):
    def __init__(self, residue_id) -> None:
        message = f"Found no reference atoms to compare against for residue: {residue_id}"

        super().__init__(message)


class ExceptionNoCommonAtoms(Exception):
    def __init__(self) -> None:
        message = f"Found no common atoms!"

        super().__init__(message)


@dataclasses.dataclass()
class Transform:
    transform: gemmi.Transform
    com_reference: np.array
    com_moving: np.array

    def apply_moving_to_reference(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_moving[0],
                                                     position[1] - self.com_moving[1],
                                                     position[2] - self.com_moving[2])
            transformed_vector = self.transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_reference[0],
                                                          transformed_vector[1] + self.com_reference[1],
                                                          transformed_vector[2] + self.com_reference[2])

        return transformed_positions

    def apply_reference_to_moving(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        inverse_transform = self.transform.inverse()
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_reference[0],
                                                     position[1] - self.com_reference[1],
                                                     position[2] - self.com_reference[2])
            transformed_vector = inverse_transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_moving[0],
                                                          transformed_vector[1] + self.com_moving[1],
                                                          transformed_vector[2] + self.com_moving[2])

        return transformed_positions

    @staticmethod
    def from_translation_rotation(translation, rotation, com_reference, com_moving):
        transform = gemmi.Transform()
        transform.vec.fromlist(translation.tolist())
        transform.mat.fromlist(rotation.as_matrix().tolist())

        return Transform(transform, com_reference, com_moving)

    @staticmethod
    def pos_to_list(pos: gemmi.Position):
        return [pos[0], pos[1], pos[2]]

    @staticmethod
    def from_atoms(dataset_selection,
                   reference_selection,
                   com_dataset,
                   com_reference,
                   ):

        # mean = np.mean(dataset_selection, axis=0)
        # mean_ref = np.mean(reference_selection, axis=0)
        # mean = np.array(com_dataset)
        # mean_ref = np.array(com_reference)
        mean = com_dataset
        mean_ref = com_reference

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = dataset_selection - mean
        de_meaned_ref = reference_selection - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)


@dataclasses.dataclass()
class Alignment:
    transforms: typing.Dict[ResidueID, Transform]

    def __getitem__(self, item: ResidueID):
        return self.transforms[item]

    def reference_to_moving(self, positions):

        transforms_list = [transform for transform in self.transforms.values()]

        reference_positions = np.vstack([transform.com_reference for transform in transforms_list])
        tree = spatial.KDTree(reference_positions)
        dist, inds = tree.query(positions)

        results = []
        for pos, ind in zip(positions, inds):
            transform = transforms_list[ind]
            transformed_pos = transform.apply_reference_to_moving({(0, 0, 0): gemmi.Position(*pos), })[(0, 0, 0)]
            results.append((transformed_pos.x, transformed_pos.y, transformed_pos.z,))

        return results

    @staticmethod
    def has_large_gap(reference: Reference, dataset: Dataset):
        try:
            Alignment.from_dataset(reference, dataset)

        except ExceptionUnmatchedAlignmentMarker as e:
            return False

        except ExceptionNoCommonAtoms as e:
            return False

        return True

    @staticmethod
    def from_dataset(reference: Reference, dataset: Dataset, marker_atom_search_radius=10.0):

        dataset_pos_list = []
        reference_pos_list = []

        # Iterate protein atoms, then pull out their atoms, and search them
        for res_id in reference.dataset.structure.protein_residue_ids():

            # Get the matchable CAs
            try:
                # Get reference residue
                ref_res_span = reference.dataset.structure[res_id]
                ref_res = ref_res_span[0]

                # Get corresponding reses
                dataset_res_span = dataset.structure[res_id]
                dataset_res = dataset_res_span[0]

                # Get the CAs
                atom_ref = ref_res["CA"][0]
                atom_dataset = dataset_res["CA"][0]

                # Get the shared atoms
                reference_pos_list.append([atom_ref.pos.x, atom_ref.pos.y, atom_ref.pos.z, ])
                dataset_pos_list.append([atom_dataset.pos.x, atom_dataset.pos.y, atom_dataset.pos.z, ])

            except Exception as e:
                print(f"WARNING: An exception occured in matching residues for alignment at residue id: {res_id}: {e}")
                continue

        dataset_atom_array = np.array(dataset_pos_list)
        reference_atom_array = np.array(reference_pos_list)

        if (reference_atom_array.shape[0] == 0) or (dataset_atom_array.shape[0] == 0):
            raise ExceptionNoCommonAtoms()

        # dataset kdtree
        dataset_tree = spatial.KDTree(dataset_atom_array)
        # Other kdtree
        reference_tree = spatial.KDTree(reference_atom_array)

        if reference_atom_array.size != dataset_atom_array.size:
            raise AlignmentUnmatchedAtomsError(reference_atom_array,
                                               dataset_atom_array,
                                               )

        transforms = {}

        # Start searching
        for res_id in reference.dataset.structure.protein_residue_ids():
            # Get reference residue
            ref_res_span = reference.dataset.structure[res_id]
            ref_res = ref_res_span[0]

            # Get ca pos in reference model
            reference_ca_pos = ref_res["CA"][0].pos

            # other selection
            reference_indexes = reference_tree.query_ball_point(
                [reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                marker_atom_search_radius,
            )
            reference_selection = reference_atom_array[reference_indexes]
            dataset_selection = dataset_atom_array[reference_indexes]

            if dataset_selection.shape[0] == 0:
                raise ExceptionUnmatchedAlignmentMarker(res_id)

            transforms[res_id] = Transform.from_atoms(
                dataset_selection,
                reference_selection,
                # com_dataset=[dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z],
                # com_reference=[reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                com_dataset=np.mean(dataset_selection, axis=0),
                com_reference=np.mean(reference_selection, axis=0),

            )

        return Alignment(transforms)

    def __iter__(self):
        for res_id in self.transforms:
            yield res_id
