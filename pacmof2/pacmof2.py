"""Core module for PACMOF2 charge prediction."""

import glob
import logging
import os
from importlib.resources import files
from typing import Optional, Union

import joblib
import numpy as np
from ase import neighborlist
from ase.io import read
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from pacmof2.data import electronegativity, first_ip, metals
from . import models

logger = logging.getLogger(__name__)


def load_models() -> tuple:
    """Load the pre-trained neutral and ionic charge prediction models.

    Returns
    -------
    tuple
        A tuple of (neutral_model, ionic_model) scikit-learn estimators.
    """
    neutral_path = files(models) / "PACMOF2_neutral.gz"
    ionic_path = files(models) / "PACMOF2_ionic.gz"

    neutral_model = joblib.load(neutral_path)
    ionic_model = joblib.load(ionic_path)

    return neutral_model, ionic_model


def get_neighbor_indices_pm(i: int, pob, cnn: CrystalNN) -> list[int]:
    """Get nearest neighbor indices for atom i using Pymatgen's CrystalNN.

    Parameters
    ----------
    i : int
        Index of the center atom.
    pob : pymatgen.core.Structure
        Pymatgen structure object.
    cnn : CrystalNN
        CrystalNN instance for neighbor detection.

    Returns
    -------
    list[int]
        List of neighbor atom indices.

    Raises
    ------
    ValueError
        If no neighbors are found.
    """
    nn_data = cnn.get_nn_data(pob, i)
    nn_info = nn_data.all_nninfo
    indices = {item["site_index"] for item in nn_info}
    if not indices:
        raise ValueError(f"Atom {pob[i]} (index {i}) has no neighbors via CrystalNN.")
    return list(indices)


def get_neighbor_indices_ase(
    i: int,
    atoms,
    nl: Optional[neighborlist.NeighborList] = None,
    cutoff=None,
    skin: float = 0.25,
) -> list[int]:
    """Get nearest neighbor indices for atom i using ASE's NeighborList.

    Parameters
    ----------
    i : int
        Index of the center atom.
    atoms : ase.Atoms
        ASE Atoms object.
    nl : NeighborList, optional
        Pre-built neighbor list. If None, one is constructed.
    cutoff : list or None
        Per-element cutoff radii. If None, natural cutoffs are used.
    skin : float
        Skin distance for neighbor list construction.

    Returns
    -------
    list[int]
        List of neighbor atom indices.

    Raises
    ------
    ValueError
        If no neighbors are found.
    """
    if nl is None:
        if cutoff is None:
            cutoff = neighborlist.natural_cutoffs(atoms)
        nl = neighborlist.NeighborList(
            cutoff, skin=skin, self_interaction=False, bothways=True
        )
        nl.update(atoms)

    indices = nl.get_neighbors(i)[0].tolist()

    if not indices:
        raise ValueError(f"Atom {atoms[i]} (index {i}) has no neighbors via ASE.")
    return list(set(indices))


def get_neighbors_hybrid(atoms, pob, cnn: CrystalNN) -> dict[int, list[int]]:
    """Build a neighbor dictionary using Pymatgen with ASE fallback.

    Tries Pymatgen's CrystalNN first for each atom. If that fails,
    falls back to ASE's NeighborList.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object.
    pob : pymatgen.core.Structure
        Pymatgen structure object.
    cnn : CrystalNN
        CrystalNN instance.

    Returns
    -------
    dict[int, list[int]]
        Mapping from atom index to list of nearest neighbor indices.
    """
    neighbor_dict = {}
    # Pre-build ASE neighbor list once for fallback use
    cutoff = neighborlist.natural_cutoffs(atoms)
    ase_nl = neighborlist.NeighborList(
        cutoff, skin=0.25, self_interaction=False, bothways=True
    )
    ase_nl.update(atoms)

    for i in range(len(atoms)):
        try:
            neighbor_dict[i] = get_neighbor_indices_pm(i, pob, cnn)
        except (ValueError, Exception):
            try:
                neighbor_dict[i] = get_neighbor_indices_ase(i, atoms, nl=ase_nl)
            except ValueError:
                raise ValueError(
                    f"Atom {i} has no neighbors in both Pymatgen and ASE methods."
                )
    return neighbor_dict


def get_second_nearest_neighbors(
    neighbor_dict: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Compute second-nearest neighbors from a neighbor dictionary.

    Parameters
    ----------
    neighbor_dict : dict[int, list[int]]
        Mapping from atom index to nearest neighbor indices.

    Returns
    -------
    dict[int, list[int]]
        Mapping from atom index to second-nearest neighbor indices.
    """
    snn_dict = {}
    for k, neighbors in neighbor_dict.items():
        neighbors_set = set(neighbors)
        snn = set()
        for n in neighbors:
            candidates = neighbor_dict.get(n, [])
            for candidate in candidates:
                if candidate != k and candidate not in neighbors_set:
                    snn.add(candidate)
        snn_dict[k] = list(snn)
    return snn_dict


def revise_nn(atoms, neighbor_dict: dict[int, list[int]]) -> dict[int, list[int]]:
    """Revise neighbor dict using chemistry-aware heuristics.

    Removes direct Carbon-Metal bonds when mediated by Oxygen
    (e.g., in carboxylate groups where C-O-Metal is the correct
    connectivity, not C-Metal).

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object.
    neighbor_dict : dict[int, list[int]]
        Neighbor dictionary to revise in place.

    Returns
    -------
    dict[int, list[int]]
        Revised neighbor dictionary.
    """
    chem_symb = atoms.get_chemical_symbols()

    for k in neighbor_dict:
        element_k = chem_symb[k]
        neighbors = neighbor_dict[k]
        to_remove = set()

        # Carbon logic: remove direct C-Metal bond if O mediates
        if element_k == "C":
            for i in neighbors:
                if chem_symb[i] == "O":
                    nn_of_O = neighbor_dict.get(i, [])
                    for j in nn_of_O:
                        if chem_symb[j] in metals and j in neighbors:
                            to_remove.add(j)

        # Metal logic: remove direct Metal-C bond if O mediates
        elif element_k in metals:
            for i in neighbors:
                if chem_symb[i] == "O":
                    nn_of_O = neighbor_dict.get(i, [])
                    for j in nn_of_O:
                        if chem_symb[j] == "C" and j in neighbors:
                            to_remove.add(j)

        if to_remove:
            neighbor_dict[k] = [n for n in neighbors if n not in to_remove]

    return neighbor_dict


def calculate_en_diff(
    i: int,
    atoms,
    neighbor_dict: dict[int, list[int]],
    ignore_idx: Optional[int] = None,
) -> float:
    """Calculate the sum of electronegativity differences between an atom and its neighbors.

    Parameters
    ----------
    i : int
        Index of the center atom.
    atoms : ase.Atoms
        ASE Atoms object.
    neighbor_dict : dict[int, list[int]]
        Neighbor dictionary.
    ignore_idx : int, optional
        Index of a neighbor to skip.

    Returns
    -------
    float
        Sum of (neighbor_EN - center_EN) for all neighbors.
    """
    symbols = atoms.get_chemical_symbols()
    center_en = electronegativity[symbols[i]]

    diffs = []
    for n_idx in neighbor_dict[i]:
        if ignore_idx is not None and n_idx == ignore_idx:
            continue
        neighbor_en = electronegativity[symbols[n_idx]]
        diffs.append(neighbor_en - center_en)

    return sum(diffs)


def get_features_wrapper(atoms, pob) -> Optional[object]:
    """Generate ML features for all atoms in a structure.

    The 7 features per atom are:
        1. First ionization potential of center atom
        2. Electronegativity of center atom
        3. Mean nearest-neighbor distance
        4. Mean nearest-neighbor electronegativity
        5. Mean nearest-neighbor ionization potential
        6. Mean second-nearest-neighbor electronegativity
        7. Sum of electronegativity differences with neighbors

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object.
    pob : pymatgen.core.Structure
        Pymatgen structure object.

    Returns
    -------
    ase.Atoms or None
        The atoms object with features stored in ``atoms.info["features"]``,
        or None if featurization failed.
    """
    cnn = CrystalNN(distance_cutoffs=(0.3, 0.6))

    try:
        # 1. Get Neighbors
        neighbor_dict = get_neighbors_hybrid(atoms, pob, cnn)

        # 2. Refine Neighbors
        neighbor_dict = revise_nn(atoms, neighbor_dict)

        # 3. Get SNN
        snn_dict = get_second_nearest_neighbors(neighbor_dict)

        # 4. Check Integrity
        if any(len(v) == 0 for v in snn_dict.values()):
            raise ValueError("Missing second nearest neighbors for some atoms.")

        # 5. Compute numerical features
        chem_symb = atoms.get_chemical_symbols()
        features_atoms = []

        for k in range(len(atoms)):
            nn_dist = atoms.get_distances(k, neighbor_dict[k], mic=True)
            nn_eneg = [electronegativity[chem_symb[i]] for i in neighbor_dict[k]]
            nn_ipot = [first_ip[chem_symb[i]] for i in neighbor_dict[k]]
            snn_eneg = [electronegativity[chem_symb[j]] for j in snn_dict[k]]
            en_diff = calculate_en_diff(k, atoms, neighbor_dict)

            features = [
                first_ip[chem_symb[k]],
                electronegativity[chem_symb[k]],
                round(np.mean(nn_dist), 4),
                round(np.mean(nn_eneg), 4),
                round(np.mean(nn_ipot), 4),
                round(np.mean(snn_eneg), 4),
                round(en_diff, 4),
            ]
            features_atoms.append(features)

        atoms.info["features"] = features_atoms
        return atoms

    except ValueError as e:
        logger.error("Featurization failed: %s", e)
        return None


def adjust_charge(
    charges: np.ndarray,
    by: str = "mean",
    net_charge: float = 0,
) -> np.ndarray:
    """Adjust predicted charges to enforce a target net charge.

    Parameters
    ----------
    charges : np.ndarray
        Array of predicted atomic charges.
    by : str
        Adjustment method: ``"mean"`` distributes the error equally,
        ``"magnitude"`` distributes proportional to absolute charge.
    net_charge : float
        Target net charge (0 for neutral MOFs).

    Returns
    -------
    np.ndarray
        Adjusted charges summing to ``net_charge``.
    """
    if by == "magnitude":
        denom = np.sum(np.abs(charges))
        if denom == 0:
            return charges
        return charges - np.sum(charges) * np.abs(charges) / denom
    elif by == "mean":
        return charges - (np.sum(charges) - net_charge) / len(charges)
    return charges


def write_cif(fileobj, images, charges: list[float]) -> None:
    """Write atoms and partial charges to a CIF file.

    Parameters
    ----------
    fileobj : str or file object
        Path to the output file or an open file object.
    images : ase.Atoms or list of ase.Atoms
        The atoms to write.
    charges : list[float]
        List of partial charges corresponding to the atoms.
    """
    # Ensure images is a list
    if hasattr(images, "get_positions"):
        images = [images]

    # Handle file opening
    if isinstance(fileobj, str):
        f = open(fileobj, "w", encoding="latin-1")
        should_close = True
    else:
        f = fileobj
        should_close = False

    try:
        for i, atoms in enumerate(images):
            # 1. Write Header and Cell Info
            f.write(f"data_image{i}\n")

            a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

            if atoms.number_of_lattice_vectors == 3:
                f.write(f"_cell_length_a       {a:g}\n")
                f.write(f"_cell_length_b       {b:g}\n")
                f.write(f"_cell_length_c       {c:g}\n")
                f.write(f"_cell_angle_alpha    {alpha:g}\n")
                f.write(f"_cell_angle_beta     {beta:g}\n")
                f.write(f"_cell_angle_gamma    {gamma:g}\n")
                f.write("\n")
                f.write("_symmetry_space_group_name_H-M    'P 1'\n")
                f.write("_symmetry_int_tables_number       1\n")
                f.write("\n")
                f.write("loop_\n")
                f.write("  _symmetry_equiv_pos_as_xyz\n")
                f.write("  'x, y, z'\n")
                f.write("\n")

            # 2. Setup Coordinate Type
            coord_type = "fract" if atoms.pbc.all() else "Cartn"

            # 3. Write Column Headers
            f.write("loop_\n")
            headers = [
                "_atom_site_label",
                "_atom_site_occupancy",
                f"_atom_site_{coord_type}_x",
                f"_atom_site_{coord_type}_y",
                f"_atom_site_{coord_type}_z",
                "_atom_site_thermal_displace_type",
                "_atom_site_B_iso_or_equiv",
                "_atom_site_type_symbol",
                "_atom_site_charge",
            ]

            for h in headers:
                f.write(f"  {h}\n")

            # 4. Prepare Data Arrays
            if coord_type == "fract":
                coords = atoms.get_scaled_positions().tolist()
            else:
                coords = atoms.get_positions().tolist()

            symbols = atoms.get_chemical_symbols()
            occupancies = [1.0] * len(symbols)

            # 5. Handle Mixed Occupancy
            if "occupancy" in atoms.info:
                occ_info = atoms.info["occupancy"]
                tags = atoms.get_tags()

                for idx, tag in enumerate(tags):
                    try:
                        site_occ = occ_info[tag]
                        original_sym = symbols[idx]

                        # Set occupancy for the primary atom
                        occupancies[idx] = site_occ.get(original_sym, 1.0)

                        # If other species exist at this tag/site, append them
                        for sym, occ in site_occ.items():
                            if sym != original_sym:
                                symbols.append(sym)
                                coords.append(coords[idx])
                                occupancies.append(occ)
                                if idx < len(charges):
                                    charges.append(charges[idx])
                                else:
                                    charges.append(0.0)
                    except KeyError:
                        pass

            # 6. Write Data Rows
            symbol_counts = {}
            for idx, (sym, pos, occ) in enumerate(zip(symbols, coords, occupancies)):
                chg = charges[idx] if idx < len(charges) else 0.0

                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
                label = f"{sym}{symbol_counts[sym]}"

                f.write(
                    f"  {label:<8} {occ:6.4f} {pos[0]:7.5f}  {pos[1]:7.5f}"
                    f"  {pos[2]:7.5f}  {'Biso':<4}  {1.0:6.3f}  {sym}"
                    f"  {chg:6.6f}\n"
                )

    finally:
        if should_close:
            f.close()


def process_single_cif(
    cif_path: str,
    output_dir: str,
    models: tuple,
    identifier: str,
    net_charge_val: float,
    adjust_method: str,
) -> bool:
    """Process a single CIF file: read, featurize, predict, and write.

    Parameters
    ----------
    cif_path : str
        Path to the input CIF file.
    output_dir : str
        Directory to write the output CIF.
    models : tuple
        Tuple of (neutral_model, ionic_model).
    identifier : str
        Suffix appended to the output filename.
    net_charge_val : float
        Net charge of the MOF (0 for neutral).
    adjust_method : str
        Charge adjustment method ("mean" or "magnitude").

    Returns
    -------
    bool
        True if processing succeeded, False otherwise.
    """
    neutral_model, ionic_model = models

    try:
        # Read
        atoms = read(cif_path)
        aaa = AseAtomsAdaptor()
        pob = aaa.get_structure(atoms)

        # Featurize
        atoms = get_features_wrapper(atoms, pob)
        if atoms is None:
            return False

        features = atoms.info["features"]

        # Prediction 1: Neutral
        raw_charges = neutral_model.predict(features)

        final_charges = None

        # Logic Branch: Neutral vs Ionic
        if net_charge_val == 0:
            final_charges = adjust_charge(raw_charges, by=adjust_method, net_charge=0)
        else:
            # Prepare features for ionic model
            natoms = len(atoms)
            net_charge_per_atom = net_charge_val / natoms

            ionic_features = []
            for i, f_row in enumerate(features):
                row = list(f_row)
                row.append(round(raw_charges[i], 4))
                row.append(round(net_charge_per_atom, 4))
                ionic_features.append(row)

            # Prediction 2: Ionic Correction
            charge_diff = ionic_model.predict(ionic_features)
            ionic_charges = charge_diff + raw_charges

            logger.info("Net charge before correction: %.4f", np.sum(ionic_charges))
            final_charges = adjust_charge(
                ionic_charges, by=adjust_method, net_charge=net_charge_val
            )
            logger.info("Net charge after correction: %.4f", np.sum(final_charges))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write Output
        base_name = os.path.splitext(os.path.basename(cif_path))[0]
        new_name = f"{base_name}{identifier}.cif"
        output_path = os.path.join(output_dir, new_name)

        logger.info("Writing CIF %s", new_name)
        write_cif(output_path, atoms, final_charges)

        return True

    except Exception as e:
        logger.error("Failed to process %s: %s", cif_path, e)
        return False


def get_charges(
    path_to_cif: str,
    output_path: str,
    identifier: str = "_pacmof",
    multiple_cifs: bool = False,
    adjust_charge_method: str = "mean",
    net_charge: Union[int, float, dict] = 0,
) -> None:
    """Predict partial atomic charges for one or more MOF CIF files.

    This is the main public API for PACMOF2. It loads the pre-trained
    models, computes features, predicts charges, and writes output CIF
    files with ``_atom_site_charge`` annotations.

    Parameters
    ----------
    path_to_cif : str
        Path to a single CIF file, or a directory containing CIF files
        when ``multiple_cifs=True``.
    output_path : str
        Directory where output CIF files will be written.
    identifier : str
        Suffix appended to output filenames (default: ``"_pacmof"``).
    multiple_cifs : bool
        If True, process all ``.cif`` files in ``path_to_cif`` directory.
    adjust_charge_method : str
        Method for enforcing net charge neutrality. Either ``"mean"``
        (distribute error equally) or ``"magnitude"`` (distribute
        proportional to absolute charge).
    net_charge : int, float, or dict
        Net charge of the MOF. Use 0 for neutral MOFs, a number for a
        single ionic MOF, or a dict mapping CIF filenames to net charges
        for batch ionic processing.

    Examples
    --------
    Neutral MOF (single file):

    >>> from pacmof2 import get_charges
    >>> get_charges("my_mof.cif", "output/")

    Ionic MOFs (batch with JSON-loaded dict):

    >>> import json
    >>> with open("net_charges.json") as f:
    ...     charges = json.load(f)
    >>> get_charges("cifs/", "output/", multiple_cifs=True, net_charge=charges)
    """
    # Configure logging for CLI/interactive use
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )

    # 1. Setup File List
    if multiple_cifs:
        cifs = sorted(glob.glob(os.path.join(path_to_cif, "*.cif")))
    else:
        cifs = [path_to_cif]

    # 2. Load Models Once
    logger.info("Loading Models...")
    neutral_model, ionic_model = load_models()
    loaded_models = (neutral_model, ionic_model)

    # 3. Process Loop
    for cif in tqdm(cifs, desc="Processing CIFs", disable=len(cifs) <= 1):
        # Determine net charge for this specific file
        current_net_charge: float = 0
        if isinstance(net_charge, dict):
            fname = os.path.basename(cif)
            current_net_charge = net_charge.get(fname, 0)
        else:
            current_net_charge = net_charge

        process_single_cif(
            cif,
            output_path,
            loaded_models,
            identifier,
            current_net_charge,
            adjust_charge_method,
        )
