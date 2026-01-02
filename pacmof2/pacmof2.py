import os
import joblib
from importlib.resources import files
from tqdm import tqdm
import glob
import numpy as np


from ase.io import read
from ase import neighborlist

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from pacmof2.data import electronegativity, first_ip, metals
from . import models


def load_models():
    # 'files(models)' returns a Traversable object that acts like a pathlib.Path
    neutral_path = files(models) / "PACMOF2_neutral.gz"
    ionic_path = files(models) / "PACMOF2_ionic.gz"

    # joblib.load accepts Path objects directly
    neutral_model = joblib.load(neutral_path)
    ionic_model = joblib.load(ionic_path)

    return neutral_model, ionic_model


def get_neighbor_indices_pm(i, pob, cnn):
    """Get indices using Pymatgen. Returns list or raises ValueError."""
    nn_data = cnn.get_nn_data(pob, i)
    nn_info = nn_data.all_nninfo
    indices = {item["site_index"] for item in nn_info}
    if not indices:
        raise ValueError(f"Atom {pob[i]} (index {i}) has no neighbors via CrystalNN.")
    return list(indices)


def get_neighbor_indices_ase(i, atoms, cutoff=None, skin=0.25):
    """Get indices using ASE. Returns list or raises ValueError."""
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


def get_neighbors_hybrid(atoms, pob, cnn):
    """Try Pymatgen first, failover to ASE."""
    neighbor_dict = {}
    for i in range(len(atoms)):
        try:
            neighbor_dict[i] = get_neighbor_indices_pm(i, pob, cnn)
        except (ValueError, Exception):
            # Fallback to ASE
            try:
                neighbor_dict[i] = get_neighbor_indices_ase(i, atoms)
            except ValueError:
                raise ValueError(
                    f"Atom {i} has no neighbors in both Pymatgen and ASE methods."
                )
    return neighbor_dict


def get_second_nearest_neighbors(neighbor_dict):
    """Optimized SNN search using sets."""
    snn_dict = {}
    for k, neighbors in neighbor_dict.items():
        neighbors_set = set(neighbors)
        snn = set()
        for n in neighbors:
            # Add neighbors of neighbors
            candidates = neighbor_dict.get(n, [])
            for candidate in candidates:
                if candidate != k and candidate not in neighbors_set:
                    snn.add(candidate)
        snn_dict[k] = list(snn)
    return snn_dict


def revise_nn(atoms, neighbor_dict):
    """
    Revise neighbor_dict to remove atoms that are in both nearest neighbor
    and second nearest neighbors.

    Logic restored:
    - For Carbon: If C connects to O, and O connects to Metal... ensure C does NOT connect to Metal directly.
    - For Metal: If Metal connects to O, and O connects to C... ensure Metal does NOT connect to C directly.
    """
    chem_symb = atoms.get_chemical_symbols()

    # We must iterate over keys, but modify the lists in place safely.
    for k in neighbor_dict:
        element_k = chem_symb[k]
        neighbors = neighbor_dict[k]

        # We collect items to remove in a set to avoid index shifting issues during iteration
        to_remove = set()

        # --- Carbon Logic ---
        if element_k == "C":
            for i in neighbors:
                # i is the potential Oxygen neighbor
                if chem_symb[i] == "O":
                    # Check Oxygen's neighbors for Metals
                    nn_of_O = neighbor_dict.get(i, [])
                    for j in nn_of_O:
                        # j is the Metal.
                        # If j is a metal AND j is currently listed as my (Carbon's) neighbor...
                        if chem_symb[j] in metals and j in neighbors:
                            to_remove.add(j)

        # --- Metal Logic ---
        elif element_k in metals:
            for i in neighbors:
                # i is the potential Oxygen neighbor
                if chem_symb[i] == "O":
                    # Check Oxygen's neighbors for Carbon
                    nn_of_O = neighbor_dict.get(i, [])
                    for j in nn_of_O:
                        # j is the Carbon.
                        # If j is a Carbon AND j is currently listed as my (Metal's) neighbor...
                        if chem_symb[j] == "C" and j in neighbors:
                            to_remove.add(j)

        # Apply removals safely
        if to_remove:
            neighbor_dict[k] = [n for n in neighbors if n not in to_remove]

    return neighbor_dict


# Feature extraction
def calculate_en_diff(i, atoms, neighbor_dict, ignore_idx=None):
    """Helper to calculate electronegativity difference."""
    symbols = atoms.get_chemical_symbols()
    center_en = electronegativity[symbols[i]]

    diffs = []
    for n_idx in neighbor_dict[i]:
        if ignore_idx is not None and n_idx == ignore_idx:
            continue
        neighbor_en = electronegativity[symbols[n_idx]]
        diffs.append(neighbor_en - center_en)

    return sum(diffs)


def get_features_wrapper(atoms, pob):
    """Orchestrates the feature generation pipeline."""
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

        # 5. compute numerical features
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
        print(f"Featurization failed: {e}")
        return None


def adjust_charge(charges, by="mean", net_charge=0):
    """Vectorized charge adjustment."""
    if by == "magnitude":
        denom = np.sum(np.abs(charges))
        if denom == 0:
            return charges
        return charges - np.sum(charges) * np.abs(charges) / denom
    elif by == "mean":
        return charges - (np.sum(charges) - net_charge) / len(charges)
    return charges


def write_cif(fileobj, images, charges):
    """
    Write atoms and partial charges to a CIF file.

    Parameters
    ----------
    fileobj : str or file object
        Path to the output file or an open file object.
    images : ase.Atoms or list of ase.Atoms
        The atoms to write.
    charges : list
        List of partial charges corresponding to the atoms.
    """
    # Ensure images is a list
    if hasattr(images, "get_positions"):
        images = [images]

    # Handle file opening (context manager simulation)
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

                        # If other species exist at this tag/site, append them to lists
                        for sym, occ in site_occ.items():
                            if sym != original_sym:
                                symbols.append(sym)
                                coords.append(coords[idx])
                                occupancies.append(occ)
                                # Default charge for split sites
                                if idx < len(charges):
                                    charges.append(charges[idx])
                                else:
                                    charges.append(0.0)
                    except KeyError:
                        pass

            # 6. Write Data Rows
            symbol_counts = {}
            for idx, (sym, pos, occ) in enumerate(zip(symbols, coords, occupancies)):
                # Handle charge indexing safely
                chg = charges[idx] if idx < len(charges) else 0.0

                # Generate Labels (e.g. C1, C2, O1)
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
                label = f"{sym}{symbol_counts[sym]}"

                f.write(
                    f"  {label:<8} {occ:6.4f} {pos[0]:7.5f}  {pos[1]:7.5f}  {pos[2]:7.5f}  {'Biso':<4}  {1.0:6.3f}  {sym}  {chg:6.6f}\n"
                )

    finally:
        if should_close:
            f.close()


def process_single_cif(
    cif_path,
    output_dir,
    models,
    identifier,
    net_charge_val,
    adjust_method,
    print_features,
):
    """
    Process a single CIF file: Read -> Featurize -> Predict -> Write.
    Returns True if successful, False otherwise.
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
            feature_suffix = "neutral"
        else:
            # Prepare features for ionic model (append neutral charge and net_charge density)
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

            print(f"Net charge before correction: {np.sum(ionic_charges):.4f}")
            final_charges = adjust_charge(
                ionic_charges, by=adjust_method, net_charge=net_charge_val
            )
            print(f"Net charge after correction: {np.sum(final_charges):.4f}")
            feature_suffix = "ionic"

        # Write Output
        base_name = os.path.splitext(os.path.basename(cif_path))[0]
        new_name = f"{base_name}{identifier}.cif"
        output_path = os.path.join(output_dir, new_name)

        print(f"Writing CIF {new_name}")
        write_cif(output_path, atoms, final_charges)

        # Optional: Write CSV
        if print_features:
            csv_name = f"features_{feature_suffix}.csv"

        return True

    except Exception as e:
        print(f"Failed to process {cif_path}: {e}")
        return False


def get_charges(
    path_to_cif,
    output_path,
    identifier="_pacmof",
    multiple_cifs=False,
    adjust_charge_method="mean",
    print_features=False,
    net_charge=0,
    models_module=None,
):
    # 1. Setup File List
    if multiple_cifs:
        cifs = sorted(glob.glob(os.path.join(path_to_cif, "*.cif")))
    else:
        cifs = [path_to_cif]

    # 2. Load Models Once
    print("Loading Models...")
    # Assuming 'models' is the imported module
    neutral_model, ionic_model = load_models()
    loaded_models = (neutral_model, ionic_model)

    # 3. Process Loop
    for cif in tqdm(cifs, desc="Processing CIFs"):
        # Determine net charge for this specific file
        current_net_charge = 0
        if isinstance(net_charge, dict):
            # Key lookup based on filename
            fname = os.path.basename(cif)
            current_net_charge = net_charge.get(fname, 0)  # Default to 0 if missing?
        else:
            current_net_charge = net_charge

        process_single_cif(
            cif,
            output_path,
            loaded_models,
            identifier,
            current_net_charge,
            adjust_charge_method,
            print_features,
        )
