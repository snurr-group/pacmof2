from ase.io import read
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import os
import joblib
from importlib import resources as impresources
from . import models
from tqdm import tqdm
import warnings

radius = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
          'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
          'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
          'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
          'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
          'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
          'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
          'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
          'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
          'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
          'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
          'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
          'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
          'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
          'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
          'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
          'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
          'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
          'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
          'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
          'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
          'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
          'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
          'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69}

# electronegativity in pauling scale from CRC Handbook of Chemistry and Physics (For elements not having pauling electronegativity, Allred Rochow electronegativity is taken)
electronegativity = {'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57,
                     'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44,
                     'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31,
                     'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58,
                     'Cl': 3.16, 'Ar': 0.00, 'K': 0.82, 'Ca': 1.00,
                     'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
                     'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
                     'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
                     'As': 2.01, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.00,
                     'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
                     'Nb': 1.60, 'Mo': 2.16, 'Tc': 2.10, 'Ru': 2.20,
                     'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
                     'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10,
                     'I': 2.66, 'Xe': 2.60, 'Cs': 0.79, 'Ba': 0.89,
                     'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
                     'Pm': 1.07, 'Sm': 1.17, 'Eu': 1.01, 'Gd': 1.20,
                     'Tb': 1.10, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
                     'Tm': 1.25, 'Yb': 1.06, 'Lu': 1.00, 'Hf': 1.30,
                     'Ta': 1.50, 'W': 1.70, 'Re': 1.90, 'Os': 2.20,
                     'Ir': 2.20, 'Pt': 2.20, 'Au': 2.40, 'Hg': 1.90,
                     'Tl': 1.80, 'Pb': 1.80, 'Bi': 1.90, 'Po': 2.00,
                     'At': 2.20, 'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90,
                     'Ac': 1.10, 'Th': 1.30, 'Pa': 1.50, 'U': 1.70,
                     'Np': 1.30, 'Pu': 1.30, 'Am': 1.30, 'Cm': 1.30}

# First ionization energy (from CRC Handbook of Chemistry and Physics)
first_ip = {'H': 13.598, 'He': 24.587, 'Li': 5.392, 'Be': 9.323,
            'B': 8.298, 'C': 11.260, 'N': 14.534, 'O': 13.618,
            'F': 17.423, 'Ne': 21.565, 'Na': 5.139, 'Mg': 7.646,
            'Al': 5.986, 'Si': 8.152, 'P': 10.487, 'S': 10.360,
            'Cl': 12.968, 'Ar': 15.760, 'K': 4.341, 'Ca': 6.113,
            'Sc': 6.561, 'Ti': 6.828, 'V': 6.746, 'Cr': 6.767,
            'Mn': 7.434, 'Fe': 7.902, 'Co': 7.881, 'Ni': 7.640,
            'Cu': 7.726, 'Zn': 9.394, 'Ga': 5.999, 'Ge': 7.899,
            'As': 9.789, 'Se': 9.752, 'Br': 11.814, 'Kr': 14.000,
            'Rb': 4.177, 'Sr': 5.695, 'Y': 6.217, 'Zr': 6.634,
            'Nb': 6.759, 'Mo': 7.092, 'Tc': 7.280, 'Ru': 7.360,
            'Rh': 7.459, 'Pd': 8.337, 'Ag': 7.576, 'Cd': 8.994,
            'In': 5.786, 'Sn': 7.344, 'Sb': 8.608, 'Te': 9.010,
            'I': 10.451, 'Xe': 12.130, 'Cs': 3.894, 'Ba': 5.212,
            'La': 5.577, 'Ce': 5.539, 'Pr': 5.473, 'Nd': 5.525,
            'Pm': 5.582, 'Sm': 5.644, 'Eu': 5.670, 'Gd': 6.150,
            'Tb': 5.864, 'Dy': 5.939, 'Ho': 6.021, 'Er': 6.108,
            'Tm': 6.184, 'Yb': 6.254, 'Lu': 5.426, 'Hf': 6.825,
            'Ta': 7.550, 'W': 7.864, 'Re': 7.833, 'Os': 8.438,
            'Ir': 8.967, 'Pt': 8.959, 'Au': 9.226, 'Hg': 10.437,
            'Tl': 6.108, 'Pb': 7.417, 'Bi': 7.286, 'Po': 8.414,
            'At': 9.318, 'Rn': 10.748, 'Fr': 4.073, 'Ra': 5.278,
            'Ac': 5.170, 'Th': 6.307, 'Pa': 5.890, 'U': 6.194,
            'Np': 6.266, 'Pu': 6.026, 'Am': 5.974, 'Cm': 5.991}

metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 
           'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
            'La', 'Ce', 'Pr', 'Nd', 'Pm','Sm','Eu','Gd','Tb']

def get_neighbor_indices_pm(i, pob, cnn):
    """ Get indices of nearest neighbors of central atom i using Pymatgen CrystalNN
    
    Parameters
    ----------
    i: int
       index of central atom
    pob: pymatgen object
         pymatgen object of framework
    cnn: CrystalNN object of pob
    
    """
    nn_data = cnn.get_nn_data(pob, i)
    nn_info = nn_data.all_nninfo

    def get_nn_info(nn_info):
        indices = []
        for index, item in enumerate(nn_info):
            indices.append(item['site_index'])

        return list(set(indices))

    indices = get_nn_info(nn_info)
    if len(indices) > 0:
        return 0, indices
    else:
        print("Atom {} with index {} does not have a neighbor".format(pob[i], i))
        return 1, indices

def get_neighbor_indices_ase(i, atoms, cutoff='default', skin=0.25):
    """ Get indices of nearest neighbors of central atom i using ASE neighborlist
    
    Parameters
    ----------
    i: int
       index of central atom
    atoms: ASE atom object
    cutoff: cutoff for ASE neighborlist class (see ASE for format)
    skin: skin distance for atoms to be considered neighbor. In this, atom is considered neighbor if distance < cutoff + 2*skin (skin is applied for each atom).
    """
    from ase import neighborlist

    if cutoff == 'default':
        cutoff = neighborlist.natural_cutoffs(atoms)

    neighborList = neighborlist.NeighborList(cutoff, skin=skin, self_interaction=False, bothways=True)
    neighborList.update(atoms)
    indices = neighborList.get_neighbors(i)[0].tolist()

    if len(indices) > 0:
        indices = list(set(indices))
        return 0, indices
    else:
        print("Atom {} with index {} does not have a neighbor".format(atoms[i], i))
        return 1, indices

def get_second_nearest_neighbors(neighbor_dict):
    """ Get a dictionary of 2nd shell neighbor for each atom. Ensure central atom not in second nearest neighbor dict.
    
    Parameters
    ----------
    neighbor_dict: dict, neighbor list dictionary    
    """

    snn_dict = {}
    for k in neighbor_dict:
        snn = []
        neighbors = neighbor_dict[k]
        for n in neighbors:
            to_add = neighbor_dict[n]
            for index in to_add:
                if index == k:
                    continue
                #if index in neighbors:
                #    continue
                if index not in snn:
                    snn.append(index)

        snn_dict[k] = snn
    return snn_dict

def revise_nn(atoms, neighbor_dict):
    """ Revise neighbor_dict to remove atoms that are in both nearest neighbor and second nearest neighbors. Only revises coordination of C and metal elements
    
    Parameters
    ----------
    atoms: ASE atom object
    neighbor_dict: dict, neighbor list dictionary  

    """
    chem_symb = atoms.get_chemical_symbols()

    for k in neighbor_dict:
        if chem_symb[k] == 'C':
            neighbors = neighbor_dict[k]
            for i in neighbors:
                if chem_symb[i] == 'O':
                    nn_neighbors = neighbor_dict[i]
                    for j in nn_neighbors:
                        if j in nn_neighbors and chem_symb[j] in metals:
                            try:
                                neighbors.remove(j)
                            except ValueError:
                                continue
            neighbor_dict[k] = neighbors

        if chem_symb[k] in metals:
            neighbors = neighbor_dict[k]
            for i in neighbors:
                if chem_symb[i] == 'O':
                    nn_neighbors = neighbor_dict[i]
                    for j in nn_neighbors:
                        if j in nn_neighbors and chem_symb[j]=='C':
                            try:
                                neighbors.remove(j)
                            except ValueError:
                                continue
            neighbor_dict[k] = neighbors
    return neighbor_dict

def get_features_cif(path_to_cif):

    def calculate_EN_diff(i, atoms, input_dict, ignore=None):
        """ Calculate auto-correlation of atom i.
            Parameters:
                i: int, atom index
                atoms: obj, ASE atom object
                input_dict: dict, either neighbor_dict (d1)
                ignore: list, atom index to ignore

        """
        en_diff = []
        if ignore == None:
            for a in input_dict[i]:
                en_diff.append(electronegativity[atoms.get_chemical_symbols()[a]] - electronegativity[atoms.get_chemical_symbols()[i]])
        else:
            for a in input_dict[i]:
                if a != ignore:
                    en_diff.append(electronegativity[atoms.get_chemical_symbols()[a]] - electronegativity[atoms.get_chemical_symbols()[i]])
        return sum(en_diff)

    def get_features(atoms, pob, neighbor_dict, snn_dict):
        chem_symb = atoms.get_chemical_symbols()
        pob_d_matrix = pob.distance_matrix
        features_atoms = []

        for k in range(len(atoms)):
            nn_dist = atoms.get_distances(k, neighbor_dict[k], mic=True)
            nn_eneg = []
            nn_ipot = []
            nn_ipot_diff = []
            snn_eneg = []
            en_diff = calculate_EN_diff(k, atoms, neighbor_dict)
            for i in neighbor_dict[k]:
                nn_eneg.append(electronegativity[chem_symb[i]])
                nn_ipot.append(first_ip[chem_symb[i]])

            for j in snn_dict[k]:
                snn_eneg.append(electronegativity[chem_symb[j]])

            features = [first_ip[chem_symb[k]],
                        electronegativity[chem_symb[k]],
                        round(np.mean(nn_dist),4),
                        round(np.mean(nn_eneg),4),
                        round(np.mean(nn_ipot),4),
                        round(np.mean(snn_eneg),4),
                        round(en_diff, 4)
                       ]
            np.round(features, 4)
            features_atoms.append(features)
        atoms.info['features'] = features_atoms
        return atoms
    
    # Main function

    # Get atom input with ASE and Pymatgen
    atoms = read(path_to_cif)
    aaa = AseAtomsAdaptor()
    pob = aaa.get_structure(atoms)
    
    warnings.filterwarnings("ignore")

    cnn = CrystalNN(distance_cutoffs=(0.3, 0.6))

    neighbor_dict = {}
    
    check_neighbor = True
    check_snn = True

    for i in range(len(atoms)):
        try:
            code, neighbor_dict[i] = get_neighbor_indices_pm(i, pob, cnn)
            if code == 1:
                code, neighbor_dict[i] = get_neighbor_indices_ase(i, atoms)
                if code == 1:
                    print("Found no neighbor for atom {}".format(atoms[i]))
                    check_neighbor == False
                    break

        # Sometimes Pymatgen Voronoi decomposition fails. Use ASE neighborlist as backup
        except ValueError:
            code, neighbor_dict[i] = get_neighbor_indices_ase(i, atoms)
            if code == 1:
                print("Found no neighbor for atom {}".format(atoms[k]))
                check_neighbor = False
                break

    if check_neighbor == False:
        print("Cannot featurize MOF {} because of missing neighbors".format(path_to_cif))
        return 1
    try:
        neighbor_dict = revise_nn(atoms,neighbor_dict)
        snn_dict = get_second_nearest_neighbors(neighbor_dict)
    except Exception as e:
        return 1
    for k in snn_dict:
        if len(snn_dict[k]) == 0:
            check_snn = False
            break

    if check_snn == False:
        print("Cannot featurize MOF {} because of missing second nearest neighbors for atom {}".format(path_to_cif, atoms[k]))
        return 1
    
    features = get_features(atoms, pob, neighbor_dict, snn_dict)
    return features

def write_cif(fileobj, images, charges, format='default'):
    def write_enc(fileobj, s):
        """Write string in latin-1 encoding."""
        fileobj.write(s.encode("latin-1"))

    from ase.utils import basestring
    from ase.parallel import paropen
    # from ase.io import cif

    """Write *images* to CIF file."""
    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'wb')

    if hasattr(images, 'get_positions'):
        images = [images]

    for i, atoms in enumerate(images):
        write_enc(fileobj, 'data_image%d\n' % i)

        a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

        if format == 'mp':

            comp_name = atoms.get_chemical_formula(mode='reduce')
            sf = split_chem_form(comp_name)
            formula_sum = ''
            ii = 0
            while ii < len(sf):
                formula_sum = formula_sum + ' ' + sf[ii] + sf[ii + 1]
                ii = ii + 2

            formula_sum = str(formula_sum)
            write_enc(fileobj, '_chemical_formula_structural       %s\n' %
                      atoms.get_chemical_formula(mode='reduce'))
            write_enc(fileobj, '_chemical_formula_sum      "%s"\n' %
                      formula_sum)

        # Do this only if there's three non-zero lattice vectors
        if atoms.number_of_lattice_vectors == 3:
            write_enc(fileobj, '_cell_length_a       %g\n' % a)
            write_enc(fileobj, '_cell_length_b       %g\n' % b)
            write_enc(fileobj, '_cell_length_c       %g\n' % c)
            write_enc(fileobj, '_cell_angle_alpha    %g\n' % alpha)
            write_enc(fileobj, '_cell_angle_beta     %g\n' % beta)
            write_enc(fileobj, '_cell_angle_gamma    %g\n' % gamma)
            write_enc(fileobj, '\n')

            write_enc(fileobj, '_symmetry_space_group_name_H-M    %s\n' %
                      '"P 1"')
            write_enc(fileobj, '_symmetry_int_tables_number       %d\n' % 1)
            write_enc(fileobj, '\n')

            write_enc(fileobj, 'loop_\n')
            write_enc(fileobj, '  _symmetry_equiv_pos_as_xyz\n')
            write_enc(fileobj, "  'x, y, z'\n")
            write_enc(fileobj, '\n')

        write_enc(fileobj, 'loop_\n')

        # Is it a periodic system?
        coord_type = 'fract' if atoms.pbc.all() else 'Cartn'

        if format == 'mp':
            write_enc(fileobj, '  _atom_site_type_symbol\n')
            write_enc(fileobj, '  _atom_site_label\n')
            write_enc(fileobj, '  _atom_site_symmetry_multiplicity\n')
            write_enc(fileobj, '  _atom_site_{0}_x\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_y\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_z\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_occupancy\n')
        else:
            write_enc(fileobj, '  _atom_site_label\n')
            write_enc(fileobj, '  _atom_site_occupancy\n')
            write_enc(fileobj, '  _atom_site_{0}_x\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_y\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_{0}_z\n'.format(coord_type))
            write_enc(fileobj, '  _atom_site_thermal_displace_type\n')
            write_enc(fileobj, '  _atom_site_B_iso_or_equiv\n')
            write_enc(fileobj, '  _atom_site_type_symbol\n')
            write_enc(fileobj, '  _atom_site_charge\n')

        if coord_type == 'fract':
            coords = atoms.get_scaled_positions().tolist()
        else:
            coords = atoms.get_positions().tolist()
        symbols = atoms.get_chemical_symbols()
        occupancies = [1 for i in range(len(symbols))]

        # try to fetch occupancies // rely on the tag - occupancy mapping
        try:
            occ_info = atoms.info['occupancy']

            for i, tag in enumerate(atoms.get_tags()):
                occupancies[i] = occ_info[tag][symbols[i]]
                # extend the positions array in case of mixed occupancy
                for sym, occ in occ_info[tag].items():
                    if sym != symbols[i]:
                        symbols.append(sym)
                        coords.append(coords[i])
                        occupancies.append(occ)
        except KeyError:
            pass

        no = {}

        for symbol, pos, occ, charge in zip(symbols, coords, occupancies, charges):
            if symbol in no:
                no[symbol] += 1
            else:
                no[symbol] = 1
            if format == 'mp':
                write_enc(fileobj,
                          '  %-2s  %4s  %4s  %7.5f  %7.5f  %7.5f  %6.1f %6.1f\n' %
                          (symbol, symbol + str(no[symbol]), 1,
                           pos[0], pos[1], pos[2], occ, charge))
            else:
                write_enc(fileobj,
                          '  %-8s %6.4f %7.5f  %7.5f  %7.5f  %4s  %6.3f  %s  %6.6f\n'
                          % ('%s%d' % (symbol, no[symbol]),
                             occ,
                             pos[0],
                             pos[1],
                             pos[2],
                             'Biso',
                             1.0,
                             symbol, charge))
    return None

def get_training_charges(fname):
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    def iscoord(line):
        if len(line) >= 6:
            if False not in map(isfloat, line[2:5]):
                return True
            else:
                return False
        else:
            return False

    coords = []
    types = []
    details = []
    charge_types = ['_atom_site_charge']
    charges = [[] for i in range(len(charge_types))]
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if 'atom' in line:
                types.append(line.strip())
            else:
                line = line.split()
                if iscoord(line):
                    coords.append(line)
        for i, typ in enumerate(charge_types):
            try:
                index = types.index(typ)
            except ValueError:
                index = None
            if index != None:
                for coord in coords:
                    charges[i].append(round(float(coord[index]), 6))
            else:
                for coord in coords:
                    charges[i].append(None)

    return charges

def get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=False, adjust_charge_method='mean', print_features=False, net_charge=0):
    """
    Get charges of one or many CIF files.
    Parameters:
        path_to_cif (str): Path to a CIF file/directory. If multiple_cifs == True, use glob to get all the CIF file in path_to_cif.
        output_path (str): Path to write output
        identifier (str): Identifier to name the new file
        multiple_cifs (bool): Specify whether multiple CIFs are in path_to_cif. If yes, only load the model once to save time.
        adjust_charge_method (str): Specify the method to adjust charge. Two options: 'mean' and 'magnitude'.
        net_charge (float or dictionary). If net_charge == 0, only PACMOF2_neutral model is used. Else, if multiple_cifs == False, then specify the net charge as float. If multiple_cifs == True, then specify a dictionary of net_charge correspending for each CIF, i.e. {'MOF1': -8, 'MOF2': -2}. 
    """ 
    model_name = 'PACMOF2_neutral.gz'
    ionic_model = 'PACMOF2_ionic.gz'
    # Adjust charge from model prediction to achieve charge neutrality.
    def adjust_charge(charges, by='mean', net_charge=0):
        if by == 'magnitude':
            return charges - np.sum(charges) * np.abs(charges) / np.sum(np.abs(charges))
        elif by == 'mean':
            return charges - (np.sum(charges) - net_charge) /len(charges)
        else:
            return charges
    
    # Neutral model
    model_file = (impresources.files(models) / model_name)

    # Ionic model
    ionic_file = (impresources.files(models) / ionic_model)

    if multiple_cifs == True:
        import glob
        cifs = sorted(glob.glob(os.path.join(path_to_cif, '*.cif' )))
        print("Loading Models {} ...".format(model_name))

        # Load neutral model
        model = joblib.load(model_file)

        if net_charge == 0:
            for cif in tqdm(cifs, desc="Number of CIFs"):
                print('Getting features from CIF {}'.format(cif))
                atoms = get_features_cif(cif)
                if atoms == 1:
                    print("Cannot featurize {}".format(path_to_cif))
                    continue
                else: 
                    features = atoms.info['features']
                    # Predict neutral charge
                    charges = model.predict(features)
                    charges = adjust_charge(charges, by=adjust_charge_method)
                    # Write output
                    cif_path = os.path.abspath(cif)
                    cif_path = os.path.basename(cif_path)
                    old_name = cif_path.split('/')[-1][:-4]
                    new_name  = old_name + identifier + '.cif'
                    path_to_output_cif = os.path.join(os.path.abspath(output_path), new_name)
                    print("Writing CIF {}".format(new_name))
                    write_cif(path_to_output_cif, atoms, charges)

                    if print_features == True:
                        with open('features_neutral.csv', 'a') as f:
                            for at in features:
                                at = [str(i) for i in at]
                                f.write(','.join(at) + '\n')
        else:
            # Load ionic model
            ionic_model = joblib.load(ionic_file)
            for cif in tqdm(cifs, desc="Number of CIFs"):
                cif_path = os.path.abspath(cif)
                cif_path = os.path.basename(cif_path)
                old_cif = cif_path.split('/')[-1]

                print('Getting features from CIF {}'.format(cif))
                atoms = get_features_cif(cif)
                if atoms == 1:
                    print("Cannot featurize {}".format(path_to_cif))
                    continue
                else:
                    features = atoms.info['features']
                    # Predict neutral charge
                    charges = model.predict(features)
                    
                    # Get net charge for each CIF from dictionary input
                    nc = net_charge[old_cif]
                    
                    # Handle cases that mixed ionic and neutral models are used.
                    # Neutral model
                    if nc == 0:
                        charges = adjust_charge(charges, by=adjust_charge_method)
                        # Write output
                        old_name = cif_path.split('/')[-1][:-4]
                        new_name  = old_name + identifier + '.cif'
                        path_to_output_cif = os.path.join(os.path.abspath(output_path), new_name)
                        print("Writing CIF {}".format(new_name))
                        write_cif(path_to_output_cif, atoms, charges)

                        if print_features == True:
                            with open('features_neutral.csv', 'a') as f:
                                for at in features:
                                    at = [str(i) for i in at]
                                    f.write(','.join(at) + '\n')
                    # Ionic model
                    else:
                        natoms = len(atoms)
                        net_charge_per_atoms = nc / natoms
                        for idx, at_f in enumerate(features):
                            # Add PACMOF charge to features for ionic model
                            at_f.append(round(charges[idx], 4))
                            # Add net charge per unit cell for ionic model
                            at_f.append(round(net_charge_per_atoms, 4))

                        charge_diff = ionic_model.predict(features)
                        ionic_charges = charge_diff + charges

                        print(ionic_charges)
                        print('Net charge before correction: {}'.format(np.sum(ionic_charges)))
                        ionic_charges = adjust_charge(ionic_charges, by=adjust_charge_method, net_charge=nc)

                        print(ionic_charges)
                        print('Net charge after correction: {}'.format(np.sum(ionic_charges)))

                        old_name = cif_path.split('/')[-1][:-4]
                        new_name  = old_name + identifier + '.cif'
                        path_to_output_cif = os.path.join(os.path.abspath(output_path), new_name)

                        print("Writing CIF {}".format(new_name))
                        write_cif(path_to_output_cif, atoms, ionic_charges)

                        if print_features == True:
                            with open('features_ionic.csv', 'a') as f:
                                for at in features:
                                    at = [str(i) for i in at]
                                    f.write(','.join(at) + '\n')

    else:        
        print('Getting features from CIF')
        atoms = get_features_cif(path_to_cif)
        if atoms == 1:
            print("Cannot featurize {}".format(path_to_cif))
        else:
            features = atoms.info['features']
            model = joblib.load(model_file)
            charges = model.predict(features)

            if print_features == True:
                with open('features_neutral.csv', 'a') as f:
                    for at in features:
                        at = [str(i) for i in at]
                        f.write(','.join(at) + '\n')

            if net_charge == 0:
                charges = adjust_charge(charges, by=adjust_charge_method)
                #print("Sum of Charges: {}".format(sum(charges)))
                cif_path = os.path.abspath(path_to_cif)
                cif_path = os.path.basename(cif_path)
                old_name = cif_path.split('/')[-1][:-4]
                new_name  = old_name + identifier + '.cif'
                path_to_output_cif = os.path.join(os.path.abspath(output_path), new_name)
                print("Writing CIF {}".format(new_name))
                write_cif(path_to_output_cif, atoms, charges)
            else:
                natoms = len(atoms)
                net_charge_per_atoms = net_charge / natoms
                for idx, at_f in enumerate(features):
                    # Add PACMOF charge to features for ionic model
                    at_f.append(round(charges[idx], 4))
                    # Add net charge per unit cell for ionic model
                    at_f.append(round(net_charge_per_atoms, 4))

                ionic_file = (impresources.files(models) / ionic_model)
                ionic_model = joblib.load(ionic_file)
                charge_diff = ionic_model.predict(features)
                ionic_charges = charge_diff + charges

                print('Net charge before correction: {}'.format(np.sum(ionic_charges)))
                ionic_charges = adjust_charge(ionic_charges, by=adjust_charge_method, net_charge=net_charge)

                print('Net charge after correction: {}'.format(np.sum(ionic_charges)))

                cif_path = os.path.abspath(path_to_cif)
                cif_path = os.path.basename(cif_path)
                old_name = cif_path.split('/')[-1][:-4]

                new_name  = old_name + identifier + '.cif'
                path_to_output_cif = os.path.join(os.path.abspath(output_path), new_name)
                print("Writing CIF {}".format(new_name))
                write_cif(path_to_output_cif, atoms, ionic_charges)

                if print_features == True:
                    with open('features_ionic.csv', 'a') as f:
                        for at in features:
                            at = [str(i) for i in at]
                            f.write(','.join(at) + '\n')
