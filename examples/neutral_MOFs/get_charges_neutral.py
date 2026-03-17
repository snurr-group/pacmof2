from pacmof2 import get_charges

path_to_cif = "ddec"
output_path = "pacmof"

# 1. Single CIF
path_to_cif = "ddec/LASYOU_clean_DDEC.cif"
get_charges(path_to_cif, output_path, identifier="_pacmof")

# 2. Multiple CIFs
path_to_cif = "ddec"
get_charges(path_to_cif, output_path, identifier="_pacmof", multiple_cifs=True)
