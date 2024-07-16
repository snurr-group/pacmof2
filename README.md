# PACMOF2: Predicting Partial Atomic Charges in Metal-Organic Frameworks

## Overview
PACMOF2 is a Python package designed to predict partial atomic charges in Metal-Organic Frameworks (MOFs) with Density Functional Theory (DFT) level accuracy. It includes two pre-trained machine learning models: `PACMOF2_neutral` for neutral MOFs and `PACMOF2_ionic` for ionic MOFs. Detailed methods and implementation can be found in our upcoming publication.
Associated data (models, DDEC6 data, PACMOF2 prediction data) for the project is available on Zenodo: https://zenodo.org/records/12747095

## Installation
PACMOF2 has been tested with Python 3.9 and requires the following dependencies:

- Pymatgen (2023.10.4)
- Atomic Simulation Environment (ASE) (3.22.1)
- Scikit-Learn (1.3.2)

First, clone the repository:
```bash
git clone https://github.com/tdpham2/pacmof2
```

### Using Anaconda

```bash
conda create -n pacmof2 python==3.9
conda activate pacmof2
conda install -c conda-forge pymatgen
conda install -c conda-forge ase=3.22.1
conda install -c conda-forge scikit-learn=1.3.2
pip install build
```

### Using Pip
Alternatively, install dependencies via pip:

```bash
pip install -r requirements.txt
```

### Downloading the Models
Due to file size limitations on GitHub, the PACMOF2 models are stored on Zenodo. Download the models and place them in the `pacmof2/models/` directory:

- [PACMOF2_ionic.gz](https://zenodo.org/records/12747095/files/PACMOF2_ionic.gz)
- [PACMOF2_neutral.gz](https://zenodo.org/records/12747095/files/PACMOF2_neutral.gz)

Or use `wget` to download the models:

```bash
wget -P pacmof2/models/ https://zenodo.org/records/12747095/files/PACMOF2_ionic.gz
wget -P pacmof2/models/ https://zenodo.org/records/12747095/files/PACMOF2_neutral.gz
```

### Installing PACMOF2
After setting up the dependencies and downloading the models, install PACMOF2:

```bash
python3 -m build
pip install .
```

## Usage
PACMOF2 can predict partial atomic charges for both neutral and ionic MOFs.

### Predicting Charges for Neutral MOFs
To predict charges for a single neutral MOF:

```python
from pacmof2 import pacmof2

path_to_cif = 'path/to/cif'
output_path = 'pacmof'
pacmof2.get_charges(path_to_cif, output_path, identifier="_pacmof")
```

To predict charges for multiple neutral MOFs in a folder:

```python
from pacmof2 import pacmof2

path_to_cif = 'path/to/cifs/folder/'
output_path = 'pacmof'
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True)
```

### Predicting Charges for Ionic MOFs
For a single ionic MOF:

```python
from pacmof2 import pacmof2

path_to_cif = 'path/to/cif'
output_path = 'pacmof'
pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', net_charge=-2)
```

For multiple ionic MOFs with net charges specified in a JSON file:

```python
from pacmof2 import pacmof2
import json

path_to_cif = 'path/to/cifs/folder'
output_path = 'pacmof'
with open('net_charges.json', 'r') as f:
    net_charges = json.load(f)

pacmof2.get_charges(path_to_cif, output_path, identifier='_pacmof', multiple_cifs=True, net_charge=net_charges)
```

## Reference
Details on the method and implementation will be updated soon.
