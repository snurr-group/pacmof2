# PACMOF2: Predicting Partial Atomic Charges in Metal-Organic Frameworks: An Extension to Ionic MOFs

## Overview
PACMOF2 is a Python package designed to predict partial atomic charges in Metal-Organic Frameworks (MOFs) with Density Functional Theory (DFT) level accuracy. It includes two pre-trained machine learning models: `PACMOF2_neutral` for neutral MOFs and `PACMOF2_ionic` for ionic MOFs. Detailed methods and implementation can be found in our upcoming publication.
Associated data (models, DDEC6 data, PACMOF2 prediction data) for the project is available on Zenodo: https://zenodo.org/records/12747095

## Installation
PACMOF2 has been tested with Python 3.9 and requires the following dependencies. Newer versions of these dependencies may work as well, but we did not test them.

- Pymatgen (2023.10.4)
- Atomic Simulation Environment (ASE) (3.22.1)
- Scikit-Learn (1.3.2)
- huggingface-hub

First, clone the repository:
```bash
git clone https://github.com/tdpham2/pacmof2
```

### Using Anaconda

```bash
conda create -n pacmof2 python==3.9
conda activate pacmof2
conda install -c conda-forge pymatgen=2023.10.4
conda install -c conda-forge ase=3.22.1
conda install -c conda-forge scikit-learn=1.3.2
pip install huggingface-hub
pip install build
```

### Using Pip
Alternatively, install dependencies via pip:

```bash
pip install -r requirements.txt
```

### Installing PACMOF2
After setting up the dependencies, install PACMOF2:

```bash
pip install -e .
```

### Downloading the Models
PACMOF2 models are available on Hugging Face and Zenodo. The package downloads
the Hugging Face model files automatically the first time predictions are run,
then reuses the local Hugging Face cache on later runs.

To prefetch the models during setup, run:

```bash
pacmof2-download-models
```

For a custom Hugging Face cache location, set `PACMOF2_HF_CACHE_DIR` so both
prefetching and later predictions use the same cache:

```bash
export PACMOF2_HF_CACHE_DIR=/path/to/cache
pacmof2-download-models
```

For offline or manually managed installs, place both files in a directory and
set `PACMOF2_MODEL_DIR` to that directory:

```bash
wget -P /path/to/pacmof2-models/ https://huggingface.co/tdphamm/PACMOF2/resolve/main/PACMOF2_ionic.gz
wget -P /path/to/pacmof2-models/ https://huggingface.co/tdphamm/PACMOF2/resolve/main/PACMOF2_neutral.gz
export PACMOF2_MODEL_DIR=/path/to/pacmof2-models
```

## Usage
PACMOF2 can predict partial atomic charges for both neutral and ionic MOFs. It can be used either as a command-line tool or as a Python library. Example scripts and CIF files are available in the `examples/` directory.

### Command-Line Interface

After installation, the `pacmof2` command is available:

```bash
# Single neutral MOF
pacmof2 path/to/file.cif -o output_dir/

# Multiple neutral MOFs in a directory
pacmof2 path/to/cifs/ -o output_dir/ --multiple

# Single ionic MOF with known net charge
pacmof2 path/to/file.cif -o output_dir/ --net-charge -2

# Multiple ionic MOFs with net charges from a JSON file
pacmof2 path/to/cifs/ -o output_dir/ --multiple --net-charges net_charges.json

# Show all options
pacmof2 --help
```

### Python API

#### Predicting Charges for Neutral MOFs

```python
from pacmof2 import get_charges

# Single CIF
get_charges('path/to/file.cif', 'output_dir/', identifier="_pacmof")

# Multiple CIFs in a folder
get_charges('path/to/cifs/', 'output_dir/', identifier='_pacmof', multiple_cifs=True)
```

#### Predicting Charges for Ionic MOFs

```python
from pacmof2 import get_charges

# Single ionic MOF
get_charges('path/to/file.cif', 'output_dir/', identifier='_pacmof', net_charge=-2)
```

For multiple ionic MOFs with net charges specified in a JSON file:

```python
from pacmof2 import get_charges
import json

with open('net_charges.json', 'r') as f:
    net_charges = json.load(f)

get_charges('path/to/cifs/', 'output_dir/', identifier='_pacmof', multiple_cifs=True, net_charge=net_charges)
```

## Reference
Our work is available on JPCC: https://pubs.acs.org/doi/10.1021/acs.jpcc.4c04879#
