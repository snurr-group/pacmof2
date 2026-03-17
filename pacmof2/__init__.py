"""
PACMOF2: Predicting Partial Atomic Charges in Metal-Organic Frameworks.

An extension to ionic MOFs using pre-trained machine learning models.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pacmof2")
except PackageNotFoundError:
    __version__ = "1.0.0"

from pacmof2.pacmof2 import get_charges

__all__ = ["get_charges", "__version__"]
