"""Package for mcot.core."""
from ._write_gifti import write_gifti
from . import surface, cifti, greyordinate, log, scripts, spherical
__version__ = '0.2.1'
scripts.directories.add(f'{__name__}._scripts', group=None)

