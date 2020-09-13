"""Package for mcot.core."""
from . import surface, cifti, greyordinate, log, scripts, spherical, _write_gifti
from ._write_gifti import write_gifti
__version__ = '0.2.3'
scripts.directories.add(f'{__name__}._scripts', group=None)

