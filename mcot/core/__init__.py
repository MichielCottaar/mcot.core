"""Package for mcot.core."""
from . import surface, cifti, greyordinate, log, scripts, spherical
from .write_gifti import write_gifti
scripts.directories.add(f'{__name__}._scripts', group=None)

