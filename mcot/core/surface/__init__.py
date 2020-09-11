"""Utilities for working with cortical meshes and defining the gyral coordinate system

Contains:

- :py:mod:`cortex`: classes to work with multiple cortical mesh spanning a cortical layer
- :py:mod:`grid`: many functions that require a grid and a cortical mesh
- :py:mod:`mesh`: basis class (`Mesh2D`) to deal with a mesh
- :py:mod:`cortical_mesh`: extends `Mesh2D` into `CorticalMesh`, which contains metadata with the anatomy described by the mesh
- :py:mod:`orientation`: computes the gyral coordinate system
- :py:mod:`radial_transition`: quantifies the transition boundary from tangential orientations in the white matter to radial orientations in the cortex
- :py:mod:`utils` few utility functions
"""
from .cortical_mesh import CorticalMesh, BrainStructure
from .mesh import Mesh2D
from .cortex import Cortex, CorticalLayer, read_HCP
from . import grid
