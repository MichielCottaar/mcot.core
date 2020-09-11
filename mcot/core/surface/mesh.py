"""Defines N-1 dimensional surfaces in N-dimensional space.

All surfaces are represented by a Mesh with points and connections (i.e. line segments or triangles) between those points.
"""
import numpy as np
from scipy import sparse, linalg
from nibabel import freesurfer, spatialimages, gifti
import nibabel as nib
from operator import xor
import datetime
import tempfile
import numba
from scipy import spatial, optimize
from six import string_types
from pathlib import Path
from .utils import signed_tetrahedral_volume
from copy import deepcopy
from loguru import logger


class Mesh(object):
    """General mesh object.

    Defines methods that are independent of the number of dimensions.
    """
    vertices = None
    faces = None
    _graph = None
    _normal = None
    _tree = None

    @property
    def nvertices(self, ):
        """
        Number of vertices on the mesh
        """
        return self.vertices.shape[1]

    @property
    def ndim(self, ):
        """
        Dimensionality of the embedding space
        """
        return self.vertices.shape[0]

    @property
    def nfaces(self, ):
        """
        Number of surface elements connecting the vertices.
        """
        return self.faces.shape[1]

    def graph_connection_point(self, dtype='bool'):
        """
        Returns the interactions between vertices and faces as a sparse matrix.

        The resulting matrix can be used to multiply a vector of size M faces to get a vector of size N vertices.

        The result of this method is cached in _graph (set _graph to None to re-compute the graph).

        :param dtype: data type of the resulting sparse matrix
        :return: (N, M) sparse matrix for N vertices and M faces, which is one if connection M interacts with N.
        """
        if self._graph is not None:
            return self._graph.astype(dtype)
        rows = self.faces.flatten()
        cols = (np.ones(self.faces.shape[0])[:, None] * np.arange(self.nfaces)[None, :]).flatten().astype('i4')
        data = np.ones(rows.size, dtype='bool')
        res = sparse.coo_matrix((data, (rows, cols)),
                                shape=(self.nvertices, self.nfaces)).tocsr()
        self._graph = res
        return res.astype(dtype)

    def graph_point_point(self, weight=None, dtype='bool', include_diagonal=True):
        """
        Converts the mesh into a graph describing the edges between the individual vertices (nodes).

        :param weight: Weights the boundaries by the distance between the vertices if set to "distance"
        :param dtype: datatype of the resulting sparse matrix (only used if `weight` is None)
        :param include_diagonal: if set to False exclude the diagonal from the sparse matrix
        :return: (N, N) sparse matrix for N vertices, which is one (or the value set by `weight`) if the vertices are connected.
        """
        pc_graph = self.graph_connection_point(dtype=dtype)
        pp_graph = pc_graph * pc_graph.T

        if not include_diagonal:
            pp_graph.setdiag(False)
            pp_graph.eliminate_zeros()

        if weight is not None:
            graph_as_coo = pp_graph.tocoo()
            if weight == 'distance':
                weight = np.sqrt(np.sum((self.vertices[:, graph_as_coo.row] - self.vertices[:, graph_as_coo.col]) ** 2, 0))
            graph_as_coo.data = weight * np.ones_like(graph_as_coo.data)
            pp_graph = graph_as_coo.tocsr()
        return pp_graph

    def graph_connection_connection(self, weight=None, dtype='bool'):
        """
        Converts the mesh into a graph, where the nodes are the faces and the edges are between those faces sharing vertices.

        :param weight: Weights the boundaries by the distance between the connection centers if set to "distance"
        :param dtype: datatype of the resulting sparse matrix (only used if `weight` is None)
        :return: (N, N) sparse matrix for N faces, which is one (or the value set by `weight`) if the faces share a vertex.
        """
        pc_graph = self.graph_connection_point(dtype=dtype)
        cc_graph = pc_graph.T * pc_graph
        if weight is not None:
            graph_as_coo = cc_graph.tocoo()
            if weight == 'distance':
                positions = np.mean(self.vertices[:, self.faces], 1)
                weight = np.sqrt(np.sum((positions[:, graph_as_coo.row] - positions[:, graph_as_coo.col]) ** 2, 0))
            graph_as_coo.data = weight * np.ones_like(graph_as_coo.data)
            cc_graph = graph_as_coo.tocsr()
        return cc_graph

    def surface_edge_distance(self, use=None, method='auto', return_predecessors=False, use_connections=False):
        """
        Returns a matrix of the shortest distances across the edges connecting the vertices.

        This is an upper limit to the true distance across the surface,
        because the path is limited to following the edges of the triangular mesh.

        This is a wrapper around `scipy.sparse.csgraph.shortest_path`.

        :param use: boolean array indicating which vertices or faces to use (default: use all)
        :param method: method used by `scipy.sparse.csgraph.shortest_path`.
        :param return_predecessors: whether to return the (N, N) predecessor matrix
        :param use_connections: compute the shortest distance between the faces rather than the vertices.
        :return: (N, N) matrix of shortest distances across the graph
        """
        if use_connections:
            graph = self.graph_connection_connection(weight="distance")
        else:
            graph = self.graph_point_point(weight="distance")
        if use is not None:
            graph = graph[use, :][:, use]
            nclusters, labels = sparse.csgraph.connected_components(graph, directed=False)
            distance = []
            for ixcluster in range(np.amax(labels) + 1):
                use = labels == ixcluster
                distance.append(sparse.csgraph.shortest_path(graph[use, :][:, use], method=method,
                                                             return_predecessors=return_predecessors))
            return labels, distance
        return sparse.csgraph.shortest_path(graph, method=method, return_predecessors=return_predecessors)

    def size_vertices(self, ):
        """
        Attributes the size of the faces to the vertices they connect.
        """
        return self.graph_connection_point() * self.size_faces() / self.faces.shape[0]

    def connected_components(self, ):
        """
        Returns a tuple with (number of connected components, labeling of connected components).
        """
        return sparse.csgraph.connected_components(self.graph_point_point())

    def closed(self, ):
        """
        Checks if the mesh is closed.
        """
        raise NotImplementedError("No generic implementation for N-dimensional mesh")

    @property
    def tree(self, ):
        """
        A KD tree used to compute the distance between the vertices defining the surface and any other vertices

        :rtype: scipy.spatial.cKDTree
        """
        if self._tree is None:
            self._tree = spatial.cKDTree(self.vertices.T)
        return self._tree

    def closest_vertex(self, points):
        """
        Finds the closest vertices on the surface for a bunch of vertices

        :param points: (ndim, nvertices) array with the reference vertices
        :return: tuple with

            - (nvertices, ) distance array
            - (nvertices, ) index array
        """
        return self.tree.query(points.T)


class Mesh1D(Mesh):
    """
    1-dimensional mesh object consisting of vertices and lines connecting these vertices

    Attributes:
    `vertices`: (M, N) array with the vertices of the curve in M-dimensional space.
    `faces`: (2, K) index array with all the line segments.
    """
    def __init__(self, vertices, faces='open'):
        """
        Creates a new curve

        :param vertices: (M, N) array with N vertices on a one-dimensional curve in M-dimensional space
        :param faces: (2, K) array with integers of which lines are connected
        If faces is:
        - 'open': defaults to connecting all vertices in order
        - 'closed': defaults to connecting all vertices in order and connect the last point to the first
        """
        self.vertices = np.asarray(vertices)
        if faces == 'open':
            faces = np.array([np.arange(self.vertices.shape[1] - 1), np.arange(1, self.vertices.shape[1])])
        elif faces == 'closed':
            faces = np.array([np.arange(self.vertices.shape[1]), np.roll(np.arange(self.vertices.shape[1]), -1)])
        self.faces = faces
        if self.ndim > self.nvertices + 3:
            raise ValueError('N(dimensions) >> N(vertices), you should probably transpose the vertices array')
        if self.faces.shape[0] != 2:
            raise ValueError('1D-mesh faces should have shape (2, K), not %s' % self.faces.shape)
        if self.vertices.ndim != 2 or self.faces.ndim != 2:
            raise ValueError('vertices and faces should be 2-dimensional')

    def size_faces(self, ):
        """
        Computes the length of the line segments connecting the vertices.
        """
        return np.sum((self.vertices[:, self.faces[0, :]] - self.vertices[:, self.faces[1, :]]) ** 2, 0)

    def as_lines(self, as_indices=False):
        """
        Return the connected vertices as a list of curves.

        :param as_indices: Returns the indices of the vertices rather than the vertices themselves
        :return: List[Array], where the array is a (L, ) array of indices if as_indices is True or (L, 2) array of vertices otherwise
        """
        lines = [[ixpoint] for ixpoint in np.arange(self.nvertices)]
        for connection in self.faces.T:
            start = None
            end = None
            for ixline, line in enumerate(lines):
                if connection[1] == line[0]:
                    end = ixline
                if connection[0] == line[-1]:
                    start = ixline
            if start != end:
                lines[start].extend(lines[end])
                lines.pop(end)
        if as_indices:
            return [np.array(line) for line in lines]
        return [self.vertices[:, np.array(line)] for line in lines]

    def closed(self, ):
        """
        Check if the mesh is closed (i.e. every vertex has zero or at least two faces).
        """
        nconn = np.sum(self.graph_connection_point(), -1)
        return (nconn != 1).all()

    def find_intersections(self, position, orientation, return_position=False):
        """
        Finds out which faces intersection with position + a * hemisphere.

        :param position: origin of the ray
        :param orientation: propagation direction of the ray
        :param return_position: if True also return the coordinates of the intersection
        :return: (K, ) boolean array with the intercepted faces
        """
        offset = self.vertices - position[:, None]
        outer_product = offset[0, ...] * orientation[1] - offset[1, ...] * orientation[0]
        intercepts = np.prod(np.sign(outer_product[self.faces]), 0) < 0
        if not return_position:
            return intercepts
        use_offsets = offset[:, self.faces][:, :, intercepts]
        if orientation[0] == 0:
            result = -use_offsets[0, 0, :] / (use_offsets[0, 1, :] - use_offsets[0, 0, :])
        else:
            nominator = use_offsets[1, 0, :] - use_offsets[0, 0, :] * orientation[1] / orientation[0]
            denominator = -(use_offsets[1, 1, :] - use_offsets[1, 0, :]) + (use_offsets[0, 1, :] - use_offsets[0, 0, :]) * orientation[1] / orientation[0]
            result = nominator / denominator
        use_points = self.vertices[:, self.faces][:, :, intercepts]
        return intercepts, result[None, :] * (use_points[:, 1, :] - use_points[:, 0, :]) + use_points[:, 0, :]

    def normal(self, ):
        """Calculates the normal of every face.

        The result of this method is cached in _normal (set to None to re-compute the normals).

        :return (N, 2): for N faces in 2-dimensional space
        """
        if self._normal is not None:
            return self._normal
        if self.ndim != 2:
            raise ValueError("Normal of 1-dimensional mesh only defined in 2D")
        normal = (self.vertices[:, self.faces[1, :]] - self.vertices[:, self.faces[0, :]])[::-1, :]
        normal[0, ...] *= -1
        normal = normal / np.sqrt(np.sum(normal ** 2, 0))[None, ...]
        self._normal = normal
        return normal


class Mesh2D(Mesh):
    """
    Triangular mesh object describing a 2-dimensional surface in M-dimensional space
    """
    def __init__(self, vertices, faces, flip_normal=False):
        """
        Defines a triangular mesh in M-dimensional space

        :param vertices: (M, N) array with the vertices of the curve in M-dimensional space.
        :param faces: (3, K) index array with all the faces.
        :param flip_normal: flips the normal when it is computed (used by `Mesh2D.apply_affine`, do not set this)
        """
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)
        self.flip_normal = flip_normal
        if self.vertices.ndim != 2 or self.faces.ndim != 2:
            raise ValueError('vertices and faces should be 2-dimensional')
        if self.ndim > self.nvertices + 3:
            raise ValueError('N(dimensions) >> N(vertices), you should probably transpose the vertices array')
        if self.faces.shape[0] != 3:
            raise ValueError('2D-mesh faces should have shape (3, K), not %s' % (self.faces.shape, ))

    def size_faces(self, ):
        """
        Compute the size of the faces in the mesh.

        :return: (3, K) array for K faces
        """
        offset = self.vertices[:, self.faces[1: :]] - self.vertices[:, self.faces[0, :]][:, None, :]
        offset_size = np.sqrt(np.sum(offset ** 2, 0))
        along_second = np.sum(offset[:, 0, :] * offset[:, 1, :], 0) / offset_size[1]
        height = np.sqrt(offset_size[0] ** 2 - along_second ** 2)
        return 0.5 * height * offset_size[1]

    @classmethod
    def read(cls, gifti_filename):
        """
        Reads a surface from a surface gifti file (i.e. ending with .surf.gii).

        If you want to store the cortical information with the cortical mesh use cortical_mesh.CorticalMesh.read

        :param gifti_filename: input filename
        """
        connections = None
        points = None
        try:
            gifti_obj = nib.load(str(gifti_filename)) if isinstance(gifti_filename, string_types + (Path, )) else gifti_filename
        except spatialimages.ImageFileError:
            points, connections = freesurfer.read_geometry(gifti_filename)
            return cls(points.T, connections.T)
        for darray in gifti_obj.darrays:
            codes = gifti.gifti.intent_codes.code
            if darray.intent == codes['pointset']:
                if points is not None:
                    raise ValueError('multiple arrays with intent "%s" found in %s' % (darray.intent, gifti_filename))
                points = darray.data
            elif darray.intent == codes['triangle']:
                if connections is not None:
                    raise ValueError('multiple arrays with intent "%s" found in %s' % (darray.intent, gifti_filename))
                connections = darray.data
        if points is None:
            raise ValueError("no array with intent 'pointset' found in %s" % gifti_filename)
        if connections is None:
            raise ValueError("no array with intent 'triangle' found in %s" % gifti_filename)
        return cls(points.T, connections.T)

    def write(self, gifti_filename, scalar_arr=None, **kwargs):
        """
        Writes a surface to a surface gifti file.

        :param gifti_filename: output filename
        :param scalar_arr: optionally include a scalar array with same length as number of vertices (as expected by FSL's probtrackX)
        :param kwargs: any keywords are added to the meta information in the GIFTI file
        """
        from . import cortical_mesh
        use_kwargs = {'Date': str(datetime.datetime.now()),
                      'encoding': 'XML',
                      'GeometricType': 'Anatomical'}
        use_kwargs.update(
                cortical_mesh.BrainStructure('Other').gifti
        )
        use_kwargs.update(kwargs)
        meta = gifti.GiftiMetaData.from_dict(use_kwargs)
        img = gifti.GiftiImage(meta=meta)
        for arr, intent, dtype in zip([self.vertices, self.faces], ['pointset', 'triangle'], ['f4', 'i4']):
            img.add_gifti_data_array(gifti.GiftiDataArray(arr.T.astype(dtype), intent, meta=meta.metadata))
        if scalar_arr is not None:
            img.add_gifti_data_array(gifti.GiftiDataArray(scalar_arr.astype('f4'), intent='shape', meta=meta.metadata))
        for da in img.darrays:
            da.encoding = 2  # Base64Binary
        nib.save(img, gifti_filename)

    def plane_projection(self, position=(0, 0, 0), orientation=(0, 0, 1)):
        """
        Returns list of ProjectedMesh of the surface projected onto a plane.

        :param position: origin of the plane on which the ProjectedMesh will be defined
        :param orientation:  normal of the plane on which the ProjectedMesh will be defined
        :return: Each of the ProjectedMesh describes an isolated intersection
        :rtype: List[ProjectedMesh]
        """
        position = np.asarray(position)
        orientation = np.asarray(orientation)
        assert position.size == 3, "3-dimensional position required"
        assert orientation.size == 3, "3-dimensional hemisphere required"
        norm_orient = orientation / np.sqrt(np.sum(orientation ** 2))

        relative_offset = np.sum((self.vertices.T - position[None, :]) * norm_orient[None, :], 1)

        vertex_above_plane = (relative_offset > 0)[self.faces.T]
        total_above_plane = np.sum(vertex_above_plane, 1)
        lonely_point = -np.ones(total_above_plane.size, dtype='i4')
        lonely_point[total_above_plane == 1] = np.where(vertex_above_plane[total_above_plane == 1, :])[1]
        lonely_point[total_above_plane == 2] = np.where(~vertex_above_plane[total_above_plane == 2, :])[1]
        other_point1 = (lonely_point + 1) % 3
        other_point2 = (lonely_point + 2) % 3
        lines = self.faces[[[lonely_point, other_point1],
                            [lonely_point, other_point2]], np.arange(lonely_point.size)]
        lines[:, :, lonely_point == -1] = -1  # (2 lines, 2 vertices, N vertices)
        lines = np.sort(lines, 1)

        routes = []
        tmp_route = np.zeros((np.sum(lonely_point != -1) * 2 + 1, 3), dtype='i8')
        while (lines != -1).any():
            idx_min, idx_max = _trace_route(lines, tmp_route)
            indices = tmp_route[idx_min:idx_max, :]
            routes.append(ProjectedMesh(self, indices.copy(), position, orientation))
        return routes

    def clean(self, ):
        """
        Returns a clean mesh

        1. Merges duplicate vertices
        2. Remove isolated vertices
        """
        mesh = deepcopy(self)
        keep_vertices = np.ones(mesh.nvertices, dtype='bool')

        while True:
            poss_vertices = mesh.faces[:, mesh.size_faces().argmin()]
            for idx1 in range(3):
                idx2 = (idx1 + 1) % 3
                idx1, idx2 = poss_vertices[idx1], poss_vertices[idx2]
                if np.allclose(self.vertices[:, idx1], self.vertices[:, idx2], rtol=1e-3):
                    logger.info(f'Merging duplicate vertices {idx1} and {idx2}')
                    mesh.vertices[:, idx2] = np.nan
                    mesh.faces[mesh.faces == idx2] = idx1
                    mesh.faces = mesh.faces[:, np.sum(mesh.faces == idx1, 0) <= 1]
                    keep_vertices[idx2] = False
                    break
            else:
                break

        in_faces = np.append(np.append(-1, mesh.faces.flatten()), mesh.nvertices)
        in_faces.sort()
        bad_jumps = (in_faces[1:] - in_faces[:-1]) > 1
        if bad_jumps.any():
            for idx_bad in np.where(bad_jumps)[0]:
                for isolated_vertex in range(in_faces[idx_bad] + 1, in_faces[idx_bad + 1]):
                    logger.info(f'Removing isolated vertex {isolated_vertex}')
                    keep_vertices[isolated_vertex] = False

        return mesh[keep_vertices]

    def closed(self, ):
        """Check if the mesh is closed (i.e. every line connecting two vertices is used in zero or at least 2 faces).

        :rtype: bool
        """
        point_connection = self.graph_connection_point().astype('i4')
        point_point = point_connection * point_connection.T
        return np.sum(point_point == 1) == 0

    def find_intersections(self, position, orientation, return_position=False):
        """
        Finds a ray intersection with the surface

        If many ray intersections are required grid.GridSurfaceIntersection.ray_intersect will be much faster

        :param position: (M, ) array with the starting point of the ray
        :param orientation: (M, ) array with the hemisphere of the ray
        :param return_position: if True returns the position of the intersection in addition to a boolean indicating whether there is one
        :return: boolean indicating whether there is an intersection (as well as the position of the intersection if `return_position` is set to True)
        """
        orientation = np.asarray(orientation)
        position = np.asarray(position)
        base_point = self.vertices[:, self.faces[0, :]]
        edge1 = self.vertices[:, self.faces[1, :]] - base_point
        edge2 = self.vertices[:, self.faces[2, :]] - base_point
        point_normal = np.cross(orientation, edge2, axis=0)
        inv_det = 1. / np.sum(edge1 * point_normal, axis=0)
        offset = position[:, None] - base_point
        intercept1 = np.sum(point_normal * offset, axis=0) * inv_det
        intercept2 = np.sum(np.cross(offset, edge1, axis=0) * orientation[:, None], axis=0) * inv_det
        intercepts = (intercept1 >= 0) & (intercept2 >= 0) & ((intercept1 + intercept2) <= 1)
        if not return_position:
            return intercepts
        position = (self.vertices[:, self.faces[0, intercepts]] + intercept1[intercepts] * edge1[:, intercepts] + intercept2[intercepts] * edge2[:, intercepts])
        return intercepts, position

    def normal(self, atpoint=False):
        """
        Calculates the normal of every connection.

        The result of this method is cached in _normal (set to None to re-compute the normals).

        :param atpoint: interpolates the normals from the vertices to the vertices (as defined by Freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferWiki/SurfaceNormal)
        :return: (Nvertex, 3) (or (Npoint, 3) if `atpoint` is set) array with the normals
        """
        if atpoint:
            face_normal = self.normal()
            face_normal[~np.isfinite(face_normal)] = 0.  # no contribution from faces with zero surface area
            norm = (self.graph_connection_point() * face_normal.T).T
            norm /= np.sqrt(np.sum(norm ** 2, 0))
            return norm
        base_point = self.vertices[:, self.faces[0, :]]
        edge1 = self.vertices[:, self.faces[1, :]] - base_point
        edge2 = self.vertices[:, self.faces[2, :]] - base_point
        unnorm = np.cross(edge1, edge2, axis=0) * (-1 if self.flip_normal else 1)
        norm = unnorm / np.sqrt(np.sum(unnorm ** 2, 0))
        return norm

    def neighbour_faces(self, ):
        """
        Find the neighbouring faces

        :return: (3, N) array with for all N faces the indices of the neighbours (padded with -1 if there are less than 3 neighbours).
        """
        neighbour_graph = (self.graph_connection_connection(dtype='i4') == 2).tocoo()
        arr = -np.ones((3, self.nfaces), dtype='i4')
        rows = neighbour_graph.row.copy()
        for ix in range(3):
            values, ixcol = np.unique(rows, return_index=True)
            arr[ix, values] = neighbour_graph.col[ixcol]
            rows[ixcol] = -1
        assert (rows == -1).all()
        return arr

    def split_mask(self, mask):
        """
        Splits a mask into contiguous surface patches.

        :param mask: (N, ) boolean array which is True for any vertices in the mask
        :return: (N, ) integer array with labels for any point on the mask (-1 for vertices outside of the mask)
        """
        if mask.size == self.nvertices:
            graph = self.graph_point_point()
        elif mask.size == self.nfaces:
            graph = self.graph_connection_connection()
        else:
            raise ValueError("mask size (%i) should either match the number of vertices or the number of faces" % (mask.size, self.nvertices, self.nfaces))
        subgraph = graph[:, mask][mask, :]
        _, labels = sparse.csgraph.connected_components(subgraph)
        res = -np.ones(mask.size, dtype=labels.dtype)
        res[mask] = labels
        return res

    def gradient(self, scalar, atpoint=False):
        """
        Computes the gradient orientations of a scalar across the surface.

        :param scalar: (K, ) array with value of scalar function for every point on the surface.
        :param atpoint: if set interpolate the gradients from the vertices to the vertices.
        :return: (3, N) array with the gradient for all N vertices.
        """
        if atpoint:
            return (self.graph_connection_point() * self.gradient(scalar).T).T / 3
        neighbour_val = scalar[self.faces]
        neighbour_pos = self.vertices[:, self.faces]

        def gradient_line(ix1, ix2):
            offset = neighbour_pos[:, ix1, :] - neighbour_pos[:, ix2, :]
            distance = np.sqrt(np.sum(offset ** 2, 0))
            return offset / distance, (neighbour_val[ix1, :] - neighbour_val[ix2, :]) / distance

        def gradient_estimate(ix):
            v1, g1 = gradient_line(ix, (ix + 1) % 3)
            v2, g2 = gradient_line(ix, (ix + 2) % 3)
            cross = np.sum(v1 * v2, 0)
            cont_v1 = (g1 - g2 * cross) / (1 - cross ** 2)
            cont_v2 = (g2 - g1 * cross) / (1 - cross ** 2)
            return cont_v1 * v1 + cont_v2 * v2
        return np.mean([gradient_estimate(ix) for ix in range(3)], 0)

    def smooth(self, nsteps, smooth_step=0.5, expand_step=None):
        """
        Smoothing algorithm with rough volume preservation

        Implements the algorithm from Rypl and Nerad, "VOLUME CONSERVATION OF 3D SURFACE TRIANGULAR MESH SMOOTHING."
        See https://pdfs.semanticscholar.org/2c88/01e50f5ecf0035e8c2bdca7976a3a5d45ee8.pdf .
        This algorithm iterates between smoothing steps and expansion steps with the expansion step sizes determined
        by the local curvature as to preserve the local volume.

        :param nsteps: number of smoothing steps
        :param smooth_step: How much the smoothing step moves the vertex to the mean of the neighbour (between 0 and 1)
        :param expand_step: How much the expansion step moves the vertex back out (default: enough to roughly preserve volume)
        :return: new smooth mesh
        """
        pp_graph = self.graph_point_point(include_diagonal=False)
        if expand_step is None:
            vf_graph = self.graph_connection_point().tocoo()
        current_mesh = self
        for _ in range(nsteps):
            shrunk_mesh = current_mesh._smooth_step(smooth_step, graph=pp_graph)
            if expand_step == 0:
                current_mesh = shrunk_mesh
                continue
            elif expand_step is None:
                all_normals = shrunk_mesh.normal(atpoint=False).T
                vertex_normals = shrunk_mesh.normal(atpoint=True).T
                inner_prod = 1 - np.einsum('ij,ij->i', all_normals[vf_graph.col, :], vertex_normals[vf_graph.row, :]) ** 2
                avg_inner_prod = np.bincount(vf_graph.row, inner_prod, minlength=self.nvertices) / np.bincount(vf_graph.row, minlength=self.nvertices)
                expand_step_size = - smooth_step / (2 * smooth_step * avg_inner_prod + 1)
                expand_step_size[~np.isfinite(expand_step_size)] = -smooth_step
            else:
                expand_step_size = expand_step
            current_mesh = shrunk_mesh._smooth_step(expand_step_size, graph=pp_graph)
        return current_mesh.clean()

    def _smooth_step(self, step_size, graph=None):
        """
        Applies a smoothing or expansion step

        :param step_size: How much the step moves to the mean of the neighbours (make negative for expansion step)
        :param graph: Optional parameter with the vertex-vertex graph
        """
        if graph is None:
            graph = self.graph_point_point(include_diagonal=False)
        nneigh = graph.dot(np.ones(self.nvertices)).flatten()
        ref_point = graph.dot(self.vertices.T) / nneigh[:, None]
        new_points = step_size * ref_point.T + (1 - step_size) * self.vertices
        return Mesh2D(new_points, self.faces, flip_normal=self.flip_normal)

    def apply_affine(self, affine):
        """
        Returns a new Mesh to which the affine transformation as been applied.

        :param affine: (4, 4) array defining the voxel->mm transformation (i.e. the transformation TOWARDS the space the surface is defined in)
        :return: new Mesh in the origin space of the affine transformation
        :rtype Mesh2D:
        """
        points = np.dot(linalg.inv(affine), np.append(self.vertices, np.ones((1, self.nvertices)), 0))[:-1, :]
        return Mesh2D(points, self.faces, flip_normal=xor(linalg.det(affine) < 0, self.flip_normal))

    def volume(self,):
        """
        Returns the signed volume of the mesh
        """
        vol = np.sum(signed_tetrahedral_volume(*self.vertices.T[self.faces, :]))
        if self.flip_normal:
            return -vol
        return vol

    def inflate(self, shift=None, volume=None):
        """
        Increases with the given shift or volume

        Moves each vertex to get the requested volume decrease/increase

        :param shift: shift of each vertex in mm
        :param volume: increase in volume in mm^3 (can be negative)
        :return: new surface
        """
        if shift is not None and volume is not None:
            raise ValueError("Only shift of volume should be given to inflate")
        if shift is None and volume is None:
            raise ValueError("Either shift or volume should be specified to inflate")
        if shift is None:
            ref_vol = self.volume()
            shift = optimize.minimize_scalar(
                lambda test_shift: (self.inflate(shift=test_shift).volume() - ref_vol - volume) ** 2
            ).x
        return Mesh2D(self.vertices + self.normal(atpoint=True) * shift, self.faces, flip_normal=self.flip_normal)

    def as_temp_file(self, ):
        """
        Returns the filename of a temporary .surf.gii file containing this mesh.

        The user is responsible for deleting this file after use.
        """
        file = tempfile.NamedTemporaryFile(suffix='.surf.gii', delete=False)
        file.close()
        self.write(file.name)
        return file.name

    def to_vtk_polydata(self, color=None):
        """
        Returns a vtk.vtkPolyData object with the mesh.

        :param color: (3, N) or (N, ) array defining the color across the mesh
        :rtype: vtk.vtkPolyData
        """
        import vtk
        import vtk.util.numpy_support as vtknp
        polydata = vtk.vtkPolyData()

        vtkpoints = vtk.vtkPoints()
        vtkpoints.SetData(vtknp.numpy_to_vtk(self.vertices.astype('f4').T.copy(), deep=1))
        polydata.SetPoints(vtkpoints)

        vtkpolys = vtk.vtkCellArray()
        packed = np.concatenate([np.ones(self.nfaces)[:, None] * 3, self.faces.T], 1).flatten().astype('i8')
        vtkpolys.SetCells(self.nvertices,
                          vtknp.numpy_to_vtkIdTypeArray(packed, deep=1))
        polydata.SetPolys(vtkpolys)

        if color is not None:
            if color.ndim == 2:
                assert color.shape == (3, self.nvertices) or color.shape == (4, self.nvertices)
                color = np.around(color.T * 255).astype('u1')
            else:
                assert color.shape == (self.nvertices,)
            polydata.GetPointData().SetScalars(vtknp.numpy_to_vtk(color, deep=1))
        return polydata

    def to_vtk_mapper(self, color=None):
        """
        Returns a vtkPolyDataMapper mapping the mesh to an object ready for plotting

        :param color: (N, 3) or (N, ) array defining the color across the mesh
        :rtype: vtk.vtkPolyDataMapper
        """
        import vtk
        polydata = self.to_vtk_polydata(color=color)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarVisibility(1)
        return mapper

    def to_vtk_actor(self, color=None, opacity=1):
        """
        Returns a vtkPolyDataActor mapping the mesh to an actor, that can plot the mesh

        :param color: (N, 3) or (N, ) array defining the color across the mesh
        :rtype: vtk.vtkActor
        """
        import vtk
        actor = vtk.vtkActor()
        actor.SetMapper(self.to_vtk_mapper(color=color))
        actor.GetProperty().SetOpacity(opacity)
        return actor

    def render(self, color=None, opacity=1., view='+x', axes_scale=0., filename=None,
               window=None, renderer=None, interact=True):
        """
        Plots the mesh on the provided vtkRenderWindow

        :param color: (N, 3) or (N, ) array defining the color across the mesh
        :param opacity: float setting the opacity of the surface
        :param view: where the object is viewed from; one of '+x', '-x', '+y', '-y', '+z', or '-z' or tuple with

            - vector pointing from the mesh center to the camera
            - vector defining the hemisphere that is up from the camera

        :param filename: if set saves the image to the given filename
        :param window: If provded the window on which the mesh will be plotted (otherwise a new window is created)
        :type window: vtk.vtkRenderWindow
        :param renderer: the VTK rendered to which the actor plotting the mesh will be added (default: a new one is created)
        :type renderer: vtk.vtkRenderer
        :param interact: if True allows interaction of the window (this will pause the evaulation)
        :return: the window the mesh is plotted on and the rendered doing the plotting
        :rtype: (vtk.vtkRenderWindow, vtk.vtkRenderer)
        """
        import vtk
        if renderer is None:
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(1, 1, 1)
            if window is None:
                window = vtk.vtkRenderWindow()
                window.SetSize(1000, 1000)
        renderer.AddActor(self.to_vtk_actor(color=color, opacity=opacity))
        window.AddRenderer(renderer)

        if axes_scale > 0:
            vtk_axes = vtk.vtkAxesActor()
            transform = vtk.vtkTransform()
            transform.Scale(axes_scale, axes_scale, axes_scale)
            vtk_axes.SetUserTransform(transform)
            vtk_axes.SetAxisLabels(0)
            renderer.AddActor(vtk_axes)

        if isinstance(view, str):
            assert len(view) == 2
            view = {
                '+x': ((1, 0, 0), (0, 0, 1)),
                '-x': ((-1, 0, 0), (0, 0, 1)),
                '+y': ((0, 1, 0), (0, 0, 1)),
                '-y': ((0, -1, 0), (0, 0, 1)),
                '+z': ((0, 0, 1), (0, 1, 0)),
                '-z': ((0, 0, -1), (0, 1, 0))
            }[view]
        camera = renderer.GetActiveCamera()
        renderer.ResetCamera()
        camera.SetPosition(np.array(camera.GetFocalPoint()) + np.array(view[0]))
        camera.SetViewUp(view[1])
        renderer.ResetCamera()

        window.Render()
        if filename is not None:
            imfilt = vtk.vtkWindowToImageFilter()
            imfilt.SetInput(window)
            imfilt.Update()
            pngwriter = vtk.vtkPNGWriter()
            pngwriter.SetInputData(imfilt.GetOutput())
            pngwriter.SetFileName(filename)
            pngwriter.Write()
        if interact:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(window)
            interactor.Start()
        return window, renderer

    def __getitem__(self, item):
        """
        Gets the surface covering a subsection of all vertices
        """
        points = self.vertices[:, item]

        new_indices = np.full(self.nvertices, -1, dtype=int)
        new_indices[item] = np.arange(points.shape[1])

        triangles = new_indices[self.faces]
        use = (triangles != -1).all(0)
        return type(self)(points, triangles[:, use], self.flip_normal)


class ProjectedMesh(object):
    """
    1-dimensional slice of a 2D mesh.
    """
    def __init__(self, mesh, indices, position, orientation):
        self.position = position
        self.orientation = orientation
        self.vertex = indices[:, 0]
        self.points = indices[:, 1:]
        self.mesh = mesh

        norm_orient = orientation / np.sqrt(np.sum(orientation ** 2))
        relative_offset = np.sum((mesh.vertices.T - position[None, :]) * norm_orient[None, :], 1)
        offset = abs(relative_offset[self.points])
        pos = mesh.vertices.T[self.points, :]
        self.location = np.sum(offset[:, ::-1, None] * pos, 1) / np.sum(offset, 1)[:, None]

    @property
    def npoints(self):
        return self.points.shape[0]

    def spanned_coordinates(self, inplane_vec, flip_other=False):
        """Computes the 2D coordinates from the `position`

        :param inplane_vec: (3, ) array with any in-plane hemisphere defining the first coordinate
        :param flip_other: if True the second coordinate is defined based on the negative of the cross product betwen the plane normal and `inplane_vec` rather than the positive
        :return: (Npoints, 2) array with the coordinates along and perpendicular to ``inplane_vec``
        """
        inplane_vec = np.asarray(inplane_vec)
        norm_inplane_vec = inplane_vec - np.sum(inplane_vec * self.orientation) * self.orientation
        norm_inplane_vec /= np.sum(norm_inplane_vec)
        other_inplane = np.cross(norm_inplane_vec, self.orientation)
        if flip_other:
            other_inplane *= -1
        return np.sum((self.location - self.position)[:, None, :] * [norm_inplane_vec, other_inplane], 2)

    def line_collection(self, inplane_vec, flip_other=False, surface_arr=None, axes=None, **kwargs):
        """Returns a matplotlib line collection of the projected surface.

        :param inplane_vec: (3, ) array defining the hemisphere that will be used as the x-coordinate (see ProjectedMesh.spanned_coordinates)
        :param flip_other: if True the y-coordinate is defined based on the negative of the cross product betwen the plane normal and `inplane_vec` rather than the positive
        :param surface_arr: (N, ) or (N, 3) array defining values for vertices on the original mesh. If set will be used to set the color along the line;
        :param axes: matplotlib axes. If set the LineCollection will be added to this plot
        :param kwargs: keywords are pased on to the creation of the LineCollection (see matplotlib.collections.LineCollection)
        :return: the new LineCollection
        :rtype: matplotlib.collections.LineCollection
        """
        coord = self.spanned_coordinates(inplane_vec, flip_other=flip_other)
        segments = np.transpose([coord[:-1, :], coord[1:, :]], (1, 0, 2))
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, **kwargs)
        if surface_arr is not None:
            point_arr = np.nanmean(surface_arr[self.points], -1)
            lc.set_array((point_arr[1:] + point_arr[:-1]) / 2)
        if axes is not None:
            axes.add_collection(lc)
        return lc


@numba.jit(nopython=True)
def _trace_route(lines, output_arr):
    """Helper function to compute the projection a Mesh2D
    """
    idx_out_start = output_arr.shape[0] // 2
    for idx_vertex in range(lines.shape[2]):
        if lines[0, 0, idx_vertex] != -1:
            idx_start = idx_vertex
            break
    output_arr[idx_out_start, 0] = idx_vertex
    output_arr[idx_out_start, 1] = lines[0, 0, idx_vertex]
    output_arr[idx_out_start, 2] = lines[0, 1, idx_vertex]
    circle_found = False
    idx_min = idx_out_start
    idx_max = idx_out_start
    for idx_line, direction_out in enumerate((1, -1)):
        found_next = True
        idx_cvert = idx_start
        idx_cline = idx_line
        idx_out = idx_out_start
        while found_next:
            idx_out += direction_out
            found_next = False
            for idx_vertex in range(lines.shape[2]):
                if idx_vertex != idx_cvert:
                    for idx_line in range(2):
                        if (lines[idx_cline, 0, idx_cvert] == lines[idx_line, 0, idx_vertex] and
                            lines[idx_cline, 1, idx_cvert] == lines[idx_line, 1, idx_vertex]):
                            found_next = True
                            break
                    if found_next:
                        break
            if not found_next:
                if direction_out == 1:
                    idx_max = idx_out
                else:
                    idx_min = idx_out + 1
                break
            output_arr[idx_out, 0] = idx_vertex
            if idx_start == idx_vertex:
                circle_found = True
                output_arr[idx_out, 1] = output_arr[idx_out_start, 1]
                output_arr[idx_out, 2] = output_arr[idx_out_start, 2]
                lines[:, :, idx_cvert] = -1
                break
            else:
                idx_cline = 1 - idx_line
                if direction_out == 1:
                    output_arr[idx_out, 1] = lines[idx_cline, 0, idx_vertex]
                    output_arr[idx_out, 2] = lines[idx_cline, 1, idx_vertex]
                else:
                    output_arr[idx_out, 1] = lines[idx_line, 0, idx_vertex]
                    output_arr[idx_out, 2] = lines[idx_line, 1, idx_vertex]
                if idx_cvert != idx_start:
                    lines[:, :, idx_cvert] = -1
                idx_cvert = idx_vertex
        if circle_found:
            idx_max = idx_out + 1
            break
    lines[:, :, idx_start] = -1
    return idx_min, idx_max




