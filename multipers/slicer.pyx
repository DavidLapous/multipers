from multipers.simplex_tree_multi import SimplexTreeMulti
import multipers
from typing import Optional,Literal
import multipers.io as mio

from multipers.slicer cimport *
import numpy as np
cimport cython
python_value_type = np.float32

cdef class SlicerNoVineSimplicial:
    cdef SimplicialNoVineMatrixTruc truc

    def get_ptr(self):
        return <intptr_t>(&self.truc)
    def __cinit__(self, st, bool vine=False):
        cdef intptr_t ptr = st.thisptr
        cdef Simplex_tree_multi_interface* st_ptr = <Simplex_tree_multi_interface*>(ptr)
        self.truc = SimplicialNoVineMatrixTruc(st_ptr)
    def get_barcode(self):
        return self.truc.get_barcode()
    def persistence_on_line(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        self.truc.compute_persistence()
        return self.truc.get_barcode()
    def compute_persistence(self,one_filtration=None):
        if one_filtration is not None:
            self.truc.set_one_filtration(one_filtration)
        self.truc.compute_persistence()
        # return self.truc.get_barcode()
    def get_barcode(self):
        return self.truc.get_barcode()
    def sliced_filtration(self,basepoint, direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        return np.asarray(self.truc.get_one_filtration())
    def __len__(self):
        return self.truc.num_generators()
    def __repr__(self):
        return self.truc.to_str().decode()
    def compute_box(self):
        cdef box_type box = self.truc.get_bounding_box()
        a = np.asarray(box.first._convert_back())
        b = np.asarray(box.second._convert_back())
        return np.array([a,b], dtype=python_value_type)
    def get_filtrations(self):
        return np.asarray([np.asarray(stuff._convert_back()) for stuff in self.truc.get_filtration_values()], dtype=python_value_type)
    def get_dimensions(self):
        return np.asarray(self.truc.get_dimensions(), dtype=np.int32)
    def get_boundaries(self):
        return tuple(tuple(b) for b in self.truc.get_boundaries())

cdef class SlicerVineSimplicial:
    cdef SimplicialVineMatrixTruc truc

    def get_ptr(self):
        return <intptr_t>(&self.truc)
    def __cinit__(self, st, bool vine=False):
        cdef intptr_t ptr = st.thisptr
        cdef Simplex_tree_multi_interface* st_ptr = <Simplex_tree_multi_interface*>(ptr)
        self.truc = SimplicialVineMatrixTruc(st_ptr)
    def vine_update(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)        
        self.truc.vineyard_update()
    def get_barcode(self):
        return self.truc.get_barcode()
    def persistence_on_line(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        self.truc.compute_persistence()
        return self.truc.get_barcode()
    def compute_persistence(self,one_filtration=None):
        if one_filtration is not None:
            self.truc.set_one_filtration(one_filtration)
        self.truc.compute_persistence()
        # return self.truc.get_barcode()
    def get_barcode(self):
        return self.truc.get_barcode()
    def sliced_filtration(self,basepoint, direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        return np.asarray(self.truc.get_one_filtration())
    def __len__(self):
        return self.truc.num_generators()
    def __repr__(self):
        return self.truc.to_str().decode()
    def compute_box(self):
        cdef box_type box = self.truc.get_bounding_box()
        a = np.asarray(box.first._convert_back())
        b = np.asarray(box.second._convert_back())
        return np.array([a,b], dtype=python_value_type)
    def get_filtrations(self):
        return np.asarray([np.asarray(stuff._convert_back()) for stuff in self.truc.get_filtration_values()], dtype=python_value_type)
    def get_dimensions(self):
        return np.asarray(self.truc.get_dimensions(), dtype=np.int32)
    def get_boundaries(self):
        return tuple(tuple(b) for b in self.truc.get_boundaries())

cdef class SlicerClement:
    cdef GeneralVineClementTruc truc
    def __cinit__(self, generator_maps, generator_dimensions, filtration_values):
        cdef uint32_t num_generators = len(generator_maps)
        cdef vector[vector[uint32_t]] c_generator_maps
        cdef vector[Finitely_critical_multi_filtration] c_filtration_values
        for stuff in generator_maps:
            c_generator_maps.push_back(<vector[uint32_t]>(stuff))
        cdef Finitely_critical_multi_filtration cf
        for f in filtration_values:
            cf.clear()
            for truc in f:
                cf.push_back(truc)
            c_filtration_values.push_back(cf)
        cdef vector[int] c_generator_dimensions = generator_dimensions
        assert num_generators == c_generator_maps.size() == c_filtration_values.size(), "Invalid input, shape do not coincide."
        self.truc = GeneralVineClementTruc(c_generator_maps,c_generator_dimensions, c_filtration_values)


    def get_ptr(self):
        return <intptr_t>(&self.truc)
    def vine_update(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)        
        self.truc.vineyard_update()
    def get_barcode(self):
        return self.truc.get_barcode()
    def persistence_on_line(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        self.truc.compute_persistence()
        return self.truc.get_barcode()
    def compute_persistence(self,one_filtration=None):
        if one_filtration is not None:
            self.truc.set_one_filtration(one_filtration)
        self.truc.compute_persistence()
        # return self.truc.get_barcode()
    def get_barcode(self):
        return self.truc.get_barcode()
    def sliced_filtration(self,basepoint, direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        return np.asarray(self.truc.get_one_filtration())
    def __len__(self):
        return self.truc.num_generators()
    def __repr__(self):
        return self.truc.to_str().decode()
    def compute_box(self):
        cdef box_type box = self.truc.get_bounding_box()
        a = np.asarray(box.first._convert_back())
        b = np.asarray(box.second._convert_back())
        return np.array([a,b], dtype=python_value_type)
    def get_filtrations(self):
        return np.asarray([np.asarray(stuff._convert_back()) for stuff in self.truc.get_filtration_values()], dtype=python_value_type)
    def get_dimensions(self):
        return np.asarray(self.truc.get_dimensions(), dtype=np.int32)
    def get_boundaries(self):
        return tuple(tuple(b) for b in self.truc.get_boundaries())

cdef class Slicer:
    cdef GeneralVineTruc truc
    def __cinit__(self, generator_maps, generator_dimensions, filtration_values):
        cdef uint32_t num_generators = len(generator_maps)
        cdef vector[vector[uint32_t]] c_generator_maps
        cdef vector[Finitely_critical_multi_filtration] c_filtration_values
        for stuff in generator_maps:
            c_generator_maps.push_back(<vector[uint32_t]>(stuff))
        cdef Finitely_critical_multi_filtration cf
        for f in filtration_values:
            cf.clear()
            for truc in f:
                cf.push_back(truc)
            c_filtration_values.push_back(cf)
        cdef vector[int] c_generator_dimensions = generator_dimensions
        assert num_generators == c_generator_maps.size() == c_filtration_values.size(), "Invalid input, shape do not coincide."
        self.truc = GeneralVineTruc(c_generator_maps,c_generator_dimensions, c_filtration_values)

    def get_ptr(self):
        return <intptr_t>(&self.truc)
    def vine_update(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)        
        self.truc.vineyard_update()
    def get_barcode(self):
        return self.truc.get_barcode()
    def persistence_on_line(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        self.truc.compute_persistence()
        return self.truc.get_barcode()
    def compute_persistence(self,one_filtration=None):
        if one_filtration is not None:
            self.truc.set_one_filtration(one_filtration)
        self.truc.compute_persistence()
        # return self.truc.get_barcode()
    def get_barcode(self):
        return self.truc.get_barcode()
    def sliced_filtration(self,basepoint, direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        return np.asarray(self.truc.get_one_filtration())
    def __len__(self):
        return self.truc.num_generators()
    def __repr__(self):
        return self.truc.to_str().decode()
    def compute_box(self):
        cdef box_type box = self.truc.get_bounding_box()
        a = np.asarray(box.first._convert_back())
        b = np.asarray(box.second._convert_back())
        return np.array([a,b], dtype=python_value_type)
    def get_filtrations(self):
        return np.asarray([np.asarray(stuff._convert_back()) for stuff in self.truc.get_filtration_values()], dtype=python_value_type)
    def get_dimensions(self):
        return np.asarray(self.truc.get_dimensions(), dtype=np.int32)
    def get_boundaries(self):
        return tuple(tuple(b) for b in self.truc.get_boundaries())

cdef class SlicerNoVine:
    cdef GeneralNoVineTruc truc
    def __cinit__(self, generator_maps, generator_dimensions, filtration_values):
        cdef uint32_t num_generators = len(generator_maps)
        cdef vector[vector[uint32_t]] c_generator_maps
        cdef vector[Finitely_critical_multi_filtration] c_filtration_values
        for stuff in generator_maps:
            c_generator_maps.push_back(<vector[uint32_t]>(stuff))
        cdef Finitely_critical_multi_filtration cf
        for f in filtration_values:
            cf.clear()
            for truc in f:
                cf.push_back(truc)
            c_filtration_values.push_back(cf)
        cdef vector[int] c_generator_dimensions = generator_dimensions
        assert num_generators == c_generator_maps.size() == c_filtration_values.size(), "Invalid input, shape do not coincide."
        self.truc = GeneralNoVineTruc(c_generator_maps,c_generator_dimensions, c_filtration_values)

    def get_ptr(self):
        return <intptr_t>(&self.truc)
    def get_barcode(self):
        return self.truc.get_barcode()
    def persistence_on_line(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        self.truc.compute_persistence()
        return self.truc.get_barcode()
    def compute_persistence(self,one_filtration=None):
        if one_filtration is not None:
            self.truc.set_one_filtration(one_filtration)
        self.truc.compute_persistence()
        # return self.truc.get_barcode()
    def get_barcode(self):
        return self.truc.get_barcode()
    def sliced_filtration(self,basepoint, direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        return np.asarray(self.truc.get_one_filtration())
    def __len__(self):
        return self.truc.num_generators()
    def __repr__(self):
        return self.truc.to_str().decode()
    def compute_box(self):
        cdef box_type box = self.truc.get_bounding_box()
        a = np.asarray(box.first._convert_back())
        b = np.asarray(box.second._convert_back())
        return np.array([a,b], dtype=python_value_type)
    def get_filtrations(self):
        return np.asarray([np.asarray(stuff._convert_back()) for stuff in self.truc.get_filtration_values()], dtype=python_value_type)
    def get_dimensions(self):
        return np.asarray(self.truc.get_dimensions(), dtype=np.int32)
    def get_boundaries(self):
        return tuple(tuple(b) for b in self.truc.get_boundaries())

cdef class SlicerVineGraph:
    cdef SimplicialVineGraphTruc truc
    def __cinit__(self, st, bool vine=False):
        cdef intptr_t ptr = st.thisptr
        cdef Simplex_tree_multi_interface* st_ptr = <Simplex_tree_multi_interface*>(ptr)
        self.truc = SimplicialVineGraphTruc(st_ptr)
    def vine_update(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)        
        self.truc.vineyard_update()
    def get_barcode(self):
        return self.truc.get_barcode()

    def get_ptr(self):
        return <intptr_t>(&self.truc)
    def persistence_on_line(self,basepoint,direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        self.truc.compute_persistence()
        return self.truc.get_barcode()
    def compute_persistence(self,one_filtration=None):
        if one_filtration is not None:
            self.truc.set_one_filtration(one_filtration)
        self.truc.compute_persistence()
        # return self.truc.get_barcode()
    def get_barcode(self):
        return self.truc.get_barcode()
    def sliced_filtration(self,basepoint, direction=None):
        basepoint = np.asarray(basepoint)
        cdef Line line
        if direction is None:
            line = Line(basepoint)
        else:
            line = Line(basepoint,direction)
        self.truc.push_to[Line](line)
        return np.asarray(self.truc.get_one_filtration())
    def __len__(self):
        return self.truc.num_generators()
    def __repr__(self):
        return self.truc.to_str().decode()
    def compute_box(self):
        cdef box_type box = self.truc.get_bounding_box()
        a = np.asarray(box.first._convert_back())
        b = np.asarray(box.second._convert_back())
        return np.array([a,b], dtype=python_value_type)
    def get_filtrations(self):
        return np.asarray([np.asarray(stuff._convert_back()) for stuff in self.truc.get_filtration_values()], dtype=python_value_type)
    def get_dimensions(self):
        return np.asarray(self.truc.get_dimensions(), dtype=np.int32)
    def get_boundaries(self):
        return tuple(tuple(b) for b in self.truc.get_boundaries())


def from_function_delaunay(
    points,
    grades,
    degree: int=-1,
    backend: Literal["matrix", "clement"] = "matrix",
    vineyard=True,
):
    """
    Given points in $\mathbb R^n$ and function grades, compute the function-delaunay
    bifiltration as a in an scc format, and converts it into a slicer.

    points : (num_pts, n) float array
    grades : (num_pts,) float array
    degree (opt) : if given, computes a minimal presentation of this homological degree first
    backend : slicer backend, e.g. "matrix", "clement"
    vineyard : bool, use a vineyard-compatible backend
    """
    blocks = mio.function_delaunay_presentation(points, grades, degree=degree)
    return multipers.Slicer(blocks, backend=backend, vineyard=vineyard)

def slicer2blocks(slicer, int degree = -1, bool reverse=True):
    """
    Convert any slicer to the block format a.k.a. scc format for python
    """
    dims = slicer.get_dimensions()
    num_empty_blocks_to_add = 1 if degree == -1 else dims.min()-degree +1
    _,counts = np.unique(dims, return_counts=True, )
    indices = np.concatenate([[0],counts], dtype=np.int32).cumsum()
    filtration_values = slicer.get_filtrations()
    filtration_values = [filtration_values[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    boundaries = slicer.get_boundaries()
    boundaries = [boundaries[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    shift = np.concatenate([[0], indices], dtype=np.int32)
    boundaries = [tuple(np.asarray(x-s, dtype=np.int32) for x in block)  for s,block in zip(shift,boundaries)]
    blocks = [tuple((f,tuple(b))) for b,f in zip(boundaries,filtration_values)]
    blocks = ([(np.empty((0,)),[])]*num_empty_blocks_to_add) + blocks
    if reverse:
        blocks.reverse()
    return blocks

def minimal_presentation(
        slicer,
        int degree = 1, 
        backend:Literal["mpfree", "2pac"]="mpfree", 
        slicer_backend:Literal["matrix","clement","graph"]="matrix",
        bool vineyard=True, 
        **minpres_kwargs
        ):
    """
    Computes a minimal presentation of the multifiltered complex given by the slicer,
    and returns it as a slicer.
    Only works for mpfree for the moment.
    """
    blocks = slicer2blocks(slicer)
    mio.scc2disk(blocks,path=mio.input_path)
    dimension = len(blocks) -2 - degree # latest  = L-1, which is empty, -1 for degree 0, -2 for degree 1 etc.
    new_blocks = mio.scc_reduce_from_str(path=mio.input_path,dimension=dimension, backend=backend, **minpres_kwargs)
    new_slicer = multipers.Slicer(new_blocks,backend=slicer_backend)
    return new_slicer
