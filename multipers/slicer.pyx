from multipers.simplex_tree_multi import SimplexTreeMulti

from multipers.slicer cimport *
import numpy as np
cimport cython


cdef class SlicerNoVineSimplcial:
    cdef SimplicialNoVineMatrixTruc truc

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

cdef class SlicerVineSimplcial:
    cdef SimplicialVineMatrixTruc truc
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


