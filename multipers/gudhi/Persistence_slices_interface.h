#pragma once

#include "mma_interface_h0.h"
#include "mma_interface_matrix.h"
#include "truc.h"
#include <gudhi/Simplex_tree/Simplex_tree_multi.h>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>

using SimplexTreeMultiOptions =
    Gudhi::multiparameter::Simplex_tree_options_multidimensional_filtration;
using BackendOptionsWithoutVine =
    Gudhi::multiparameter::interface::No_vine_multi_persistence_options<>;

enum Column_types_strs {
  LIST,
  SET,
  HEAP,
  VECTOR,
  NAIVE_VECTOR,
  UNORDERED_SET,
  INTRUSIVE_LIST,
  INTRUSIVE_SET
};

using Available_columns = Gudhi::persistence_matrix::Column_types;

template <Available_columns col>
using BackendOptionsWithVine =
    Gudhi::multiparameter::interface::Multi_persistence_options<col>;
;
using SimplicialStructure =
    Gudhi::multiparameter::interface::SimplicialStructure;
using PresentationStructure =
    Gudhi::multiparameter::interface::PresentationStructure;

using MatrixBackendNoVine =
    Gudhi::multiparameter::interface::Persistence_backend_matrix<
        BackendOptionsWithoutVine, SimplicialStructure>;

template <Available_columns col, class Structure = SimplicialStructure>
using MatrixBackendVine =
    Gudhi::multiparameter::interface::Persistence_backend_matrix<
        BackendOptionsWithVine<col>, Structure>;
using GraphBackendVine =
    Gudhi::multiparameter::interface::Persistence_backend_h0<
        SimplicialStructure>;

using Filtration_value = Gudhi::multiparameter::multi_filtrations::
    Finitely_critical_multi_filtration<float>;

using SimplicialNoVineMatrixTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendNoVine, SimplicialStructure, Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendVine<col, PresentationStructure>, PresentationStructure,
    Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialVineMatrixTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendVine<col>, SimplicialStructure, Filtration_value>;
using SimplicialVineGraphTruc = Gudhi::multiparameter::interface::Truc<
    GraphBackendVine, SimplicialStructure, Filtration_value>;
