#pragma once

#include "mma_interface_h0.h"
#include "mma_interface_matrix.h"
#include "truc.h"
#include <gudhi/Simplex_tree/Simplex_tree_multi.h>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>

template <typename Filtration>
using SimplexTreeMultiOptions =
    Gudhi::multiparameter::Simplex_tree_options_multidimensional_filtration<Filtration>;

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
template <Available_columns col>
using BackendOptionsWithoutVine =
    Gudhi::multiparameter::interface::No_vine_multi_persistence_options<col>;

template <Available_columns col>
using ClementBackendOptionsWithVine =
    Gudhi::multiparameter::interface::Multi_persistence_Clement_options<col>;

using SimplicialStructure =
    Gudhi::multiparameter::interface::SimplicialStructure;
using PresentationStructure =
    Gudhi::multiparameter::interface::PresentationStructure;

template <Available_columns col, class Structure = SimplicialStructure>
using MatrixBackendNoVine =
    Gudhi::multiparameter::interface::Persistence_backend_matrix<
        BackendOptionsWithoutVine<col>, Structure>;

template <Available_columns col, class Structure = SimplicialStructure>
using MatrixBackendVine =
    Gudhi::multiparameter::interface::Persistence_backend_matrix<
        BackendOptionsWithVine<col>, Structure>;

template <Available_columns col, class Structure = SimplicialStructure>
using ClementMatrixBackendVine =
    Gudhi::multiparameter::interface::Persistence_backend_matrix<
        ClementBackendOptionsWithVine<col>, Structure>;
using GraphBackendVine =
    Gudhi::multiparameter::interface::Persistence_backend_h0<
        SimplicialStructure>;

using Filtration_value = Gudhi::multiparameter::multi_filtrations::
    Finitely_critical_multi_filtration<float>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialNoVineMatrixTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendNoVine<col>, SimplicialStructure, Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendVine<col, PresentationStructure>, PresentationStructure,
    Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralNoVineTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendNoVine<col, PresentationStructure>, PresentationStructure,
    Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineClementTruc = Gudhi::multiparameter::interface::Truc<
    ClementMatrixBackendVine<col, PresentationStructure>, PresentationStructure,
    Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialVineMatrixTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendVine<col>, SimplicialStructure, Filtration_value>;
using SimplicialVineGraphTruc = Gudhi::multiparameter::interface::Truc<
    GraphBackendVine, SimplicialStructure, Filtration_value>;

// multicrititcal
using KCriticalFiltrationValue =
    Gudhi::multiparameter::multi_filtrations::KCriticalFiltration<float>;
template <Available_columns col = Available_columns::INTRUSIVE_SET>
using KCriticalVineTruc = Gudhi::multiparameter::interface::Truc<
    MatrixBackendVine<col, PresentationStructure>, PresentationStructure,
    KCriticalFiltrationValue>;

template <bool is_vine,
          Available_columns col = Available_columns::INTRUSIVE_SET>
using Matrix_interface =
    std::conditional_t<is_vine, MatrixBackendVine<col, PresentationStructure>,
                       MatrixBackendNoVine<col, PresentationStructure>>;

template <bool is_kcritical, typename value_type>
using filtration_options = std::conditional_t<
    is_kcritical,
    Gudhi::multiparameter::multi_filtrations::KCriticalFiltration<value_type>,
    Gudhi::multiparameter::multi_filtrations::
        Finitely_critical_multi_filtration<value_type>>;

template <bool is_vine, bool is_kcritical, typename value_type,
          Available_columns col = Available_columns::INTRUSIVE_SET>
using MatrixTrucPythonInterface = Gudhi::multiparameter::interface::Truc<
    Matrix_interface<is_vine, col>, PresentationStructure,
    filtration_options<is_kcritical, value_type>>;
