#pragma once

#include "mma_interface_h0.h"
#include "mma_interface_matrix.h"
#include "mma_interface_coh.h"
#include <type_traits>  // For static_assert
#include "truc.h"
#include <gudhi/Simplex_tree_multi.h>
#include <gudhi/One_critical_filtration.h>
#include <gudhi/Multi_critical_filtration.h>

template <typename Filtration>
using SimplexTreeMultiOptions = Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<Filtration>;

enum Column_types_strs { LIST, SET, HEAP, VECTOR, NAIVE_VECTOR, UNORDERED_SET, INTRUSIVE_LIST, INTRUSIVE_SET };

using Available_columns = Gudhi::persistence_matrix::Column_types;

template <Available_columns col>
using BackendOptionsWithVine = Gudhi::multiparameter::truc_interface::Multi_persistence_options<col>;
template <Available_columns col>
using BackendOptionsWithoutVine = Gudhi::multiparameter::truc_interface::No_vine_multi_persistence_options<col>;

template <Available_columns col>
using ClementBackendOptionsWithVine = Gudhi::multiparameter::truc_interface::Multi_persistence_Clement_options<col>;

using SimplicialStructure = Gudhi::multiparameter::truc_interface::SimplicialStructure;
using PresentationStructure = Gudhi::multiparameter::truc_interface::PresentationStructure;

template <Available_columns col, class Structure = SimplicialStructure>
using MatrixBackendNoVine =
    Gudhi::multiparameter::truc_interface::Persistence_backend_matrix<BackendOptionsWithoutVine<col>, Structure>;

template <Available_columns col, class Structure = SimplicialStructure>
using MatrixBackendVine =
    Gudhi::multiparameter::truc_interface::Persistence_backend_matrix<BackendOptionsWithVine<col>, Structure>;

template <Available_columns col, class Structure = SimplicialStructure>
using ClementMatrixBackendVine =
    Gudhi::multiparameter::truc_interface::Persistence_backend_matrix<ClementBackendOptionsWithVine<col>, Structure>;
using GraphBackendVine = Gudhi::multiparameter::truc_interface::Persistence_backend_h0<SimplicialStructure>;

using Filtration_value = Gudhi::multi_filtration::One_critical_filtration<float>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialNoVineMatrixTruc =
    Gudhi::multiparameter::truc_interface::Truc<MatrixBackendNoVine<col>, SimplicialStructure, Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineTruc = Gudhi::multiparameter::truc_interface::
    Truc<MatrixBackendVine<col, PresentationStructure>, PresentationStructure, Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralNoVineTruc = Gudhi::multiparameter::truc_interface::
    Truc<MatrixBackendNoVine<col, PresentationStructure>, PresentationStructure, Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineClementTruc = Gudhi::multiparameter::truc_interface::
    Truc<ClementMatrixBackendVine<col, PresentationStructure>, PresentationStructure, Filtration_value>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialVineMatrixTruc =
    Gudhi::multiparameter::truc_interface::Truc<MatrixBackendVine<col>, SimplicialStructure, Filtration_value>;
using SimplicialVineGraphTruc =
    Gudhi::multiparameter::truc_interface::Truc<GraphBackendVine, SimplicialStructure, Filtration_value>;

// multicrititcal
using Multi_critical_filtrationValue = Gudhi::multi_filtration::Multi_critical_filtration<float>;
template <Available_columns col = Available_columns::INTRUSIVE_SET>
using KCriticalVineTruc = Gudhi::multiparameter::truc_interface::
    Truc<MatrixBackendVine<col, PresentationStructure>, PresentationStructure, Multi_critical_filtrationValue>;

template <bool is_vine, Available_columns col = Available_columns::INTRUSIVE_SET>
using Matrix_interface = std::conditional_t<is_vine,
                                            MatrixBackendVine<col, PresentationStructure>,
                                            MatrixBackendNoVine<col, PresentationStructure>>;

template <bool is_kcritical, typename value_type>
using filtration_options = std::conditional_t<is_kcritical,
                                              Gudhi::multi_filtration::Multi_critical_filtration<value_type>,
                                              Gudhi::multi_filtration::One_critical_filtration<value_type>>;

template <bool is_vine,
          bool is_kcritical,
          typename value_type,
          Available_columns col = Available_columns::INTRUSIVE_SET>
using MatrixTrucPythonInterface = Gudhi::multiparameter::truc_interface::
    Truc<Matrix_interface<is_vine, col>, PresentationStructure, filtration_options<is_kcritical, value_type>>;

enum class BackendsEnum { Matrix, Graph, Clement, GudhiCohomology };

// Create a template metafunction to simplify the type selection
template <BackendsEnum backend, bool is_vine, Available_columns col>
struct PersBackendOptsImpl;

template <bool is_vine, Available_columns col>
struct PersBackendOptsImpl<BackendsEnum::Matrix, is_vine, col> {
  using type = Matrix_interface<is_vine, col>;
};

template <bool is_vine, Available_columns col>
struct PersBackendOptsImpl<BackendsEnum::Clement, is_vine, col> {
  static_assert(is_vine, "Clement is vine");
  using type = ClementMatrixBackendVine<col, PresentationStructure>;
};

template <bool is_vine, Available_columns col>
struct PersBackendOptsImpl<BackendsEnum::GudhiCohomology, is_vine, col> {
  static_assert(!is_vine, "Gudhi is not vine");
  using type = Gudhi::multiparameter::truc_interface::Persistence_backend_cohomology<PresentationStructure>;
};

template <bool is_vine, Available_columns col>
struct PersBackendOptsImpl<BackendsEnum::Graph, is_vine, col> {
  static_assert(is_vine, "Graph backend requires is_vine to be true");
  using type = GraphBackendVine;
};

// Helper alias to extract the type
template <BackendsEnum backend, bool is_vine, Available_columns col>
using PersBackendOpts = typename PersBackendOptsImpl<backend, is_vine, col>::type;

template <BackendsEnum backend>
using StructureStuff = std::conditional_t<backend == BackendsEnum::Graph, SimplicialStructure, PresentationStructure>;

template <BackendsEnum backend,
          bool is_vine,
          bool is_kcritical,
          typename value_type,
          Available_columns col = Available_columns::INTRUSIVE_SET>
using TrucPythonInterface = Gudhi::multiparameter::truc_interface::
    Truc<PersBackendOpts<backend, is_vine, col>, StructureStuff<backend>, filtration_options<is_kcritical, value_type>>;
