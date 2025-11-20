#pragma once

#include <array>
#include <cstdint>

#include "gudhi/Multi_parameter_filtration.h"
#include "gudhi/multi_simplex_tree_helpers.h"
#include "gudhi/persistence_matrix_options.h"
#include "gudhi/Multi_parameter_filtered_complex.h"
#include "gudhi/Multi_persistence/Persistence_interface_matrix.h"
#include "gudhi/Multi_persistence/Persistence_interface_cohomology.h"
#include "gudhi/Dynamic_multi_parameter_filtration.h"
#include "gudhi/Degree_rips_bifiltration.h"
#include "gudhi/Slicer.h"

#include "tmp_h0_pers/mma_interface_h0.h"

namespace multipers::tmp_interface {

template <typename Filtration>
using SimplexTreeMultiOptions = Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<Filtration>;

enum Column_types_strs : std::uint8_t {
  LIST,
  SET,
  HEAP,
  VECTOR,
  NAIVE_VECTOR,
  UNORDERED_SET,
  INTRUSIVE_LIST,
  INTRUSIVE_SET
};

enum Filtration_containers_strs : std::uint8_t {
  Dynamic_multi_parameter_filtration,
  Multi_parameter_filtration,
  Degree_rips_bifiltration
};

using Available_columns = Gudhi::persistence_matrix::Column_types;

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct Multi_persistence_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = true;
  static const bool can_retrieve_representative_cycles = true;
};

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct Multi_persistence_Clement_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = true;
  static const bool is_of_boundary_type = false;
  static const Gudhi::persistence_matrix::Column_indexation_types column_indexation_type =
      Gudhi::persistence_matrix::Column_indexation_types::POSITION;
  static const bool can_retrieve_representative_cycles = true;
};

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct No_vine_multi_persistence_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = false;
};

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET,
          bool row_access = true>
struct fix_presentation_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_row_access = row_access;
  static const bool has_map_column_container = false;
  static const bool has_removable_columns = false;  // WARN : idx will change if map is not true
};

template <Available_columns col>
using BackendOptionsWithVine = Multi_persistence_options<col>;
template <Available_columns col>
using BackendOptionsWithoutVine = No_vine_multi_persistence_options<col>;

template <Available_columns col>
using ClementBackendOptionsWithVine = Multi_persistence_Clement_options<col>;

// using SimplicialStructure = Gudhi::multiparameter::truc_interface::SimplicialStructure;
template <typename Filtration>
using StructureStuff = Gudhi::multi_persistence::Multi_parameter_filtered_complex<Filtration>;

template <Available_columns col>
using MatrixBackendNoVine = Gudhi::multi_persistence::Persistence_interface_matrix<BackendOptionsWithoutVine<col>>;

template <Available_columns col>
using MatrixBackendVine = Gudhi::multi_persistence::Persistence_interface_matrix<BackendOptionsWithVine<col>>;

template <Available_columns col>
using ClementMatrixBackendVine =
    Gudhi::multi_persistence::Persistence_interface_matrix<ClementBackendOptionsWithVine<col>>;
template <typename Filtration>
using GraphBackendVine = Gudhi::multiparameter::truc_interface::Persistence_backend_h0<StructureStuff<Filtration>>;

template <typename value_type = float>
using Filtration_value = Gudhi::multi_filtration::Multi_parameter_filtration<value_type, false, true>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialNoVineMatrixTruc = Gudhi::multi_persistence::Slicer<Filtration_value<>, MatrixBackendNoVine<col>>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineTruc = Gudhi::multi_persistence::Slicer<Filtration_value<>, MatrixBackendVine<col>>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralNoVineTruc = Gudhi::multi_persistence::Slicer<Filtration_value<>, MatrixBackendNoVine<col>>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using GeneralVineClementTruc = Gudhi::multi_persistence::Slicer<Filtration_value<>, ClementMatrixBackendVine<col>>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using SimplicialVineMatrixTruc = Gudhi::multi_persistence::Slicer<Filtration_value<>, MatrixBackendVine<col>>;
using SimplicialVineGraphTruc =
    Gudhi::multi_persistence::Slicer<Filtration_value<>, GraphBackendVine<Filtration_value<>>>;

// multi-critical
template <typename value_type = float>
using Multi_critical_filtration_value = Gudhi::multi_filtration::Multi_parameter_filtration<value_type>;

template <Available_columns col = Available_columns::INTRUSIVE_SET>
using KCriticalVineTruc = Gudhi::multi_persistence::Slicer<Multi_critical_filtration_value<>, MatrixBackendVine<col>>;

template <bool is_vine, Available_columns col = Available_columns::INTRUSIVE_SET>
using Matrix_interface = std::conditional_t<is_vine, MatrixBackendVine<col>, MatrixBackendNoVine<col>>;

template <Filtration_containers_strs fil_container,  bool is_k_critical, typename value_type>
using filtration_options = std::conditional_t<fil_container == Filtration_containers_strs::Dynamic_multi_parameter_filtration,
                                            Gudhi::multi_filtration::Dynamic_multi_parameter_filtration<value_type, false, !is_k_critical>,
                                            std::conditional_t<fil_container == Filtration_containers_strs::Multi_parameter_filtration,
                                                               Gudhi::multi_filtration::Multi_parameter_filtration<value_type, false, !is_k_critical>,
                                                               Gudhi::multi_filtration::Degree_rips_bifiltration<value_type, false, !is_k_critical>>>;

template <bool is_vine,
          bool is_k_critical,
          typename value_type,
          Available_columns col = Available_columns::INTRUSIVE_SET,
          Filtration_containers_strs filt_cont = Filtration_containers_strs::Multi_parameter_filtration>
using MatrixTrucPythonInterface =
    Gudhi::multi_persistence::Slicer<filtration_options<filt_cont,is_k_critical, value_type>, Matrix_interface<is_vine, col>>;

enum class BackendsEnum : std::uint8_t { Matrix, Graph, Clement, GudhiCohomology };

// Create a template metafunction to simplify the type selection
template <BackendsEnum backend, bool is_vine, Available_columns col, typename Filtration>
struct PersBackendOptsImpl;

template <bool is_vine, Available_columns col, typename Filtration>
struct PersBackendOptsImpl<BackendsEnum::Matrix, is_vine, col, Filtration> {
  using type = Matrix_interface<is_vine, col>;
};

template <bool is_vine, Available_columns col, typename Filtration>
struct PersBackendOptsImpl<BackendsEnum::Clement, is_vine, col, Filtration> {
  static_assert(is_vine, "Clement is vine");
  using type = ClementMatrixBackendVine<col>;
};

template <bool is_vine, Available_columns col, typename Filtration>
struct PersBackendOptsImpl<BackendsEnum::GudhiCohomology, is_vine, col, Filtration> {
  static_assert(!is_vine, "Gudhi is not vine");
  using type = Gudhi::multi_persistence::Persistence_interface_cohomology<Filtration>;
};

template <bool is_vine, Available_columns col, typename Filtration>
struct PersBackendOptsImpl<BackendsEnum::Graph, is_vine, col, Filtration> {
  static_assert(is_vine, "Graph backend requires is_vine to be true");
  using type = GraphBackendVine<Filtration>;
};

// Helper alias to extract the type
template <BackendsEnum backend, bool is_vine, Available_columns col, typename Filtration>
using PersBackendOpts = typename PersBackendOptsImpl<backend, is_vine, col, Filtration>::type;

template <BackendsEnum backend,
          bool is_vine,
          bool is_k_critical,
          typename value_type,
          Available_columns col = Available_columns::INTRUSIVE_SET,
          Filtration_containers_strs filt_cont = Filtration_containers_strs::Multi_parameter_filtration>
using TrucPythonInterface = Gudhi::multi_persistence::Slicer<
    filtration_options<filt_cont,is_k_critical, value_type>,
    PersBackendOpts<backend, is_vine, col, filtration_options<filt_cont,is_k_critical, value_type>>>;

//for python
template<typename T>
using One_critical_filtration = Gudhi::multi_filtration::Multi_parameter_filtration<T, false, true>;
template<typename T>
using Multi_critical_filtration = Gudhi::multi_filtration::Multi_parameter_filtration<T, false, false>;

template <typename T>
using Bar = std::array<T, 2>;
template <class Slicer, typename T = typename Slicer::T>
using Barcode = typename Slicer::template Flat_barcode<T>;
template <class Slicer, typename T = typename Slicer::T>
using Dim_barcode = typename Slicer::template Multi_dimensional_flat_barcode<T>;

template<class Slicer>
std::string slicer_to_str(Slicer& s)
{
  std::stringstream stream;
  stream << s;
  return stream.str();
}

}  // namespace multipers::tmp_interface
