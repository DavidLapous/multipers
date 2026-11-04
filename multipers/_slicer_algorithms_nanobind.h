#ifndef MP_PY_SLICER_ALGO_NANOBIND_H_INCLUDED
#define MP_PY_SLICER_ALGO_NANOBIND_H_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gudhi/multiparameter_module_approximation.h>
#include <gudhi/Multi_persistence/tensor_helpers.h>
#include <gudhi/hilbert_signed_measure.h>
#include <gudhi/rank_invariant.h>
#include <python_interfaces/numpy_utils.h>
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "gudhi/Multi_persistence/Box.h"
#include "gudhi/Module_interface.h"
#include "nanobind_array_utils.hpp"
#include "nanobind_dense_array_utils.hpp"
#include "nanobind_object_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpnb {

using tensor_dtype = int32_t;
using indices_type = int32_t;
using signed_measure_type = std::pair<std::vector<indices_type>, std::vector<tensor_dtype>>;

using multipers::nanobind_dense_utils::matrix_from_array;
using multipers::nanobind_dense_utils::vector_from_array;
using multipers::nanobind_helpers::dispatch_slicer_by_template_id;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_utils::cast_vector;
using multipers::nanobind_utils::lowercase_copy;
using multipers::nanobind_utils::numpy_dtype_name;
using multipers::nanobind_utils::template_id_of;
using multipers::nanobind_helpers::is_slicer_object;
using multipers::nanobind_utils::owned_array;

inline bool has_slicer_template_id(const nb::handle& input) { return is_slicer_object(input); }

template <typename... Ds>
inline nb::object get_slicer_class(type_list<Ds...>,
                            bool is_vineyard,
                            bool is_k_critical,
                            const nb::handle& dtype,
                            std::string col,
                            std::string pers_backend,
                            std::string filtration_container) {
  std::string dtype_name = numpy_dtype_name(dtype);
  col = lowercase_copy(std::move(col));
  pers_backend = lowercase_copy(std::move(pers_backend));
  filtration_container = lowercase_copy(std::move(filtration_container));
  bool matched = false;
  nb::object result;
  (
      [&]<typename D>() {
        if (!matched && D::is_vine == is_vineyard && D::is_kcritical == is_k_critical && D::dtype_name == dtype_name &&
            lowercase_copy(std::string(D::column_type)) == col &&
            lowercase_copy(std::string(D::backend_type)) == pers_backend &&
            lowercase_copy(std::string(D::filtration_container)) == filtration_container) {
          result = nb::borrow<nb::object>(nb::type<typename D::interface>());
          matched = true;
        }
      }.template operator()<Ds>(),
      ...);
  if (!matched) {
    throw nb::value_error("Unimplemented slicer combination.");
  }
  return result;
}

// inline nb::object get_slicer_class_from_template_id(int template_id) {
//   return dispatch_slicer_by_template_id(template_id, [&]<typename Desc>() -> nb::object {
//     return nb::borrow<nb::object>(nb::type<typename Desc::interface>());
//   });
// }

template <typename... Ds>
inline nb::tuple compute_hilbert_signed_measure(type_list<Ds...>,
                                                nb::handle slicer,
                                                const std::vector<indices_type>& full_shape,
                                                const std::vector<indices_type>& degrees,
                                                bool zero_pad,
                                                indices_type n_jobs,
                                                bool verbose,
                                                bool ignore_inf) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  for (const auto extent : full_shape) {
    if (extent <= indices_type{zero_pad ? 1 : 0}) [[unlikely]] {
      throw std::runtime_error("Internal error: invalid Hilbert grid shape.");
    }
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::tuple {
    auto& wrapper = nb::cast<typename D::interface&>(slicer);
    if (wrapper.get_number_of_parameters() < 2)
      throw std::logic_error("Not implemented for " + std::to_string(wrapper.get_number_of_parameters()) +
                             " < 2 parameters.");
    if (full_shape.size() != wrapper.get_number_of_parameters()) [[unlikely]]
      throw std::runtime_error("Internal error: invalid Hilbert grid shape.");
    Gudhi::multi_persistence::Signed_measure<indices_type, tensor_dtype> sm;
    {
      nanobind::gil_scoped_release release;
      sm = Gudhi::multi_persistence::compute_hilbert_signed_measure<tensor_dtype, indices_type>(
          wrapper.get_slicer(), full_shape, degrees, zero_pad, ignore_inf, verbose, n_jobs);
    }
    return nanobind::make_tuple(
        _wrap_as_numpy_array(
            std::move(sm.coordinates), sm.coordinates.size() / sm.numberOfParameters, sm.numberOfParameters),
        _wrap_as_numpy_array(std::move(sm.values), sm.values.size()));
  });
}

template <typename... Ds>
inline nb::tuple compute_hilbert_signed_measure_sparse(type_list<Ds...>,
                                                       nb::handle slicer,
                                                       const std::vector<indices_type>& grid_shape,
                                                       const std::vector<indices_type>& degrees,
                                                       bool zero_pad,
                                                       indices_type n_jobs,
                                                       bool verbose,
                                                       bool ignore_inf) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  for (const auto extent : grid_shape) {
    if (extent <= indices_type{zero_pad ? 1 : 0}) [[unlikely]] {
      throw std::runtime_error("Internal error: invalid sparse Hilbert grid shape.");
    }
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::tuple {
    auto& wrapper = nb::cast<typename D::interface&>(slicer);
    if (wrapper.get_number_of_parameters() < 2)
      throw std::logic_error("Not implemented for " + std::to_string(wrapper.get_number_of_parameters()) +
                             " < 2 parameters.");
    if (grid_shape.size() != wrapper.get_number_of_parameters()) [[unlikely]]
      throw std::runtime_error("Internal error: invalid Hilbert grid shape.");
    Gudhi::multi_persistence::Signed_measure<indices_type, tensor_dtype> sm;
    {
      nb::gil_scoped_release release;
      sm = Gudhi::multi_persistence::compute_sparse_hilbert_signed_measure<tensor_dtype, indices_type>(
          wrapper.get_slicer(), grid_shape, degrees, zero_pad, ignore_inf, verbose, n_jobs);
    }
    return nanobind::make_tuple(
        _wrap_as_numpy_array(
            std::move(sm.coordinates), sm.coordinates.size() / sm.numberOfParameters, sm.numberOfParameters),
        _wrap_as_numpy_array(std::move(sm.values), sm.values.size()));
  });
}

template <typename... Ds>
inline nb::tuple compute_rank_tensor(type_list<Ds...>,
                                     nb::handle slicer,
                                     const std::vector<indices_type>& grid_shape,
                                     const std::vector<indices_type>& degrees,
                                     indices_type n_jobs,
                                     bool ignore_inf) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::tuple {
    auto& wrapper = nb::cast<typename D::interface&>(slicer);
    if (grid_shape.size() != wrapper.get_number_of_parameters()) [[unlikely]] {
      throw std::runtime_error("Internal error: invalid fixed rank tensor shape dispatch.");
    }
    std::pair<std::vector<tensor_dtype>, std::vector<indices_type>> tensor;
    {
      nb::gil_scoped_release release;
      tensor = Gudhi::multi_persistence::compute_rank_invariant<tensor_dtype, indices_type>(
          wrapper.get_slicer(), grid_shape, degrees, ignore_inf, n_jobs);
    }
    return nanobind::make_tuple(_wrap_as_numpy_array(std::move(tensor.first), tensor.first.size()),
                                _wrap_as_numpy_array(std::move(tensor.second), tensor.second.size()));
  });
}

template <typename... Ds>
inline nb::tuple compute_rank_signed_measure_sparse(type_list<Ds...>,
                                                    nb::handle slicer,
                                                    const std::vector<indices_type>& grid_shape,
                                                    const std::vector<indices_type>& degrees,
                                                    bool zero_pad,
                                                    indices_type n_jobs,
                                                    bool ignore_inf) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::tuple {
    auto& wrapper = nb::cast<typename D::interface&>(slicer);
    if (grid_shape.size() != wrapper.get_number_of_parameters()) [[unlikely]] {
      throw std::runtime_error("Internal error: invalid fixed rank tensor shape dispatch.");
    }
    Gudhi::multi_persistence::Signed_measure<indices_type, tensor_dtype> sm;
    {
      nb::gil_scoped_release release;
      sm = Gudhi::multi_persistence::compute_sparse_rank_signed_measure<tensor_dtype, indices_type>(
          wrapper.get_slicer(), grid_shape, degrees, zero_pad, ignore_inf, n_jobs);
    }
    return nanobind::make_tuple(
        _wrap_as_numpy_array(
            std::move(sm.coordinates), sm.coordinates.size() / sm.numberOfParameters, sm.numberOfParameters),
        _wrap_as_numpy_array(std::move(sm.values), sm.values.size()));
  });
}

template <typename Desc>
inline Gudhi::multi_persistence::Module_interface<double> module_approximation_from_desc(
    typename Desc::interface& wrapper,
    const std::vector<double>& direction,
    double max_error,
    Gudhi::multi_persistence::Box<double> box,
    bool threshold,
    bool complete,
    bool verbose,
    int n_jobs) {
  if constexpr (!Desc::enable_module_approximation) {
    throw std::runtime_error("Unsupported slicer type for module approximation.");
  } else {
    Gudhi::multi_persistence::Module<double> mod;
    {
      nb::gil_scoped_release release;
      mod = Gudhi::multi_persistence::multiparameter_module_approximation(wrapper.get_slicer(),
                                                                          max_error,
                                                                          box.get_lower_corner(),
                                                                          box.get_upper_corner(),
                                                                          direction,
                                                                          threshold,
                                                                          complete,
                                                                          verbose,
                                                                          n_jobs);
    }
    return {std::move(mod), box};
  }
}

template <typename... Ds>
inline Gudhi::multi_persistence::Module_interface<double> compute_module_approximation_from_slicer(
    type_list<Ds...>,
    nb::handle slicer,
    const std::vector<double>& direction,
    double max_error,
    Gudhi::multi_persistence::Box<double> box,
    bool threshold,
    bool complete,
    bool verbose,
    int n_jobs) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type for module approximation.");
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer),
                                        [&]<typename D>() -> Gudhi::multi_persistence::Module_interface<double> {
                                          auto& wrapper = nb::cast<typename D::interface&>(slicer);
                                          return module_approximation_from_desc<D>(
                                              wrapper, direction, max_error, box, threshold, complete, verbose, n_jobs);
                                        });
}

}  // namespace mpnb

#endif  // MP_PY_SLICER_ALGO_NANOBIND_H_INCLUDED


