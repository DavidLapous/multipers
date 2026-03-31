#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ext_interface/nanobind_registry_helpers.hpp"
#include "nanobind_array_utils.hpp"
#include "nanobind_dense_array_utils.hpp"
#include "multi_parameter_rank_invariant/function_rips.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpfrn {

using tensor_dtype = int32_t;
using indices_type = int32_t;

using multipers::nanobind_dense_utils::cast_vector_from_array;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_const_simplextree_wrapper;
using multipers::nanobind_utils::owned_array;

template <typename Func>
decltype(auto) visit_simplextree(const nb::handle& simplextree, Func&& func) {
  if (!is_simplextree_object(simplextree)) {
    throw nb::type_error("Expected a multipers SimplexTreeMulti.");
  }
  return visit_const_simplextree_wrapper(simplextree, std::forward<Func>(func));
}

inline indices_type grid_rows(const nb::handle& simplextree) {
  return visit_simplextree(simplextree, [&]<typename Desc>(const auto& wrapper) -> indices_type {
    nb::ndarray<> axis = nb::cast<nb::ndarray<>>(wrapper.filtration_grid[0]);
    return static_cast<indices_type>(axis.shape(0));
  });
}

inline indices_type num_parameters(const nb::handle& simplextree) {
  return visit_simplextree(simplextree, [&]<typename Desc>(const auto& wrapper) -> indices_type {
    return static_cast<indices_type>(wrapper.tree.num_parameters());
  });
}

inline intptr_t simplextree_ptr(const nb::handle& simplextree) {
  return visit_simplextree(simplextree, [&]<typename Desc>(const auto& wrapper) -> intptr_t {
    return reinterpret_cast<intptr_t>(&wrapper.tree);
  });
}

inline std::vector<indices_type> flatten_points(const std::vector<std::vector<indices_type>>& points,
                                                size_t num_columns) {
  std::vector<indices_type> flat;
  flat.reserve(points.size() * num_columns);
  for (const auto& row : points) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

}  // namespace mpfrn

NB_MODULE(_function_rips_nanobind, m) {
  m.def(
      "get_degree_rips",
      [](nb::object target,
         nb::ndarray<nb::numpy, const int8_t, nb::ndim<1>, nb::c_contig> gudhi_state,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> degrees) -> nb::object {
        auto degree_vector = mpfrn::cast_vector_from_array<int>(degrees);
        auto target_ptr = mpfrn::simplextree_ptr(target);
        {
          nb::gil_scoped_release release;
          Gudhi::multiparameter::function_rips::get_degree_rips_st_python(
              reinterpret_cast<const char*>(gudhi_state.data()), gudhi_state.shape(0), target_ptr, degree_vector);
        }
        return target;
      },
      "target"_a,
      "gudhi_state"_a,
      "degrees"_a);

  m.def(
      "function_rips_surface",
      [](nb::object simplextree,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> homological_degrees,
         bool mobius_inversion,
         bool zero_pad,
         mpfrn::indices_type n_jobs) {
        auto degree_vector = mpfrn::cast_vector_from_array<mpfrn::indices_type>(homological_degrees);
        const auto I = mpfrn::grid_rows(simplextree);
        const auto J = mpfrn::num_parameters(simplextree);
        std::vector<mpfrn::tensor_dtype> container(static_cast<size_t>(degree_vector.size()) * I * J, 0);
        {
          nb::gil_scoped_release release;
          Gudhi::multiparameter::function_rips::compute_function_rips_surface_python<mpfrn::tensor_dtype,
                                                                                     mpfrn::indices_type>(
              mpfrn::simplextree_ptr(simplextree),
              container.data(),
              degree_vector,
              I,
              J,
              mobius_inversion,
              zero_pad,
              n_jobs);
        }
        return mpfrn::owned_array<mpfrn::tensor_dtype>(
            std::move(container), {degree_vector.size(), static_cast<size_t>(I), static_cast<size_t>(J)});
      },
      "simplextree"_a,
      "homological_degrees"_a,
      "mobius_inversion"_a = true,
      "zero_pad"_a = false,
      "n_jobs"_a = 0);

  m.def(
      "function_rips_signed_measure",
      [](nb::object simplextree,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> homological_degrees,
         bool mobius_inversion,
         bool zero_pad,
         mpfrn::indices_type n_jobs) {
        auto degree_vector = mpfrn::cast_vector_from_array<mpfrn::indices_type>(homological_degrees);
        const auto I = mpfrn::grid_rows(simplextree);
        const auto J = mpfrn::num_parameters(simplextree);
        std::vector<mpfrn::tensor_dtype> container(static_cast<size_t>(degree_vector.size()) * I * J, 0);
        std::pair<std::vector<std::vector<mpfrn::indices_type>>, std::vector<mpfrn::tensor_dtype>> out;
        {
          nb::gil_scoped_release release;
          out = Gudhi::multiparameter::function_rips::compute_function_rips_signed_measure_python<mpfrn::tensor_dtype,
                                                                                                  mpfrn::indices_type>(
              mpfrn::simplextree_ptr(simplextree),
              container.data(),
              degree_vector,
              I,
              J,
              mobius_inversion,
              zero_pad,
              n_jobs);
        }
        const size_t num_columns = out.first.empty() ? 3u : out.first.front().size();
        auto flat_points = mpfrn::flatten_points(out.first, num_columns);
        return nb::make_tuple(
            mpfrn::owned_array<mpfrn::indices_type>(std::move(flat_points), {out.first.size(), num_columns}),
            mpfrn::owned_array<mpfrn::tensor_dtype>(std::move(out.second), {out.second.size()}));
      },
      "simplextree"_a,
      "homological_degrees"_a,
      "mobius_inversion"_a = true,
      "zero_pad"_a = false,
      "n_jobs"_a = 0);
}
