#include <cstdint>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <gudhi/multi_simplex_tree_helpers.h>
#include <python_interfaces/numpy_utils.h>

#include "ext_interface/nanobind_registry_helpers.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpfrn {

using multipers::nanobind_helpers::visit_simplextree_wrapper;

using value_type = typename Gudhi::multi_persistence::Simplex_tree_std::Filtration_value;
using flat_multifiltration = multipers::tmp_interface::Degree_rips_bifiltration<value_type>;
using degree_rips_interface = Gudhi::multi_persistence::Multi_simplex_tree_interface<flat_multifiltration>;

template <typename simplex_tree_type>
inline void get_degree_rips_st_python(const char *buffer_start,
                                      const std::size_t buffer_size,
                                      simplex_tree_type &st_multi_python_container,
                                      nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> degrees) {
  typename simplex_tree_type::Base st_multi;
  {
    nanobind::gil_scoped_release release;
    Gudhi::multi_persistence::Simplex_tree_std st_std;
    st_std.deserialize(buffer_start, buffer_size);
    if (st_std.dimension() != 1)
      throw std::invalid_argument("only works for 1-dimensional simplex trees.");
    st_multi = Gudhi::multi_persistence::build_degree_filtration<typename simplex_tree_type::Base>(st_std,
                                                                                                   Numpy_span(degrees));
  }
  st_multi_python_container = std::move(st_multi);
}

}  // namespace mpfrn

NB_MODULE(_function_rips_nanobind, m) {
  m.def(
      "get_degree_rips",
      [](nb::object target,
         nb::ndarray<nb::numpy, const int8_t, nb::ndim<1>, nb::c_contig> gudhi_state,
         nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> degrees) -> nb::object {
        mpfrn::visit_simplextree_wrapper(target, [&]<typename Desc>(auto &wrapper) {
          using Interface = typename Desc::interface_type;
          if constexpr (std::is_same_v<Interface, mpfrn::degree_rips_interface>) {
            mpfrn::get_degree_rips_st_python(
                reinterpret_cast<const char *>(gudhi_state.data()), gudhi_state.shape(0), wrapper, degrees);
          } else {
            throw nb::type_error("get_degree_rips expects the canonical degree-Rips SimplexTreeMulti target.");
          }
        });
        return target;
      },
      "target"_a,
      "gudhi_state"_a,
      "degrees"_a);
}
