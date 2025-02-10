#pragma once

#include "Simplex_tree_multi_interface.h"
#include <cstdint>
#include <vector>

namespace Gudhi {
namespace multiparameter {
namespace differentiation {

using signed_measure_indices = std::vector<std::vector<int32_t>>;
template <typename Filtration>
using interface_multi = python_interface::interface_multi<Filtration>;
template <typename Filtration>
using signed_measure_pts = std::vector<std::vector<typename Filtration::value_type>>;
template <typename Filtration>
using idx_map_type = std::vector<std::map<typename Filtration::value_type, int32_t>>;

// O(num_simplices)
template <typename Filtration>
idx_map_type<Filtration> build_idx_map(interface_multi<Filtration> &st, const std::vector<int> &simplices_dimensions) {
  auto num_parameters = st.get_number_of_parameters();
  if (static_cast<int>(simplices_dimensions.size()) < num_parameters) throw;
  int max_dim = *std::max_element(simplices_dimensions.begin(), simplices_dimensions.end());
  int min_dim = *std::min_element(simplices_dimensions.begin(), simplices_dimensions.end());
  max_dim = min_dim >= 0 ? max_dim : st.dimension();

  idx_map_type<Filtration> idx_map(num_parameters);
  auto splx_idx = 0u;
  for (auto sh : st.complex_simplex_range()) {  // order has to be retrieved later, so I'm
                                                // not sure that skeleton iterator is well
                                                // suited
    const auto &splx_filtration = st.filtration(sh);
    const auto splx_dim = st.dimension(sh);
    if (splx_dim <= max_dim)
      for (auto i = 0u; i < splx_filtration.size(); i++) {
        if (simplices_dimensions[i] != splx_dim and simplices_dimensions[i] != -1) continue;
        auto f = splx_filtration[i];
        idx_map[i].try_emplace(f, splx_idx);
      }
    splx_idx++;
  }
  return idx_map;
}

template <typename Filtration, typename... Args>
idx_map_type<Filtration> build_idx_map(const intptr_t simplextree_ptr, Args... args) {
  auto &st_multi = get_simplextree_from_pointer<interface_multi<Filtration>>(simplextree_ptr);
  return build_idx_map(st_multi, args...);
}

// O(signed_measure_size*num_parameters)
template <typename Filtration>
std::pair<signed_measure_indices, signed_measure_indices> get_signed_measure_indices(
    const idx_map_type<Filtration> &idx_map,
    const signed_measure_pts<Filtration> &pts) {
  using value_type = typename Filtration::value_type;
  std::size_t num_pts = pts.size();
  std::size_t num_parameters = idx_map.size();
  signed_measure_indices out_indices(
      num_pts,
      std::vector<int32_t>(num_parameters,
                           -1));  // -1 to be able from indicies to get if the pt is found or not
  signed_measure_indices out_unmapped_values;
  for (auto pt_idx = 0u; pt_idx < num_pts; pt_idx++) {
    auto &pt = pts[pt_idx];
    auto &pt_indices = out_indices[pt_idx];

    for (std::size_t parameter = 0u; parameter < num_parameters; parameter++) {
      value_type f = pt[parameter];
      const std::map<value_type, int32_t> &parameter_map = idx_map[parameter];
      auto it = parameter_map.find(f);
      if (it == parameter_map.end())
        out_unmapped_values.push_back({static_cast<int32_t>(pt_idx), static_cast<int32_t>(parameter)});
      else
        pt_indices[parameter] = it->second;
    }
  }
  return {out_indices, out_unmapped_values};  // TODO return a ptr for python
}

}  // namespace differentiation
}  // namespace multiparameter
}  // namespace Gudhi
