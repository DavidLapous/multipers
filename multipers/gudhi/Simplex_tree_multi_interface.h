/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which
 *is released under MIT. See file LICENSE or go to
 *https://gudhi.inria.fr/licensing/ for full license details. Author(s): Vincent
 *Rouvreau
 *
 *    Copyright (C) 2016 Inria
 *
 *    Modification(s):
 *		- 2022/11 David Loiseaux, Hannah Schreiber : adapt for
 *multipersistence.
 *      - YYYY/MM Author: Description of the modification
 */

#pragma once

#include "Simplex_tree_interface.h"
#include <cstdint>
#include <gudhi/Simplex_tree/Simplex_tree_multi.h>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>
#include "multiparameter_module_approximation/format_python-cpp.h"

#include <iostream>
#include <utility> // std::pair
#include <vector>

namespace Gudhi::multiparameter::python_interface {

using interface_std =
    Simplex_tree<Simplex_tree_options_for_python>; // Interface not necessary
                                                   // (smaller so should do less
                                                   // segfaults)

template <class simplextreeinterface>
simplextreeinterface &
get_simplextree_from_pointer(const uintptr_t splxptr) { // DANGER
  simplextreeinterface &st = *(simplextreeinterface *)(splxptr);
  return st;
}
template <typename Filtration, typename value_type = typename Filtration::value_type>
class Simplex_tree_multi_interface
    : public Simplex_tree_interface<
          Simplex_tree_options_multidimensional_filtration<Filtration>> {
public:
  using SimplexTreeOptions =
      Simplex_tree_options_multidimensional_filtration<Filtration>;
  using Python_filtration_type =
      std::vector<typename SimplexTreeOptions::value_type>; // TODO :
                                                            // std::conditional
  using Base = Simplex_tree<SimplexTreeOptions>;
  using Filtration_value = typename Base::Filtration_value;
  using Vertex_handle = typename Base::Vertex_handle;
  using Simplex_handle = typename Base::Simplex_handle;
  using Insertion_result = typename std::pair<Simplex_handle, bool>;
  using Simplex = std::vector<Vertex_handle>;
  using Simplex_and_filtration =
      std::pair<Simplex, Filtration_value*>;
  using Filtered_simplices = std::vector<Simplex_and_filtration>;
  using Skeleton_simplex_iterator = typename Base::Skeleton_simplex_iterator;
  using Complex_simplex_iterator = typename Base::Complex_simplex_iterator;
  using Extended_filtration_data = typename Base::Extended_filtration_data;
  using Boundary_simplex_iterator = typename Base::Boundary_simplex_iterator;
  typedef bool (*blocker_func_t)(Simplex simplex, void *user_data);
  using euler_chars_type = std::vector<int>;

public:
  static_assert(std::is_same<typename Base::Filtration_value, Filtration_value>::value,
                "value_type has to be the same as Filtration::value_type. This is only a hack for python");
  Extended_filtration_data efd;

  bool find_simplex(const Simplex &simplex) {
    return (Base::find(simplex) != Base::null_simplex());
  }

  void assign_simplex_filtration(const Simplex &simplex,
                                 const Filtration_value &filtration) {
    Base::assign_filtration(Base::find(simplex), filtration);
    Base::clear_filtration();
  }

  bool insert(const Simplex &simplex, const Filtration_value &filtration) {
    Insertion_result result =
        Base::insert_simplex_and_subfaces(simplex, filtration);
    if (result.first != Base::null_simplex())
      Base::clear_filtration();
    return (result.second);
  }

  // Do not interface this function, only used in alpha complex interface for
  // complex creation
  bool insert_simplex(const Simplex &simplex,
                      const Filtration_value &filtration) {
    Insertion_result result = Base::insert_simplex(simplex, filtration);
    return (result.second);
  }
  //   bool insert_simplex(const Simplex& simplex, const Python_filtration_type&
  //   filtration ) {
  // 	Filtration_value& filtration_ = *(Filtration_value*)(&filtration); //
  // Jardinage for no copy. 	Insertion_result result =
  // Base::insert_simplex(simplex, filtration); 	return (result.second);
  //   }

  // Do not interface this function, only used in interface for complex creation
  bool insert_simplex_and_subfaces(const Simplex &simplex,
                                   const Filtration_value &filtration) {
    Insertion_result result =
        Base::insert_simplex_and_subfaces(simplex, filtration);
    return (result.second);
  }
  //   bool insert_simplex_and_subfaces(const Simplex& simplex, const
  //   Python_filtration_type& filtration ) {
  // 	Filtration_value& filtration_ = *(Filtration_value*)(&filtration); //
  // Jardinage for no copy. 	Insertion_result result =
  // Base::insert_simplex_and_subfaces(simplex, filtration); 	return
  // (result.second);
  //   }

  // Do not interface this function, only used in strong witness interface for
  // complex creation
  bool insert_simplex(const std::vector<std::size_t> &simplex,
                       const typename Filtration_value::OneCritical &filtration) {
    Insertion_result result = Base::insert_simplex(simplex, filtration);
    return (result.second);
  }

  // Do not interface this function, only used in strong witness interface for
  // complex creation
  bool insert_simplex_and_subfaces(const std::vector<std::size_t> &simplex,
                                   const Filtration_value &filtration) {
    Insertion_result result =
        Base::insert_simplex_and_subfaces(simplex, filtration);
    return (result.second);
  }
  typename SimplexTreeOptions::Filtration_value *
  simplex_filtration(const Simplex &simplex) {
    auto &filtration = Base::filtration_mutable(Base::find(simplex));
    return &filtration; // We return the pointer to get a numpy view
                           // afterward
  }

  Simplex_and_filtration get_simplex_and_filtration(Simplex_handle f_simplex) {
    // Simplex simplex;
    // for (auto vertex : Base::simplex_vertex_range(f_simplex)) {
    // //   simplex.insert(simplex.begin(), vertex); // why not push back ?
    // }
    auto it = Base::simplex_vertex_range(f_simplex);
    Simplex simplex(it.begin(), it.end());
    std::reverse(simplex.begin(), simplex.end());
    return std::make_pair(std::move(simplex),
                          &Base::filtration_mutable(f_simplex));
  }

  Filtered_simplices get_star(const Simplex &simplex) {
    Filtered_simplices star;
    for (auto f_simplex : Base::star_simplex_range(Base::find(simplex))) {
      Simplex simplex_star;
      for (auto vertex : Base::simplex_vertex_range(f_simplex)) {
        simplex_star.insert(simplex_star.begin(), vertex);
      }
      star.push_back(std::make_pair(simplex_star,
                                    &Base::filtration_mutable(f_simplex)));
    }
    return star;
  }

  Filtered_simplices get_cofaces(const Simplex &simplex, int dimension) {
    Filtered_simplices cofaces;
    for (auto f_simplex :
         Base::cofaces_simplex_range(Base::find(simplex), dimension)) {
      Simplex simplex_coface;
      for (auto vertex : Base::simplex_vertex_range(f_simplex)) {
        simplex_coface.insert(simplex_coface.begin(), vertex);
      }
      cofaces.push_back(std::make_pair(
          simplex_coface, &Base::filtration_mutable(f_simplex)));
    }
    return cofaces;
  }

  void compute_extended_filtration() {
    throw std::logic_error("Incompatible with multipers");
  }

  Simplex_tree_multi_interface *collapse_edges(int nb_collapse_iteration) {
    throw std::logic_error("Incompatible with multipers");
  }

  // ######################## MULTIPERS STUFF
  void set_keys_to_enumerate() {
    int count = 0;
    for (auto sh : Base::filtration_simplex_range())
      Base::assign_key(sh, count++);
  }

  int get_key(const Simplex &simplex) { return Base::key(Base::find(simplex)); }

  void set_key(const Simplex &simplex, int key) {
    Base::assign_key(Base::find(simplex), key);
    return;
  }

  // Fills a parameter with a lower-star filtration
  void
  fill_lowerstar(const std::vector<typename Filtration::value_type> &filtration,
                 int axis) {
    /* constexpr value_type minus_inf =
     * -1*std::numeric_limits<value_type>::infinity(); */
    std::vector<value_type> filtration_values_of_vertex;
    for (auto &SimplexHandle : Base::complex_simplex_range()) {
      auto &current_birth = Base::filtration_mutable(SimplexHandle);
      /* value_type to_assign = minus_inf; */
      filtration_values_of_vertex.clear();
      for (auto vertex : Base::simplex_vertex_range(SimplexHandle)) {
        /* to_assign = std::max(filtration[vertex], to_assign); */
        if (std::isnan(filtration[vertex]))
          std::cerr << "Invalid filtration for vertex " << vertex << " !!"
                    << std::endl;
        filtration_values_of_vertex.push_back(filtration[vertex]);
      }
      value_type to_assign =
          *std::max_element(filtration_values_of_vertex.begin(),
                            filtration_values_of_vertex.end());
      /* if (to_assign >10 || to_assign < -10 ) */
      /*   std::cout <<"to_assign : "<< to_assign << std::endl; */
      current_birth[axis] = to_assign;
      // Base::assign_filtration(SimplexHandle, current_birth);
    }
  }

  using simplices_list = std::vector<std::vector<int>>;
  simplices_list get_simplices_of_dimension(int dimension) {
    simplices_list simplex_list;
    simplex_list.reserve(Base::num_simplices());
    for (auto simplexhandle : Base::skeleton_simplex_range(dimension)) {
      if (Base::dimension(simplexhandle) == dimension) {
        std::vector<int> simplex;
        simplex.reserve(dimension + 1);
        for (int vertex : Base::simplex_vertex_range(simplexhandle))
          simplex.push_back(vertex);
        simplex_list.push_back(simplex);
      }
    }
    /*	simplex_list.shrink_to_fit();*/
    return simplex_list;
  }
  using edge_list =
      std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>>;
  edge_list get_edge_list() {
    edge_list simplex_list;
    simplex_list.reserve(Base::num_simplices());
    for (auto &simplexHandle : Base::skeleton_simplex_range(1)) {
      if (Base::dimension(simplexHandle) == 1) {
        std::pair<int, int> simplex;
        auto it = Base::simplex_vertex_range(simplexHandle).begin();
        simplex = {*it, *(++it)};
        const auto &f = Base::filtration(simplexHandle);
        simplex_list.push_back({simplex, {static_cast<double>(f[0]), static_cast<double>(f[1])}});
      }
    }
    return simplex_list;
  }
  void
  resize_all_filtrations(int num) { 
    if (num < 0)
      return;
    for (const auto &SimplexHandle : Base::complex_simplex_range()) {
      if constexpr (Filtration::is_multi_critical){
        for (auto& stuff : Base::filtration_mutable(SimplexHandle))
          stuff.resize(num);
      }
      else
        Base::filtration_mutable(SimplexHandle).resize(num);
    }
  }
  void from_std(intptr_t ptr, int dimension, const Filtration &default_values) {
    interface_std &st = get_simplextree_from_pointer<interface_std>(ptr);
    multify(st, *this, dimension, default_values);
  }
  template <typename Line_like>
  void to_std(intptr_t ptr, const Line_like &line, int dimension) {
    interface_std &st = get_simplextree_from_pointer<interface_std>(ptr);

    for (const auto &simplex_handle : this->complex_simplex_range()) {
      std::vector<int> simplex;
      for (auto vertex : this->simplex_vertex_range(simplex_handle))
        simplex.push_back(vertex);

      // Filtration as double
      const auto &f = this->filtration(simplex_handle).template astype<typename interface_std::Filtration_value>();

      typename interface_std::Filtration_value new_filtration = line.push_forward(f)[dimension];
      st.insert_simplex(simplex, new_filtration);
    }
  }
  void to_std_linear_projection(intptr_t ptr, std::vector<double> linear_form) {
    interface_std &st = get_simplextree_from_pointer<interface_std>(ptr);
    linear_projection(st, *this, linear_form);
  }

  void squeeze_filtration_inplace(const std::vector<std::vector<double>> &grid,
                          const bool coordinate_values = true) {
    std::size_t num_parameters = this->get_number_of_parameters();
    if (grid.size() != num_parameters) {
      throw std::invalid_argument(
          "Grid and simplextree do not agree on number of parameters.");
    }
    for (const auto &simplex_handle : this->complex_simplex_range()) {
      auto &simplex_filtration = this->filtration_mutable(simplex_handle);
      const auto& coords = simplex_filtration.coordinates_in_grid(grid);
      if (coordinate_values) 
        simplex_filtration = coords.template astype<value_type>();
      else 
        simplex_filtration = coords.evaluate_in_grid(grid).template astype<value_type>();
    }
  }
  
  void squeeze_filtration(
      const intptr_t outptr,
      const std::vector<std::vector<double>> &grid) { // TODO : this is const but GUDHI
    const bool verbose = false;
    using st_coord_type = Simplex_tree_multi_interface<typename Filtration::template Base<std::int32_t>, int32_t> ;
    st_coord_type& out = *(st_coord_type*) outptr; // TODO : maybe fix this.
    std::vector<int> simplex_vertex;
    for (const auto &simplex_handle : this->complex_simplex_range()) {
      const auto &simplex_filtration = this->filtration(simplex_handle);
      if constexpr (verbose)
         std::cout << "Filtration " << simplex_filtration << "\n";
      const auto& coords = simplex_filtration.coordinates_in_grid(grid);
      if constexpr (verbose)
         std::cout << "Coords " << coords << "\n";
      for (auto s : this->simplex_vertex_range(simplex_handle))
        simplex_vertex.push_back(s);
      out.insert_simplex(simplex_vertex, coords);
      if constexpr (verbose)
         std::cout << "Coords in st" << out.filtration(out.find(simplex_vertex)) << std::endl;
      simplex_vertex.clear();
    }
  }


  std::vector<std::vector<std::vector<value_type>>> // dim, pts, param
  get_filtration_values(const std::vector<int> &degrees) {
    using multi_filtration_grid = std::vector<std::vector<value_type>>;
    int num_parameters = this->get_number_of_parameters();
    std::vector<multi_filtration_grid> out(
        degrees.size(), multi_filtration_grid(num_parameters));
    std::vector<int> degree_index(this->dimension() + 1);
    int count = 0;
    for (auto degree : degrees) {
      degree_index[degree] = count++;
      out[degree_index[degree]].reserve(this->num_simplices());
    }

    for (const auto &simplex_handle : this->complex_simplex_range()) {
      const auto &filtration = this->filtration(simplex_handle);
      const auto degree = this->dimension(simplex_handle);
      if (std::find(degrees.begin(), degrees.end(), degree) == degrees.end())
        continue;
      // for (int parameter = 0; parameter < num_parameters; parameter++) {
      //   out[degree_index[degree]][parameter].push_back(filtration[parameter]);
      // }
      if constexpr (Filtration::is_multi_critical)
        for (std::size_t i = 0; i < filtration.num_generators(); i++)
          for (int parameter = 0; parameter < num_parameters; parameter++)
            out[degree_index[degree]][parameter].push_back(filtration[i][parameter]);
      else
          for (int parameter = 0; parameter < num_parameters; parameter++)
            out[degree_index[degree]][parameter].push_back(filtration[parameter]);
    }
    return out;
  }

  using boundary_type = std::vector<unsigned int>;
  using boundary_matrix = std::vector<boundary_type>;

  using scc_type = mma::scc_type<SimplexTreeOptions>;
  scc_type inline simplextree_to_scc() { return Gudhi::multiparameter::mma::simplextree_to_scc(*this); }
  using kscc_type = mma::kscc_type<SimplexTreeOptions>;
  kscc_type inline kcritical_simplextree_to_scc() {
    if constexpr (SimplexTreeOptions::Filtration_value::is_multi_critical)
      return Gudhi::multiparameter::mma::kcritical_simplextree_to_scc(*this);
    else
      return {};
  }

  using function_scc_type =
      std::vector<std::pair<std::vector<std::vector<std::vector<value_type>>>,
                            boundary_matrix>>;
  function_scc_type inline function_simplextree_to_scc() {
    return Gudhi::multiparameter::mma::function_simplextree_to_scc(*this);

}


using flattened_scc_type =
    std::pair<std::vector<std::vector<value_type>>,
              std::vector<std::vector<unsigned int>>>;

flattened_scc_type inline simplextree_to_ordered_bf(){
  return Gudhi::multiparameter::mma::simplextree_to_ordered_bf(*this);
  }

  // Diff / grid stuff

  using idx_map_type = std::vector<std::map<typename Filtration::value_type, int32_t>>;
  inline idx_map_type build_idx_map(const std::vector<int> &simplices_dimensions){
    static_assert(!Filtration::is_multi_critical, "Multicritical not supported yet");
  auto num_parameters = this->get_number_of_parameters();
  if (static_cast<int>(simplices_dimensions.size()) < num_parameters)
    throw;
  int max_dim = *std::max_element(simplices_dimensions.begin(),
                                  simplices_dimensions.end());
  int min_dim = *std::min_element(simplices_dimensions.begin(),
                                  simplices_dimensions.end());
  max_dim = min_dim >= 0 ? max_dim : this->dimension();

  idx_map_type idx_map(num_parameters);
  auto splx_idx = 0u;
  for (auto sh :
       this->complex_simplex_range()) { // order has to be retrieved later, so I'm
                                     // not sure that skeleton iterator is well
                                     // suited
    const auto &splx_filtration = this->filtration(sh);
    const auto splx_dim = this->dimension(sh);
    if (splx_dim <= max_dim)
      for (auto i = 0u; i < splx_filtration.size(); i++) {
        if (simplices_dimensions[i] != splx_dim and
            simplices_dimensions[i] != -1)
          continue;
        auto f = splx_filtration[i];
        idx_map[i].try_emplace(f, splx_idx);
      }
    splx_idx++;
  }
  return idx_map;
  };

  using pts_indices_type = std::vector<std::vector<int32_t>>;
  static std::pair<pts_indices_type, pts_indices_type> inline get_pts_indices(
      const idx_map_type &idx_map,
      const std::vector<std::vector<typename Filtration::value_type>> &pts) {
    static_assert(!Filtration::is_multi_critical, "Multicritical not supported yet");
    std::size_t num_pts = pts.size();
    std::size_t num_parameters = idx_map.size();
    pts_indices_type out_indices(
        num_pts, std::vector<int32_t>(num_parameters,
                                      -1)); // -1 to be able from indicies to
                                            // get if the pt is found or not
    pts_indices_type out_unmapped_values;
    for (auto pt_idx = 0u; pt_idx < num_pts; pt_idx++) {
      auto &pt = pts[pt_idx];
      auto &pt_indices = out_indices[pt_idx];

      for (std::size_t parameter = 0u; parameter < num_parameters;
           parameter++) {
        value_type f = pt[parameter];
        const std::map<value_type, int32_t> &parameter_map = idx_map[parameter];
        auto it = parameter_map.find(f);
        if (it == parameter_map.end())
          out_unmapped_values.push_back(
              {static_cast<int32_t>(pt_idx), static_cast<int32_t>(parameter)});
        else
          pt_indices[parameter] = it->second;
      }
    }
    return {out_indices, out_unmapped_values}; // TODO return a ptr for python
  }

  std::pair<pts_indices_type, pts_indices_type> inline pts_to_indices(
      const std::vector<std::vector<typename Filtration::value_type>> &pts,
      const std::vector<int> &simplices_dimensions) {
    return get_pts_indices(this->build_idx_map(simplices_dimensions), pts);
  }
};

template <typename Filtration>
using interface_multi = Simplex_tree_multi_interface<
    Simplex_tree_options_multidimensional_filtration<Filtration>>;

// Wrappers of the functions in Simplex_tree_multi.h, to deal with the "pointer
// only" python interface
template <typename Filtration>
void inline flatten_diag_from_ptr(
    const uintptr_t splxptr, const uintptr_t newsplxptr,
    const std::vector<typename Filtration::value_type> basepoint,
    int dimension) { // for python
  auto &st = get_simplextree_from_pointer<interface_std>(newsplxptr);
  auto &st_multi =
      get_simplextree_from_pointer<interface_multi<Filtration>>(splxptr);
  flatten_diag(st, st_multi, basepoint, dimension);
}
template <typename Filtration>
void inline multify_from_ptr(uintptr_t splxptr, uintptr_t newsplxptr,
                             const int dimension,
                             const Filtration &default_values) { // for python
  auto &st = get_simplextree_from_pointer<interface_std>(splxptr);
  auto &st_multi =
      get_simplextree_from_pointer<interface_multi<Filtration>>(newsplxptr);
  multify(st, st_multi, dimension, default_values);
}
template <typename Filtration>
void inline flatten_from_ptr(uintptr_t splxptr, uintptr_t newsplxptr,
                             const int dimension = 0) { // for python
  auto &st = get_simplextree_from_pointer<interface_std>(newsplxptr);
  auto &st_multi =
      get_simplextree_from_pointer<interface_multi<Filtration>>(splxptr);
  flatten(st, st_multi, dimension);
}

template <typename Filtration, typename... Args>
void inline linear_projection_from_ptr(const uintptr_t ptr,
                                       const uintptr_t ptr_multi,
                                       Args... args) {
  auto &st = get_simplextree_from_pointer<interface_std>(ptr);
  auto &st_multi =
      get_simplextree_from_pointer<interface_multi<Filtration>>(ptr_multi);
  linear_projection(st, st_multi, args...);
}

template <typename Filtration, typename... Args>
void inline squeeze_filtration_from_ptr(uintptr_t splxptr, Args... args) {
  Simplex_tree<options_multi<Filtration>> &st_multi =
      *(Gudhi::Simplex_tree<options_multi<Filtration>> *)(splxptr);
  squeeze_filtration(st_multi, args...);
  return;
}

template <typename Filtration, typename... Args>
inline std::vector<multi_filtration_grid<Filtration>>
get_filtration_values_from_ptr(uintptr_t splxptr, Args... args) {
  Simplex_tree<options_multi<Filtration>> &st_multi =
      *(Gudhi::Simplex_tree<options_multi<Filtration>> *)(splxptr);
  return get_filtration_values(st_multi, args...);
}

// Final types
//
//

template <typename Filtration>
using Simplex_tree_multi_simplex_handle = typename
    Simplex_tree_multi_interface<Filtration>::Simplex_handle;

template <typename Filtration>
using Simplex_tree_multi_simplices_iterator = typename 
    Simplex_tree_multi_interface<Filtration>::Complex_simplex_iterator;
template <typename Filtration>
using Simplex_tree_multi_skeleton_iterator = typename 
    Simplex_tree_multi_interface<Filtration>::Skeleton_simplex_iterator;
template <typename Filtration>
using Simplex_tree_multi_boundary_iterator = typename 
    Simplex_tree_multi_interface<Filtration>::Boundary_simplex_iterator;
} // namespace Gudhi::multiparameter::python_interface
