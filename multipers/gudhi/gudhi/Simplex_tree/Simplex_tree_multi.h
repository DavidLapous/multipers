/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which
 * is released under MIT. See file LICENSE or go to
 * https://gudhi.inria.fr/licensing/ for full license details. Author(s): David
 * Loiseaux, Hannah Schreiber
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */
#ifndef SIMPLEX_TREE_MULTI_H_
#define SIMPLEX_TREE_MULTI_H_

#include <algorithm>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>
#include <gudhi/Simplex_tree/multi_filtrations/Line.h>
#include <tbb/parallel_for.h>

namespace Gudhi::multiparameter {
/** Model of SimplexTreeOptions, with a multiparameter filtration. */
template <typename Filtration>
struct Simplex_tree_options_multidimensional_filtration {
public:
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  using Filtration_value = Filtration;
  typedef typename Filtration::value_type value_type;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = false;
  static const bool link_nodes_by_label = false;
  static const bool stable_simplex_handles = false;
  static const bool is_multi_parameter = true;
};

template <
    typename Filtration =
        typename multi_filtrations::Finitely_critical_multi_filtration<float>>
using options_multi =
    Simplex_tree_options_multidimensional_filtration<Filtration>;
using options_std = Simplex_tree_options_full_featured;
using simplextree_std = Simplex_tree<options_std>;
template <
    typename Filtration =
        typename multi_filtrations::Finitely_critical_multi_filtration<float>>
using simplextree_multi = Simplex_tree<options_multi<Filtration>>;
// template <typename Filtration = typename
// multi_filtrations::Finitely_critical_multi_filtration<float>> using
// value_type =
// Simplex_tree_options_multidimensional_filtration<Filtration>::value_type;
template <
    typename Filtration =
        typename multi_filtrations::Finitely_critical_multi_filtration<float>>
using multi_filtration_type =
    std::vector<typename options_multi<Filtration>::value_type>;
template <
    typename Filtration =
        typename multi_filtrations::Finitely_critical_multi_filtration<float>>
using multi_filtration_grid = std::vector<multi_filtration_type<Filtration>>;

/**
 * \brief Turns a 1-parameter simplextree into a multiparameter simplextree,
 * and keeps the 1-filtration in the 1st axis.
 * Default values can be specified.
 * \ingroup multiparameter
 * \tparam simplextree_std A non-multi simplextree
 * \tparam simplextree_multi A multi simplextree
 * \param st Simplextree to copy
 * \param st_multi Multiparameter simplextree container to fill.
 * \param default_values If given, this vector is assume to be of size
 * `num_parameters-1` and contains the default values of axes `1` to
 * `num_parameters`.
 * */
template <class simplextree_std, class simplextree_multi>
void multify(simplextree_std &st, simplextree_multi &st_multi,
             const int num_parameters,
             const typename simplextree_multi::Options::Filtration_value
                 &default_values = {}) {
  typename simplextree_multi::Options::Filtration_value f(num_parameters);
  static_assert(
      !simplextree_std::Options::is_multi_parameter &&
          simplextree_multi::Options::is_multi_parameter,
      "Can only convert non-multiparameter to multiparameter simplextree.");
  unsigned int num_default_values;
  if constexpr (simplextree_multi::Options::Filtration_value::
                    is_multi_critical) {
    num_default_values = default_values[0].size();
  } else {
    num_default_values = default_values.size();
  }
  for (auto i = 0u; i < std::min(num_default_values,
                                 static_cast<unsigned int>(num_parameters - 1));
       i++)
    if constexpr (simplextree_multi::Options::Filtration_value::
                      is_multi_critical) {
      f[0][i + 1] = default_values[0][i];
    } else {
      f[i + 1] = default_values[i];
    }

  std::vector<int> simplex;
  simplex.reserve(st.dimension() + 1);
  for (auto &simplex_handle : st.complex_simplex_range()) {
    simplex.clear();
    for (auto vertex : st.simplex_vertex_range(simplex_handle))
      simplex.push_back(vertex);

    if (num_parameters > 0) {
      if constexpr (simplextree_multi::Options::Filtration_value::
                        is_multi_critical) {
        f[0][0] = st.filtration(simplex_handle);
      } else {
        f[0] = st.filtration(simplex_handle);
      }
    }
    st_multi.insert_simplex(simplex, f);
  }
  st_multi.set_number_of_parameters(num_parameters);
}

/**
 * \brief Turns a multiparameter-parameter simplextree into a 1-parameter
 * simplextree. \ingroup multiparameter \tparam simplextree_std A non-multi
 * simplextree \tparam simplextree_multi A multi simplextree \param st
 * Simplextree to fill. \param st_multi Multiparameter simplextree to convert
 * into a 1 parameter simplex tree. \param dimension The filtration parameter to
 * put into the 1 parameter simplextree.
 * */
template <class simplextree_std, class simplextree_multi>
void flatten(simplextree_std &st, simplextree_multi &st_multi,
             const int dimension = 0) {
  static_assert(
      !simplextree_std::Options::is_multi_parameter &&
          simplextree_multi::Options::is_multi_parameter,
      "Can only convert multiparameter to non-multiparameter simplextree.");
  for (const auto &simplex_handle : st_multi.complex_simplex_range()) {
    std::vector<int> simplex;
    typename simplextree_multi::Options::value_type f;
    for (auto vertex : st_multi.simplex_vertex_range(simplex_handle))
      simplex.push_back(vertex);
    if constexpr (simplextree_multi::Filtration_value::is_multi_critical) {
      f = dimension >= 0 ? st_multi.filtration(simplex_handle)[0][dimension]
                         : 0;
    } else {
      f = dimension >= 0 ? st_multi.filtration(simplex_handle)[dimension] : 0;
    }
    st.insert_simplex(simplex, f);
  }
}

/**
 * \brief Applies a linear form (given by a scalar product, via Riesz
 * representation) to the filtration values of the multiparameter simplextree to
 * get a 1 parameter simplextree. \ingroup multiparameter \tparam
 * simplextree_std A non-multi simplextree \tparam simplextree_multi A multi
 * simplextree \param st Simplextree, with the same simplicial complex as
 * st_multi, whose filtration has to be filled. \param st_multi Multiparameter
 * simplextree to convert into a 1 parameter simplex tree. \param linear_form
 * the linear form to apply.
 * */
template <class simplextree_std, class simplextree_multi>
void linear_projection(simplextree_std &st, simplextree_multi &st_multi,
                       const std::vector<double> &linear_form) {
  static_assert(
      !simplextree_std::Options::is_multi_parameter &&
          simplextree_multi::Options::is_multi_parameter,
      "Can only convert multiparameter to non-multiparameter simplextree.");
  auto sh = st.complex_simplex_range().begin();
  auto sh_multi = st_multi.complex_simplex_range().begin();
  auto end = st.complex_simplex_range().end();
  typename simplextree_multi::Options::Filtration_value multi_filtration;
  for (; sh != end; ++sh, ++sh_multi) {
    multi_filtration = st_multi.filtration(*sh_multi);
    auto projected_filtration = multi_filtration.linear_projection(linear_form);
    st.assign_filtration(*sh, projected_filtration);
  }
}

/**
 * \brief Pushes the filtration values of a multiparameter simplextree to a
 * diagonal line, to get a 1 parameter simplextree. \ingroup multiparameter
 * \tparam simplextree_std A non-multi simplextree
 * \tparam simplextree_multi A multiparameter simplextree
 * \param st Simplextree to fill.
 * \param st_multi Multiparameter simplextree to convert into a 1 parameter
 * simplex tree. \param basepoint The basepoint of the diagonal line. \param
 * dimension The coordinate of the line to choose as a 1 parameter filtration
 * (they are all equivalent).
 * */
template <class simplextree_std, class simplextree_multi>
void flatten_diag(
    simplextree_std &st, simplextree_multi &st_multi,
    const std::vector<typename simplextree_multi::Options::value_type>
        &basepoint,
    int dimension) {
  static_assert(
      !simplextree_std::Options::is_multi_parameter &&
          simplextree_multi::Options::is_multi_parameter,
      "Can only convert multiparameter to non-multiparameter simplextree.");
  assert(dimension >= 0);
  multi_filtrations::Line<typename simplextree_multi::Options::value_type> l(
      basepoint);
  for (const auto &simplex_handle : st_multi.complex_simplex_range()) {
    std::vector<int> simplex;
    for (auto vertex : st_multi.simplex_vertex_range(simplex_handle))
      simplex.push_back(vertex);

    std::vector<typename simplextree_multi::Options::value_type> f =
        st_multi.filtration(simplex_handle);
    typename simplextree_multi::Options::value_type new_filtration =
        l.push_forward(f)[dimension];
    st.insert_simplex(simplex, new_filtration);
  }
}

/**
 * \brief Given a point on a multiparameter discrete grid, pushes the point onto
 * this grid. Turns the input point as the closest grid point, as coordinates in
 * this grid. \ingroup multiparameter \tparam vector_like Vector like class
 * \param x The point to push on the grid.
 * \param grid The multiparameter grid. A vector of size `num_parameters`, whose
 * elements are the elements of the grid for this axis.
 * */
// DEPRECATED : replaced by Filtration::coordinate_in_grid{,_inplace}
// template <typename U, typename T>
// inline std::vector<int64_t> find_coordinates(const std::vector<U> &x,
//                              const std::vector<std::vector<T>> &grid) {
//   std::vector<int64_t> coordinates(x.size());
//   for (auto parameter = 0u; parameter < grid.size(); parameter++) {
//     // my guess is that it's memory bottlenecked, so parallel not useful.
//     /* tbb::parallel_for( */
//     /*     static_cast<std::size_t>(0u), grid.size(), [&](std::size_t
//     parameter)
//      * { */
//     const auto &filtration = grid[parameter]; // assumes its sorted
//     const T to_project = static_cast<T>(x[parameter]);
//     if constexpr (std::numeric_limits<T>::has_infinity)
//       if (to_project ==
//           std::numeric_limits<T>::infinity()) {
//         coordinates[parameter] =
//             std::numeric_limits<int64_t>::max();
//         continue;
//         /* return; */
//       }
//     if (to_project >= filtration.back()) [[unlikely]] {
//       coordinates[parameter] = filtration.size() - 1;
//       continue;
//     } // deals with infinite value at the end of the grid
//
//     coordinates[parameter] = std::distance(
//         filtration.begin(),
//         std::lower_bound(filtration.begin(), filtration.end(), to_project));
//   }
//     return coordinates;
// }

/**
 * \brief Pushes all of the filtration values of a simplextree onto a grid, c.f.
 * \ref find_coordinates. \ingroup multiparameter \param st_multi Multiparameter
 * simplex tree to squeeze on the grid. \param grid The multiparameter grid. A
 * vector of size `num_parameters`, whose elements are the elements of the grid
 * for this axis. \param coordinate_values If set to true the filtration values
 * will be turned into coordinates in this grid instead of points in this grid.
 * */

template <class simplextree_multi>
void squeeze_filtration(
    simplextree_multi &st_multi,
    const multi_filtration_grid<
        typename simplextree_multi::Options::Filtration_value> &grid,
    bool coordinate_values = true) {
  static_assert(simplextree_multi::Options::is_multi_parameter,
                "Only works for multiparameter simplextrees.");
  auto num_parameters =
      static_cast<unsigned int>(st_multi.get_number_of_parameters());
  if (grid.size() != num_parameters) {
    throw std::invalid_argument(
        "Grid and simplextree do not agree on number of parameters.");
  }
  for (const auto &simplex_handle : st_multi.complex_simplex_range()) {
    auto &simplex_filtration = st_multi.filtration_mutable(simplex_handle);
    find_coordinates(
        simplex_filtration,
        grid); // turns the simplexfiltration into coords in the grid
    if (!coordinate_values) {
      for (auto parameter = 0u; parameter < num_parameters; parameter++)
        simplex_filtration[parameter] =
            grid[parameter][simplex_filtration[parameter]];
    }
  }
  return;
}

// retrieves the filtration values of a simplextree. Useful to generate a grid.
/**
 * \brief Retrieves all of the filtration values, for each simplex dimension, of
 * the simplextree. Useful to generate grids. \ingroup multiparameter \param
 * st_multi Simplextree on which filtration values are exctracted. \param
 * degrees Only the simpleces of these dimension will be taken into account.
 * Useful for, e.g., Rips filtrations.
 * */
template <class simplextree_multi>
std::vector<multi_filtration_grid<
    typename simplextree_multi::Options::Filtration_value>>
get_filtration_values(simplextree_multi &st_multi,
                      const std::vector<int> &degrees) {
  using multi_filtration_grid = multi_filtration_grid<
      typename simplextree_multi::Options::Filtration_value>;
  static_assert(simplextree_multi::Options::is_multi_parameter,
                "Only works for multiparameter simplextrees.");
  int num_parameters = st_multi.get_number_of_parameters();
  std::vector<multi_filtration_grid> out(degrees.size(),
                                         multi_filtration_grid(num_parameters));
  std::vector<int> degree_index(st_multi.dimension() + 1);
  int count = 0;
  for (auto degree : degrees) {
    degree_index[degree] = count++;
    out[degree_index[degree]].reserve(st_multi.num_simplices());
  }

  for (const auto &simplex_handle : st_multi.complex_simplex_range()) {
    const auto &filtration = st_multi.filtration(simplex_handle);
    const auto degree = st_multi.dimension(simplex_handle);
    if (std::find(degrees.begin(), degrees.end(), degree) == degrees.end())
      continue;
    for (int parameter = 0; parameter < num_parameters; parameter++) {
      out[degree_index[degree]][parameter].push_back(filtration[parameter]);
    }
  }
  return out;
}

} // namespace Gudhi::multiparameter

#endif // SIMPLEX_TREE_MULTI_H_
