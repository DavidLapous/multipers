/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Vincent Rouvreau
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef SIMPLEX_TREE_SIMPLEX_TREE_OPTIONS_H_
#define SIMPLEX_TREE_SIMPLEX_TREE_OPTIONS_H_

#include <gudhi/Simplex_tree/indexing_tag.h>

#include <cstdint>
#include <type_traits>  // void_t
#include <cmath>        // std::isnan

namespace Gudhi {

/** \addtogroup simplex_tree 
 * Pre-defined options for the Simplex_tree.
 *  @{
 */

/** Model of SimplexTreeOptions.
 * 
 * Default options version of the Simplex_tree.
 * 
 * Maximum number of simplices to compute persistence is <CODE>std::numeric_limits<std::uint32_t>::max()</CODE>
 * (about 4 billions of simplices). */
struct Simplex_tree_options_default {
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  typedef double Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = false;
  static const bool link_nodes_by_label = false;
  static const bool stable_simplex_handles = false;
  static const bool is_multi_parameter = false;
};

/** Model of SimplexTreeOptions.
 * 
 * Maximum number of simplices to compute persistence is <CODE>std::numeric_limits<std::uint32_t>::max()</CODE>
 * (about 4 billions of simplices). */
struct Simplex_tree_options_full_featured {
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  typedef double Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = false;
  static const bool link_nodes_by_label = true;
  static const bool stable_simplex_handles = true;
  static const bool is_multi_parameter = false;
};

/** Model of SimplexTreeOptions.
 * 
 * Minimal version of the Simplex_tree. No filtration values are stored and it is impossible to compute persistence
 * with these options. */
struct Simplex_tree_options_minimal {
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  typedef double Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = false;
  static const bool store_filtration = false;
  static const bool contiguous_vertices = false;
  static const bool link_nodes_by_label = false;
  static const bool stable_simplex_handles = false;
  static const bool is_multi_parameter = false;
};

/** @private @brief Model of SimplexTreeOptions, faster than `Simplex_tree_options_default` but note the unsafe
 * `contiguous_vertices` option.
 * 
 * Maximum number of simplices to compute persistence is <CODE>std::numeric_limits<std::uint32_t>::max()</CODE>
 * (about 4 billions of simplices). */
struct Simplex_tree_options_fast_persistence {
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  typedef float Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = true;
  static const bool link_nodes_by_label = false;
  static const bool stable_simplex_handles = false;
  static const bool is_multi_parameter = false;
};

/** @}*/  // end addtogroup simplex_tree

struct No_simplex_data {};

// Nested `type` is O::Simplex_data if that exists, No_simplex_data otherwise
template <class, class=void> struct Get_simplex_data_type { typedef No_simplex_data type; };
template <class O>
struct Get_simplex_data_type<O, std::void_t<typename O::Simplex_data>> { typedef typename O::Simplex_data type; };

/**
 * @brief Given two filtration values at which a simplex exists, stores in the first value the minimal union of births
 * generating a lifetime including those two values.
 * This is the overload for when @ref FiltrationValue is an arithmetic type, like double, int etc.
 * Because the filtration values are totally ordered then, the union is simply the minimum of the two values.
 *
 * NaN values are not supported.
 */
template <typename Arithmetic_filtration_value>
bool unify_lifetimes(Arithmetic_filtration_value& f1, Arithmetic_filtration_value f2)
{
  if (f2 < f1){
    f1 = f2;
    return true;
  }
  return false;
}

/**
 * @brief Given two filtration values, stores in the first value the lowest common upper bound of the two values.
 * If a filtration value has value `NaN`, it should be considered as the lowest value possible.
 * This is the overload for when @ref FiltrationValue is an arithmetic type, like double, float, int etc.
 * Because the filtration values are totally ordered then, the upper bound is always the maximum of the two values.
 */
template <typename Arithmetic_filtration_value>
bool intersect_lifetimes(Arithmetic_filtration_value& f1, Arithmetic_filtration_value f2)
{
  if constexpr (std::is_floating_point_v<Arithmetic_filtration_value>) {
    if (std::isnan(f1)) {
      f1 = f2;
      return !std::isnan(f2);
    }

    // Computes the max while handling NaN as lowest value.
    if (!(f1 < f2)) return false;

    f1 = f2;
    return true;
  } else {
    // NaN not possible.
    if (f1 < f2){
      f1 = f2;
      return true;
    }
    return false;
  }
}

}  // namespace Gudhi

#endif  // SIMPLEX_TREE_SIMPLEX_TREE_OPTIONS_H_
