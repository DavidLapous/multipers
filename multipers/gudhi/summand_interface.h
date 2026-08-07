/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux, Hannah Schreiber
 *
 *    Copyright (C) 2026 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file summand_interface.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains @ref Gudhi::multi_persistence::Summand methods for python bindings.
 */

#ifndef MP_PY_SUMMAND_H_INCLUDED
#define MP_PY_SUMMAND_H_INCLUDED

#include <cstddef>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <gudhi/Multi_persistence/Summand.h>
#include <python_interfaces/numpy_utils.h>

namespace Gudhi {
namespace multi_persistence {

/**
 * @private
 */
template <typename T, class Corners>
inline auto compute_flat_corners(const Corners& corners) {
  std::vector<T> res(corners.num_generators() * corners.num_parameters());
  {
    nanobind::gil_scoped_release release;
    Gudhi::Simple_mdspan view(res.data(), corners.num_generators(), corners.num_parameters());
    for (std::size_t g = 0; g < corners.num_generators(); ++g) {
      for (std::size_t p = 0; p < corners.num_parameters(); ++p) {
        view(g, p) = corners(g, p);
      }
    }
  }
  return _wrap_as_numpy_array(std::move(res), corners.num_generators(), corners.num_parameters());
}

/**
 * @private
 */
template <typename T>
inline Summand<T> deserialize_summand_from_python(
    const nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy>& state) {
  Summand<T> sum;
  {
    nanobind::gil_scoped_release release;
    deserialize_value_from_char_buffer(sum, state.data());
  }
  return sum;
}

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_SUMMAND_H_INCLUDED
