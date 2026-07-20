/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2026 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file slicer_interface_helpers.h
 * @author Hannah Schreiber
 * @brief Contains helpers for the @ref Gudhi::multi_persistence::Slicer_interface class for python bindings.
 */

#ifndef MP_PY_SLICER_HELPERS_H_INCLUDED
#define MP_PY_SLICER_HELPERS_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <gudhi/Slicer.h>
#include <gudhi/Degree_rips_bifiltration.h>
#include <gudhi/Dynamic_multi_parameter_filtration.h>
#include <gudhi/Multi_parameter_filtration.h>
#include <python_interfaces/numpy_utils.h>
#include <python_interfaces/construction_utils.h>

namespace Gudhi {
namespace multi_persistence {
namespace detail {

template <typename T, typename... Ts>
inline constexpr bool _all_same_v = (std::is_same_v<T, Ts> && ...);

enum class Array_dtype : std::uint8_t { INT32, INT64, FLOAT32, FLOAT64, EMPTY, UNKNOWN };

template <typename U>
inline bool _is_dtype(const nanobind::dlpack::dtype &dt) {
  auto expected = nanobind::dtype<U>();
  return dt.code == expected.code && dt.bits == expected.bits && dt.lanes == expected.lanes;
}

inline Array_dtype _get_dtype(const nanobind::dlpack::dtype &dt) {
  if (_is_dtype<std::int64_t>(dt)) return Array_dtype::INT64;
  if (_is_dtype<std::int32_t>(dt)) return Array_dtype::INT32;
  if (_is_dtype<double>(dt)) return Array_dtype::FLOAT64;
  if (_is_dtype<float>(dt)) return Array_dtype::FLOAT32;
  return Array_dtype::UNKNOWN;
}

inline Array_dtype _get_dtype(nanobind::handle obj) {
  // special case of ndarray
  if (nanobind::ndarray<> arr; nanobind::try_cast<nanobind::ndarray<>>(obj, arr)) {
    return _get_dtype(arr.dtype());
  }

  // terminal case of recursion
  if (nanobind::isinstance<nanobind::int_>(obj)) return Array_dtype::INT64;
  if (nanobind::isinstance<nanobind::float_>(obj)) return Array_dtype::FLOAT64;

  // recursion on first element
  if (nanobind::isinstance<nanobind::iterable>(obj)) {
    auto it = nanobind::cast<nanobind::iterable>(obj);
    auto begin = it.begin();
    if (begin != it.end()) return _get_dtype(*begin);
    return Array_dtype::EMPTY;
  }

  return Array_dtype::UNKNOWN;
}

template <typename F>
inline auto _dispatch_dtype(nanobind::handle data, F &&func) {
  using R_int32 = decltype(func.template operator()<std::int32_t>());
  using R_int64 = decltype(func.template operator()<std::int64_t>());
  using R_float32 = decltype(func.template operator()<float>());
  using R_float64 = decltype(func.template operator()<double>());

  // the case were some are equal but not all should not happen in our use case
  using Union = std::conditional_t<_all_same_v<R_int32, R_int64, R_float32, R_float64>,
                                   R_int32,
                                   std::variant<R_int32, R_int64, R_float32, R_float64>>;

  Array_dtype dtype = _get_dtype(data);
  switch (dtype) {
    case Array_dtype::INT32:
      return Union(std::forward<F>(func).template operator()<std::int32_t>());
    case Array_dtype::INT64:
      return Union(std::forward<F>(func).template operator()<std::int64_t>());
    case Array_dtype::FLOAT32:
      return Union(std::forward<F>(func).template operator()<float>());
    case Array_dtype::FLOAT64:
      return Union(std::forward<F>(func).template operator()<double>());
    case Array_dtype::EMPTY:
      // type does not matter for now then
      return Union(std::forward<F>(func).template operator()<double>());
    default:
      throw nanobind::type_error("Unsupported element type.");
  }
}

inline auto _get_compatible_generator_maps(nanobind::iterable maps) {
  return _dispatch_dtype(maps, [&]<typename U>() {
    return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<
        std::vector<nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>>,
        std::vector<std::vector<U>>>(
        maps, "Generator maps must be either iterable[iterable[U]] or iterable[ndarray[U, ndim=1]] (contiguous).");
  });
}

inline auto _get_compatible_generator_dimensions(nanobind::iterable dimensions) {
  return _dispatch_dtype(dimensions, [&]<typename U>() {
    return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<
        nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>,
        std::vector<U>>(dimensions,
                        "Generator dimensions must be either iterable[U] or ndarray[U, ndim=1] (contiguous).");
  });
}

inline auto _get_compatible_filtration_values(nanobind::iterable filts) {
  return _dispatch_dtype(filts, [&]<typename U>() {
    using Seq1_t = std::vector<std::vector<U>>;
    using Seq2_t = std::vector<std::vector<std::vector<U>>>;
    using Ten1_t = std::vector<nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>>;
    using Ten2_t = std::vector<nanobind::ndarray<const U, nanobind::ndim<2>>>;
    return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<Ten1_t, Ten2_t, Seq1_t, Seq2_t>(
        filts,
        "Filtration values must be one of: iterable[iterable[U]], iterable[iterable[iterable[U]]], "
        "iterable[ndarray[U, ndim=1]] (contiguous), or iterable[ndarray[U, ndim=2]].");
  });
}

template <class MultiFiltrationValue>
constexpr bool _is_degree_rips() {
  using T = typename MultiFiltrationValue::value_type;
  constexpr bool co = MultiFiltrationValue::has_negative_cones();
  constexpr bool oneCrit = MultiFiltrationValue::ensures_1_criticality();

  return std::is_same_v<MultiFiltrationValue, multi_filtration::Degree_rips_bifiltration<T, co, oneCrit>>;
}

template <class MultiFiltrationValue>
constexpr bool _is_dynamic() {
  using T = typename MultiFiltrationValue::value_type;
  constexpr bool co = MultiFiltrationValue::has_negative_cones();
  constexpr bool oneCrit = MultiFiltrationValue::ensures_1_criticality();

  return std::is_same_v<MultiFiltrationValue, multi_filtration::Dynamic_multi_parameter_filtration<T, co, oneCrit>>;
}

template <class MultiFiltrationValue>
constexpr bool _is_flat() {
  using T = typename MultiFiltrationValue::value_type;
  constexpr bool co = MultiFiltrationValue::has_negative_cones();
  constexpr bool oneCrit = MultiFiltrationValue::ensures_1_criticality();

  return std::is_same_v<MultiFiltrationValue, multi_filtration::Multi_parameter_filtration<T, co, oneCrit>>;
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::object _get_raw_filtration_data(
    const multi_filtration::Dynamic_multi_parameter_filtration<T, Co, OneCritical> &f,
    bool copy) {
  if constexpr (OneCritical) {
    if (copy) {
      std::vector<T> copy(f[0].begin(), f[0].end());
      return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_parameters()));
    }
    return nanobind::cast(nanobind::ndarray<const T, nanobind::numpy>(&f(0, 0), {f.num_parameters()}));
  } else {
    return Gudhi::python::_build_tuple(f.num_generators(), [&](std::size_t g) -> nanobind::object {
      if (copy) {
        std::vector<T> copy(f[g].begin(), f[g].end());
        return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_parameters()));
      }
      return nanobind::cast(nanobind::ndarray<const T, nanobind::numpy>(&f(g, 0), {f.num_parameters()}));
    });
  }
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::object _get_raw_filtration_data(
    const multi_filtration::Multi_parameter_filtration<T, Co, OneCritical> &f,
    bool copy) {
  if constexpr (OneCritical) {
    if (copy) {
      std::vector<T> copy(f.begin(), f.end());
      return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_parameters()));
    }
    return nanobind::cast(nanobind::ndarray<const T, nanobind::numpy>(&f(0, 0), {f.num_parameters()}));
  } else {
    if (copy) {
      std::vector<T> copy(f.begin(), f.end());
      return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_generators(), f.num_parameters()));
    }
    return nanobind::cast(
        nanobind::ndarray<const T, nanobind::numpy>(&f(0, 0), {f.num_generators(), f.num_parameters()}));
  }
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::object _get_raw_filtration_data(
    const multi_filtration::Degree_rips_bifiltration<T, Co, OneCritical> &f,
    bool copy) {
  if (copy) {
    std::vector<T> copy(f.begin(), f.end());
    return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_generators()));
  }
  return nanobind::cast(nanobind::ndarray<const T, nanobind::numpy>(&f(0, 0), {f.num_generators()}));
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::tuple _get_compact_filtration_data(
    const std::vector<multi_filtration::Degree_rips_bifiltration<T, Co, OneCritical>> &filts) {
  std::vector<T> values;
  std::vector<std::int64_t> startIndices(filts.size() + 1, 0);

  {
    nanobind::gil_scoped_release release;
    for (std::size_t i = 0; i < filts.size(); ++i) {
      startIndices[i + 1] = startIndices[i] + filts[i].num_generators();
    }
    values.resize(startIndices.back());
    for (std::size_t i = 0; i < filts.size(); ++i) {
      const auto &f = filts[i];
      std::copy(&f(0, 0), &f(0, f.num_parameters() - 1), values.begin() + startIndices[i]);
    }
  }

  return nanobind::make_tuple(_wrap_as_numpy_array(std::move(startIndices), startIndices.size()),
                              _wrap_as_numpy_array(std::move(values), startIndices.back()));
}

}  // namespace detail
}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_SLICER_HELPERS_H_INCLUDED
