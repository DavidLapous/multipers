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
 * @file interface_helpers.h
 * @author Hannah Schreiber
 * @brief Contains helpers for the @ref Gudhi::multi_persistence::Slicer_interface class for python bindings.
 */

#ifndef MP_PY_INTERFACE_HELPERS_H_INCLUDED
#define MP_PY_INTERFACE_HELPERS_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <cstring>
// #include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/iterator_range.hpp>

#include <gudhi/Slicer.h>
#include <gudhi/Multi_filtration/Flat_array_filtration.h>
#include <gudhi/Multi_filtration/Nested_array_filtration.h>
#include <gudhi/Multi_filtration/Degree_bifiltration.h>
#include <gudhi/Multi_parameter_filtration_value.h>
#include <python_interfaces/numpy_utils.h>
#include <python_interfaces/construction_utils.h>

namespace Gudhi {
namespace multi_persistence {
namespace detail {

inline constexpr bool _is_host_device_type(int device_type) {
  return device_type == nanobind::device::cpu::value || device_type == nanobind::device::cuda_host::value ||
         device_type == nanobind::device::rocm_host::value;
}

template <typename... Args>
inline void _require_cpu_array(const nanobind::ndarray<Args...> &array) {
  if (!_is_host_device_type(array.device_type()))
    throw nanobind::type_error("Native persistence inputs must be CPU arrays.");
}

template <typename T, typename... Ts>
inline constexpr bool _all_same_v = (std::is_same_v<T, Ts> && ...);

enum class Array_dtype : std::uint8_t { INT32, INT64, UINT32, UINT64, FLOAT32, FLOAT64, EMPTY, UNKNOWN };

template <typename U>
inline bool _is_dtype(const nanobind::dlpack::dtype &dt) {
  auto expected = nanobind::dtype<U>();
  return dt.code == expected.code && dt.bits == expected.bits && dt.lanes == expected.lanes;
}

inline Array_dtype _get_dtype(const nanobind::dlpack::dtype &dt) {
  if (_is_dtype<std::uint64_t>(dt)) return Array_dtype::UINT64;
  if (_is_dtype<std::uint32_t>(dt)) return Array_dtype::UINT32;
  if (_is_dtype<std::int64_t>(dt)) return Array_dtype::INT64;
  if (_is_dtype<std::int32_t>(dt)) return Array_dtype::INT32;
  if (_is_dtype<double>(dt)) return Array_dtype::FLOAT64;
  if (_is_dtype<float>(dt)) return Array_dtype::FLOAT32;
  return Array_dtype::UNKNOWN;
}

inline Array_dtype _get_dtype(nanobind::handle obj, int depth = 0) {
  constexpr int maxRecursionDepth = 32;  // same limit than for numpy, seems reasonable
  if (depth > maxRecursionDepth) {
    throw nanobind::value_error("Exceeded maximum nesting depth while inferring dtype.");
  }

  // special case of ndarray
  if (nanobind::ndarray<> arr; nanobind::try_cast<nanobind::ndarray<>>(obj, arr)) {
    return _get_dtype(arr.dtype());
  }

  // terminal case of recursion
  if (nanobind::isinstance<nanobind::int_>(obj)) return Array_dtype::INT64;
  if (nanobind::isinstance<nanobind::float_>(obj)) return Array_dtype::FLOAT64;
  // to avoid weird inf recursion because e.g. str[0] yield another str in python
  if (PyUnicode_Check(obj.ptr()) || PyBytes_Check(obj.ptr()) || PyByteArray_Check(obj.ptr())) {
    throw nanobind::type_error("Expected an arithmetic dtype: got str/bytes-like object.");
  }

  // recursion on first element
  if (nanobind::isinstance<nanobind::iterable>(obj)) {
    if (!nanobind::hasattr(obj, "__getitem__")) throw nanobind::type_error("Container has to support subscripting.");
    if (!nanobind::hasattr(obj, "__len__")) throw nanobind::type_error("Container has to support __len__.");
    if (nanobind::len(obj) != 0) return _get_dtype(obj[0], depth + 1);
    return Array_dtype::EMPTY;
  }

  return Array_dtype::UNKNOWN;
}

// template <typename F>
// inline auto _dispatch_int_dtype(nanobind::handle data, F &&func) {
//   using R_int32 = decltype(func.template operator()<std::int32_t>());
//   using R_int64 = decltype(func.template operator()<std::int64_t>());
//   using R_uint32 = decltype(func.template operator()<std::uint32_t>());
//   using R_uint64 = decltype(func.template operator()<std::uint64_t>());

//   using Union = std::conditional_t<_all_same_v<R_int32, R_int64, R_uint32, R_uint64>,
//                                    R_uint32,
//                                    std::variant<R_int32, R_int64, R_uint32, R_uint64>>;

//   Array_dtype dtype = _get_dtype(data);
//   switch (dtype) {
//     case Array_dtype::INT32:
//       return Union(std::forward<F>(func).template operator()<std::int32_t>());
//     case Array_dtype::UINT32:
//       return Union(std::forward<F>(func).template operator()<std::uint32_t>());
//     case Array_dtype::INT64:
//       return Union(std::forward<F>(func).template operator()<std::int64_t>());
//     case Array_dtype::UINT64:
//       return Union(std::forward<F>(func).template operator()<std::uint64_t>());
//     case Array_dtype::EMPTY:
//       // type does not matter for now then
//       return Union(std::forward<F>(func).template operator()<std::uint32_t>());
//     default:
//       std::stringstream errMsg;
//       errMsg << "Unsupported integer type: ";
//       if (dtype == Array_dtype::FLOAT32)
//         errMsg << "FLOAT32";
//       else if (dtype == Array_dtype::FLOAT64)
//         errMsg << "FLOAT64";
//       else
//         errMsg << "UNKNOWN";
//       errMsg << ".";
//       throw nanobind::type_error(errMsg.str().c_str());
//   }
// }

// template <typename F>
// inline auto _dispatch_float_dtype(nanobind::handle data, F &&func) {
//   using R_float32 = decltype(func.template operator()<float>());
//   using R_float64 = decltype(func.template operator()<double>());

//   using Union = std::conditional_t<std::is_same_v<R_float32, R_float64>, R_float32, std::variant<R_float32,
//   R_float64>>;

//   Array_dtype dtype = _get_dtype(data);
//   switch (dtype) {
//     case Array_dtype::FLOAT32:
//       return Union(std::forward<F>(func).template operator()<float>());
//     case Array_dtype::FLOAT64:
//       return Union(std::forward<F>(func).template operator()<double>());
//     case Array_dtype::EMPTY:
//       // type does not matter for now then
//       return Union(std::forward<F>(func).template operator()<float>());
//     default:
//       std::stringstream errMsg;
//       errMsg << "Unsupported floating point type: ";
//       if (dtype == Array_dtype::INT32)
//         errMsg << "INT32";
//       else if (dtype == Array_dtype::INT64)
//         errMsg << "INT64";
//       else if (dtype == Array_dtype::UINT32)
//         errMsg << "UINT32";
//       else if (dtype == Array_dtype::UINT64)
//         errMsg << "UINT64";
//       else
//         errMsg << "UNKNOWN";
//       errMsg << ".";
//       throw nanobind::type_error(errMsg.str().c_str());
//   }
// }

template <typename F, typename F_empty, typename F_unknown>
inline auto _dispatch_dtype(nanobind::handle data, F &&func, F_empty &&funcEmpty, F_unknown &&funcUnkown) {
  using R_int32 = decltype(func.template operator()<std::int32_t>());
  using R_int64 = decltype(func.template operator()<std::int64_t>());
  using R_uint32 = decltype(func.template operator()<std::uint32_t>());
  using R_uint64 = decltype(func.template operator()<std::uint64_t>());
  using R_float32 = decltype(func.template operator()<float>());
  using R_float64 = decltype(func.template operator()<double>());

  // Only allow _all_same_v to be true to avoid std::variant compilation overhead?
  using Union = std::conditional_t<_all_same_v<R_int32, R_int64, R_uint32, R_uint64, R_float32, R_float64>,
                                   R_uint32,
                                   std::variant<R_int32, R_int64, R_uint32, R_uint64, R_float32, R_float64>>;

  Array_dtype dtype = _get_dtype(data);
  switch (dtype) {
    case Array_dtype::INT32:
      return Union(std::forward<F>(func).template operator()<std::int32_t>());
    case Array_dtype::UINT32:
      return Union(std::forward<F>(func).template operator()<std::uint32_t>());
    case Array_dtype::INT64:
      return Union(std::forward<F>(func).template operator()<std::int64_t>());
    case Array_dtype::UINT64:
      return Union(std::forward<F>(func).template operator()<std::uint64_t>());
    case Array_dtype::FLOAT32:
      return Union(std::forward<F>(func).template operator()<float>());
    case Array_dtype::FLOAT64:
      return Union(std::forward<F>(func).template operator()<double>());
    case Array_dtype::EMPTY:
      return Union(std::forward<F_empty>(funcEmpty)());
    default:
      return Union(std::forward<F_unknown>(funcUnkown)());
  }
}

// Number of items of a sequence (uses __len__).
inline std::size_t _sequence_size(nanobind::handle obj) {
  Py_ssize_t n = PySequence_Size(obj.ptr());
  if (n < 0) {
    PyErr_Clear();
    throw std::invalid_argument("Expected a sequence (with __len__ and __getitem__), got object of type '" +
                                std::string(nanobind::type_name(obj.type()).c_str()) + "'.");
  }
  return static_cast<std::size_t>(n);
}

// It is only a hint: ragged or malformed input gives a wrong estimate and should be reported later
// uses __len__ and __getitem__
template <typename U>
static std::size_t _estimate_flat_sequence_size(nanobind::handle values, int dim) {
  if (dim <= 0) return 0;

  std::size_t total = 1;
  nanobind::object current = nanobind::borrow(values);
  for (int level = 0; level < dim; ++level) {
    Py_ssize_t n = PySequence_Size(current.ptr());
    if (n < 0) {
      // no __len__
      PyErr_Clear();
      return 0;
    }
    total *= static_cast<std::size_t>(n);
    if (n == 0 || level + 1 == dim) return total;  // empty, or deepest level reached

    PyObject *raw_first = PySequence_GetItem(current.ptr(), 0);
    if (!raw_first) {
      // when there is no __getitem__ but there was a __len__ so n >= 0
      // or __getitem__ does not accept 0 as key etc.
      PyErr_Clear();
      return total;
    }
    current = nanobind::steal(raw_first);
    if (U scalar; nanobind::try_cast<U>(current, scalar)) return total;  // leaf reached
  }
  return total;
}

// Calls fun(nanobind::object item) for every item of a sequence, by index.
// No iterator object is created, and lists/tuples skip the generic call machinery.
template <typename F>
inline void _for_each_sequence_item(nanobind::handle obj, F &&fun) {
  PyObject *p = obj.ptr();
  // str/bytes problematic, better to reject directly
  if (PyUnicode_Check(p) || PyBytes_Check(p) || PyByteArray_Check(p))
    throw std::invalid_argument("Expected a numeric sequence, got a string/bytes object.");

  // List + Tuple special case
  // Size is re-read every iteration and items are (cheaply) borrowed, so a callback
  // that runs Python code and mutates the container cannot cause dangling pointers.
  if (PyList_CheckExact(p)) {
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(p); ++i) fun(nanobind::borrow(PyList_GET_ITEM(p, i)));
    return;
  }
  if (PyTuple_CheckExact(p)) {
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(p); ++i) fun(nanobind::borrow(PyTuple_GET_ITEM(p, i)));
    return;
  }

  // Generic sequence: __len__ once, then __getitem__(i). No copy into a temporary list.
  const std::size_t n = _sequence_size(obj);
  for (std::size_t i = 0; i < n; ++i) {
    PyObject *item = PySequence_GetItem(p, static_cast<Py_ssize_t>(i));
    if (!item) throw nanobind::python_error();
    fun(nanobind::steal(item));
  }
}

// Calls fun for every item of the iterable obj (does not assume __len__ and __getitem__, just __iter__)
template <typename F>
inline void _for_each_python_item(nanobind::handle obj, F &&fun) {
  PyObject *raw_iter = PyObject_GetIter(obj.ptr());
  if (!raw_iter) {
    PyErr_Clear();
    throw std::runtime_error("Expected an iterable, got object of type '" +
                             std::string(nanobind::type_name(obj.type()).c_str()) + "'.");
  }
  nanobind::object iter = nanobind::steal(raw_iter);

  while (true) {
    PyObject *item = PyIter_Next(iter.ptr());
    if (!item) {
      if (PyErr_Occurred()) throw nanobind::python_error();
      break;
    }
    fun(nanobind::steal(item));
  }
}

// uncommenting in the _get_compatible_* methods gives much more possibilities, but it also takes much more
// compile time (and binary size). I had to split the _dispatch_dtype method into a int and a float version
// for the same reason

// inline auto _get_compatible_generator_maps(nanobind::iterable maps) {
//   return _dispatch_int_dtype(maps, [&]<typename U>() {
//     // return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<
//     //     std::vector<nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>>,
//     //     std::vector<std::vector<U>>>(
//     //     maps, "Generator maps must be either iterable[iterable[U]] or iterable[ndarray[U, ndim=1]]
//     (contiguous)."); std::vector<std::vector<U>> res; if (nanobind::try_cast(maps, res, false)) return res; throw
//     std::invalid_argument("Generator maps must be iterable[iterable[U]].");
//   });
// }

// inline auto _get_compatible_generator_dimensions(nanobind::iterable dimensions) {
//   return _dispatch_int_dtype(dimensions, [&]<typename U>() {
//     // return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<
//     //     nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>,
//     //     std::vector<U>>(dimensions,
//     //                     "Generator dimensions must be either iterable[U] or ndarray[U, ndim=1] (contiguous).");
//     nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig> res;
//     if (nanobind::try_cast(dimensions, res, false)) return Numpy_span(res);
//     throw std::invalid_argument("Generator dimensions must be ndarray[U, ndim=1] (contiguous).");
//   });
// }

// template <typename T, bool is_kcritical>
// inline auto _get_compatible_filtration_values(nanobind::iterable filts) {
//   auto convert = [&]<typename U>() {
//     if constexpr (is_kcritical) {
//       using Seq2_t = std::vector<std::vector<std::vector<U>>>;
//       using Ten2_t = std::vector<nanobind::ndarray<const U, nanobind::ndim<2>>>;
//       return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<Ten2_t, Seq2_t>(
//             filts,
//             "Filtration values must be one of: iterable[iterable[U]], iterable[iterable[iterable[U]]], "
//             "iterable[ndarray[U, ndim=1]] (contiguous), or iterable[ndarray[U, ndim=2]]."
//             /* "Filtration values must be either iterable[ndarray[U, ndim=1]] (contiguous) or iterable[ndarray[U, "
//             "ndim=2]]." */);
//     } else {
//       using Seq1_t = std::vector<std::vector<U>>;
//       using Ten1_t = nanobind::ndarray<const U, nanobind::ndim<2>>;
//       return Gudhi::python::_convert_iterable_to_cpp_type_and_wrap_ndarrays<Ten1_t, Seq1_t>(
//             filts,
//             "Filtration values must be one of: iterable[iterable[U]], iterable[iterable[iterable[U]]], "
//             "iterable[ndarray[U, ndim=1]] (contiguous), or iterable[ndarray[U, ndim=2]]."
//             /* "Filtration values must be either iterable[ndarray[U, ndim=1]] (contiguous) or iterable[ndarray[U, "
//             "ndim=2]]." */);
//     }
//   };
//   if constexpr (std::is_floating_point_v<T>) {
//     return _dispatch_float_dtype(filts, convert);
//   } else {
//     return _dispatch_int_dtype(filts, convert);
//   }
// }

template <class MultiFiltrationValue>
constexpr bool _is_degree_rips() {
  using T = typename MultiFiltrationValue::value_type;
  using SP = typename MultiFiltrationValue::Storage_policy;

  return std::is_same_v<SP, multi_filtration::Degree_bifiltration<T>>;
}

template <class MultiFiltrationValue>
constexpr bool _is_dynamic() {
  using T = typename MultiFiltrationValue::value_type;
  using SP = typename MultiFiltrationValue::Storage_policy;

  return std::is_same_v<SP, multi_filtration::Nested_array_filtration<T>>;
}

template <class MultiFiltrationValue>
constexpr bool _is_flat() {
  using T = typename MultiFiltrationValue::value_type;
  using SP = typename MultiFiltrationValue::Storage_policy;

  return std::is_same_v<SP, multi_filtration::Flat_array_filtration<T>>;
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::object _get_raw_filtration_data(
    nanobind::handle owner,
    multi_filtration::Multi_parameter_filtration_value<multi_filtration::Nested_array_filtration<T>, Co, OneCritical>
        &f,
    bool copy) {
  if constexpr (OneCritical) {
    if (copy) {
      std::vector<T> copy(f.begin(0), f.end(0));
      return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_parameters()));
    }
    return nanobind::cast(_wrap_view_as_numpy_array<false>(owner, &f(0, 0), f.num_parameters()));
  } else {
    return Gudhi::python::_build_tuple(f.num_generators(), [&](std::size_t g) -> nanobind::object {
      if (copy) {
        std::vector<T> copy(f.begin(g), f.end(g));
        return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_parameters()));
      }
      return nanobind::cast(_wrap_view_as_numpy_array<false>(owner, &f(g, 0), f.num_parameters()));
    });
  }
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::object _get_raw_filtration_data(
    nanobind::handle owner,
    multi_filtration::Multi_parameter_filtration_value<multi_filtration::Flat_array_filtration<T>, Co, OneCritical> &f,
    bool copy) {
  auto &container = f.get_underlying_container();
  if constexpr (OneCritical) {
    if (copy) {
      std::vector<T> copy(container.begin(), container.end());
      return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_parameters()));
    }
    return nanobind::cast(_wrap_view_as_numpy_array<false>(owner, container.data(), f.num_parameters()));
  } else {
    if (copy) {
      std::vector<T> copy(container.begin(), container.end());
      return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_generators(), f.num_parameters()));
    }
    return nanobind::cast(
        _wrap_view_as_numpy_array<false>(owner, container.data(), f.num_generators(), f.num_parameters()));
  }
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::object _get_raw_filtration_data(
    nanobind::handle owner,
    multi_filtration::Multi_parameter_filtration_value<multi_filtration::Degree_bifiltration<T>, Co, OneCritical> &f,
    bool copy) {
  auto &container = f.get_underlying_container();
  if (copy) {
    std::vector<T> copy(container.begin(), container.end());
    return nanobind::cast(_wrap_as_numpy_array(std::move(copy), f.num_generators()));
  }
  return nanobind::cast(_wrap_view_as_numpy_array<false>(owner, container.data(), f.num_generators()));
}

template <typename T, bool Co, bool OneCritical>
inline nanobind::tuple _get_compact_filtration_data(
    const std::vector<
        multi_filtration::Multi_parameter_filtration_value<multi_filtration::Degree_bifiltration<T>, Co, OneCritical>>
        &filts) {
  std::vector<T> values;
  std::vector<std::int64_t> startIndices(filts.size() + 1, 0);

  {
    nanobind::gil_scoped_release release;
    for (std::size_t i = 0; i < filts.size(); ++i) {
      startIndices[i + 1] = startIndices[i] + filts[i].num_generators();
    }
    values.resize(startIndices.back());
    for (std::size_t i = 0; i < filts.size(); ++i) {
      const auto &container = filts[i].get_underlying_container();
      std::copy(container.begin(), container.end(), values.begin() + startIndices[i]);
    }
  }

  return nanobind::make_tuple(_wrap_as_numpy_array(std::move(startIndices), startIndices.size()),
                              _wrap_as_numpy_array(std::move(values), values.size()));
}

template <class MultiFiltrationValue>
inline auto _get_filtration_array(const MultiFiltrationValue &f) {
  std::vector<typename MultiFiltrationValue::value_type> values(f.num_generators() * f.num_parameters());
  Gudhi::Simple_mdspan view(values.data(), f.num_generators(), f.num_parameters());
  {
    nanobind::gil_scoped_release release;
    for (std::size_t g = 0; g < f.num_generators(); ++g) {
      for (std::size_t p = 0; p < f.num_parameters(); ++p) {
        view(g, p) = f(g, p);
      }
    }
  }
  if constexpr (MultiFiltrationValue::ensures_1_criticality()) {
    return _wrap_as_numpy_array(std::move(values), f.num_parameters());
  } else {
    return _wrap_as_numpy_array(std::move(values), f.num_generators(), f.num_parameters());
  }
}

template <typename T, typename U>
struct Flat_2D_array_span {
  using Del_array = nanobind::ndarray<const T, nanobind::ndim<1>, nanobind::any_contig>;
  using Data_array = nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>;
  using Del_view = decltype(std::declval<Del_array>().view());
  using Data_view = decltype(std::declval<Data_array>().view());

  Flat_2D_array_span(Del_array delimiters, Data_array flatData)
      : delimiters_(delimiters.view()), flatData_(flatData.view()) {}

  std::size_t size() const { return delimiters_.shape(0) - 1; }

  auto operator[](std::size_t i) const {
    if (i >= size()) throw std::out_of_range("Index is out of range for flat 2D range.");
    return Numpy_span(&flatData_(delimiters_(i)), &flatData_(delimiters_(i + 1)));
  }

  Del_view delimiters_;
  Data_view flatData_;
};

// careful: single pass
template <typename T>
class Py_iterable_iterator
    : public boost::iterator_facade<Py_iterable_iterator<T>, T, boost::single_pass_traversal_tag, T> {
 public:
  Py_iterable_iterator() = default;

  explicit Py_iterable_iterator(nanobind::handle iterable) {
    PyObject *it = PyObject_GetIter(iterable.ptr());
    if (!it) {
      PyErr_Format(
          PyExc_TypeError, "Expected an iterable of numerical, got '%.200s'.", Py_TYPE(iterable.ptr())->tp_name);
      throw nanobind::python_error();
    }
    it_ = nanobind::steal(it);
    advance();
  }

 private:
  friend class boost::iterator_core_access;

  nanobind::object it_;   // shared between copies
  nanobind::object cur_;  // null object == end

  void advance() {
    PyObject *next = PyIter_Next(it_.ptr());
    if (next) {
      cur_ = nanobind::steal(next);
    } else {
      cur_ = nanobind::object();
      if (PyErr_Occurred()) throw nanobind::python_error();
    }
  }

  T dereference() const { return nanobind::cast<T>(cur_); }

  void increment() { advance(); }

  bool equal(const Py_iterable_iterator &o) const { return cur_.is_valid() == o.cur_.is_valid(); }
};

template <typename T>
inline boost::iterator_range<Py_iterable_iterator<T>> as_cpp_range(nanobind::handle iterable) {
  return {Py_iterable_iterator<T>(iterable), Py_iterable_iterator<T>()};
}

}  // namespace detail
}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_INTERFACE_HELPERS_H_INCLUDED
