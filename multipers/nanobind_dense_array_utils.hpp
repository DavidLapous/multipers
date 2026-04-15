#pragma once

#include <nanobind/ndarray.h>

#include <cstddef>
#include <vector>

namespace multipers::nanobind_dense_utils {

template <typename T>
std::vector<T> vector_from_array(
    const nanobind::ndarray<nanobind::numpy, const T, nanobind::ndim<1>, nanobind::c_contig>& array) {
  return std::vector<T>(array.data(), array.data() + array.shape(0));
}

template <typename Out, typename In>
std::vector<Out> cast_vector_from_array(
    const nanobind::ndarray<nanobind::numpy, const In, nanobind::ndim<1>, nanobind::c_contig>& array) {
  std::vector<Out> out;
  out.reserve(array.shape(0));
  for (size_t i = 0; i < array.shape(0); ++i) {
    out.push_back(static_cast<Out>(array(i)));
  }
  return out;
}

template <typename T>
std::vector<std::vector<T>> matrix_from_array(
    const nanobind::ndarray<nanobind::numpy, const T, nanobind::ndim<2>, nanobind::c_contig>& array) {
  const auto view = array.view();
  std::vector<std::vector<T>> out(array.shape(0), std::vector<T>(array.shape(1)));
  for (size_t i = 0; i < array.shape(0); ++i) {
    for (size_t j = 0; j < array.shape(1); ++j) {
      out[i][j] = view(i, j);
    }
  }
  return out;
}

}  // namespace multipers::nanobind_dense_utils
