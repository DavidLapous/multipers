#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace multipers::nanobind_utils {

template <typename T>
void delete_vector_capsule(void* ptr) noexcept {
  delete static_cast<std::vector<T>*>(ptr);
}

template <typename T>
nanobind::ndarray<nanobind::numpy, T> owned_array(std::vector<T>&& values, std::initializer_list<size_t> shape) {
  auto* storage = new std::vector<T>(std::move(values));
  nanobind::capsule owner(storage, &delete_vector_capsule<T>);
  return nanobind::ndarray<nanobind::numpy, T>(storage->data(), shape, owner);
}

template <typename T>
nanobind::ndarray<nanobind::numpy, T> owned_array(std::vector<T>&& values, const std::vector<size_t>& shape) {
  auto* storage = new std::vector<T>(std::move(values));
  nanobind::capsule owner(storage, &delete_vector_capsule<T>);
  return nanobind::ndarray<nanobind::numpy, T>(storage->data(), shape.size(), shape.data(), owner);
}

template <typename T>
nanobind::ndarray<nanobind::numpy, T> view_array(T* ptr, std::initializer_list<size_t> shape, nanobind::handle owner) {
  return nanobind::ndarray<nanobind::numpy, T>(ptr, shape, owner);
}

template <typename Func>
nanobind::tuple tuple_from_size(size_t size, Func&& make_item) {
  nanobind::tuple out = nanobind::steal<nanobind::tuple>(PyTuple_New(static_cast<Py_ssize_t>(size)));
  if (!out.is_valid()) {
    throw nanobind::python_error();
  }
  for (size_t i = 0; i < size; ++i) {
    nanobind::object item = make_item(i);
    PyTuple_SET_ITEM(out.ptr(), static_cast<Py_ssize_t>(i), item.release().ptr());
  }
  return out;
}

}  // namespace multipers::nanobind_utils
