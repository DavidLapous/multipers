#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace multipers::nanobind_utils {

inline std::string lowercase_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline std::string numpy_dtype_name(const nanobind::handle& dtype) {
  nanobind::object np = nanobind::module_::import_("numpy");
  return nanobind::cast<std::string>(np.attr("dtype")(nanobind::borrow(dtype)).attr("name"));
}

inline nanobind::object numpy_dtype_type(std::string_view name) {
  nanobind::object np = nanobind::module_::import_("numpy");
  return np.attr("dtype")(std::string(name)).attr("type");
}

inline bool has_template_id(const nanobind::handle& input) { return nanobind::hasattr(input, "_template_id"); }

inline int template_id_of(const nanobind::handle& input) {
  if (!has_template_id(input)) {
    throw nanobind::type_error("Object does not expose a template id.");
  }
  return nanobind::cast<int>(input.attr("_template_id"));
}

template <typename T>
std::vector<T> vector_from_handle(const nanobind::handle& h) {
  nanobind::object obj = nanobind::borrow(h);
  if (nanobind::hasattr(obj, "shape")) {
    nanobind::object np = nanobind::module_::import_("numpy");
    obj = np.attr("asarray")(obj).attr("reshape")(-1);
  }
  return nanobind::cast<std::vector<T>>(obj);
}

template <typename T>
std::vector<T> cast_vector(const nanobind::handle& h) {
  return vector_from_handle<T>(h);
}

template <typename T>
std::vector<std::vector<T>> matrix_from_handle(const nanobind::handle& h) {
  nanobind::object obj = nanobind::borrow(h);
  if (nanobind::hasattr(obj, "shape")) {
    nanobind::object np = nanobind::module_::import_("numpy");
    obj = np.attr("asarray")(obj);
  }
  return nanobind::cast<std::vector<std::vector<T>>>(obj);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(const nanobind::handle& h) {
  return matrix_from_handle<T>(h);
}

template <typename T>
std::vector<std::vector<std::vector<T>>> tensor3_from_handle(const nanobind::handle& h) {
  return nanobind::cast<std::vector<std::vector<std::vector<T>>>>(h);
}

template <typename T>
std::vector<std::vector<std::vector<T>>> cast_tensor3(const nanobind::handle& h) {
  return tensor3_from_handle<T>(h);
}

}  // namespace multipers::nanobind_utils
