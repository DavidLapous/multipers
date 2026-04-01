#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace multipers::nanobind_utils {

inline nanobind::module_ numpy_module() { return nanobind::module_::import_("numpy"); }

inline std::string lowercase_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline std::string numpy_dtype_name(const nanobind::handle& dtype) {
  nanobind::module_ np = numpy_module();
  return nanobind::cast<std::string>(np.attr("dtype")(nanobind::borrow(dtype)).attr("name"));
}

inline nanobind::object numpy_dtype_type(std::string_view name) {
  nanobind::module_ np = numpy_module();
  std::string dtype_name(name);
  return np.attr(dtype_name.c_str());
}

inline bool has_template_id(const nanobind::handle& input) { return nanobind::hasattr(input, "_template_id"); }

inline int template_id_of(const nanobind::handle& input) {
  if (!has_template_id(input)) {
    throw nanobind::type_error("Object does not expose a template id.");
  }
  return nanobind::cast<int>(input.attr("_template_id"));
}

inline std::optional<int> maybe_template_id_of(const nanobind::handle& input) {
  nanobind::object template_id = nanobind::getattr(input, "_template_id", nanobind::none());
  if (template_id.is_none()) {
    return std::nullopt;
  }
  try {
    return nanobind::cast<int>(template_id);
  } catch (const nanobind::cast_error&) {
    return std::nullopt;
  } catch (const nanobind::python_error&) {
    return std::nullopt;
  }
}

template <typename T>
std::vector<T> vector_from_handle(const nanobind::handle& h) {
  nanobind::object obj = nanobind::borrow(h);
  if (nanobind::hasattr(obj, "shape")) {
    nanobind::module_ np = numpy_module();
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
    nanobind::module_ np = numpy_module();
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
