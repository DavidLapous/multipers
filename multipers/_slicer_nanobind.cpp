#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>

#include "Persistence_slices_interface.h"
#include "gudhi/Multi_parameter_filtered_complex.h"
#include "gudhi/slicer_helpers.h"
#include "multi_parameter_rank_invariant/hilbert_function.h"
#include "multi_parameter_rank_invariant/rank_invariant.h"
#include "multiparameter_module_approximation/approximation.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpnb {

using tensor_dtype = int32_t;
using indices_type = int32_t;
using signed_measure_type = std::pair<std::vector<std::vector<indices_type>>, std::vector<tensor_dtype>>;

template <typename Slicer>
struct PySlicer {
  Slicer truc;
  nb::object filtration_grid;
  int minpres_degree;

  PySlicer() : filtration_grid(nb::none()), minpres_degree(-1) {}
};

template <typename... Types>
struct type_list {};

#include "_slicer_nanobind_registry.inc"

template <typename Func>
decltype(auto) dispatch_slicer_by_template_id(int template_id, Func&& func) {
  switch (template_id) {
#define MP_SLICER_CASE(desc) \
  case desc::template_id:    \
    return std::forward<Func>(func).template operator()<desc>();
    MP_FOR_EACH_SLICER_DESC(MP_SLICER_CASE)
#undef MP_SLICER_CASE
    default:
      throw nb::type_error("Unknown slicer template id.");
  }
}

template <typename Func>
decltype(auto) dispatch_simplextree_by_template_id(int template_id, Func&& func) {
  switch (template_id) {
#define MP_SIMPLEXTREE_CASE(desc) \
  case desc::template_id:         \
    return std::forward<Func>(func).template operator()<desc>();
    MP_FOR_EACH_SIMPLEXTREE_DESC(MP_SIMPLEXTREE_CASE)
#undef MP_SIMPLEXTREE_CASE
    default:
      throw nb::type_error("Unknown SimplexTreeMulti template id.");
  }
}

template <typename T>
void delete_vector_capsule(void* ptr) noexcept {
  delete static_cast<std::vector<T>*>(ptr);
}

template <typename T>
nb::ndarray<nb::numpy, T> owned_array(std::vector<T>&& values, std::initializer_list<size_t> shape) {
  auto* storage = new std::vector<T>(std::move(values));
  nb::capsule owner(storage, &delete_vector_capsule<T>);
  return nb::ndarray<nb::numpy, T>(storage->data(), shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T> view_array(T* ptr, std::initializer_list<size_t> shape, nb::handle owner) {
  return nb::ndarray<nb::numpy, T>(ptr, shape, owner);
}

inline nb::tuple signed_measure_to_python(const signed_measure_type& sm, size_t width) {
  std::vector<indices_type> flat_pts;
  flat_pts.reserve(sm.first.size() * width);
  for (const auto& row : sm.first) {
    flat_pts.insert(flat_pts.end(), row.begin(), row.end());
  }
  std::vector<tensor_dtype> weights(sm.second.begin(), sm.second.end());
  return nb::make_tuple(nb::cast(owned_array<indices_type>(std::move(flat_pts), {sm.first.size(), width})),
                        nb::cast(owned_array<tensor_dtype>(std::move(weights), {sm.second.size()})));
}

template <typename T>
nb::ndarray<nb::numpy, T> filled_array(size_t size, T value) {
  return owned_array<T>(std::vector<T>(size, value), {size});
}

template <typename T>
std::vector<T> cast_vector(const nb::handle& h) {
  nb::object np = nb::module_::import_("numpy");
  nb::object obj = nb::borrow(h);
  if (nb::hasattr(obj, "shape")) {
    obj = np.attr("asarray")(obj).attr("reshape")(-1);
  }
  return nb::cast<std::vector<T>>(obj);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(const nb::handle& h) {
  nb::object np = nb::module_::import_("numpy");
  nb::object obj = nb::borrow(h);
  if (nb::hasattr(obj, "shape")) {
    obj = np.attr("asarray")(obj);
  }
  return nb::cast<std::vector<std::vector<T>>>(obj);
}

template <typename T>
std::vector<std::vector<std::vector<T>>> cast_tensor3(const nb::handle& h) {
  return nb::cast<std::vector<std::vector<std::vector<T>>>>(h);
}

inline bool is_none_or_empty(const nb::handle& h) {
  if (!h.is_valid() || h.is_none()) {
    return true;
  }
  if (nb::hasattr(h, "__len__")) {
    return nb::len(h) == 0;
  }
  return false;
}

inline std::string lowercase_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline std::string numpy_dtype_name(const nb::handle& dtype) {
  nb::object np = nb::module_::import_("numpy");
  return nb::cast<std::string>(np.attr("dtype")(nb::borrow(dtype)).attr("name"));
}

inline nb::object numpy_dtype_type(std::string_view name);

inline bool has_template_id(const nb::handle& input) { return nb::hasattr(input, "_template_id"); }

inline int template_id_of(const nb::handle& input) { return nb::cast<int>(input.attr("_template_id")); }

inline void print_flush(const std::string& message, std::string end = "\n") {
  nb::module_::import_("builtins").attr("print")(message, "end"_a = end, "flush"_a = true);
}

struct SlicerRuntimeInfo {
  bool is_squeezed;
  nb::object filtration_grid;
};

inline bool has_slicer_template_id(const nb::handle& input) {
  return has_template_id(input) && nb::hasattr(input, "col_type") && nb::hasattr(input, "pers_backend");
}

SlicerRuntimeInfo get_slicer_runtime_info(const nb::handle& input) {
  if (!has_slicer_template_id(input)) {
    throw nb::value_error("First argument must be a simplextree or a slicer !");
  }
  return dispatch_slicer_by_template_id(template_id_of(input), [&]<typename Desc>() -> SlicerRuntimeInfo {
    const auto& wrapper = nb::cast<const typename Desc::wrapper&>(input);
    return SlicerRuntimeInfo{!is_none_or_empty(wrapper.filtration_grid), wrapper.filtration_grid};
  });
}

nb::object compute_filtration_bounds(const nb::handle& input) {
  if (!has_slicer_template_id(input)) {
    throw nb::value_error("First argument must be a simplextree or a slicer !");
  }
  return dispatch_slicer_by_template_id(template_id_of(input), [&]<typename Desc>() -> nb::object {
    using Wrapper = typename Desc::wrapper;
    using Value = typename Desc::value_type;
    const auto& wrapper = nb::cast<const Wrapper&>(input);
    size_t num_parameters = Desc::is_degree_rips ? size_t(2) : wrapper.truc.get_number_of_parameters();
    std::vector<Value> mins(num_parameters);
    std::vector<Value> maxs(num_parameters);
    bool initialized = false;
    {
      nb::gil_scoped_release release;
      const auto& filtrations = wrapper.truc.get_filtration_values();
      for (size_t i = 0; i < filtrations.size(); ++i) {
        const auto& filtration = filtrations[i];
        if constexpr (Desc::is_degree_rips) {
          for (size_t g = 0; g < filtration.num_generators(); ++g) {
            Value values[2] = {filtration(g, 0), static_cast<Value>(g)};
            if (!initialized) {
              mins.assign(values, values + 2);
              maxs.assign(values, values + 2);
              initialized = true;
            } else {
              for (size_t p = 0; p < 2; ++p) {
                mins[p] = std::min(mins[p], values[p]);
                maxs[p] = std::max(maxs[p], values[p]);
              }
            }
          }
        } else {
          for (size_t g = 0; g < filtration.num_generators(); ++g) {
            for (size_t p = 0; p < num_parameters; ++p) {
              Value value = filtration.is_finite() ? filtration(g, p) : filtration(g, 0);
              if (!initialized) {
                std::fill(mins.begin(), mins.end(), value);
                std::fill(maxs.begin(), maxs.end(), value);
                initialized = true;
              }
              mins[p] = std::min(mins[p], value);
              maxs[p] = std::max(maxs[p], value);
            }
          }
        }
      }
    }
    nb::object np = nb::module_::import_("numpy");
    if (!initialized) {
      return np.attr("empty")(nb::make_tuple(2, 0), "dtype"_a = numpy_dtype_type(Desc::dtype_name));
    }
    return np.attr("array")(nb::make_tuple(nb::cast(mins), nb::cast(maxs)),
                            "dtype"_a = numpy_dtype_type(Desc::dtype_name));
  });
}

template <typename Desc>
bool slicer_class_matches(bool is_vineyard,
                          bool is_k_critical,
                          const std::string& dtype_name,
                          const std::string& col,
                          const std::string& pers_backend,
                          const std::string& filtration_container) {
  return Desc::is_vine == is_vineyard && Desc::is_kcritical == is_k_critical && Desc::dtype_name == dtype_name &&
         lowercase_copy(std::string(Desc::column_type)) == col &&
         lowercase_copy(std::string(Desc::backend_type)) == pers_backend &&
         lowercase_copy(std::string(Desc::filtration_container)) == filtration_container;
}

template <typename... Desc>
nb::object get_slicer_class(type_list<Desc...>,
                            bool is_vineyard,
                            bool is_k_critical,
                            const nb::handle& dtype,
                            std::string col,
                            std::string pers_backend,
                            std::string filtration_container) {
  std::string dtype_name = numpy_dtype_name(dtype);
  col = lowercase_copy(std::move(col));
  pers_backend = lowercase_copy(std::move(pers_backend));
  filtration_container = lowercase_copy(std::move(filtration_container));
  bool matched = false;
  nb::object result;
  (
      [&] {
        if (!matched && slicer_class_matches<Desc>(
                            is_vineyard, is_k_critical, dtype_name, col, pers_backend, filtration_container)) {
          result = nb::module_::import_("multipers._slicer_nanobind").attr(Desc::python_name.data());
          matched = true;
        }
      }(),
      ...);
  if (!matched) {
    throw nb::value_error("Unimplemented slicer combination.");
  }
  return result;
}

inline nb::object prepare_box_object(const std::vector<std::vector<double>>& box) {
  return nb::module_::import_("numpy").attr("array")(nb::cast(box));
}

template <typename Wrapper>
nb::object self_handle(Wrapper& self) {
  return nb::find(self);
}

template <typename Concrete, typename Rows>
typename Concrete::Filtration_value make_kcritical_filtration(const Rows& rows) {
  using TargetValue = typename Concrete::Filtration_value::value_type;

  size_t num_parameters = rows.empty() ? 0 : rows.front().size();
  typename Concrete::Filtration_value filtration(num_parameters);
  auto inf = Concrete::Filtration_value::inf(num_parameters);
  filtration.push_to_least_common_upper_bound(inf, false);
  for (const auto& row : rows) {
    if constexpr (std::is_same_v<typename Rows::value_type::value_type, TargetValue>) {
      filtration.add_generator(row);
    } else {
      std::vector<TargetValue> converted(row.begin(), row.end());
      filtration.add_generator(converted);
    }
  }
  return filtration;
}

template <typename Wrapper, typename Concrete, typename SourceInterface>
bool try_build_kcritical_from_simplextree_scc(Wrapper& self, SourceInterface* st_ptr, const nb::handle& source) {
  using Complex = Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value>;
  Concrete built;
  {
    nb::gil_scoped_release release;
    auto blocks = st_ptr->kcritical_simplextree_to_scc();
    size_t total_size = 0;
    std::vector<size_t> block_sizes;
    block_sizes.reserve(blocks.size());
    for (const auto& block : blocks) {
      block_sizes.push_back(block.first.size());
      total_size += block.first.size();
    }

    typename Complex::Boundary_container boundaries;
    typename Complex::Dimension_container dimensions;
    typename Complex::Filtration_value_container filtrations;
    boundaries.reserve(total_size);
    dimensions.reserve(total_size);
    filtrations.reserve(total_size);

    size_t shift = 0;
    size_t previous_block_size = 0;
    for (size_t dim = 0; dim < blocks.size(); ++dim) {
      const auto& block_filtrations = blocks[dim].first;
      const auto& block_boundaries = blocks[dim].second;
      for (size_t i = 0; i < block_filtrations.size(); ++i) {
        typename Complex::Boundary boundary;
        boundary.reserve(block_boundaries[i].size());
        for (unsigned int value : block_boundaries[i]) {
          boundary.push_back(static_cast<uint32_t>(value + shift));
        }
        std::sort(boundary.begin(), boundary.end());
        boundaries.push_back(std::move(boundary));
        dimensions.push_back(static_cast<int>(dim));
        filtrations.push_back(make_kcritical_filtration<Concrete>(block_filtrations[i]));
      }
      shift += previous_block_size;
      previous_block_size = block_sizes[dim];
    }

    built = Concrete(Complex(std::move(boundaries), std::move(dimensions), std::move(filtrations)));
  }
  self.truc = std::move(built);
  if (nb::hasattr(source, "filtration_grid")) {
    self.filtration_grid = source.attr("filtration_grid");
  } else {
    self.filtration_grid = nb::none();
  }
  self.minpres_degree = -1;
  return true;
}

inline bool is_simplextree_multi(const nb::handle& source) {
  auto checker = nb::module_::import_("multipers.simplex_tree_multi").attr("is_simplextree_multi");
  return nb::cast<bool>(checker(source));
}

template <typename Desc, typename Wrapper, typename Concrete>
void build_from_simplextree_desc(Wrapper& self, const nb::handle& source, bool is_function_simplextree, intptr_t ptr) {
  using SourceInterface = typename Desc::interface_type;
  auto* st_ptr = reinterpret_cast<SourceInterface*>(ptr);
  if constexpr (Desc::is_kcritical) {
    if (!is_function_simplextree) {
      (void)try_build_kcritical_from_simplextree_scc<Wrapper, Concrete>(self, st_ptr, source);
      return;
    }
  }
  {
    nb::gil_scoped_release release;
    self.truc = Gudhi::multi_persistence::build_slicer_from_simplex_tree<Concrete>(*st_ptr);
  }
  if (nb::hasattr(source, "filtration_grid")) {
    self.filtration_grid = source.attr("filtration_grid");
  } else {
    self.filtration_grid = nb::none();
  }
  self.minpres_degree = -1;
}

template <typename Wrapper, typename Concrete>
bool try_build_from_multipers_simplextree(Wrapper& self, const nb::handle& source) {
  if (!is_simplextree_multi(source) || !has_template_id(source)) {
    return false;
  }

  bool is_function_simplextree =
      nb::hasattr(source, "_is_function_simplextree") ? nb::cast<bool>(source.attr("_is_function_simplextree")) : false;
  intptr_t ptr = nb::cast<intptr_t>(source.attr("thisptr"));
  dispatch_simplextree_by_template_id(template_id_of(source), [&]<typename Desc>() {
    build_from_simplextree_desc<Desc, Wrapper, Concrete>(self, source, is_function_simplextree, ptr);
  });
  return true;
}

template <typename Barcode, typename Value>
nb::tuple dim_barcode_to_tuple(const Barcode& barcode) {
  size_t dims = barcode.size();
  nb::tuple out = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)dims));
  for (size_t dim = 0; dim < dims; ++dim) {
    auto& bc = barcode[dim];
    std::vector<Value> flat;
    flat.reserve(bc.size() * 2);
    auto* data = bc.data();
    for (size_t i = 0; i < bc.size(); ++i) {
      flat.push_back(data[i][0]);
      flat.push_back(data[i][1]);
    }
    nb::object value = nb::cast(owned_array<Value>(std::move(flat), {bc.size(), size_t(2)}));
    PyTuple_SET_ITEM(out.ptr(), (Py_ssize_t)dim, value.release().ptr());
  }
  return out;
}

template <typename Wrapper>
nb::object dimensions_array(Wrapper& self) {
  std::vector<int> dims;
  {
    nb::gil_scoped_release release;
    dims = self.truc.get_dimensions();
  }
  std::vector<int32_t> out(dims.begin(), dims.end());
  return nb::cast(owned_array<int32_t>(std::move(out), {dims.size()}));
}

template <typename Wrapper>
nb::object boundaries_object(Wrapper& self, bool packed) {
  std::vector<uint64_t> indptr;
  std::vector<uint32_t> indices;
  size_t num_rows = 0;
  {
    nb::gil_scoped_release release;
    const auto& boundaries = self.truc.get_boundaries();
    num_rows = boundaries.size();
    indptr.assign(num_rows + 1, 0);
    size_t total_size = 0;
    for (size_t i = 0; i < num_rows; ++i) {
      total_size += boundaries[i].size();
      indptr[i + 1] = total_size;
    }
    indices.reserve(total_size);
    for (const auto& row : boundaries) {
      indices.insert(indices.end(), row.begin(), row.end());
    }
  }

  size_t total_size = indices.size();
  auto indptr_arr = owned_array<uint64_t>(std::move(indptr), {num_rows + 1});
  auto indices_arr = owned_array<uint32_t>(std::move(indices), {total_size});
  if (packed) {
    return nb::make_tuple(indptr_arr, indices_arr);
  }

  nb::tuple out = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)num_rows));
  auto indptr_view = indptr_arr.template view<uint64_t, nb::ndim<1>>();
  auto indices_view = indices_arr.template view<uint32_t, nb::ndim<1>>();
  for (size_t i = 0; i < num_rows; ++i) {
    uint64_t start = indptr_view(i);
    uint64_t stop = indptr_view(i + 1);
    std::vector<uint32_t> row(stop - start);
    for (uint64_t j = start; j < stop; ++j) {
      row[j - start] = indices_view(j);
    }
    nb::object value = nb::cast(owned_array<uint32_t>(std::move(row), {size_t(stop - start)}));
    PyTuple_SET_ITEM(out.ptr(), (Py_ssize_t)i, value.release().ptr());
  }
  return out;
}

template <typename Wrapper, typename Value, bool IsKCritical, bool IsDegreeRips>
nb::object filtration_value_to_python(Wrapper& self, size_t idx, bool copy, bool raw) {
  auto& filtration = self.truc.get_filtration_value(idx);
  nb::object owner = self_handle(self);

  if constexpr (IsDegreeRips) {
    size_t k = filtration.num_generators();
    if (raw) {
      if (copy) {
        std::vector<Value> values(k);
        for (size_t i = 0; i < k; ++i) {
          values[i] = filtration(i, 0);
        }
        return nb::cast(owned_array<Value>(std::move(values), {k}));
      }
      return nb::cast(view_array<Value>(&filtration(0, 0), {k}, owner));
    }

    std::vector<Value> values(k * 2);
    for (size_t i = 0; i < k; ++i) {
      values[2 * i] = filtration(i, 0);
      values[2 * i + 1] = static_cast<Value>(i);
    }
    return nb::cast(owned_array<Value>(std::move(values), {k, size_t(2)}));
  } else if constexpr (IsKCritical) {
    size_t k = filtration.num_generators();
    size_t p = filtration.num_parameters();
    nb::list out;
    if (!filtration.is_finite()) {
      for (size_t i = 0; i < k; ++i) {
        out.append(filled_array<Value>(p, filtration(i, 0)));
      }
      return out;
    }
    for (size_t i = 0; i < k; ++i) {
      if (copy) {
        std::vector<Value> row(p);
        for (size_t j = 0; j < p; ++j) {
          row[j] = filtration(i, j);
        }
        out.append(owned_array<Value>(std::move(row), {p}));
      } else {
        out.append(view_array<Value>(&filtration(i, 0), {p}, owner));
      }
    }
    return out;
  } else {
    size_t p = filtration.num_parameters();
    if (!filtration.is_finite()) {
      return nb::cast(filled_array<Value>(p, filtration(0, 0)));
    }
    if (copy) {
      std::vector<Value> row(p);
      std::memcpy(row.data(), &filtration(0, 0), p * sizeof(Value));
      return nb::cast(owned_array<Value>(std::move(row), {p}));
    }
    return nb::cast(view_array<Value>(&filtration(0, 0), {p}, owner));
  }
}

template <typename Wrapper, typename Value, bool IsKCritical, bool IsDegreeRips>
nb::object pack_filtrations(Wrapper& self, bool raw) {
  auto& filtrations = self.truc.get_filtration_values();
  size_t num_stuff = filtrations.size();
  std::vector<int64_t> indptr;
  indptr.resize(num_stuff + 1, 0);

  if constexpr (!IsKCritical) {
    throw std::runtime_error("packed=True is only available for k-critical filtrations.");
  } else if constexpr (IsDegreeRips) {
    size_t total = 0;
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < num_stuff; ++i) {
        total += filtrations[i].num_generators();
        indptr[i + 1] = static_cast<int64_t>(total);
      }
    }
    auto indptr_arr = owned_array<int64_t>(std::move(indptr), {num_stuff + 1});
    if (raw) {
      std::vector<Value> grades(total);
      {
        nb::gil_scoped_release release;
        size_t offset = 0;
        for (size_t i = 0; i < num_stuff; ++i) {
          size_t k = filtrations[i].num_generators();
          for (size_t g = 0; g < k; ++g) {
            grades[offset + g] = filtrations[i](g, 0);
          }
          offset += k;
        }
      }
      return nb::make_tuple(indptr_arr, owned_array<Value>(std::move(grades), {total}));
    }
    std::vector<Value> grades(total * 2);
    {
      nb::gil_scoped_release release;
      size_t offset = 0;
      for (size_t i = 0; i < num_stuff; ++i) {
        size_t k = filtrations[i].num_generators();
        for (size_t g = 0; g < k; ++g) {
          grades[2 * (offset + g)] = filtrations[i](g, 0);
          grades[2 * (offset + g) + 1] = static_cast<Value>(g);
        }
        offset += k;
      }
    }
    return nb::make_tuple(indptr_arr, owned_array<Value>(std::move(grades), {total, size_t(2)}));
  } else {
    size_t total = 0;
    size_t num_parameters = self.truc.get_number_of_parameters();
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < num_stuff; ++i) {
        total += filtrations[i].num_generators();
        indptr[i + 1] = static_cast<int64_t>(total);
      }
    }
    std::vector<Value> grades(total * num_parameters);
    {
      nb::gil_scoped_release release;
      size_t offset = 0;
      for (size_t i = 0; i < num_stuff; ++i) {
        size_t k = filtrations[i].num_generators();
        for (size_t g = 0; g < k; ++g) {
          for (size_t p = 0; p < num_parameters; ++p) {
            grades[(offset + g) * num_parameters + p] = filtrations[i](g, p);
          }
        }
        offset += k;
      }
    }
    return nb::make_tuple(owned_array<int64_t>(std::move(indptr), {num_stuff + 1}),
                          owned_array<Value>(std::move(grades), {total, num_parameters}));
  }
}

template <typename Wrapper, typename Value, bool IsKCritical, bool IsDegreeRips>
nb::object copy_filtrations(Wrapper& self, bool raw) {
  auto& filtrations = self.truc.get_filtration_values();
  size_t num_stuff = filtrations.size();

  if constexpr (!IsKCritical && !IsDegreeRips) {
    size_t num_parameters = self.truc.get_number_of_parameters();
    std::vector<Value> out(num_stuff * num_parameters);
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < num_stuff; ++i) {
        if (!filtrations[i].is_finite()) {
          std::fill_n(out.data() + i * num_parameters, num_parameters, filtrations[i](0, 0));
        } else if (num_parameters > 0) {
          std::memcpy(out.data() + i * num_parameters, &filtrations[i](0, 0), num_parameters * sizeof(Value));
        }
      }
    }
    return nb::cast(owned_array<Value>(std::move(out), {num_stuff, num_parameters}));
  } else if constexpr (IsDegreeRips) {
    std::vector<std::vector<Value>> copied(num_stuff);
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < num_stuff; ++i) {
        size_t k = filtrations[i].num_generators();
        if (raw) {
          copied[i].resize(k);
          for (size_t g = 0; g < k; ++g) {
            copied[i][g] = filtrations[i](g, 0);
          }
        } else {
          copied[i].resize(k * 2);
          for (size_t g = 0; g < k; ++g) {
            copied[i][2 * g] = filtrations[i](g, 0);
            copied[i][2 * g + 1] = static_cast<Value>(g);
          }
        }
      }
    }
    nb::list out;
    for (size_t i = 0; i < num_stuff; ++i) {
      if (raw) {
        out.append(owned_array<Value>(std::move(copied[i]), {copied[i].size()}));
      } else {
        out.append(owned_array<Value>(std::move(copied[i]), {copied[i].size() / 2, size_t(2)}));
      }
    }
    return out;
  } else {
    std::vector<std::vector<Value>> copied(num_stuff);
    std::vector<size_t> ks(num_stuff, 0);
    std::vector<size_t> ps(num_stuff, 0);
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < num_stuff; ++i) {
        auto& filtration = filtrations[i];
        size_t k = filtration.num_generators();
        size_t p = filtration.num_parameters();
        ks[i] = k;
        ps[i] = p;
        copied[i].resize(k * p);
        if (!filtration.is_finite()) {
          for (size_t g = 0; g < k; ++g) {
            std::fill_n(copied[i].data() + g * p, p, filtration(g, 0));
          }
        } else {
          for (size_t g = 0; g < k; ++g) {
            for (size_t j = 0; j < p; ++j) {
              copied[i][g * p + j] = filtration(g, j);
            }
          }
        }
      }
    }
    nb::list out;
    for (size_t i = 0; i < num_stuff; ++i) {
      out.append(owned_array<Value>(std::move(copied[i]), {ks[i], ps[i]}));
    }
    return out;
  }
}

template <typename Wrapper, typename Value, bool IsKCritical, bool IsDegreeRips>
nb::object filtration_values_array(Wrapper& self) {
  auto& filtrations = self.truc.get_filtration_values();
  size_t num_parameters = self.truc.get_number_of_parameters();
  size_t total = 0;
  {
    nb::gil_scoped_release release;
    for (size_t i = 0; i < filtrations.size(); ++i) {
      total += filtrations[i].num_generators();
    }
  }

  std::vector<Value> out;
  if constexpr (IsDegreeRips) {
    out.resize(total * 2);
    {
      nb::gil_scoped_release release;
      size_t offset = 0;
      for (size_t i = 0; i < filtrations.size(); ++i) {
        for (size_t g = 0; g < filtrations[i].num_generators(); ++g) {
          out[2 * (offset + g)] = filtrations[i](g, 0);
          out[2 * (offset + g) + 1] = static_cast<Value>(g);
        }
        offset += filtrations[i].num_generators();
      }
    }
    return nb::cast(owned_array<Value>(std::move(out), {total, size_t(2)}));
  } else {
    out.resize(total * num_parameters);
    {
      nb::gil_scoped_release release;
      size_t offset = 0;
      for (size_t i = 0; i < filtrations.size(); ++i) {
        for (size_t g = 0; g < filtrations[i].num_generators(); ++g) {
          for (size_t p = 0; p < num_parameters; ++p) {
            out[(offset + g) * num_parameters + p] = filtrations[i](g, p);
          }
        }
        offset += filtrations[i].num_generators();
      }
    }
    return nb::cast(owned_array<Value>(std::move(out), {total, num_parameters}));
  }
}

template <typename TargetWrapper, typename TargetConcrete>
bool try_copy_from_existing(TargetWrapper& self, const nb::handle& source) {
  if (!has_slicer_template_id(source)) {
    return false;
  }
  dispatch_slicer_by_template_id(template_id_of(source), [&]<typename Desc>() {
    const auto& other = nb::cast<const typename Desc::wrapper&>(source);
    {
      nb::gil_scoped_release release;
      self.truc = TargetConcrete(other.truc);
    }
    self.filtration_grid = other.filtration_grid;
    self.minpres_degree = other.minpres_degree;
  });
  return true;
}

template <typename Filtration>
std::vector<std::vector<typename Filtration::value_type>> sorted_generators(const Filtration& filtration) {
  using Value = typename Filtration::value_type;
  std::vector<std::vector<Value>> out(filtration.num_generators(), std::vector<Value>(filtration.num_parameters()));
  for (size_t g = 0; g < filtration.num_generators(); ++g) {
    for (size_t p = 0; p < filtration.num_parameters(); ++p) {
      out[g][p] = filtration(g, p);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

template <typename Concrete>
bool equal_kcritical_filtrations(const Concrete& lhs, const Concrete& rhs) {
  const auto& lhs_filtrations = lhs.get_filtration_values();
  const auto& rhs_filtrations = rhs.get_filtration_values();
  if (lhs_filtrations.size() != rhs_filtrations.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs_filtrations.size(); ++i) {
    if (lhs_filtrations[i].num_parameters() != rhs_filtrations[i].num_parameters() ||
        lhs_filtrations[i].num_generators() != rhs_filtrations[i].num_generators()) {
      return false;
    }
    if (sorted_generators(lhs_filtrations[i]) != sorted_generators(rhs_filtrations[i])) {
      return false;
    }
  }
  return true;
}

template <typename Wrapper, typename Concrete, bool IsKCritical>
bool wrapper_equals(Wrapper& self, const nb::handle& other) {
  if (!other.is_valid() || other.is_none()) {
    return false;
  }
  if (!is_none_or_empty(self.filtration_grid)) {
    return nb::cast<bool>(self_handle(self).attr("unsqueeze")().attr("__eq__")(other));
  }
  if (nb::hasattr(other, "is_squeezed") && nb::cast<bool>(other.attr("is_squeezed"))) {
    return nb::cast<bool>(self_handle(self).attr("__eq__")(other.attr("unsqueeze")()));
  }

  Wrapper rhs;
  if (!try_copy_from_existing<Wrapper, Concrete>(rhs, other)) {
    return false;
  }
  if (self.truc.get_dimensions() != rhs.truc.get_dimensions()) {
    return false;
  }
  if (self.truc.get_boundaries() != rhs.truc.get_boundaries()) {
    return false;
  }
  if constexpr (IsKCritical) {
    return equal_kcritical_filtrations(self.truc, rhs.truc);
  } else {
    return self.truc.get_filtration_values() == rhs.truc.get_filtration_values();
  }
}

template <typename Wrapper>
void reset_python_state(Wrapper& self) {
  self.filtration_grid = nb::none();
  self.minpres_degree = -1;
}

template <typename Wrapper, typename Concrete, typename Value, bool IsKCritical>
Wrapper construct_from_generator_data(nb::object generator_maps,
                                      nb::object generator_dimensions,
                                      nb::object filtration_values) {
  Wrapper out;
  if (is_none_or_empty(generator_maps)) {
    return out;
  }

  std::vector<std::vector<uint32_t>> boundaries;
  boundaries.reserve(nb::len(generator_maps));
  for (nb::handle row_handle : nb::iter(generator_maps)) {
    boundaries.push_back(cast_vector<uint32_t>(row_handle));
  }

  auto dims = cast_vector<int>(generator_dimensions);
  if (boundaries.size() != dims.size()) {
    throw std::runtime_error("Invalid input, shape do not coincide.");
  }

  std::vector<typename Concrete::Filtration_value> c_filtrations;
  c_filtrations.reserve(boundaries.size());

  if constexpr (IsKCritical) {
    std::vector<std::vector<std::vector<Value>>> py_filtrations;
    py_filtrations.reserve(nb::len(filtration_values));
    for (nb::handle rows_handle : nb::iter(filtration_values)) {
      py_filtrations.push_back(cast_matrix<Value>(rows_handle));
    }
    if (py_filtrations.size() != boundaries.size()) {
      throw std::runtime_error("Invalid input, shape do not coincide.");
    }
    size_t num_parameters = 0;
    if (!py_filtrations.empty() && !py_filtrations[0].empty()) {
      num_parameters = py_filtrations[0][0].size();
    }
    for (const auto& rows : py_filtrations) {
      typename Concrete::Filtration_value filtration(num_parameters);
      auto inf = Concrete::Filtration_value::inf(num_parameters);
      filtration.push_to_least_common_upper_bound(inf, false);
      for (const auto& row : rows) {
        filtration.add_generator(row);
      }
      c_filtrations.push_back(std::move(filtration));
    }
  } else {
    auto py_filtrations = cast_matrix<Value>(filtration_values);
    if (py_filtrations.size() != boundaries.size()) {
      throw std::runtime_error("Invalid input, shape do not coincide.");
    }
    for (const auto& row : py_filtrations) {
      c_filtrations.emplace_back(row);
    }
  }

  Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value> cpx(
      boundaries, dims, c_filtrations);
  out.truc = Concrete(cpx);
  reset_python_state(out);
  return out;
}

inline nb::object numpy_dtype_type(std::string_view name) {
  nb::object np = nb::module_::import_("numpy");
  return np.attr("dtype")(std::string(name)).attr("type");
}

template <typename Desc>
typename Desc::wrapper construct_from_source(nb::object source) {
  using Wrapper = typename Desc::wrapper;
  using Concrete = typename Desc::concrete;
  using Value = typename Desc::value_type;

  Wrapper out;
  if (is_none_or_empty(source)) {
    return out;
  }
  if (try_copy_from_existing<Wrapper, Concrete>(out, source)) {
    return out;
  }
  if (is_simplextree_multi(source)) {
    if (try_build_from_multipers_simplextree<Wrapper, Concrete>(out, source)) {
      return out;
    }
    throw std::runtime_error("Unsupported SimplexTreeMulti input type.");
  }

  auto helper = nb::module_::import_("multipers._slicer_meta").attr("_blocks2boundary_dimension_grades");
  nb::object blocks = nb::hasattr(source, "_to_scc") ? source.attr("_to_scc")() : source;
  nb::tuple converted = nb::cast<nb::tuple>(helper(blocks, "inplace"_a = false, "is_kcritical"_a = Desc::is_kcritical));
  return construct_from_generator_data<Wrapper, Concrete, Value, Desc::is_kcritical>(
      converted[0], converted[1], converted[2]);
}

template <typename Desc, typename Class>
void bind_grid_methods(Class& cls) {
  if constexpr (Desc::has_grid_methods) {
    using Wrapper = typename Desc::wrapper;
    using Concrete = typename Desc::concrete;
    using Value = typename Desc::value_type;
    using TargetWrapper = typename Desc::coarsened_wrapper;

    cls.def("coarsen_on_grid_copy",
            [](Wrapper& self, std::vector<std::vector<Value>> grid) {
              TargetWrapper out;
              {
                nb::gil_scoped_release release;
                out.truc = build_slicer_coarsen_on_grid(self.truc, grid);
              }
              out.filtration_grid = self.filtration_grid;
              out.minpres_degree = self.minpres_degree;
              return out;
            })
        .def(
            "compute_kernel_projective_cover",
            [](Wrapper& self, nb::object dim_obj) {
              Wrapper out;
              if (self.truc.get_number_of_cycle_generators() == 0) {
                out.filtration_grid = self.filtration_grid;
                out.minpres_degree = self.minpres_degree;
                return out;
              }
              int dim = dim_obj.is_none() ? self.truc.get_dimension(0) : nb::cast<int>(dim_obj);
              {
                nb::gil_scoped_release release;
                out.truc = build_slicer_from_projective_cover_kernel(self.truc, dim);
              }
              out.filtration_grid = self.filtration_grid;
              out.minpres_degree = self.minpres_degree;
              return out;
            },
            "dim"_a = nb::none());
  }
}

template <typename Desc, typename Class>
void bind_vine_methods(Class& cls) {
  if constexpr (Desc::is_vine) {
    using Wrapper = typename Desc::wrapper;
    using Value = typename Desc::value_type;

    cls.def(
           "vine_update",
           [](Wrapper& self, nb::object basepoint, nb::object direction) -> Wrapper& {
             std::vector<Value> bp = cast_vector<Value>(basepoint);
             std::vector<Value> dir;
             bool has_direction = !direction.is_none();
             if (has_direction) {
               dir = cast_vector<Value>(direction);
             }
             {
               nb::gil_scoped_release release;
               if (has_direction) {
                 self.truc.push_to(Gudhi::multi_persistence::Line<Value>(bp, dir));
               } else {
                 self.truc.push_to(Gudhi::multi_persistence::Line<Value>(bp));
               }
               self.truc.update_persistence_computation();
             }
             return self;
           },
           "basepoint"_a,
           "direction"_a = nb::none(),
           nb::rv_policy::reference_internal)
        .def(
            "get_representative_cycles",
            [](Wrapper& self, bool update, nb::object idx_obj) {
              std::vector<int64_t> requested;
              bool filter_cycles = !idx_obj.is_none();
              if (filter_cycles) {
                requested = cast_vector<int64_t>(idx_obj);
              }
              std::vector<std::vector<std::vector<std::vector<uint32_t>>>> out_cpp;
              {
                nb::gil_scoped_release release;
                auto cycle_idx = self.truc.get_representative_cycles(update);
                std::vector<std::vector<size_t>> selected_indices(cycle_idx.size());
                if (!filter_cycles) {
                  for (size_t i = 0; i < cycle_idx.size(); ++i) {
                    selected_indices[i].resize(cycle_idx[i].size());
                    for (size_t j = 0; j < cycle_idx[i].size(); ++j) {
                      selected_indices[i][j] = j;
                    }
                  }
                } else {
                  std::vector<size_t> offsets(cycle_idx.size() + 1, 0);
                  for (size_t i = 0; i < cycle_idx.size(); ++i) {
                    offsets[i + 1] = offsets[i] + cycle_idx[i].size();
                  }
                  size_t total_cycles = offsets.back();
                  for (int64_t raw_idx : requested) {
                    int64_t normalized = raw_idx;
                    if (normalized < 0) {
                      normalized += static_cast<int64_t>(total_cycles);
                    }
                    if (normalized < 0 || normalized >= static_cast<int64_t>(total_cycles)) {
                      throw nb::index_error("Representative cycle index out of range.");
                    }
                    size_t current = static_cast<size_t>(normalized);
                    auto it = std::upper_bound(offsets.begin(), offsets.end(), current);
                    size_t dim = static_cast<size_t>(std::distance(offsets.begin(), it) - 1);
                    selected_indices[dim].push_back(current - offsets[dim]);
                  }
                }
                out_cpp.resize(cycle_idx.size());
                for (size_t i = 0; i < cycle_idx.size(); ++i) {
                  out_cpp[i].resize(selected_indices[i].size());
                }
                tbb::parallel_for(size_t(0), cycle_idx.size(), [&](size_t i) {
                  for (size_t j = 0; j < selected_indices[i].size(); ++j) {
                    size_t selected_idx = selected_indices[i][j];
                    if (!cycle_idx[i][selected_idx].empty()) {
                      if (self.truc.get_boundary(cycle_idx[i][selected_idx][0]).empty()) {
                        out_cpp[i][j] = {std::vector<uint32_t>{}};
                      } else {
                        out_cpp[i][j].resize(cycle_idx[i][selected_idx].size());
                        for (size_t k = 0; k < cycle_idx[i][selected_idx].size(); ++k) {
                          out_cpp[i][j][k] = self.truc.get_boundary(cycle_idx[i][selected_idx][k]);
                        }
                      }
                    }
                  }
                });
              }
              nb::list out;
              for (size_t i = 0; i < out_cpp.size(); ++i) {
                nb::list dim_cycles;
                for (size_t j = 0; j < out_cpp[i].size(); ++j) {
                  nb::list cycle;
                  for (size_t k = 0; k < out_cpp[i][j].size(); ++k) {
                    auto boundary = std::move(out_cpp[i][j][k]);
                    cycle.append(nb::cast(owned_array<uint32_t>(std::move(boundary), {boundary.size()})));
                  }
                  dim_cycles.append(cycle);
                }
                out.append(dim_cycles);
              }
              return out;
            },
            "update"_a = true,
            "idx"_a = nb::none())
        .def(
            "get_most_persistent_cycle",
            [](Wrapper& self, int dim, bool update, bool idx) {
              std::vector<uint32_t> cycle_idx;
              std::vector<std::vector<uint32_t>> out_cpp;
              {
                nb::gil_scoped_release release;
                cycle_idx = self.truc.get_most_persistent_cycle(dim, update);
                if (!idx && !cycle_idx.empty()) {
                  if (self.truc.get_boundary(cycle_idx[0]).empty()) {
                    out_cpp.push_back(std::vector<uint32_t>{});
                  } else {
                    out_cpp.resize(cycle_idx.size());
                    for (size_t k = 0; k < cycle_idx.size(); ++k) {
                      out_cpp[k] = self.truc.get_boundary(cycle_idx[k]);
                    }
                  }
                }
              }
              if (idx) {
                return nb::cast(owned_array<uint32_t>(std::move(cycle_idx), {cycle_idx.size()}));
              }
              nb::list out;
              for (size_t k = 0; k < out_cpp.size(); ++k) {
                auto boundary = std::move(out_cpp[k]);
                out.append(nb::cast(owned_array<uint32_t>(std::move(boundary), {boundary.size()})));
              }
              return nb::object(out);
            },
            "dim"_a = 1,
            "update"_a = true,
            "idx"_a = false)
        .def("get_permutation", [](Wrapper& self) -> nb::object {
          std::vector<uint32_t> order;
          {
            nb::gil_scoped_release release;
            order = self.truc.get_current_order();
          }
          return nb::cast(owned_array<uint32_t>(std::move(order), {order.size()}));
        });
  }
}

template <typename Desc>
void bind_slicer_class(nb::module_& m, nb::list& available_slicers) {
  using Wrapper = typename Desc::wrapper;
  using Concrete = typename Desc::concrete;
  using Value = typename Desc::value_type;

  auto cls = nb::class_<Wrapper>(m, Desc::python_name.data())
      .def(nb::init<>())
      .def(
          nb::new_([](nb::object source) {
            return construct_from_source<Desc>(source);
          }),
          "source"_a = nb::none())
      .def(
          nb::new_([](nb::object generator_maps, nb::object generator_dimensions, nb::object filtration_values) {
            return construct_from_generator_data<Wrapper, Concrete, Value, Desc::is_kcritical>(
                generator_maps,
                generator_dimensions,
                filtration_values);
          }),
          "generator_maps"_a,
          "generator_dimensions"_a,
          "filtration_values"_a)
      .def_prop_rw(
          "filtration_grid",
          [](Wrapper& self) -> nb::object { return self.filtration_grid; },
          [](Wrapper& self, nb::object value) {
            self.filtration_grid = value.is_none() ? nb::none() : value;
          },
          nb::arg("value").none())
      .def_rw("minpres_degree", &Wrapper::minpres_degree)
      .def("get_ptr", [](Wrapper& self) -> intptr_t { return reinterpret_cast<intptr_t>(&self.truc); })
      .def("_from_ptr", [](Wrapper& self, intptr_t slicer_ptr) -> Wrapper& {
        self.truc = *reinterpret_cast<Concrete*>(slicer_ptr);
        return self;
      }, nb::rv_policy::reference_internal)
      .def("__len__", [](Wrapper& self) -> int { return self.truc.get_number_of_cycle_generators(); })
      .def_prop_ro("num_generators", [](const Wrapper& self) -> int { return self.truc.get_number_of_cycle_generators(); })
      .def_prop_ro("num_parameters", [](const Wrapper& self) -> int { return self.truc.get_number_of_parameters(); })
      .def_prop_ro("dtype", [](const Wrapper&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def_prop_ro("_template_id", [](const Wrapper&) -> int { return Desc::template_id; })
      .def_prop_ro("col_type", [](const Wrapper&) -> std::string { return std::string(Desc::column_type); })
      .def_prop_ro("filtration_container", [](const Wrapper&) -> std::string { return std::string(Desc::filtration_container); })
      .def_prop_ro("is_vine", [](const Wrapper&) -> bool { return Desc::is_vine; })
      .def_prop_ro("is_kcritical", [](const Wrapper&) -> bool { return Desc::is_kcritical; })
      .def_prop_ro("pers_backend", [](const Wrapper&) -> std::string { return std::string(Desc::backend_type); })
      .def_prop_ro("ftype", [](const Wrapper&) -> std::string { return std::string(Desc::filtration_type); })
      .def("__eq__", [](Wrapper& self, nb::handle other) -> bool { return wrapper_equals<Wrapper, Concrete, Desc::is_kcritical>(self, other); })
      .def_static("_inf_value", []() {
        if constexpr (std::is_floating_point_v<Value>) {
          return nb::cast(std::numeric_limits<Value>::infinity());
        }
        return nb::cast(std::numeric_limits<Value>::max());
      })
      .def("get_dimensions", [](Wrapper& self) -> nb::object { return dimensions_array(self); })
      .def("get_boundaries", [](Wrapper& self, bool packed) -> nb::object { return boundaries_object(self, packed); }, "packed"_a = false)
      .def("get_filtration", [](Wrapper& self, int idx, bool raw) {
        Py_ssize_t n = static_cast<Py_ssize_t>(self.truc.get_number_of_cycle_generators());
        Py_ssize_t i = idx;
        if (i < 0) {
          i += n;
        }
        if (i < 0 || i >= n) {
          throw nb::index_error("Generator index out of range.");
        }
        return filtration_value_to_python<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, static_cast<size_t>(i), false, raw);
      }, "idx"_a, "raw"_a = false)
      .def("_get_filtrations_impl", [](Wrapper& self, bool raw, bool view, bool packed) -> nb::object {
        if (packed) {
          return pack_filtrations<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, raw);
        }
        if (view) {
          nb::list out;
          size_t n = self.truc.get_number_of_cycle_generators();
          for (size_t i = 0; i < n; ++i) {
            out.append(filtration_value_to_python<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, i, false, raw));
          }
          return nb::object(out);
        }
        return copy_filtrations<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, raw);
      }, "raw"_a = false, "view"_a = false, "packed"_a = false)
      .def("get_filtrations_values", [](Wrapper& self) -> nb::object {
        return filtration_values_array<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self);
      })
      .def("build_from_simplex_tree", [](Wrapper& self, nb::object st) -> Wrapper& {
        if (try_build_from_multipers_simplextree<Wrapper, Concrete>(self, st)) {
          return self;
        }
        throw std::runtime_error("Unsupported SimplexTreeMulti input type.");
      }, nb::rv_policy::reference_internal)
      .def("_build_from_scc_file", [](Wrapper& self, std::string path, bool rivet_compatible, bool reverse, int shift_dimension) -> Wrapper& {
        {
          nb::gil_scoped_release release;
          self.truc = Gudhi::multi_persistence::build_slicer_from_scc_file<Concrete>(path, rivet_compatible, reverse, shift_dimension);
        }
        return self;
      }, "path"_a, "rivet_compatible"_a = false, "reverse"_a = false, "shift_dimension"_a = 0, nb::rv_policy::reference_internal)
      .def("_to_scc_raw", [](Wrapper& self, std::string path, int degree, bool rivet_compatible, bool ignore_last_generators, bool strip_comments, bool reverse) -> void {
        {
          nb::gil_scoped_release release;
          write_slicer_to_scc_file(path, self.truc, degree, rivet_compatible, ignore_last_generators, strip_comments, reverse);
        }
      }, "path"_a, "degree"_a = -1, "rivet_compatible"_a = false, "ignore_last_generators"_a = false, "strip_comments"_a = false, "reverse"_a = false)
      .def("push_to_line", [](Wrapper& self, nb::object basepoint, nb::object direction) -> Wrapper& {
        std::vector<Value> bp = cast_vector<Value>(basepoint);
        std::vector<Value> dir;
        bool has_direction = !direction.is_none();
        if (has_direction) {
          dir = cast_vector<Value>(direction);
        }
        {
          nb::gil_scoped_release release;
          if (has_direction) {
            self.truc.push_to(Gudhi::multi_persistence::Line<Value>(bp, dir));
          } else {
            self.truc.push_to(Gudhi::multi_persistence::Line<Value>(bp));
          }
        }
        return self;
      }, "basepoint"_a, "direction"_a = nb::none(), nb::rv_policy::reference_internal)
      .def("set_slice", [](Wrapper& self, nb::object values) -> Wrapper& {
        auto c_values = cast_vector<Value>(values);
        {
          nb::gil_scoped_release release;
          self.truc.set_slice(c_values);
        }
        return self;
      }, nb::rv_policy::reference_internal)
      .def("initialize_persistence_computation", [](Wrapper& self, bool ignore_infinite_filtration_values) -> Wrapper& {
        {
          nb::gil_scoped_release release;
          self.truc.initialize_persistence_computation(ignore_infinite_filtration_values);
        }
        return self;
      }, "ignore_infinite_filtration_values"_a = true, nb::rv_policy::reference_internal)
      .def("update_persistence_computation", [](Wrapper& self, bool ignore_infinite_filtration_values) -> Wrapper& {
        {
          nb::gil_scoped_release release;
          self.truc.update_persistence_computation(ignore_infinite_filtration_values);
        }
        return self;
      }, "ignore_infinite_filtration_values"_a = false, nb::rv_policy::reference_internal)
      .def("get_barcode", [](Wrapper& self) -> nb::tuple {
        using Barcode = decltype(self.truc.template get_flat_barcode<true, Value, false>());
        Barcode barcode;
        {
          nb::gil_scoped_release release;
          barcode = self.truc.template get_flat_barcode<true, Value, false>();
        }
        return dim_barcode_to_tuple<Barcode, Value>(barcode);
      })
      .def("get_barcode_idx", [](Wrapper& self) -> nb::tuple {
        using Barcode = decltype(self.truc.template get_flat_barcode<true, int, true>());
        Barcode barcode;
        {
          nb::gil_scoped_release release;
          barcode = self.truc.template get_flat_barcode<true, int, true>();
        }
        return dim_barcode_to_tuple<Barcode, int>(barcode);
      })
      .def("get_current_filtration", [](Wrapper& self) -> nb::object {
        std::vector<Value> current;
        {
          nb::gil_scoped_release release;
          current = self.truc.get_slice();
        }
        return nb::cast(owned_array<Value>(std::move(current), {current.size()}));
      })
      .def("prune_above_dimension", [](Wrapper& self, int max_dimension) -> Wrapper& {
        {
          nb::gil_scoped_release release;
          self.truc.prune_above_dimension(max_dimension);
        }
        return self;
      }, nb::rv_policy::reference_internal)
      .def("coarsen_on_grid_inplace", [](Wrapper& self, std::vector<std::vector<Value>> grid, bool coordinates) -> Wrapper& {
        {
          nb::gil_scoped_release release;
          self.truc.coarsen_on_grid(grid, coordinates);
        }
        return self;
      }, nb::rv_policy::reference_internal)
      .def("to_colexical", [](Wrapper& self, bool return_permutation) {
        decltype(build_permuted_slicer(self.truc)) stuff;
        {
          nb::gil_scoped_release release;
          stuff = build_permuted_slicer(self.truc);
        }
        Wrapper out;
        out.truc = std::move(stuff.first);
        out.filtration_grid = self.filtration_grid;
        out.minpres_degree = self.minpres_degree;
        if (!return_permutation) {
          return nb::object(nb::cast(out));
        }
        std::vector<uint32_t> perm(stuff.second.begin(), stuff.second.end());
        return nb::object(nb::make_tuple(nb::cast(out), owned_array<uint32_t>(std::move(perm), {stuff.second.size()})));
      }, "return_permutation"_a = false)
      .def("permute_generators", [](Wrapper& self, std::vector<uint32_t> permutation) {
        Wrapper out;
        {
          nb::gil_scoped_release release;
          out.truc = build_permuted_slicer(self.truc, permutation);
        }
        out.filtration_grid = self.filtration_grid;
        out.minpres_degree = self.minpres_degree;
        return out;
      })
      .def("copy", [](Wrapper& self) -> Wrapper { return Wrapper(self); })
      .def("_info_string", [](Wrapper& self) -> std::string { return multipers::tmp_interface::slicer_to_str(self.truc); });

  bind_grid_methods<Desc>(cls);
  bind_vine_methods<Desc>(cls);
  available_slicers.append(cls);
}

template <typename... Desc>
void bind_all_slicers(type_list<Desc...>, nb::module_& m, nb::list& available_slicers) {
  (bind_slicer_class<Desc>(m, available_slicers), ...);
}

template <typename... Desc>
nb::tuple compute_hilbert_signed_measure(type_list<Desc...>,
                                         nb::handle slicer,
                                         std::vector<indices_type>& container,
                                         const std::vector<indices_type>& full_shape,
                                         const std::vector<indices_type>& degrees,
                                         size_t width,
                                         bool zero_pad,
                                         indices_type n_jobs,
                                         bool verbose,
                                         bool ignore_inf) {
  bool matched = false;
  nb::tuple result;
  (
      [&] {
        if (!matched && nb::isinstance<typename Desc::wrapper>(slicer)) {
          auto& wrapper = nb::cast<typename Desc::wrapper&>(slicer);
          signed_measure_type sm;
          {
            nb::gil_scoped_release release;
            sm = Gudhi::multiparameter::hilbert_function::get_hilbert_signed_measure(
                wrapper.truc, container.data(), full_shape, degrees, zero_pad, n_jobs, verbose, ignore_inf);
          }
          result = signed_measure_to_python(sm, width);
          matched = true;
        }
      }(),
      ...);
  if (!matched) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  return result;
}

template <typename... Desc>
nb::tuple compute_rank_tensor(type_list<Desc...>,
                              nb::handle slicer,
                              std::vector<tensor_dtype>& container,
                              const std::vector<indices_type>& full_shape,
                              const std::vector<indices_type>& degrees,
                              size_t total,
                              indices_type n_jobs,
                              bool ignore_inf) {
  bool matched = false;
  nb::tuple result;
  (
      [&] {
        if (!matched && nb::isinstance<typename Desc::wrapper>(slicer)) {
          auto& wrapper = nb::cast<typename Desc::wrapper&>(slicer);
          {
            nb::gil_scoped_release release;
            Gudhi::multiparameter::rank_invariant::compute_rank_invariant_python(
                wrapper.truc, container.data(), full_shape, degrees, n_jobs, ignore_inf);
          }
          result =
              nb::make_tuple(nb::cast(owned_array<tensor_dtype>(std::move(container), {total})), nb::cast(full_shape));
          matched = true;
        }
      }(),
      ...);
  if (!matched) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  return result;
}

template <typename Desc>
nb::object module_approximation_from_desc(typename Desc::wrapper& wrapper,
                                          const std::vector<double>& direction,
                                          double max_error,
                                          Gudhi::multi_persistence::Box<double> box,
                                          bool threshold,
                                          bool complete,
                                          bool verbose,
                                          int n_jobs,
                                          const nb::object& mma_module) {
  if constexpr (!Desc::enable_module_approximation) {
    throw std::runtime_error("Unsupported slicer type for module approximation.");
  } else {
    Gudhi::multi_persistence::Module<double> mod;
    {
      nb::gil_scoped_release release;
      mod = Gudhi::multiparameter::mma::multiparameter_module_approximation(
          wrapper.truc, direction, max_error, box, threshold, complete, verbose, n_jobs);
    }
    nb::object out = mma_module.attr("PyModule_f64")();
    out.attr("_from_ptr")(reinterpret_cast<intptr_t>(new Gudhi::multi_persistence::Module<double>(std::move(mod))));
    return out;
  }
}

template <typename... Desc>
nb::object compute_module_approximation_from_slicer(type_list<Desc...>,
                                                    nb::handle slicer,
                                                    const std::vector<double>& direction,
                                                    double max_error,
                                                    Gudhi::multi_persistence::Box<double> box,
                                                    bool threshold,
                                                    bool complete,
                                                    bool verbose,
                                                    int n_jobs,
                                                    const nb::object& mma_module) {
  bool matched = false;
  nb::object result;
  (
      [&] {
        if (!matched && nb::isinstance<typename Desc::wrapper>(slicer)) {
          auto& wrapper = nb::cast<typename Desc::wrapper&>(slicer);
          result = module_approximation_from_desc<Desc>(
              wrapper, direction, max_error, box, threshold, complete, verbose, n_jobs, mma_module);
          matched = true;
        }
      }(),
      ...);
  if (!matched) {
    throw std::runtime_error("Unsupported slicer type for module approximation.");
  }
  return result;
}

inline nb::object module_approximation_single_input(nb::object input,
                                                    nb::object box,
                                                    double max_error,
                                                    int nlines,
                                                    bool from_coordinates,
                                                    bool complete,
                                                    bool threshold,
                                                    bool verbose,
                                                    bool ignore_warnings,
                                                    nb::handle direction_handle,
                                                    nb::handle swap_box_coords_handle,
                                                    int n_jobs) {
  nb::object np = nb::module_::import_("numpy");
  nb::object mp_logs = nb::module_::import_("multipers.logs");
  nb::object direction = nb::borrow(direction_handle);
  nb::object swap_box_coords = nb::borrow(swap_box_coords_handle);

  if (is_simplextree_multi(input)) {
    input = nb::module_::import_("multipers._slicer_meta")
                .attr("Slicer")(input, "backend"_a = "matrix", "vineyard"_a = true);
  }
  if (!has_slicer_template_id(input)) {
    throw nb::value_error("First argument must be a simplextree or a slicer !");
  }
  SlicerRuntimeInfo runtime_info = get_slicer_runtime_info(input);

  std::vector<double> c_direction = cast_vector<double>(direction);
  bool is_degenerate = false;
  for (double value : c_direction) {
    if (value < 0) {
      throw nb::value_error("Got an invalid negative direction.");
    }
    if (value == 0) {
      is_degenerate = true;
    }
  }
  if (is_degenerate && !ignore_warnings) {
    mp_logs.attr("warn_geometry")(
        "Got a degenerate direction. This function may fail if the first line is not generic.");
  }

  if (from_coordinates && !runtime_info.is_squeezed) {
    if (verbose) {
      print_flush("Preparing filtration (squeeze)... ", "");
    }
    if (!ignore_warnings) {
      mp_logs.attr("warn_copy")("Got a non-squeezed input with `from_coordinates=True`.");
    }
    input = input.attr("grid_squeeze")();
    runtime_info = get_slicer_runtime_info(input);
    if (verbose) {
      print_flush("Done.");
    }
  }

  nb::object unsqueeze_grid = nb::none();
  if (runtime_info.is_squeezed) {
    if (verbose) {
      print_flush("Preparing filtration (unsqueeze)... ", "");
    }
    if (from_coordinates) {
      unsqueeze_grid = nb::module_::import_("multipers.grids")
                           .attr("sanitize_grid")(runtime_info.filtration_grid, "numpyfy"_a = true, "add_inf"_a = true);
      input = input.attr("astype")("dtype"_a = np.attr("float64"));
      runtime_info = get_slicer_runtime_info(input);
      if (c_direction.empty()) {
        nb::tuple unsqueeze_grid_tuple = nb::cast<nb::tuple>(unsqueeze_grid);
        c_direction.resize(nb::len(unsqueeze_grid_tuple));
        double norm_sq = 0.;
        for (size_t i = 0; i < c_direction.size(); ++i) {
          nb::object axis_grid = unsqueeze_grid_tuple[(Py_ssize_t)i];
          c_direction[i] = 1.0 / static_cast<double>(PyObject_Length(axis_grid.ptr()));
          norm_sq += c_direction[i] * c_direction[i];
        }
        double norm = std::sqrt(norm_sq);
        for (double& value : c_direction) {
          value /= norm;
        }
        direction = np.attr("array")(nb::cast(c_direction), "dtype"_a = np.attr("float64"));
      }
      if (verbose) {
        print_flush("Updated  `direction=...`, and `max_error=...` ", "");
      }
    } else {
      if (!ignore_warnings) {
        mp_logs.attr("warn_copy")("Got a squeezed input.");
      }
      input = input.attr("unsqueeze")();
      runtime_info = get_slicer_runtime_info(input);
    }
    if (verbose) {
      print_flush("Done.");
    }
  }

  if (nb::cast<size_t>(box.attr("size")) == 0) {
    if (verbose) {
      print_flush("No box given. Using filtration bounds to infer it.");
    }
    box = compute_filtration_bounds(input);
    if (verbose) {
      print_flush("Using inferred box.");
    }
  }

  if (nb::cast<int>(box.attr("ndim")) != 2) {
    throw nb::value_error("Invalid box dimension. Expected ndim == 2.");
  }

  std::vector<std::vector<double>> c_box = cast_matrix<double>(box);
  std::vector<double> scales(c_box[0].size(), 0.);
  double max_scale = 0.;
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = c_box[1][i] - c_box[0][i];
    max_scale = std::max(max_scale, scales[i]);
  }
  if (max_scale != 0.) {
    for (double& value : scales) {
      value /= max_scale;
    }
  }
  if (std::any_of(scales.begin(), scales.end(), [](double value) { return value < 0.1; })) {
    mp_logs.attr("warn_geometry")(
        "Squewed filtration detected. Consider rescaling the filtration for interpretable results.");
  }

  bool has_trivial_box_coord = false;
  for (size_t i = 0; i < c_box[0].size(); ++i) {
    if (c_box[1][i] == c_box[0][i]) {
      has_trivial_box_coord = true;
      c_box[1][i] += 1.0;
    }
  }
  if (has_trivial_box_coord && !ignore_warnings) {
    mp_logs.attr("warn_geometry")("Got trivial box coordinates.");
  }

  for (int idx : cast_vector<int>(swap_box_coords)) {
    std::swap(c_box[0][idx], c_box[1][idx]);
  }
  box = prepare_box_object(c_box);

  size_t num_parameters = c_box[0].size();
  if (!c_direction.empty() && c_direction.size() != num_parameters) {
    throw nb::value_error("Invalid line direction size.");
  }

  double prod = 0.;
  for (size_t i = 0; i < num_parameters; ++i) {
    if (!c_direction.empty() && c_direction[i] == 0.) {
      continue;
    }
    double term = 1.;
    for (size_t j = 0; j < num_parameters; ++j) {
      if (i == j) {
        continue;
      }
      term *= std::abs(c_box[1][j] - c_box[0][j]);
    }
    prod += term;
  }
  if (max_error <= 0) {
    max_error = std::pow(prod / static_cast<double>(nlines), 1.0 / static_cast<double>(num_parameters - 1));
  }
  double estimated_nlines = prod / std::pow(max_error, static_cast<double>(num_parameters - 1));
  if (!ignore_warnings && estimated_nlines >= 10000.) {
    throw nb::value_error(
        "Warning: the number of lines may be too high. Try to increase the precision parameter or set "
        "ignore_warnings=True.");
  }

  return nb::module_::import_("multipers.multiparameter_module_approximation")
      .attr("module_approximation_from_slicer")(input,
                                                "box"_a = box,
                                                "max_error"_a = max_error,
                                                "complete"_a = complete,
                                                "threshold"_a = threshold,
                                                "verbose"_a = verbose,
                                                "direction"_a = direction,
                                                "unsqueeze_grid"_a = unsqueeze_grid,
                                                "n_jobs"_a = n_jobs);
}

template <typename Desc>
void bind_bitmap_builder(nb::module_& m) {
  if constexpr (Desc::enable_bitmap_builder) {
    using Wrapper = typename Desc::wrapper;
    using Concrete = typename Desc::concrete;
    using Value = typename Desc::value_type;

    std::string name = std::string("_build_bitmap_") + std::string(Desc::short_value_type);
    m.def(
        name.c_str(),
        [](nb::handle image_handle, nb::handle shape_handle) {
          auto image = cast_matrix<Value>(image_handle);
          auto shape = cast_vector<unsigned int>(shape_handle);
          if (image.empty()) {
            return Wrapper();
          }
          std::vector<typename Concrete::Filtration_value> vertices;
          vertices.reserve(image.size());
          for (const auto& row : image) {
            vertices.emplace_back(row.begin(), row.end());
          }
          Wrapper out;
          {
            nb::gil_scoped_release release;
            out.truc = Gudhi::multi_persistence::build_slicer_from_bitmap<Concrete>(vertices, shape);
          }
          reset_python_state(out);
          return out;
        },
        "image"_a,
        "shape"_a);
  }
}

template <typename... Desc>
void bind_bitmap_builders(type_list<Desc...>, nb::module_& m) {
  (bind_bitmap_builder<Desc>(m), ...);
}

}  // namespace mpnb

NB_MODULE(_slicer_nanobind, m) {
  m.doc() = "nanobind slicer bindings";
  nb::list available_slicers;

  mpnb::bind_all_slicers(mpnb::SlicerDescriptorList{}, m, available_slicers);

  m.def(
      "_compute_hilbert_signed_measure",
      [](nb::handle slicer,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         bool zero_pad,
         mpnb::indices_type n_jobs,
         bool verbose,
         bool ignore_inf) {
        auto grid_shape = mpnb::cast_vector<mpnb::indices_type>(grid_shape_handle);
        auto degrees = mpnb::cast_vector<mpnb::indices_type>(degrees_handle);
        std::vector<mpnb::indices_type> full_shape;
        full_shape.reserve(grid_shape.size() + 1);
        full_shape.push_back((mpnb::indices_type)degrees.size());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        size_t width = grid_shape.size() + 1;
        size_t total = 1;
        for (mpnb::indices_type value : full_shape) {
          total *= (size_t)value;
        }
        std::vector<mpnb::tensor_dtype> container(total, 0);
        return mpnb::compute_hilbert_signed_measure(mpnb::SlicerDescriptorList{},
                                                    slicer,
                                                    container,
                                                    full_shape,
                                                    degrees,
                                                    width,
                                                    zero_pad,
                                                    n_jobs,
                                                    verbose,
                                                    ignore_inf);
      },
      "slicer"_a,
      "grid_shape"_a,
      "degrees"_a,
      "zero_pad"_a = false,
      "n_jobs"_a = 0,
      "verbose"_a = false,
      "ignore_inf"_a = true);

  m.def(
      "_compute_rank_tensor",
      [](nb::handle slicer,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         mpnb::indices_type n_jobs,
         bool ignore_inf) {
        auto grid_shape = mpnb::cast_vector<mpnb::indices_type>(grid_shape_handle);
        auto degrees = mpnb::cast_vector<mpnb::indices_type>(degrees_handle);
        std::vector<mpnb::indices_type> full_shape;
        full_shape.reserve(1 + 2 * grid_shape.size());
        full_shape.push_back((mpnb::indices_type)degrees.size());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        size_t total = 1;
        for (mpnb::indices_type value : full_shape) {
          total *= (size_t)value;
        }
        std::vector<mpnb::tensor_dtype> container(total, 0);
        return mpnb::compute_rank_tensor(
            mpnb::SlicerDescriptorList{}, slicer, container, full_shape, degrees, total, n_jobs, ignore_inf);
      },
      "slicer"_a,
      "grid_shape"_a,
      "degrees"_a,
      "n_jobs"_a = 0,
      "ignore_inf"_a = true);

  m.def(
      "_compute_module_approximation_from_slicer",
      [](nb::handle slicer,
         nb::handle direction_handle,
         double max_error,
         nb::handle box_handle,
         bool threshold,
         bool complete,
         bool verbose,
         int n_jobs) {
        auto direction = mpnb::cast_vector<double>(direction_handle);
        auto box_values = mpnb::cast_matrix<double>(box_handle);
        Gudhi::multi_persistence::Box<double> box(box_values[0], box_values[1]);
        auto mma_module = nb::module_::import_("multipers._mma_nanobind");
        return mpnb::compute_module_approximation_from_slicer(mpnb::SlicerDescriptorList{},
                                                              slicer,
                                                              direction,
                                                              max_error,
                                                              box,
                                                              threshold,
                                                              complete,
                                                              verbose,
                                                              n_jobs,
                                                              mma_module);
      },
      "slicer"_a,
      "direction"_a,
      "max_error"_a,
      "box"_a,
      "threshold"_a = false,
      "complete"_a = true,
      "verbose"_a = false,
      "n_jobs"_a = -1);

  m.def(
      "_module_approximation_single_input",
      [](nb::object input,
         nb::object box,
         double max_error,
         int nlines,
         bool from_coordinates,
         bool complete,
         bool threshold,
         bool verbose,
         bool ignore_warnings,
         nb::handle direction,
         nb::handle swap_box_coords,
         int n_jobs) {
        return mpnb::module_approximation_single_input(input,
                                                       box,
                                                       max_error,
                                                       nlines,
                                                       from_coordinates,
                                                       complete,
                                                       threshold,
                                                       verbose,
                                                       ignore_warnings,
                                                       direction,
                                                       swap_box_coords,
                                                       n_jobs);
      },
      "input"_a,
      "box"_a = nb::none(),
      "max_error"_a = -1,
      "nlines"_a = 557,
      "from_coordinates"_a = false,
      "complete"_a = true,
      "threshold"_a = false,
      "verbose"_a = false,
      "ignore_warnings"_a = false,
      "direction"_a = nb::make_tuple(),
      "swap_box_coords"_a = nb::make_tuple(),
      "n_jobs"_a = -1);

  m.def(
      "_get_slicer_class",
      [](bool is_vineyard,
         bool is_k_critical,
         nb::handle dtype,
         std::string col,
         std::string pers_backend,
         std::string filtration_container) {
        return mpnb::get_slicer_class(mpnb::SlicerDescriptorList{},
                                      is_vineyard,
                                      is_k_critical,
                                      dtype,
                                      std::move(col),
                                      std::move(pers_backend),
                                      std::move(filtration_container));
      },
      "is_vineyard"_a,
      "is_k_critical"_a,
      "dtype"_a,
      "col"_a,
      "pers_backend"_a,
      "filtration_container"_a);

  mpnb::bind_bitmap_builders(mpnb::SlicerDescriptorList{}, m);

  m.attr("available_slicers") = available_slicers;
}
