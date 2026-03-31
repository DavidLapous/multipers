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
#include <unordered_set>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>

#include "Persistence_slices_interface.h"
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "gudhi/Multi_parameter_filtered_complex.h"
#include "gudhi/slicer_helpers.h"
#include "multi_parameter_rank_invariant/hilbert_function.h"
#include "multi_parameter_rank_invariant/rank_invariant.h"
#include "multiparameter_module_approximation/approximation.h"
#include "nanobind_array_utils.hpp"
#include "nanobind_dense_array_utils.hpp"
#include "nanobind_mma_registry_helpers.hpp"
#include "nanobind_object_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpnb {

using tensor_dtype = int32_t;
using indices_type = int32_t;
using signed_measure_type = std::pair<std::vector<std::vector<indices_type>>, std::vector<tensor_dtype>>;

using multipers::nanobind_dense_utils::matrix_from_array;
using multipers::nanobind_dense_utils::vector_from_array;
using multipers::nanobind_helpers::dispatch_simplextree_by_template_id;
using multipers::nanobind_helpers::dispatch_slicer_by_template_id;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::is_slicer_object;
using multipers::nanobind_helpers::PySlicer;
using multipers::nanobind_helpers::simplextree_wrapper_t;
using multipers::nanobind_helpers::SimplexTreeDescriptorList;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_helpers::visit_const_slicer_wrapper;
using multipers::nanobind_helpers::visit_simplextree_wrapper;
using multipers::nanobind_mma_helpers::canonical_double_mma_desc;
using multipers::nanobind_mma_helpers::module_wrapper_t;
using multipers::nanobind_utils::cast_matrix;
using multipers::nanobind_utils::cast_tensor3;
using multipers::nanobind_utils::cast_vector;
using multipers::nanobind_utils::has_template_id;
using multipers::nanobind_utils::lowercase_copy;
using multipers::nanobind_utils::numpy_dtype_name;
using multipers::nanobind_utils::numpy_dtype_type;
using multipers::nanobind_utils::owned_array;
using multipers::nanobind_utils::template_id_of;
using multipers::nanobind_utils::tuple_from_size;
using multipers::nanobind_utils::view_array;

template <typename Desc>
inline constexpr bool is_kcritical_contiguous_f64_matrix_slicer_v =
    std::is_same_v<typename Desc::value_type, double> && !Desc::is_vine && Desc::is_kcritical &&
    !Desc::is_degree_rips && Desc::column_type == std::string_view("UNORDERED_SET") &&
    Desc::backend_type == std::string_view("Matrix") && Desc::filtration_container == std::string_view("Contiguous");

template <typename List>
struct kcritical_contiguous_f64_matrix_slicer_desc_impl;

template <>
struct kcritical_contiguous_f64_matrix_slicer_desc_impl<type_list<>> {
  using type = void;
  static constexpr bool found = false;
  static constexpr int matches = 0;
};

template <typename Head, typename... Tail>
struct kcritical_contiguous_f64_matrix_slicer_desc_impl<type_list<Head, Tail...>> {
  using tail = kcritical_contiguous_f64_matrix_slicer_desc_impl<type_list<Tail...>>;
  static constexpr bool is_match = is_kcritical_contiguous_f64_matrix_slicer_v<Head>;
  static constexpr bool found = is_match || tail::found;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
  using type = std::conditional_t<is_match, Head, typename tail::type>;
};

using KcriticalContiguousF64MatrixSlicerDesc =
    typename kcritical_contiguous_f64_matrix_slicer_desc_impl<SlicerDescriptorList>::type;

static_assert(!std::is_void_v<KcriticalContiguousF64MatrixSlicerDesc>,
              "Expected exactly one k-critical contiguous float64 matrix slicer template.");
static_assert(kcritical_contiguous_f64_matrix_slicer_desc_impl<SlicerDescriptorList>::matches == 1,
              "k-critical contiguous float64 matrix slicer template must be unique.");

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

inline bool is_none_or_empty(const nb::handle& h) {
  if (!h.is_valid() || h.is_none()) {
    return true;
  }
  if (nb::hasattr(h, "__len__")) {
    return nb::len(h) == 0;
  }
  return false;
}

template <typename Wrapper>
void ensure_sorted_filtration_grid(const Wrapper& self) {
  if (is_none_or_empty(self.filtration_grid)) {
    return;
  }
  for (nb::handle row_handle : nb::iter(self.filtration_grid)) {
    auto row = cast_vector<double>(row_handle);
    for (size_t i = 1; i < row.size(); ++i) {
      if (row[i] < row[i - 1]) {
        throw nb::value_error("Found non-sorted grid.");
      }
    }
  }
}

template <typename Wrapper>
Wrapper& make_filtration_non_decreasing_inplace(Wrapper& self, bool safe) {
  if (safe && !is_none_or_empty(self.filtration_grid)) {
    ensure_sorted_filtration_grid(self);
  }

  {
    nb::gil_scoped_release release;
    auto& filtrations = self.truc.get_filtration_values();
    const auto& boundaries = self.truc.get_boundaries();
    const bool ordered = self.truc.get_filtered_complex().is_ordered_by_dimension();

    bool modified = true;
    while (modified) {
      modified = false;
      for (size_t i = boundaries.size(); i-- > 0;) {
        for (auto b : boundaries[i]) {
          modified |= intersect_lifetimes(filtrations[b], filtrations[i]);
        }
      }
      if (ordered) {
        break;
      }
    }
  }
  return self;
}

inline bool has_slicer_template_id(const nb::handle& input) { return is_slicer_object(input); }

template <typename... Ds>
nb::object get_slicer_class(type_list<Ds...>,
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
      [&]<typename D>() {
        if (!matched && D::is_vine == is_vineyard && D::is_kcritical == is_k_critical && D::dtype_name == dtype_name &&
            lowercase_copy(std::string(D::column_type)) == col &&
            lowercase_copy(std::string(D::backend_type)) == pers_backend &&
            lowercase_copy(std::string(D::filtration_container)) == filtration_container) {
          result = nb::borrow<nb::object>(nb::type<typename D::wrapper>());
          matched = true;
        }
      }.template operator()<Ds>(),
      ...);
  if (!matched) {
    throw nb::value_error("Unimplemented slicer combination.");
  }
  return result;
}

inline nb::object get_slicer_class_from_template_id(int template_id) {
  return dispatch_slicer_by_template_id(template_id, [&]<typename Desc>() -> nb::object {
    return nb::borrow<nb::object>(nb::type<typename Desc::wrapper>());
  });
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

template <typename Wrapper, typename Concrete, typename SourceWrapper>
bool try_build_kcritical_from_simplextree_scc(Wrapper& self, SourceWrapper& source) {
  using Complex = Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value>;
  Concrete built;
  {
    nb::gil_scoped_release release;
    auto blocks = source.tree.kcritical_simplextree_to_scc();
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
  self.filtration_grid = source.filtration_grid;
  self.generator_basis = nb::none();
  self.minpres_degree = -1;
  return true;
}

inline bool is_simplextree_multi(const nb::handle& source) { return is_simplextree_object(source); }

template <typename Desc, typename Wrapper, typename Concrete>
void build_from_simplextree_desc(Wrapper& self, simplextree_wrapper_t<Desc>& source) {
  if constexpr (Desc::is_kcritical) {
    (void)try_build_kcritical_from_simplextree_scc<Wrapper, Concrete>(self, source);
    return;
  }
  {
    nb::gil_scoped_release release;
    self.truc = Gudhi::multi_persistence::build_slicer_from_simplex_tree<Concrete>(source.tree);
  }
  self.filtration_grid = source.filtration_grid;
  self.generator_basis = nb::none();
  self.minpres_degree = -1;
}

template <typename Wrapper, typename Concrete>
bool try_build_from_multipers_simplextree(Wrapper& self, const nb::handle& source) {
  if (!is_simplextree_object(source)) {
    return false;
  }
  visit_simplextree_wrapper(source, [&]<typename D>(simplextree_wrapper_t<D>& wrapper) {
    build_from_simplextree_desc<D, Wrapper, Concrete>(self, wrapper);
  });
  return true;
}

template <typename Barcode, typename Value>
nb::tuple dim_barcode_to_tuple(const Barcode& barcode) {
  size_t dims = barcode.size();
  return tuple_from_size(dims, [&](size_t dim) -> nb::object {
    const auto& bc = barcode[dim];
    std::vector<Value> flat;
    flat.reserve(bc.size() * 2);
    auto* data = bc.data();
    for (size_t i = 0; i < bc.size(); ++i) {
      flat.push_back(data[i][0]);
      flat.push_back(data[i][1]);
    }
    return nb::cast(owned_array<Value>(std::move(flat), {bc.size(), size_t(2)}));
  });
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

  auto indptr_view = indptr_arr.template view<uint64_t, nb::ndim<1>>();
  auto indices_view = indices_arr.template view<uint32_t, nb::ndim<1>>();
  return tuple_from_size(num_rows, [&](size_t i) -> nb::object {
    uint64_t start = indptr_view(i);
    uint64_t stop = indptr_view(i + 1);
    std::vector<uint32_t> row(stop - start);
    for (uint64_t j = start; j < stop; ++j) {
      row[j - start] = indices_view(j);
    }
    return nb::cast(owned_array<uint32_t>(std::move(row), {size_t(stop - start)}));
  });
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
  visit_const_slicer_wrapper(source, [&]<typename D>(const typename D::wrapper& other) {
    {
      nb::gil_scoped_release release;
      self.truc = TargetConcrete(other.truc);
    }
    self.filtration_grid = other.filtration_grid;
    self.generator_basis = other.generator_basis;
    self.minpres_degree = other.minpres_degree;
  });
  return true;
}

template <typename TargetDesc, typename SourceDesc>
typename TargetDesc::wrapper construct_from_slicer_wrapper(const typename SourceDesc::wrapper& source) {
  using Wrapper = typename TargetDesc::wrapper;
  using Concrete = typename TargetDesc::concrete;
  Wrapper out;
  {
    nb::gil_scoped_release release;
    out.truc = Concrete(source.truc);
  }
  out.filtration_grid = source.filtration_grid;
  out.generator_basis = source.generator_basis;
  out.minpres_degree = source.minpres_degree;
  return out;
}

template <typename TargetDesc, typename SourceDesc>
typename TargetDesc::wrapper construct_from_simplextree_wrapper(simplextree_wrapper_t<SourceDesc>& source) {
  using Wrapper = typename TargetDesc::wrapper;
  using Concrete = typename TargetDesc::concrete;
  Wrapper out;
  build_from_simplextree_desc<SourceDesc, Wrapper, Concrete>(out, source);
  return out;
}

template <typename TargetDesc, typename Class, typename... SourceDesc>
void bind_slicer_source_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(nb::new_([](const typename SourceDesc::wrapper& source) {
             return construct_from_slicer_wrapper<TargetDesc, SourceDesc>(source);
           }),
           "source"_a),
   ...);
}

template <typename TargetDesc, typename Class, typename... SourceDesc>
void bind_simplextree_source_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(nb::new_([](simplextree_wrapper_t<SourceDesc>& source) {
             return construct_from_simplextree_wrapper<TargetDesc, SourceDesc>(source);
           }),
           "source"_a),
   ...);
}

template <typename TargetDesc, typename Class>
void bind_typed_source_constructors(Class& cls) {
  bind_slicer_source_constructors<TargetDesc>(cls, SlicerDescriptorList{});
  bind_simplextree_source_constructors<TargetDesc>(cls, SimplexTreeDescriptorList{});
}

template <typename Wrapper>
void reset_python_state(Wrapper& self);

template <typename Wrapper, typename Concrete>
Wrapper construct_from_scc_file(const std::string& path, int shift_dimension) {
  Wrapper out;
  {
    nb::gil_scoped_release release;
    out.truc = Gudhi::multi_persistence::build_slicer_from_scc_file<Concrete>(path, false, false, shift_dimension);
  }
  reset_python_state(out);
  return out;
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
  self.generator_basis = nb::none();
  self.minpres_degree = -1;
}

struct GeneratorBasisData {
  bool active = false;
  int degree = -1;
  std::vector<std::vector<uint32_t>> columns;
  std::vector<std::vector<uint32_t>> row_boundaries;
};

template <typename Wrapper>
GeneratorBasisData extract_generator_basis(const Wrapper& self) {
  GeneratorBasisData out;
  if (self.generator_basis.is_none()) {
    return out;
  }
  nb::dict basis = nb::cast<nb::dict>(self.generator_basis);
  if (!basis.contains("degree") || !basis.contains("columns") || !basis.contains("row_boundaries")) {
    throw std::runtime_error("Invalid `_generator_basis`: expected keys `degree`, `columns`, and `row_boundaries`.");
  }
  out.degree = nb::cast<int>(basis["degree"]);
  out.columns = nb::cast<std::vector<std::vector<uint32_t>>>(basis["columns"]);
  out.row_boundaries = nb::cast<std::vector<std::vector<uint32_t>>>(basis["row_boundaries"]);
  out.active = true;
  return out;
}

inline std::vector<std::vector<uint32_t>> expand_cycle_in_generator_basis(const std::vector<uint32_t>& cycle,
                                                                          const GeneratorBasisData& basis) {
  std::vector<uint8_t> active_rows(basis.row_boundaries.size(), 0);
  for (uint32_t generator_idx : cycle) {
    if (generator_idx >= basis.columns.size()) {
      throw std::runtime_error("Representative cycle refers to a generator outside `_generator_basis`.");
    }
    for (uint32_t row_idx : basis.columns[generator_idx]) {
      if (row_idx >= basis.row_boundaries.size()) {
        throw std::runtime_error("`_generator_basis` column support refers to a row outside `row_boundaries`.");
      }
      active_rows[row_idx] ^= 1;
    }
  }

  std::vector<std::vector<uint32_t>> out;
  for (size_t row_idx = 0; row_idx < active_rows.size(); ++row_idx) {
    if (active_rows[row_idx] != 0) {
      out.push_back(basis.row_boundaries[row_idx]);
    }
  }
  return out;
}

inline bool cycle_intersects_points(const std::vector<std::vector<uint32_t>>& cycle,
                                    const std::unordered_set<uint32_t>& intersect_points) {
  if (intersect_points.empty()) {
    return false;
  }
  for (const auto& simplex : cycle) {
    for (uint32_t vertex : simplex) {
      if (intersect_points.contains(vertex)) {
        return true;
      }
    }
  }
  return false;
}

inline constexpr uint32_t kSlicerSerializationMagic = 0x4d50534c;
inline constexpr uint32_t kSlicerSerializationVersion = 1;

enum class SlicerSerializationMode : uint32_t {
  OneCritical = 0,
  KCritical = 1,
  DegreeRips = 2,
};

template <typename T>
void append_scalar(std::vector<uint8_t>& buffer, T value) {
  size_t offset = buffer.size();
  buffer.resize(offset + sizeof(T));
  std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

template <typename T>
void append_vector(std::vector<uint8_t>& buffer, const std::vector<T>& values) {
  if (values.empty()) {
    return;
  }
  size_t offset = buffer.size();
  buffer.resize(offset + values.size() * sizeof(T));
  std::memcpy(buffer.data() + offset, values.data(), values.size() * sizeof(T));
}

inline void ensure_serialized_bytes_available(const uint8_t* ptr, const uint8_t* end, size_t num_bytes) {
  if ((size_t)(end - ptr) < num_bytes) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
}

template <typename T>
T read_scalar(const uint8_t*& ptr, const uint8_t* end) {
  ensure_serialized_bytes_available(ptr, end, sizeof(T));
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  ptr += sizeof(T);
  return value;
}

template <typename T>
std::vector<T> read_vector(const uint8_t*& ptr, const uint8_t* end, size_t count) {
  std::vector<T> out(count);
  if (count == 0) {
    return out;
  }
  ensure_serialized_bytes_available(ptr, end, count * sizeof(T));
  std::memcpy(out.data(), ptr, count * sizeof(T));
  ptr += count * sizeof(T);
  return out;
}

template <typename Wrapper, typename Value, bool IsKCritical, bool IsDegreeRips>
nb::object serialized_state(Wrapper& self) {
  std::vector<uint64_t> boundary_indptr;
  std::vector<uint32_t> boundary_flat;
  std::vector<int32_t> dimensions;
  std::vector<int64_t> grade_indptr;
  std::vector<Value> grades_flat;
  size_t num_generators = 0;
  uint64_t encoded_num_parameters = 0;
  uint64_t filtration_rows = 0;

  {
    nb::gil_scoped_release release;
    const auto& boundaries = self.truc.get_boundaries();
    const auto& dims = self.truc.get_dimensions();
    const auto& filtrations = self.truc.get_filtration_values();

    num_generators = boundaries.size();
    boundary_indptr.assign(num_generators + 1, 0);
    dimensions.reserve(dims.size());

    size_t total_boundary_size = 0;
    for (size_t i = 0; i < num_generators; ++i) {
      total_boundary_size += boundaries[i].size();
      boundary_indptr[i + 1] = total_boundary_size;
      dimensions.push_back((int32_t)dims[i]);
    }
    boundary_flat.reserve(total_boundary_size);
    for (const auto& row : boundaries) {
      boundary_flat.insert(boundary_flat.end(), row.begin(), row.end());
    }

    if constexpr (IsKCritical) {
      encoded_num_parameters = IsDegreeRips ? uint64_t(2) : (uint64_t)self.truc.get_number_of_parameters();
      grade_indptr.assign(num_generators + 1, 0);
      size_t total_rows = 0;
      for (size_t i = 0; i < num_generators; ++i) {
        total_rows += filtrations[i].num_generators();
        grade_indptr[i + 1] = (int64_t)total_rows;
      }
      filtration_rows = (uint64_t)total_rows;
      grades_flat.resize(total_rows * encoded_num_parameters);
      size_t offset = 0;
      for (size_t i = 0; i < num_generators; ++i) {
        size_t k = filtrations[i].num_generators();
        for (size_t g = 0; g < k; ++g) {
          if constexpr (IsDegreeRips) {
            grades_flat[2 * (offset + g)] = filtrations[i](g, 0);
            grades_flat[2 * (offset + g) + 1] = static_cast<Value>(g);
          } else {
            for (size_t p = 0; p < encoded_num_parameters; ++p) {
              grades_flat[(offset + g) * encoded_num_parameters + p] = filtrations[i](g, p);
            }
          }
        }
        offset += k;
      }
    } else {
      encoded_num_parameters = (uint64_t)self.truc.get_number_of_parameters();
      filtration_rows = (uint64_t)num_generators;
      grades_flat.resize(num_generators * encoded_num_parameters);
      for (size_t i = 0; i < num_generators; ++i) {
        if (!filtrations[i].is_finite()) {
          std::fill_n(grades_flat.data() + i * encoded_num_parameters, encoded_num_parameters, filtrations[i](0, 0));
        } else if (encoded_num_parameters > 0) {
          std::memcpy(grades_flat.data() + i * encoded_num_parameters,
                      &filtrations[i](0, 0),
                      encoded_num_parameters * sizeof(Value));
        }
      }
    }
  }

  std::vector<uint8_t> buffer;
  buffer.reserve(3 * sizeof(uint32_t) + 4 * sizeof(uint64_t) + boundary_indptr.size() * sizeof(uint64_t) +
                 boundary_flat.size() * sizeof(uint32_t) + dimensions.size() * sizeof(int32_t) +
                 grade_indptr.size() * sizeof(int64_t) + grades_flat.size() * sizeof(Value));
  append_scalar<uint32_t>(buffer, kSlicerSerializationMagic);
  append_scalar<uint32_t>(buffer, kSlicerSerializationVersion);
  append_scalar<uint32_t>(buffer,
                          static_cast<uint32_t>(IsDegreeRips ? SlicerSerializationMode::DegreeRips
                                                             : (IsKCritical ? SlicerSerializationMode::KCritical
                                                                            : SlicerSerializationMode::OneCritical)));
  append_scalar<uint64_t>(buffer, (uint64_t)num_generators);
  append_scalar<uint64_t>(buffer, (uint64_t)boundary_flat.size());
  append_scalar<uint64_t>(buffer, encoded_num_parameters);
  append_scalar<uint64_t>(buffer, filtration_rows);
  append_vector<uint64_t>(buffer, boundary_indptr);
  append_vector<uint32_t>(buffer, boundary_flat);
  append_vector<int32_t>(buffer, dimensions);
  if constexpr (IsKCritical) {
    append_vector<int64_t>(buffer, grade_indptr);
  }
  append_vector<Value>(buffer, grades_flat);
  return nb::cast(owned_array<uint8_t>(std::move(buffer), {buffer.size()}));
}

template <typename Wrapper, typename Concrete, typename Value, bool IsKCritical, bool IsDegreeRips>
void load_state(Wrapper& self, nb::handle state) {
  auto buffer = cast_vector<uint8_t>(state);
  const uint8_t* ptr = buffer.data();
  const uint8_t* end = ptr + buffer.size();

  uint32_t magic = read_scalar<uint32_t>(ptr, end);
  uint32_t version = read_scalar<uint32_t>(ptr, end);
  uint32_t mode = read_scalar<uint32_t>(ptr, end);
  if (magic != kSlicerSerializationMagic || version != kSlicerSerializationVersion) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  uint32_t expected_mode = static_cast<uint32_t>(
      IsDegreeRips ? SlicerSerializationMode::DegreeRips
                   : (IsKCritical ? SlicerSerializationMode::KCritical : SlicerSerializationMode::OneCritical));
  if (mode != expected_mode) {
    throw std::runtime_error("Serialized slicer state does not match target type.");
  }

  uint64_t num_generators = read_scalar<uint64_t>(ptr, end);
  uint64_t boundary_flat_size = read_scalar<uint64_t>(ptr, end);
  uint64_t encoded_num_parameters = read_scalar<uint64_t>(ptr, end);
  uint64_t filtration_rows = read_scalar<uint64_t>(ptr, end);

  auto boundary_indptr = read_vector<uint64_t>(ptr, end, (size_t)num_generators + 1);
  auto boundary_flat = read_vector<uint32_t>(ptr, end, (size_t)boundary_flat_size);
  auto dimensions32 = read_vector<int32_t>(ptr, end, (size_t)num_generators);
  if (boundary_indptr.empty() || boundary_indptr.back() != boundary_flat_size) {
    throw std::runtime_error("Invalid serialized slicer boundaries.");
  }

  std::vector<std::vector<uint32_t>> boundaries((size_t)num_generators);
  for (size_t i = 0; i < (size_t)num_generators; ++i) {
    uint64_t begin = boundary_indptr[i];
    uint64_t finish = boundary_indptr[i + 1];
    if (begin > finish || finish > boundary_flat.size()) {
      throw std::runtime_error("Invalid serialized slicer boundaries.");
    }
    boundaries[i].assign(boundary_flat.begin() + (ptrdiff_t)begin, boundary_flat.begin() + (ptrdiff_t)finish);
  }

  std::vector<int> dimensions(dimensions32.begin(), dimensions32.end());
  std::vector<typename Concrete::Filtration_value> c_filtrations;
  c_filtrations.reserve((size_t)num_generators);

  if constexpr (IsKCritical) {
    auto grade_indptr = read_vector<int64_t>(ptr, end, (size_t)num_generators + 1);
    auto grades_flat = read_vector<Value>(ptr, end, (size_t)(filtration_rows * encoded_num_parameters));
    if (grade_indptr.empty() || grade_indptr.back() != (int64_t)filtration_rows) {
      throw std::runtime_error("Invalid serialized slicer filtrations.");
    }
    for (size_t i = 0; i < (size_t)num_generators; ++i) {
      int64_t begin = grade_indptr[i];
      int64_t finish = grade_indptr[i + 1];
      if (begin > finish || finish > (int64_t)filtration_rows) {
        throw std::runtime_error("Invalid serialized slicer filtrations.");
      }
      typename Concrete::Filtration_value filtration((size_t)encoded_num_parameters);
      auto inf = Concrete::Filtration_value::inf((size_t)encoded_num_parameters);
      filtration.push_to_least_common_upper_bound(inf, false);
      for (int64_t row = begin; row < finish; ++row) {
        std::vector<Value> grade((size_t)encoded_num_parameters);
        size_t offset = (size_t)row * (size_t)encoded_num_parameters;
        if (!grade.empty()) {
          std::memcpy(grade.data(), grades_flat.data() + offset, grade.size() * sizeof(Value));
        }
        filtration.add_generator(grade);
      }
      c_filtrations.push_back(std::move(filtration));
    }
  } else {
    auto grades_flat = read_vector<Value>(ptr, end, (size_t)(num_generators * encoded_num_parameters));
    for (size_t i = 0; i < (size_t)num_generators; ++i) {
      std::vector<Value> grade((size_t)encoded_num_parameters);
      size_t offset = i * (size_t)encoded_num_parameters;
      if (!grade.empty()) {
        std::memcpy(grade.data(), grades_flat.data() + offset, grade.size() * sizeof(Value));
      }
      c_filtrations.emplace_back(grade);
    }
  }

  if (ptr != end) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }

  if (num_generators == 0) {
    self.truc = Concrete();
    reset_python_state(self);
    return;
  }

  Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value> cpx(
      boundaries, dimensions, c_filtrations);
  self.truc = Concrete(cpx);
  reset_python_state(self);
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

inline std::vector<std::vector<uint32_t>> boundaries_from_generator_maps(const nb::handle& generator_maps) {
  std::vector<std::vector<uint32_t>> boundaries;
  boundaries.reserve(nb::len(generator_maps));
  for (nb::handle row_handle : nb::iter(generator_maps)) {
    boundaries.push_back(cast_vector<uint32_t>(row_handle));
  }
  return boundaries;
}

template <typename Wrapper, typename Concrete, typename Value, typename Index>
Wrapper construct_from_dense_generator_data(nb::iterable generator_maps,
                                            nb::ndarray<nb::numpy, const Index, nb::ndim<1>> generator_dimensions,
                                            nb::ndarray<nb::numpy, const Value, nb::ndim<2>> filtration_values) {
  Wrapper out;
  if (nb::len(generator_maps) == 0) {
    return out;
  }

  auto boundaries = boundaries_from_generator_maps(generator_maps);
  const size_t num_generators = boundaries.size();
  if (generator_dimensions.shape(0) != num_generators || filtration_values.shape(0) != num_generators) {
    throw std::runtime_error("Invalid input, shape do not coincide.");
  }

  std::vector<int> dims;
  dims.reserve(num_generators);
  for (size_t i = 0; i < num_generators; ++i) {
    dims.push_back((int)generator_dimensions(i));
  }

  std::vector<typename Concrete::Filtration_value> c_filtrations;
  c_filtrations.reserve(num_generators);
  const auto view = filtration_values.view();
  const size_t num_parameters = filtration_values.shape(1);
  for (size_t i = 0; i < num_generators; ++i) {
    std::vector<Value> row(num_parameters);
    for (size_t p = 0; p < num_parameters; ++p) {
      row[p] = view(i, p);
    }
    c_filtrations.emplace_back(row);
  }

  Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value> cpx(
      boundaries, dims, c_filtrations);
  out.truc = Concrete(cpx);
  reset_python_state(out);
  return out;
}

template <typename Class, typename Wrapper, typename Concrete, typename Value, bool IsKCritical, typename Index>
void bind_dense_generator_data_overloads(Class& cls) {
  if constexpr (!IsKCritical) {
    cls.def(nb::new_([](nb::iterable generator_maps,
                        nb::ndarray<nb::numpy, const Index, nb::ndim<1>> generator_dimensions,
                        nb::ndarray<nb::numpy, const Value, nb::ndim<2>> filtration_values) {
              return construct_from_dense_generator_data<Wrapper, Concrete, Value>(
                  generator_maps, generator_dimensions, filtration_values);
            }),
            "generator_maps"_a,
            "generator_dimensions"_a,
            "filtration_values"_a);
  }
}

template <typename Wrapper, typename Concrete, typename Value>
Wrapper construct_kcritical_from_packed(
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
    nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat,
    nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> generator_dimensions,
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> grade_indptr,
    nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades_flat) {
  Wrapper out;
  if (boundary_indptr.shape(0) == 0) {
    return out;
  }

  const size_t num_generators = (size_t)boundary_indptr.shape(0) - 1;
  if ((size_t)generator_dimensions.shape(0) != num_generators || (size_t)grade_indptr.shape(0) != num_generators + 1) {
    throw std::runtime_error("Invalid packed input, shape do not coincide.");
  }

  std::vector<std::vector<uint32_t>> boundaries(num_generators);
  const int64_t* boundary_ptr = boundary_indptr.data();
  const int32_t* boundary_vals = boundary_flat.data();
  for (size_t i = 0; i < num_generators; ++i) {
    const int64_t begin = boundary_ptr[i];
    const int64_t end = boundary_ptr[i + 1];
    auto& row = boundaries[i];
    row.reserve((size_t)std::max<int64_t>(end - begin, 0));
    for (int64_t idx = begin; idx < end; ++idx) {
      row.push_back((uint32_t)boundary_vals[idx]);
    }
  }

  std::vector<int> dims;
  dims.reserve(num_generators);
  for (size_t i = 0; i < num_generators; ++i) {
    dims.push_back((int)generator_dimensions(i));
  }

  const size_t num_parameters = grades_flat.shape(1);
  std::vector<typename Concrete::Filtration_value> filtrations;
  filtrations.reserve(num_generators);
  const int64_t* grade_ptr = grade_indptr.data();
  for (size_t i = 0; i < num_generators; ++i) {
    typename Concrete::Filtration_value filtration(num_parameters);
    auto inf = Concrete::Filtration_value::inf(num_parameters);
    filtration.push_to_least_common_upper_bound(inf, false);
    const int64_t begin = grade_ptr[i];
    const int64_t end = grade_ptr[i + 1];
    for (int64_t row = begin; row < end; ++row) {
      std::vector<Value> grade(num_parameters);
      for (size_t p = 0; p < num_parameters; ++p) {
        grade[p] = static_cast<Value>(grades_flat((size_t)row, p));
      }
      filtration.add_generator(grade);
    }
    filtrations.push_back(std::move(filtration));
  }

  Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value> cpx(
      boundaries, dims, filtrations);
  out.truc = Concrete(cpx);
  reset_python_state(out);
  return out;
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
              out.generator_basis = self.generator_basis;
              out.minpres_degree = self.minpres_degree;
              return out;
            })
        .def(
            "compute_kernel_projective_cover",
            [](Wrapper& self, nb::object dim_obj) {
              Wrapper out;
              if (self.truc.get_number_of_cycle_generators() == 0) {
                out.filtration_grid = self.filtration_grid;
                out.generator_basis = self.generator_basis;
                out.minpres_degree = self.minpres_degree;
                return out;
              }
              int dim = dim_obj.is_none() ? self.truc.get_dimension(0) : nb::cast<int>(dim_obj);
              {
                nb::gil_scoped_release release;
                out.truc = build_slicer_from_projective_cover_kernel(self.truc, dim);
              }
              out.filtration_grid = self.filtration_grid;
              out.generator_basis = self.generator_basis;
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
            [](Wrapper& self, bool update, nb::object idx_obj, nb::object intersect_points_obj) {
              std::vector<int64_t> requested;
              bool filter_cycles = !idx_obj.is_none();
              if (filter_cycles) {
                requested = cast_vector<int64_t>(idx_obj);
              }
              std::unordered_set<uint32_t> intersect_points;
              const bool filter_points = !intersect_points_obj.is_none();
              if (filter_points) {
                auto requested_points = cast_vector<uint32_t>(intersect_points_obj);
                intersect_points.insert(requested_points.begin(), requested_points.end());
              }
              GeneratorBasisData generator_basis = extract_generator_basis(self);
              std::vector<std::vector<std::vector<std::vector<uint32_t>>>> out_cpp;
              std::vector<std::vector<uint8_t>> keep_mask;
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
                keep_mask.resize(cycle_idx.size());
                for (size_t i = 0; i < cycle_idx.size(); ++i) {
                  out_cpp[i].resize(selected_indices[i].size());
                  keep_mask[i].assign(selected_indices[i].size(), 0);
                }
                tbb::parallel_for(size_t(0), cycle_idx.size(), [&](size_t i) {
                  for (size_t j = 0; j < selected_indices[i].size(); ++j) {
                    size_t selected_idx = selected_indices[i][j];
                    if (!cycle_idx[i][selected_idx].empty()) {
                      if (generator_basis.active && static_cast<int>(i) == generator_basis.degree) {
                        out_cpp[i][j] = expand_cycle_in_generator_basis(cycle_idx[i][selected_idx], generator_basis);
                      } else if (self.truc.get_boundary(cycle_idx[i][selected_idx][0]).empty()) {
                        out_cpp[i][j] = {std::vector<uint32_t>{}};
                      } else {
                        out_cpp[i][j].resize(cycle_idx[i][selected_idx].size());
                        for (size_t k = 0; k < cycle_idx[i][selected_idx].size(); ++k) {
                          out_cpp[i][j][k] = self.truc.get_boundary(cycle_idx[i][selected_idx][k]);
                        }
                      }
                      if (!filter_points || cycle_intersects_points(out_cpp[i][j], intersect_points)) {
                        keep_mask[i][j] = 1;
                      }
                    } else if (!filter_points) {
                      keep_mask[i][j] = 1;
                    }
                  }
                });
              }
              nb::list out;
              for (size_t i = 0; i < out_cpp.size(); ++i) {
                nb::list dim_cycles;
                for (size_t j = 0; j < out_cpp[i].size(); ++j) {
                  if (!keep_mask[i][j]) {
                    continue;
                  }
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
            "idx"_a = nb::none(),
            "intersect_points"_a = nb::none())
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
typename Desc::wrapper construct_from_supported_source(nb::object source) {
  using Wrapper = typename Desc::wrapper;
  using Concrete = typename Desc::concrete;

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
  throw nb::type_error(
      "Slicer construction from Python SCC/block iterables has been removed. "
      "Construct from a SimplexTreeMulti, an existing slicer, an SCC file path, "
      "or explicit (generator_maps, generator_dimensions, filtration_values) data.");
}

template <typename Desc>
void bind_slicer_class(nb::module_& m, nb::list& available_slicers) {
  using Wrapper = typename Desc::wrapper;
  using Concrete = typename Desc::concrete;
  using Value = typename Desc::value_type;

  auto cls =
      nb::class_<Wrapper>(m, Desc::python_name.data())
          .def(nb::init<>())
          .def(nb::new_([](nb::object source) { return construct_from_supported_source<Desc>(source); }),
               "source"_a = nb::none())
          .def(nb::new_([](std::string path, int shift_dimension) {
                 return construct_from_scc_file<Wrapper, Concrete>(path, shift_dimension);
               }),
               "path"_a,
               "shift_dimension"_a = 0)
          .def(nb::new_([](nb::object generator_maps, nb::object generator_dimensions, nb::object filtration_values) {
                 return construct_from_generator_data<Wrapper, Concrete, Value, Desc::is_kcritical>(
                     generator_maps, generator_dimensions, filtration_values);
               }),
               "generator_maps"_a,
               "generator_dimensions"_a,
               "filtration_values"_a);

  bind_typed_source_constructors<Desc>(cls);

  bind_dense_generator_data_overloads<decltype(cls), Wrapper, Concrete, Value, Desc::is_kcritical, int32_t>(cls);
  bind_dense_generator_data_overloads<decltype(cls), Wrapper, Concrete, Value, Desc::is_kcritical, int64_t>(cls);

  cls.def_prop_rw(
         "filtration_grid",
         [](Wrapper& self) -> nb::object { return self.filtration_grid; },
         [](Wrapper& self, nb::object value) { self.filtration_grid = value.is_none() ? nb::none() : value; },
         nb::arg("value").none())
      .def_prop_rw(
          "_generator_basis",
          [](Wrapper& self) -> nb::object { return self.generator_basis; },
          [](Wrapper& self, nb::object value) { self.generator_basis = value.is_none() ? nb::none() : value; },
          nb::arg("value").none())
      .def_rw("minpres_degree", &Wrapper::minpres_degree)
      .def("get_ptr", [](Wrapper& self) -> intptr_t { return reinterpret_cast<intptr_t>(&self.truc); })
      .def(
          "_from_ptr",
          [](Wrapper& self, intptr_t slicer_ptr) -> Wrapper& {
            self.truc = *reinterpret_cast<Concrete*>(slicer_ptr);
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("_serialize_state",
           [](Wrapper& self) -> nb::object {
             return serialized_state<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self);
           })
      .def(
          "_deserialize_state",
          [](Wrapper& self, nb::handle state) -> Wrapper& {
            load_state<Wrapper, Concrete, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, state);
            return self;
          },
          "state"_a,
          nb::rv_policy::reference_internal)
      .def("__len__", [](Wrapper& self) -> int { return self.truc.get_number_of_cycle_generators(); })
      .def_prop_ro("num_generators",
                   [](const Wrapper& self) -> int { return self.truc.get_number_of_cycle_generators(); })
      .def_prop_ro("num_parameters", [](const Wrapper& self) -> int { return self.truc.get_number_of_parameters(); })
      .def_prop_ro("dtype", [](const Wrapper&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def_prop_ro("_template_id", [](const Wrapper&) -> int { return Desc::template_id; })
      .def_prop_ro("col_type", [](const Wrapper&) -> std::string { return std::string(Desc::column_type); })
      .def_prop_ro("filtration_container",
                   [](const Wrapper&) -> std::string { return std::string(Desc::filtration_container); })
      .def_prop_ro("is_vine", [](const Wrapper&) -> bool { return Desc::is_vine; })
      .def_prop_ro("is_kcritical", [](const Wrapper&) -> bool { return Desc::is_kcritical; })
      .def_prop_ro("pers_backend", [](const Wrapper&) -> std::string { return std::string(Desc::backend_type); })
      .def_prop_ro("ftype", [](const Wrapper&) -> std::string { return std::string(Desc::filtration_type); })
      .def("__eq__",
           [](Wrapper& self, nb::handle other) -> bool {
             return wrapper_equals<Wrapper, Concrete, Desc::is_kcritical>(self, other);
           })
      .def_static("_inf_value",
                  []() {
                    if constexpr (std::is_floating_point_v<Value>) {
                      return nb::cast(std::numeric_limits<Value>::infinity());
                    }
                    return nb::cast(std::numeric_limits<Value>::max());
                  })
      .def("get_dimensions", [](Wrapper& self) -> nb::object { return dimensions_array(self); })
      .def(
          "get_boundaries",
          [](Wrapper& self, bool packed) -> nb::object { return boundaries_object(self, packed); },
          "packed"_a = false)
      .def(
          "get_filtration",
          [](Wrapper& self, int idx, bool raw) {
            Py_ssize_t n = static_cast<Py_ssize_t>(self.truc.get_number_of_cycle_generators());
            Py_ssize_t i = idx;
            if (i < 0) {
              i += n;
            }
            if (i < 0 || i >= n) {
              throw nb::index_error("Generator index out of range.");
            }
            return filtration_value_to_python<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(
                self, static_cast<size_t>(i), false, raw);
          },
          "idx"_a,
          "raw"_a = false)
      .def(
          "_get_filtrations_impl",
          [](Wrapper& self, bool raw, bool view, bool packed) -> nb::object {
            if (packed) {
              return pack_filtrations<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, raw);
            }
            if (view) {
              nb::list out;
              size_t n = self.truc.get_number_of_cycle_generators();
              for (size_t i = 0; i < n; ++i) {
                out.append(filtration_value_to_python<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(
                    self, i, false, raw));
              }
              return nb::object(out);
            }
            return copy_filtrations<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self, raw);
          },
          "raw"_a = false,
          "view"_a = false,
          "packed"_a = false)
      .def("get_filtrations_values",
           [](Wrapper& self) -> nb::object {
             return filtration_values_array<Wrapper, Value, Desc::is_kcritical, Desc::is_degree_rips>(self);
           })
      .def(
          "build_from_simplex_tree",
          [](Wrapper& self, nb::object st) -> Wrapper& {
            if (try_build_from_multipers_simplextree<Wrapper, Concrete>(self, st)) {
              return self;
            }
            throw std::runtime_error("Unsupported SimplexTreeMulti input type.");
          },
          nb::rv_policy::reference_internal)
      .def(
          "_build_from_scc_file",
          [](Wrapper& self, std::string path, bool rivet_compatible, bool reverse, int shift_dimension) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.truc = Gudhi::multi_persistence::build_slicer_from_scc_file<Concrete>(
                  path, rivet_compatible, reverse, shift_dimension);
            }
            return self;
          },
          "path"_a,
          "rivet_compatible"_a = false,
          "reverse"_a = false,
          "shift_dimension"_a = 0,
          nb::rv_policy::reference_internal)
      .def(
          "_to_scc_raw",
          [](Wrapper& self,
             std::string path,
             int degree,
             bool rivet_compatible,
             bool ignore_last_generators,
             bool strip_comments,
             bool reverse) -> void {
            {
              nb::gil_scoped_release release;
              write_slicer_to_scc_file(
                  path, self.truc, degree, rivet_compatible, ignore_last_generators, strip_comments, reverse);
            }
          },
          "path"_a,
          "degree"_a = -1,
          "rivet_compatible"_a = false,
          "ignore_last_generators"_a = false,
          "strip_comments"_a = false,
          "reverse"_a = false)
      .def(
          "push_to_line",
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
            }
            return self;
          },
          "basepoint"_a,
          "direction"_a = nb::none(),
          nb::rv_policy::reference_internal)
      .def(
          "set_slice",
          [](Wrapper& self, nb::object values) -> Wrapper& {
            auto c_values = cast_vector<Value>(values);
            {
              nb::gil_scoped_release release;
              self.truc.set_slice(c_values);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "initialize_persistence_computation",
          [](Wrapper& self, bool ignore_infinite_filtration_values) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.truc.initialize_persistence_computation(ignore_infinite_filtration_values);
            }
            return self;
          },
          "ignore_infinite_filtration_values"_a = true,
          nb::rv_policy::reference_internal)
      .def(
          "update_persistence_computation",
          [](Wrapper& self, bool ignore_infinite_filtration_values) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.truc.update_persistence_computation(ignore_infinite_filtration_values);
            }
            return self;
          },
          "ignore_infinite_filtration_values"_a = false,
          nb::rv_policy::reference_internal)
      .def("get_barcode",
           [](Wrapper& self) -> nb::tuple {
             using Barcode = decltype(self.truc.template get_flat_barcode<true, Value, false>());
             Barcode barcode;
             {
               nb::gil_scoped_release release;
               barcode = self.truc.template get_flat_barcode<true, Value, false>();
             }
             return dim_barcode_to_tuple<Barcode, Value>(barcode);
           })
      .def("get_barcode_idx",
           [](Wrapper& self) -> nb::tuple {
             using Barcode = decltype(self.truc.template get_flat_barcode<true, int, true>());
             Barcode barcode;
             {
               nb::gil_scoped_release release;
               barcode = self.truc.template get_flat_barcode<true, int, true>();
             }
             return dim_barcode_to_tuple<Barcode, int>(barcode);
           })
      .def("get_current_filtration",
           [](Wrapper& self) -> nb::object {
             std::vector<Value> current;
             {
               nb::gil_scoped_release release;
               current = self.truc.get_slice();
             }
             return nb::cast(owned_array<Value>(std::move(current), {current.size()}));
           })
      .def(
          "prune_above_dimension",
          [](Wrapper& self, int max_dimension) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.truc.prune_above_dimension(max_dimension);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "_make_filtration_non_decreasing_raw",
          [](Wrapper& self, bool safe) -> Wrapper& { return make_filtration_non_decreasing_inplace(self, safe); },
          "safe"_a = true,
          nb::rv_policy::reference_internal)
      .def(
          "coarsen_on_grid_inplace",
          [](Wrapper& self, std::vector<std::vector<Value>> grid, bool coordinates) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.truc.coarsen_on_grid(grid, coordinates);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "to_colexical",
          [](Wrapper& self, bool return_permutation) {
            decltype(build_permuted_slicer(self.truc)) stuff;
            {
              nb::gil_scoped_release release;
              stuff = build_permuted_slicer(self.truc);
            }
            Wrapper out;
            out.truc = std::move(stuff.first);
            out.filtration_grid = self.filtration_grid;
            out.generator_basis = self.generator_basis;
            out.minpres_degree = self.minpres_degree;
            if (!return_permutation) {
              return nb::object(nb::cast(out));
            }
            std::vector<uint32_t> perm(stuff.second.begin(), stuff.second.end());
            return nb::object(
                nb::make_tuple(nb::cast(out), owned_array<uint32_t>(std::move(perm), {stuff.second.size()})));
          },
          "return_permutation"_a = false)
      .def("permute_generators",
           [](Wrapper& self, std::vector<uint32_t> permutation) {
             Wrapper out;
             {
               nb::gil_scoped_release release;
               out.truc = build_permuted_slicer(self.truc, permutation);
             }
             out.filtration_grid = self.filtration_grid;
             out.generator_basis = self.generator_basis;
             out.minpres_degree = self.minpres_degree;
             return out;
           })
      .def("copy", [](Wrapper& self) -> Wrapper { return Wrapper(self); })
      .def("_info_string",
           [](Wrapper& self) -> std::string { return multipers::tmp_interface::slicer_to_str(self.truc); });

  bind_grid_methods<Desc>(cls);
  bind_vine_methods<Desc>(cls);
  available_slicers.append(cls);
}

template <typename... Desc>
void bind_all_slicers(type_list<Desc...>, nb::module_& m, nb::list& available_slicers) {
  (bind_slicer_class<Desc>(m, available_slicers), ...);
}

template <typename... Ds>
nb::tuple compute_hilbert_signed_measure(type_list<Ds...>,
                                         nb::handle slicer,
                                         std::vector<indices_type>& container,
                                         const std::vector<indices_type>& full_shape,
                                         const std::vector<indices_type>& degrees,
                                         size_t width,
                                         bool zero_pad,
                                         indices_type n_jobs,
                                         bool verbose,
                                         bool ignore_inf) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::tuple {
    auto& wrapper = nb::cast<typename D::wrapper&>(slicer);
    signed_measure_type sm;
    {
      nb::gil_scoped_release release;
      sm = Gudhi::multiparameter::hilbert_function::get_hilbert_signed_measure(
          wrapper.truc, container.data(), full_shape, degrees, zero_pad, n_jobs, verbose, ignore_inf);
    }
    return signed_measure_to_python(sm, width);
  });
}

template <typename... Ds>
nb::tuple compute_rank_tensor(type_list<Ds...>,
                              nb::handle slicer,
                              std::vector<tensor_dtype>& container,
                              const std::vector<indices_type>& full_shape,
                              const std::vector<indices_type>& degrees,
                              size_t total,
                              indices_type n_jobs,
                              bool ignore_inf) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type.");
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::tuple {
    auto& wrapper = nb::cast<typename D::wrapper&>(slicer);
    {
      nb::gil_scoped_release release;
      Gudhi::multiparameter::rank_invariant::compute_rank_invariant_python(
          wrapper.truc, container.data(), full_shape, degrees, n_jobs, ignore_inf);
    }
    return nb::make_tuple(nb::cast(owned_array<tensor_dtype>(std::move(container), {total})), nb::cast(full_shape));
  });
}

template <typename Desc>
nb::object module_approximation_from_desc(typename Desc::wrapper& wrapper,
                                          const std::vector<double>& direction,
                                          double max_error,
                                          Gudhi::multi_persistence::Box<double> box,
                                          bool threshold,
                                          bool complete,
                                          bool verbose,
                                          int n_jobs) {
  if constexpr (!Desc::enable_module_approximation) {
    throw std::runtime_error("Unsupported slicer type for module approximation.");
  } else {
    Gudhi::multi_persistence::Module<double> mod;
    {
      nb::gil_scoped_release release;
      mod = Gudhi::multiparameter::mma::multiparameter_module_approximation(
          wrapper.truc, direction, max_error, box, threshold, complete, verbose, n_jobs);
    }
    module_wrapper_t<canonical_double_mma_desc> out;
    out.mod = std::move(mod);
    return nb::cast(out);
  }
}

template <typename... Ds>
nb::object compute_module_approximation_from_slicer(type_list<Ds...>,
                                                    nb::handle slicer,
                                                    const std::vector<double>& direction,
                                                    double max_error,
                                                    Gudhi::multi_persistence::Box<double> box,
                                                    bool threshold,
                                                    bool complete,
                                                    bool verbose,
                                                    int n_jobs) {
  if (!has_slicer_template_id(slicer)) {
    throw std::runtime_error("Unsupported slicer type for module approximation.");
  }
  return dispatch_slicer_by_template_id(template_id_of(slicer), [&]<typename D>() -> nb::object {
    auto& wrapper = nb::cast<typename D::wrapper&>(slicer);
    return module_approximation_from_desc<D>(wrapper, direction, max_error, box, threshold, complete, verbose, n_jobs);
  });
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
      "build_kcritical_contiguous_slicer_from_packed_f64",
      [](nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> generator_dimensions,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> grade_indptr,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades_flat) {
        return mpnb::construct_kcritical_from_packed<mpnb::KcriticalContiguousF64MatrixSlicerDesc::wrapper,
                                                     mpnb::KcriticalContiguousF64MatrixSlicerDesc::concrete,
                                                     double>(
            boundary_indptr, boundary_flat, generator_dimensions, grade_indptr, grades_flat);
      },
      "boundary_indptr"_a,
      "boundary_flat"_a,
      "generator_dimensions"_a,
      "grade_indptr"_a,
      "grades_flat"_a);

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
         nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> direction,
         double max_error,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> box_array,
         bool threshold,
         bool complete,
         bool verbose,
         int n_jobs) {
        auto direction_values = mpnb::vector_from_array(direction);
        auto box_values = mpnb::matrix_from_array(box_array);
        Gudhi::multi_persistence::Box<double> box(box_values[0], box_values[1]);
        return mpnb::compute_module_approximation_from_slicer(mpnb::SlicerDescriptorList{},
                                                              slicer,
                                                              direction_values,
                                                              max_error,
                                                              box,
                                                              threshold,
                                                              complete,
                                                              verbose,
                                                              n_jobs);
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
        return nb::module_::import_("multipers.multiparameter_module_approximation")
            .attr("_module_approximation_single_input")(input,
                                                           box,
                                                           max_error,
                                                           nlines,
                                                           from_coordinates,
                                                           complete,
                                                           threshold,
                                                           verbose,
                                                           ignore_warnings,
                                                           nb::borrow(direction),
                                                           nb::borrow(swap_box_coords),
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

  m.def("_get_slicer_class_from_template_id", &mpnb::get_slicer_class_from_template_id, "template_id"_a);

  mpnb::bind_bitmap_builders(mpnb::SlicerDescriptorList{}, m);

  m.attr("available_slicers") = available_slicers;
}
