#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "Persistence_slices_interface.h"
#include "gudhi/Multi_persistence/Box.h"
#include "gudhi/Multi_persistence/Line.h"
#include "gudhi/Multi_persistence/Module.h"
#include "gudhi/Multi_persistence/Summand.h"
#include "gudhi/Multi_persistence/module_helpers.h"
#include <python_interfaces/numpy_utils.h>
#include "nanobind_array_utils.hpp"
#include "nanobind_dense_array_utils.hpp"
#include "nanobind_mma_registry_helpers.hpp"
#include "nanobind_object_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpmma {

using multipers::nanobind_dense_utils::cast_vector_from_array;
using multipers::nanobind_dense_utils::matrix_from_array;
using multipers::nanobind_dense_utils::vector_from_array;
using multipers::nanobind_mma_helpers::dispatch_mma_by_template_id;
using multipers::nanobind_mma_helpers::is_mma_module_object;
using multipers::nanobind_mma_helpers::MMADescriptorList;
using multipers::nanobind_mma_helpers::type_list;
using multipers::nanobind_utils::matrix_from_handle;
using multipers::nanobind_utils::numpy_dtype_type;
using multipers::nanobind_utils::owned_array;
using multipers::nanobind_utils::tuple_from_size;
using multipers::nanobind_utils::vector_from_handle;

template <typename T>
nb::ndarray<nb::numpy, T> corner_matrix_to_python(std::vector<T>&& flat, size_t rows, size_t cols) {
  return owned_array<T>(std::move(flat), {rows, cols});
}

template <typename T>
nb::ndarray<nb::numpy, T> corner_matrix_to_python(const std::vector<T>& flat, size_t rows, size_t cols) {
  return owned_array<T>(std::vector<T>(flat.begin(), flat.end()), {rows, cols});
}

template <typename T>
nb::ndarray<nb::numpy, T> corner_pair_to_python(const std::vector<T>& lower, const std::vector<T>& upper) {
  std::vector<T> flat;
  flat.reserve(lower.size() + upper.size());
  flat.insert(flat.end(), lower.begin(), lower.end());
  flat.insert(flat.end(), upper.begin(), upper.end());
  return owned_array<T>(std::move(flat), {size_t(2), lower.size()});
}

template <typename T>
nb::tuple dump_summand(const Gudhi::multi_persistence::Summand<T>& summand) {
  auto births = summand.compute_flat_upset();
  auto deaths = summand.compute_flat_downset();
  const size_t num_parameters = static_cast<size_t>(summand.get_number_of_parameters());
  return nb::make_tuple(
      corner_matrix_to_python<T>(
          std::move(births), static_cast<size_t>(summand.get_number_of_birth_corners()), num_parameters),
      corner_matrix_to_python<T>(
          std::move(deaths), static_cast<size_t>(summand.get_number_of_death_corners()), num_parameters),
      summand.get_dimension());
}

template <typename T>
nb::tuple barcode_to_python(
    const std::vector<std::vector<typename Gudhi::multi_persistence::Module<T>::Bar>>& barcode) {
  return tuple_from_size(barcode.size(), [&](size_t dim) -> nb::object {
    const auto& bars = barcode[dim];
    std::vector<T> flat;
    flat.reserve(bars.size() * 2);
    for (const auto& bar : bars) {
      flat.push_back(bar[0]);
      flat.push_back(bar[1]);
    }
    return nb::cast(owned_array<T>(std::move(flat), {bars.size(), size_t(2)}));
  });
}

template <typename T>
Gudhi::multi_persistence::Summand<T> summand_from_dump(nb::handle dump) {
  nb::tuple tpl = nb::cast<nb::tuple>(dump);
  auto births = matrix_from_handle<T>(tpl[0]);
  auto deaths = matrix_from_handle<T>(tpl[1]);
  int degree = nb::cast<int>(tpl[2]);
  std::vector<T> flat_births;
  std::vector<T> flat_deaths;
  size_t num_parameters = births.empty() ? 0 : births[0].size();
  for (const auto& row : births) {
    flat_births.insert(flat_births.end(), row.begin(), row.end());
  }
  for (const auto& row : deaths) {
    flat_deaths.insert(flat_deaths.end(), row.begin(), row.end());
  }
  return Gudhi::multi_persistence::Summand<T>(flat_births, flat_deaths, (int)num_parameters, degree);
}

template <typename T>
nb::tuple dump_module(const Gudhi::multi_persistence::Module<T>& module) {
  auto box = module.get_box();
  std::vector<T> lower(box.get_lower_corner().begin(), box.get_lower_corner().end());
  std::vector<T> upper(box.get_upper_corner().begin(), box.get_upper_corner().end());
  std::vector<T> flat_box;
  flat_box.reserve(lower.size() + upper.size());
  flat_box.insert(flat_box.end(), lower.begin(), lower.end());
  flat_box.insert(flat_box.end(), upper.begin(), upper.end());
  nb::object box_arr = nb::cast(owned_array<T>(std::move(flat_box), {size_t(2), box.get_lower_corner().size()}));

  nb::tuple summands = tuple_from_size(
      module.size(), [&](size_t i) -> nb::object { return dump_summand<T>(module.get_summand((unsigned int)i)); });
  return nb::make_tuple(box_arr, summands);
}

template <typename T>
Gudhi::multi_persistence::Box<T> box_from_rows(const std::vector<std::vector<T>>& rows) {
  if (rows.size() != 2) {
    throw std::runtime_error("Expected a box with shape (2, n_parameters).");
  }
  return Gudhi::multi_persistence::Box<T>(rows[0], rows[1]);
}

template <typename T>
Gudhi::multi_persistence::Box<T> box_from_array(nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box) {
  return box_from_rows<T>(matrix_from_array(box));
}

template <typename T>
Gudhi::multi_persistence::Line<T> line_from_vectors(const std::vector<T>& basepoint, const std::vector<T>* direction) {
  return direction == nullptr ? Gudhi::multi_persistence::Line<T>(basepoint)
                              : Gudhi::multi_persistence::Line<T>(basepoint, *direction);
}

template <typename T>
nb::tuple barcode_from_line_impl(Gudhi::multi_persistence::Module<T>& self,
                                 const std::vector<T>& basepoint,
                                 const std::vector<T>* direction,
                                 int degree) {
  auto line = line_from_vectors(basepoint, direction);
  decltype(self.get_barcode_from_line(line, degree)) barcode;
  {
    nb::gil_scoped_release release;
    barcode = self.get_barcode_from_line(line, degree);
  }
  return barcode_to_python<T>(barcode);
}

template <typename T, class GridRange>
Gudhi::multi_persistence::Module<T>& evaluate_in_grid_impl(Gudhi::multi_persistence::Module<T>& self,
                                                           const GridRange& grid) {
  {
    nb::gil_scoped_release release;
    self.evaluate_in_grid(grid);
  }
  return self;
}

template <typename T, class RandomAccessValueRange1, class RandomAccessValueRange2>
auto compute_landscapes_box_impl(Gudhi::multi_persistence::Module<T>& self,
                                                      int degree,
                                                      const RandomAccessValueRange1& ks,
                                                      const Gudhi::multi_persistence::Box<T>& box,
                                                      const RandomAccessValueRange2& resolution,
                                                      int n_jobs) {
  std::vector<T> out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_set_of_module_landscapes(self, degree, ks, box, resolution, n_jobs);
  }
  return _wrap_as_numpy_array(std::move(out), ks.size(), resolution[0], resolution[1]);
}

template <typename T, class RandomAccessValueRange, class RandomAccessArray>
auto compute_landscapes_grid_impl(Gudhi::multi_persistence::Module<T>& self,
                                                       int degree,
                                                       const RandomAccessValueRange& ks,
                                                       const std::vector<RandomAccessArray>& grid,
                                                       int n_jobs) {
  std::vector<T> out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_set_of_module_landscapes(self, degree, ks, grid, n_jobs);
  }
  return _wrap_as_numpy_array(std::move(out), ks.size(), grid[0].size(), grid[1].size());
}

template <typename T, class RandomAccessPointRange, class DimensionRange>
nb::ndarray<nb::numpy, T> compute_pixels_impl(Gudhi::multi_persistence::Module<T>& self,
                                              const RandomAccessPointRange& coordinates,
                                              const DimensionRange& degrees,
                                              const Gudhi::multi_persistence::Box<T>& box,
                                              T delta,
                                              T p,
                                              bool normalize,
                                              int n_jobs) {
  std::vector<T> out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_module_pixels(self, coordinates, degrees, box, delta, p, normalize, n_jobs);
  }
  return owned_array<T>(std::move(out), {degrees.size(), coordinates.size()});
}

template <typename T, class RandomAccessPointRange>
nb::ndarray<nb::numpy, T> distance_to_impl(Gudhi::multi_persistence::Module<T>& self,
                                           const RandomAccessPointRange& pts,
                                           bool signed_distance,
                                           int n_jobs) {
  std::vector<T> out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_module_distances_to(self, pts, signed_distance, n_jobs);
  }
  return owned_array<T>(std::move(out), {pts.size(), self.size()});
}

template <typename T>
Gudhi::multi_persistence::Module<T>& set_box_impl(Gudhi::multi_persistence::Module<T>& self,
                                                  const Gudhi::multi_persistence::Box<T>& box) {
  {
    nb::gil_scoped_release release;
    self.set_box(box);
  }
  return self;
}

template <typename T>
Gudhi::multi_persistence::Module<T>& rescale_impl(Gudhi::multi_persistence::Module<T>& self,
                                                  const std::vector<T>& factors,
                                                  int degree) {
  {
    nb::gil_scoped_release release;
    self.rescale(factors, degree);
  }
  return self;
}

template <typename T>
Gudhi::multi_persistence::Module<T>& translate_impl(Gudhi::multi_persistence::Module<T>& self,
                                                    const std::vector<T>& factors,
                                                    int degree) {
  {
    nb::gil_scoped_release release;
    self.translate(factors, degree);
  }
  return self;
}

template <typename T>
Gudhi::multi_persistence::Module<T> module_from_dump(nb::handle dump) {
  nb::tuple tpl = nb::cast<nb::tuple>(dump);
  auto box = matrix_from_handle<T>(tpl[0]);
  std::vector<T> lower = box.empty() ? std::vector<T>{} : box[0];
  std::vector<T> upper = box.size() < 2 ? std::vector<T>{} : box[1];
  Gudhi::multi_persistence::Module<T> out;
  out.set_box(Gudhi::multi_persistence::Box<T>(lower, upper));
  for (nb::handle summand_dump : nb::iter(tpl[1])) {
    out.add_summand(summand_from_dump<T>(summand_dump));
  }
  return out;
}

template <typename T>
std::vector<std::vector<T>> filtration_values_from_module(const Gudhi::multi_persistence::Module<T>& module,
                                                          bool unique) {
  std::vector<std::vector<T>> values;
  size_t num_parameters = module.get_box().get_lower_corner().size();
  values.resize(num_parameters);
  for (const auto& summand : module) {
    auto births = summand.compute_flat_upset();
    auto deaths = summand.compute_flat_downset();
    const size_t birth_count = static_cast<size_t>(summand.get_number_of_birth_corners());
    const size_t death_count = static_cast<size_t>(summand.get_number_of_death_corners());
    for (size_t p = 0; p < num_parameters; ++p) {
      for (size_t row = 0; row < birth_count; ++row) {
        const T value = births[row * num_parameters + p];
        if constexpr (!std::numeric_limits<T>::has_infinity) {
          values[p].push_back(value);
        } else if (std::isfinite(static_cast<double>(value))) {
          values[p].push_back(value);
        }
      }
      for (size_t row = 0; row < death_count; ++row) {
        const T value = deaths[row * num_parameters + p];
        if constexpr (!std::numeric_limits<T>::has_infinity) {
          values[p].push_back(value);
        } else if (std::isfinite(static_cast<double>(value))) {
          values[p].push_back(value);
        }
      }
    }
  }
  if (unique) {
    for (auto& vals : values) {
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      if constexpr (std::numeric_limits<T>::has_infinity) {
        const T pos_inf = std::numeric_limits<T>::infinity();
        const T neg_inf = -pos_inf;
        if (!vals.empty() && vals.back() == pos_inf) vals.pop_back();
        if (!vals.empty() && vals.front() == neg_inf) vals.erase(vals.begin());
      }
    }
  }
  return values;
}

template <typename T>
int32_t grid_coord_from_value(T value, const std::vector<T>& axis) {
  if (axis.empty()) {
    throw std::runtime_error("Grid axes must be non-empty.");
  }
  if constexpr (std::numeric_limits<T>::has_infinity) {
    const T pos_inf = std::numeric_limits<T>::infinity();
    const T neg_inf = -pos_inf;
    if (std::isnan(static_cast<double>(value)) || value == pos_inf) {
      return static_cast<int32_t>(axis.size() - 1);
    }
    if (value == neg_inf) {
      return 0;
    }
  }
  if (value >= axis.back()) {
    return static_cast<int32_t>(axis.size() - 1);
  }
  if (value <= axis.front()) {
    return 0;
  }
  return static_cast<int32_t>(std::distance(axis.begin(), std::lower_bound(axis.begin(), axis.end(), value)));
}

template <typename T>
std::vector<int32_t> flat_idx_from_corners(const std::vector<T>& corners,
                                           size_t num_parameters,
                                           const std::vector<std::vector<T>>& grid) {
  if (grid.size() != num_parameters) {
    throw std::runtime_error("Grid dimension does not match module number of parameters.");
  }
  if (num_parameters == 0) {
    return {};
  }
  if (corners.size() % num_parameters != 0) {
    throw std::runtime_error("Corner array size is not divisible by number of parameters.");
  }

  const size_t num_rows = corners.size() / num_parameters;
  std::vector<int32_t> out(corners.size());
  for (size_t row = 0; row < num_rows; ++row) {
    const size_t offset = row * num_parameters;
    bool all_pos_inf_or_nan = true;
    bool all_neg_inf = true;
    if constexpr (std::numeric_limits<T>::has_infinity) {
      const T pos_inf = std::numeric_limits<T>::infinity();
      const T neg_inf = -pos_inf;
      for (size_t p = 0; p < num_parameters; ++p) {
        const T value = corners[offset + p];
        all_pos_inf_or_nan &= std::isnan(static_cast<double>(value)) || value == pos_inf;
        all_neg_inf &= value == neg_inf;
      }
    } else {
      all_pos_inf_or_nan = false;
      all_neg_inf = false;
    }

    if (all_pos_inf_or_nan) {
      for (size_t p = 0; p < num_parameters; ++p) {
        out[offset + p] = static_cast<int32_t>(grid[p].size() - 1);
      }
      continue;
    }
    if (all_neg_inf) {
      for (size_t p = 0; p < num_parameters; ++p) {
        out[offset + p] = 0;
      }
      continue;
    }

    for (size_t p = 0; p < num_parameters; ++p) {
      out[offset + p] = grid_coord_from_value(corners[offset + p], grid[p]);
    }
  }
  return out;
}

template <typename T>
nb::tuple to_flat_idx_impl(const Gudhi::multi_persistence::Module<T>& module, const std::vector<std::vector<T>>& grid) {
  const size_t num_parameters = module.get_box().get_lower_corner().size();
  const size_t num_summands = module.size();

  std::vector<int32_t> birth_sizes;
  std::vector<int32_t> death_sizes;
  std::vector<T> flat_births;
  std::vector<T> flat_deaths;
  std::vector<int32_t> size_matrix;
  std::vector<int32_t> birth_coords;
  std::vector<int32_t> death_coords;
  size_t birth_rows = 0;
  size_t death_rows = 0;

  {
    nb::gil_scoped_release release;
    birth_sizes.reserve(num_summands);
    death_sizes.reserve(num_summands);
    for (size_t i = 0; i < num_summands; ++i) {
      const auto summand = module.get_summand((unsigned int)i);
      birth_sizes.push_back(static_cast<int32_t>(summand.get_number_of_birth_corners()));
      death_sizes.push_back(static_cast<int32_t>(summand.get_number_of_death_corners()));

      auto births = summand.compute_birth_list();
      auto deaths = summand.compute_death_list();
      flat_births.insert(flat_births.end(), births.begin(), births.end());
      flat_deaths.insert(flat_deaths.end(), deaths.begin(), deaths.end());
    }

    size_matrix.reserve(2 * num_summands);
    size_matrix.insert(size_matrix.end(), birth_sizes.begin(), birth_sizes.end());
    size_matrix.insert(size_matrix.end(), death_sizes.begin(), death_sizes.end());

    birth_rows = num_parameters == 0 ? size_t(0) : flat_births.size() / num_parameters;
    death_rows = num_parameters == 0 ? size_t(0) : flat_deaths.size() / num_parameters;
    birth_coords = flat_idx_from_corners(flat_births, num_parameters, grid);
    death_coords = flat_idx_from_corners(flat_deaths, num_parameters, grid);
  }

  return nb::make_tuple(owned_array<int32_t>(std::move(size_matrix), {size_t(2), num_summands}),
                        owned_array<int32_t>(std::move(birth_coords), {birth_rows, num_parameters}),
                        owned_array<int32_t>(std::move(death_coords), {death_rows, num_parameters}));
}

template <typename T>
struct ModuleSummandIterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = Gudhi::multi_persistence::Summand<T>;
  using difference_type = std::ptrdiff_t;

  Gudhi::multi_persistence::Module<T>* module = nullptr;
  size_t index = 0;

  value_type operator*() const { return module->get_summand((unsigned int)index); }

  ModuleSummandIterator& operator++() {
    ++index;
    return *this;
  }

  bool operator==(const ModuleSummandIterator& other) const { return module == other.module && index == other.index; }

  bool operator!=(const ModuleSummandIterator& other) const { return !(*this == other); }
};

template <typename Wrapper>
nb::object self_handle(Wrapper& self) {
  return nb::find(self);
}

template <typename Desc, typename Class>
void bind_float_module_methods(Class& cls) {
  if constexpr (Desc::is_float) {
    using T = typename Desc::value_type;
    using Box = Gudhi::multi_persistence::Box<T>;
    using Module = Gudhi::multi_persistence::Module<T>;

    cls.def(
           "_get_barcode_from_line",
           [](Module& self,
              nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> basepoint,
              nb::object direction,
              int degree) -> nb::tuple {
             auto basepoint_values = vector_from_array(basepoint);
             if (direction.is_none()) {
               return barcode_from_line_impl<T>(self, basepoint_values, nullptr, degree);
             }
             auto direction_array = nb::cast<nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig>>(direction);
             auto direction_values = vector_from_array(direction_array);
             return barcode_from_line_impl<T>(self, basepoint_values, &direction_values, degree);
           },
           "basepoint"_a,
           "direction"_a = nb::none(),
           "degree"_a = -1)
        .def(
            "evaluate_in_grid",
            [](Module& self, nb::handle grid_handle) -> Module& {
              return evaluate_in_grid_impl<T>(self, matrix_from_handle<T>(grid_handle));
            },
            nb::rv_policy::reference_internal)
        .def(
            "evaluate_in_grid",
            [](Module& self, nb::ndarray<const T, nb::ndim<2>, nb::any_contig> grid) -> Module& {
              return evaluate_in_grid_impl<T>(self, Numpy_2d_span<T>(grid));
            },
            nb::rv_policy::reference_internal)
        .def(
            "_compute_landscapes_box",
            [](Module& self,
               int degree,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> resolution,
               int n_jobs) {
              return compute_landscapes_box_impl<T>(self,
                                                    degree,
                                                    make_element_range(ks.data(), ks.view(), false),
                                                    box_from_array<T>(box),
                                                    make_element_range(resolution.data(), resolution.view(), false),
                                                    n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "box"_a,
            "resolution"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_box",
            [](Module& self,
               int degree,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> resolution,
               int n_jobs) {
              return compute_landscapes_box_impl<T>(self,
                                                    degree,
                                                    make_element_range(ks.data(), ks.view(), false),
                                                    box_from_array<T>(box),
                                                    make_element_range(resolution.data(), resolution.view(), false),
                                                    n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "box"_a,
            "resolution"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_grid",
            [](Module& self, int degree, nb::handle ks_handle, nb::handle grid_handle, int n_jobs) {
              auto ks_in = vector_from_handle<int>(ks_handle);
              std::vector<unsigned int> ks;
              ks.reserve(ks_in.size());
              for (int k : ks_in) ks.push_back((unsigned int)k);
              return compute_landscapes_grid_impl<T>(self, degree, ks, matrix_from_handle<T>(grid_handle), n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "grid"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_grid",
            [](Module& self,
               int degree,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> grid,
               int n_jobs) {
              return compute_landscapes_grid_impl<T>(
                  self,
                  degree,
                  make_element_range(ks.data(), ks.view(), false),
                  multipers::nanobind_dense_utils::non_regular_matrix_from_array(grid),
                  n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "grid"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_grid",
            [](Module& self,
               int degree,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> grid,
               int n_jobs) {
              return compute_landscapes_grid_impl<T>(
                  self,
                  degree,
                  make_element_range(ks.data(), ks.view(), false),
                  multipers::nanobind_dense_utils::non_regular_matrix_from_array(grid),
                  n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "grid"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_pixels",
            [](Module& self,
               nb::handle coordinates_handle,
               nb::handle degrees_handle,
               nb::handle box_handle,
               T delta,
               T p,
               bool normalize,
               int n_jobs) {
              return compute_pixels_impl<T>(self,
                                            matrix_from_handle<T>(coordinates_handle),
                                            vector_from_handle<int>(degrees_handle),
                                            box_from_rows<T>(matrix_from_handle<T>(box_handle)),
                                            delta,
                                            p,
                                            normalize,
                                            n_jobs);
            },
            "coordinates"_a,
            "degrees"_a,
            "box"_a,
            "delta"_a,
            "p"_a,
            "normalize"_a = false,
            "n_jobs"_a = 0)
        .def(
            "_compute_pixels",
            [](Module& self,
               nb::ndarray<const T, nb::ndim<2>, nb::any_contig> coordinates,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> degrees,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               T delta,
               T p,
               bool normalize,
               int n_jobs) {
              return compute_pixels_impl<T>(self,
                                            Numpy_2d_span<T>(coordinates),
                                            make_element_range(degrees.data(), degrees.view(), false),
                                            box_from_array<T>(box),
                                            delta,
                                            p,
                                            normalize,
                                            n_jobs);
            },
            "coordinates"_a,
            "degrees"_a,
            "box"_a,
            "delta"_a,
            "p"_a,
            "normalize"_a = false,
            "n_jobs"_a = 0)
        .def(
            "_compute_pixels",
            [](Module& self,
               nb::ndarray<const T, nb::ndim<2>, nb::any_contig> coordinates,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> degrees,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               T delta,
               T p,
               bool normalize,
               int n_jobs) {
              return compute_pixels_impl<T>(self,
                                            Numpy_2d_span<T>(coordinates),
                                            make_element_range(degrees.data(), degrees.view(), false),
                                            box_from_array<T>(box),
                                            delta,
                                            p,
                                            normalize,
                                            n_jobs);
            },
            "coordinates"_a,
            "degrees"_a,
            "box"_a,
            "delta"_a,
            "p"_a,
            "normalize"_a = false,
            "n_jobs"_a = 0)
        .def(
            "distance_to",
            [](Module& self, nb::handle pts_handle, bool signed_distance, int n_jobs) {
              return distance_to_impl<T>(self, matrix_from_handle<T>(pts_handle), signed_distance, n_jobs);
            },
            "pts"_a,
            "signed"_a = false,
            "n_jobs"_a = 0)
        .def(
            "distance_to",
            [](Module& self,
               nb::ndarray<const T, nb::ndim<2>, nb::any_contig> pts,
               bool signed_distance,
               int n_jobs) { return distance_to_impl<T>(self, Numpy_2d_span<T>(pts), signed_distance, n_jobs); },
            "pts"_a,
            "signed"_a = false,
            "n_jobs"_a = 0)
        .def("get_interleavings",
             [](Module& self) -> nb::ndarray<nb::numpy, T> {
               std::vector<T> interleavings;
               {
                 nb::gil_scoped_release release;
                 interleavings = Gudhi::multi_persistence::compute_module_interleavings(self, self.get_box());
               }
               return owned_array<T>(std::move(interleavings), {interleavings.size()});
             })
        .def(
            "get_interleavings",
            [](Module& self,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box) -> nb::ndarray<nb::numpy, T> {
              std::vector<T> interleavings;
              {
                nb::gil_scoped_release release;
                interleavings = Gudhi::multi_persistence::compute_module_interleavings(self, box_from_array<T>(box));
              }
              return owned_array<T>(std::move(interleavings), {interleavings.size()});
            },
            "box"_a);
  }
}

template <typename Desc>
void bind_summand_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Summand = Gudhi::multi_persistence::Summand<T>;

  nb::class_<Summand>(m, Desc::summand_name.data())
      .def(nb::init<>())
      .def("get_birth_list",
           [](Summand& self) -> nb::ndarray<nb::numpy, T> {
             std::vector<T> births;
             const size_t num_parameters = static_cast<size_t>(self.get_number_of_parameters());
             const size_t num_birth_corners = static_cast<size_t>(self.get_number_of_birth_corners());
             {
               nb::gil_scoped_release release;
               births = self.compute_flat_upset();
             }
             return corner_matrix_to_python<T>(std::move(births), num_birth_corners, num_parameters);
           })
      .def("get_death_list",
           [](Summand& self) -> nb::ndarray<nb::numpy, T> {
             std::vector<T> deaths;
             const size_t num_parameters = static_cast<size_t>(self.get_number_of_parameters());
             const size_t num_death_corners = static_cast<size_t>(self.get_number_of_death_corners());
             {
               nb::gil_scoped_release release;
               deaths = self.compute_flat_downset();
             }
             return corner_matrix_to_python<T>(std::move(deaths), num_death_corners, num_parameters);
           })
      .def_prop_ro("degree", [](const Summand& self) -> int { return self.get_dimension(); })
      .def("get_bounds",
           [](Summand& self) -> nb::ndarray<nb::numpy, T> {
             std::pair<std::vector<T>, std::vector<T>> cbounds;
             {
               nb::gil_scoped_release release;
               auto bounds = self.compute_bounds();
               auto cpp_bounds = bounds.get_bounding_corners();
               cbounds.first.assign(cpp_bounds.first.begin(), cpp_bounds.first.end());
               cbounds.second.assign(cpp_bounds.second.begin(), cpp_bounds.second.end());
             }
             return corner_pair_to_python<T>(cbounds.first, cbounds.second);
           })
      .def("num_parameters", [](Summand& self) -> int { return self.get_number_of_parameters(); })
      .def_prop_ro("_template_id", [](const Summand&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const Summand&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def("__eq__", [](Summand& self, Summand& other) { return self == other; });
}

template <typename Desc>
void bind_box_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Box = Gudhi::multi_persistence::Box<T>;

  nb::class_<Box>(m, Desc::box_name.data())
      .def(nb::new_([](nb::handle bottom, nb::handle top) {
             auto lower = vector_from_handle<T>(bottom);
             auto upper = vector_from_handle<T>(top);
             return Box(lower, upper);
           }),
           "bottomCorner"_a,
           "topCorner"_a)
      .def(nb::new_([](nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> bottom,
                       nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> top) {
             return Box(vector_from_array(bottom), vector_from_array(top));
           }),
           "bottomCorner"_a,
           "topCorner"_a)
      .def_prop_ro("num_parameters", [](const Box& self) -> int { return self.get_lower_corner().size(); })
      .def("contains",
           [](Box& self, nb::handle x) {
             auto values = vector_from_handle<T>(x);
             return self.contains(values);
           })
      .def("contains",
           [](Box& self, nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> x) {
             return self.contains(vector_from_array(x));
           })
      .def("get",
           [](Box& self) -> nb::ndarray<nb::numpy, T> {
             auto lower = std::vector<T>(self.get_lower_corner().begin(), self.get_lower_corner().end());
             auto upper = std::vector<T>(self.get_upper_corner().begin(), self.get_upper_corner().end());
             return corner_pair_to_python<T>(lower, upper);
            })
       .def("to_multipers",
            [](Box& self) -> nb::ndarray<nb::numpy, T> {
             auto lower = std::vector<T>(self.get_lower_corner().begin(), self.get_lower_corner().end());
             auto upper = std::vector<T>(self.get_upper_corner().begin(), self.get_upper_corner().end());
             std::vector<T> flat;
             flat.reserve(lower.size() * 2);
             for (size_t i = 0; i < lower.size(); ++i) {
               flat.push_back(lower[i]);
               flat.push_back(upper[i]);
             }
              return owned_array<T>(std::move(flat), {size_t(2), lower.size()});
           })
      .def_prop_ro("_template_id", [](const Box&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const Box&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); });
}

template <typename Desc>
void bind_module_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Module = Gudhi::multi_persistence::Module<T>;

  std::string iterator_name = std::string("_PyModuleIterator_") + std::string(Desc::short_name);

  auto module_cls =
      nb::class_<Module>(m, Desc::module_name.data())
          .def(nb::init<>())
          .def(
              "_from_ptr",
              [](Module& self, intptr_t ptr) -> Module& {
                auto* other = reinterpret_cast<Module*>(ptr);
                self = std::move(*other);
                delete other;
                return self;
              },
              nb::rv_policy::reference_internal)
          .def("__len__", [](Module& self) -> int { return self.size(); })
          .def("__eq__", [](Module& self, Module& other) { return self == other; })
          .def_prop_ro("_template_id", [](const Module&) -> int { return Desc::template_id; })
          .def("__getitem__",
               [](Module& self, nb::object key) -> nb::object {
                 if (nb::isinstance<nb::slice>(key)) {
                   auto [start, stop, step, length] = nb::cast<nb::slice>(key).compute(self.size());
                   if (start == 0 && stop == static_cast<Py_ssize_t>(self.size()) && step == 1 &&
                       length == self.size()) {
                     return self_handle(self);
                   }
                   throw nb::index_error("Only [:] slices are supported.");
                 }
                 int index = nb::cast<int>(key);
                 size_t size = self.size();
                 if (size == 0) {
                   throw nb::index_error("Module is empty.");
                 }
                 if (index < 0) {
                   index += (int)size;
                 }
                 if (index < 0 || index >= (int)size) {
                   throw nb::index_error("Summand index out of range.");
                 }
                 return nb::cast(self.get_summand((unsigned int)index));
               })
          .def(
              "__iter__",
              [iterator_name](Module& self) {
                return nb::make_iterator(nb::type<Module>(),
                                         iterator_name.c_str(),
                                         ModuleSummandIterator<T>{&self, 0},
                                         ModuleSummandIterator<T>{&self, self.size()});
              },
              nb::keep_alive<0, 1>())
          .def(
              "merge",
              [](Module& self, Module& other, int dim) -> Module& {
                Module c_other = other;
                {
                  nb::gil_scoped_release release;
                  for (const auto& summand : c_other) self.add_summand(summand, dim);
                }
                return self;
              },
              "other"_a,
              "dim"_a = -1,
              nb::rv_policy::reference_internal)
          .def("permute_summands",
               [](Module& self, nb::ndarray<const int, nb::ndim<1>, nb::c_contig, nb::device::cpu> permutation) {
                 Module out;
                 {
                   nb::gil_scoped_release release;
                   out = Gudhi::multi_persistence::build_permuted_module(
                       self, make_element_range(permutation.data(), permutation.view(), false));
                 }
                 return out;
               })
          .def(
              "set_box",
              [](Module& self, nb::handle box_handle) -> Module& {
                return set_box_impl<T>(self, box_from_rows<T>(matrix_from_handle<T>(box_handle)));
              },
              nb::rv_policy::reference_internal)
          .def(
              "set_box",
              [](Module& self, nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box) -> Module& {
                return set_box_impl<T>(self, box_from_array<T>(box));
              },
              nb::rv_policy::reference_internal)
          .def("get_module_of_degree",
               [](Module& self, int degree) {
                 Module out;
                 {
                   nb::gil_scoped_release release;
                   out.set_box(self.get_box());
                   for (const auto& summand : self)
                     if (summand.get_dimension() == degree) out.add_summand(summand);
                 }
                 return out;
               })
          .def("get_module_of_degrees",
               [](Module& self, nb::handle degrees_handle) {
                 auto degrees = vector_from_handle<int>(degrees_handle);
                 Module out;
                 {
                   nb::gil_scoped_release release;
                   out.set_box(self.get_box());
                   for (const auto& summand : self) {
                     for (int degree : degrees) {
                       if (degree == summand.get_dimension()) {
                         out.add_summand(summand);
                         break;
                       }
                     }
                   }
                 }
                 return out;
               })
          .def("_get_dump", [](Module& self) -> nb::tuple { return dump_module<T>(self); })
          .def(
              "dump",
              [](Module& self, nb::object path) -> nb::tuple {
                nb::tuple dump = dump_module<T>(self);
                if (!path.is_none()) {
                  nb::object builtins = nb::module_::import_("builtins");
                  nb::object pickle = nb::module_::import_("pickle");
                  nb::object handle = builtins.attr("open")(path, "wb");
                  try {
                    pickle.attr("dump")(dump, handle);
                  } catch (...) {
                    handle.attr("close")();
                    throw;
                  }
                  handle.attr("close")();
                }
                return dump;
              },
              "path"_a = nb::none())
          .def(
              "_load_dump",
              [](Module& self, nb::handle dump) -> Module& {
                self = module_from_dump<T>(dump);
                return self;
              },
              nb::rv_policy::reference_internal)
          .def("__getstate__", [](Module& self) -> nb::tuple { return dump_module<T>(self); })
          .def("__setstate__", [](Module& self, nb::handle state) { new (&self) Module(module_from_dump<T>(state)); })
          .def(
              "_add_mmas",
              [](Module& self, nb::iterable mmas) -> Module& {
                for (nb::handle item : mmas) {
                  Module c_other = nb::cast<Module&>(item);
                  {
                    nb::gil_scoped_release release;
                    for (const auto& summand : c_other) self.add_summand(summand);
                  }
                }
                return self;
              },
              nb::rv_policy::reference_internal)
           .def("get_bottom",
                [](Module& self) -> nb::ndarray<nb::numpy, T> {
                  return owned_array<T>(
                      std::vector<T>(self.get_box().get_lower_corner().begin(), self.get_box().get_lower_corner().end()),
                      {self.get_box().get_lower_corner().size()});
                })
           .def("get_top",
                [](Module& self) -> nb::ndarray<nb::numpy, T> {
                  return owned_array<T>(
                      std::vector<T>(self.get_box().get_upper_corner().begin(), self.get_box().get_upper_corner().end()),
                      {self.get_box().get_upper_corner().size()});
                })
           .def("get_box",
                [](Module& self) -> nb::ndarray<nb::numpy, T> {
                 auto lower =
                     std::vector<T>(self.get_box().get_lower_corner().begin(), self.get_box().get_lower_corner().end());
                 auto upper =
                     std::vector<T>(self.get_box().get_upper_corner().begin(), self.get_box().get_upper_corner().end());
                 std::vector<T> flat_box;
                 flat_box.reserve(lower.size() + upper.size());
                 flat_box.insert(flat_box.end(), lower.begin(), lower.end());
                 flat_box.insert(flat_box.end(), upper.begin(), upper.end());
                  return owned_array<T>(std::move(flat_box), {size_t(2), self.get_box().get_lower_corner().size()});
               })
          .def_prop_ro("max_degree", [](const Module& self) -> int { return self.get_max_dimension(); })
          .def_prop_ro("num_parameters",
                       [](const Module& self) -> int { return self.get_box().get_lower_corner().size(); })
           .def("get_bounds",
                [](Module& self) -> nb::ndarray<nb::numpy, T> {
                  std::pair<std::vector<T>, std::vector<T>> cbounds;
                  {
                    nb::gil_scoped_release release;
                    auto bounds = self.compute_bounds();
                    auto cpp_bounds = bounds.get_bounding_corners();
                    cbounds.first.assign(cpp_bounds.first.begin(), cpp_bounds.first.end());
                    cbounds.second.assign(cpp_bounds.second.begin(), cpp_bounds.second.end());
                  }
                  return corner_pair_to_python<T>(cbounds.first, cbounds.second);
                })
          .def(
              "rescale",
              [](Module& self, nb::handle factors, int degree) -> Module& {
                return rescale_impl<T>(self, vector_from_handle<T>(factors), degree);
              },
              "rescale_factors"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "rescale",
              [](Module& self,
                 nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> factors,
                 int degree) -> Module& { return rescale_impl<T>(self, vector_from_array(factors), degree); },
              "rescale_factors"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "translate",
              [](Module& self, nb::handle factors, int degree) -> Module& {
                return translate_impl<T>(self, vector_from_handle<T>(factors), degree);
              },
              "translation"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "translate",
              [](Module& self,
                 nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> factors,
                 int degree) -> Module& { return translate_impl<T>(self, vector_from_array(factors), degree); },
              "translation"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "get_filtration_values",
              [](Module& self, bool unique) -> std::vector<std::vector<T>> {
                return filtration_values_from_module<T>(self, unique);
              },
              "unique"_a = true)
          .def(
              "to_flat_idx",
              [](const Module& self, nb::handle grid_handle)
                  -> nb::tuple { return to_flat_idx_impl<T>(self, matrix_from_handle<T>(grid_handle)); },
              "grid"_a)
          .def(
              "to_flat_idx",
              [](const Module& self, nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> grid) -> nb::tuple {
                return to_flat_idx_impl<T>(self, matrix_from_array(grid));
              },
              "grid"_a)
          .def("get_dimensions", [](Module& self) -> nb::ndarray<nb::numpy, int32_t> {
            std::vector<int32_t> dims;
            dims.reserve(self.size());
            for (size_t i = 0; i < self.size(); ++i)
              dims.push_back((int32_t)self.get_summand((unsigned int)i).get_dimension());
            return owned_array<int32_t>(std::move(dims), {self.size()});
          });

  bind_float_module_methods<Desc>(module_cls);

  module_cls.def_prop_ro("dtype", [](const Module&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); });

  std::string from_dump_name = std::string(Desc::from_dump_name);
  m.def(
      from_dump_name.c_str(),
      [](nb::handle dump) {
        Module out;
        out = module_from_dump<T>(dump);
        return out;
      },
      "dump"_a);
}

template <typename Desc>
void bind_mma_type(nb::module_& m) {
  bind_summand_class<Desc>(m);
  bind_box_class<Desc>(m);
  bind_module_class<Desc>(m);
}

template <typename... Desc>
void bind_all_mma(type_list<Desc...>, nb::module_& m) {
  (bind_mma_type<Desc>(m), ...);
}

inline bool is_mma(nb::handle stuff) { return is_mma_module_object(stuff); }

}  // namespace mpmma

NB_MODULE(_mma_nanobind, m) {
  m.doc() = "nanobind MMA bindings";
  mpmma::bind_all_mma(mpmma::MMADescriptorList{}, m);
  m.def("is_mma", [](nb::handle stuff) { return mpmma::is_mma(stuff); });
}
