#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
using multipers::nanobind_mma_helpers::module_wrapper_t;
using multipers::nanobind_mma_helpers::PyModule;
using multipers::nanobind_mma_helpers::type_list;
using multipers::nanobind_utils::has_template_id;
using multipers::nanobind_utils::matrix_from_handle;
using multipers::nanobind_utils::numpy_dtype_type;
using multipers::nanobind_utils::owned_array;
using multipers::nanobind_utils::template_id_of;
using multipers::nanobind_utils::tuple_from_size;
using multipers::nanobind_utils::vector_from_handle;

template <typename T>
nb::object corner_matrix_to_python(std::vector<T>&& flat, size_t rows, size_t cols) {
  return nb::cast(owned_array<T>(std::move(flat), {rows, cols}));
}

template <typename T>
nb::object corner_matrix_to_python(const std::vector<T>& flat, size_t rows, size_t cols) {
  return nb::cast(owned_array<T>(std::vector<T>(flat.begin(), flat.end()), {rows, cols}));
}

template <typename T>
nb::tuple dump_summand(const Gudhi::multi_persistence::Summand<T>& summand) {
  auto births = summand.compute_birth_list();
  auto deaths = summand.compute_death_list();
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
Gudhi::multi_persistence::Box<T> box_from_handle(nb::handle h) {
  return box_from_rows<T>(matrix_from_handle<T>(h));
}

template <typename T>
Gudhi::multi_persistence::Line<T> line_from_vectors(const std::vector<T>& basepoint, const std::vector<T>* direction) {
  return direction == nullptr ? Gudhi::multi_persistence::Line<T>(basepoint)
                              : Gudhi::multi_persistence::Line<T>(basepoint, *direction);
}

template <typename T>
nb::tuple barcode_from_line_impl(PyModule<T>& self,
                                 const std::vector<T>& basepoint,
                                 const std::vector<T>* direction,
                                 int degree) {
  auto line = line_from_vectors(basepoint, direction);
  decltype(self.mod.get_barcode_from_line(line, degree)) barcode;
  {
    nb::gil_scoped_release release;
    barcode = self.mod.get_barcode_from_line(line, degree);
  }
  return barcode_to_python<T>(barcode);
}

template <typename T>
PyModule<T>& evaluate_in_grid_impl(PyModule<T>& self, const std::vector<std::vector<T>>& grid) {
  {
    nb::gil_scoped_release release;
    self.mod.evaluate_in_grid(grid);
  }
  return self;
}

template <typename T>
nb::object compute_landscapes_box_impl(PyModule<T>& self,
                                       int degree,
                                       const std::vector<unsigned int>& ks,
                                       const Gudhi::multi_persistence::Box<T>& box,
                                       const std::vector<unsigned int>& resolution,
                                       int n_jobs) {
  decltype(Gudhi::multi_persistence::compute_set_of_module_landscapes(
      self.mod, degree, ks, box, resolution, n_jobs)) out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_set_of_module_landscapes(self.mod, degree, ks, box, resolution, n_jobs);
  }
  return nb::cast(out);
}

template <typename T>
nb::object compute_landscapes_grid_impl(PyModule<T>& self,
                                        int degree,
                                        const std::vector<unsigned int>& ks,
                                        const std::vector<std::vector<T>>& grid,
                                        int n_jobs) {
  decltype(Gudhi::multi_persistence::compute_set_of_module_landscapes(self.mod, degree, ks, grid, n_jobs)) out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_set_of_module_landscapes(self.mod, degree, ks, grid, n_jobs);
  }
  return nb::cast(out);
}

template <typename T>
nb::object compute_pixels_impl(PyModule<T>& self,
                               const std::vector<std::vector<T>>& coordinates,
                               const std::vector<int>& degrees,
                               const Gudhi::multi_persistence::Box<T>& box,
                               T delta,
                               T p,
                               bool normalize,
                               int n_jobs) {
  std::vector<std::vector<T>> out;
  {
    nb::gil_scoped_release release;
    out = Gudhi::multi_persistence::compute_module_pixels(
        self.mod, coordinates, degrees, box, delta, p, normalize, n_jobs);
  }
  return nb::cast(out);
}

template <typename T>
nb::object distance_to_impl(PyModule<T>& self,
                            const std::vector<std::vector<T>>& pts,
                            bool signed_distance,
                            int n_jobs) {
  std::vector<T> out(pts.size() * self.mod.size());
  {
    nb::gil_scoped_release release;
    Gudhi::multi_persistence::compute_module_distances_to(self.mod, out.data(), pts, signed_distance, n_jobs);
  }
  return nb::cast(owned_array<T>(std::move(out), {pts.size(), self.mod.size()}));
}

template <typename T>
PyModule<T>& set_box_impl(PyModule<T>& self, const Gudhi::multi_persistence::Box<T>& box) {
  {
    nb::gil_scoped_release release;
    self.mod.set_box(box);
  }
  return self;
}

template <typename T>
PyModule<T>& rescale_impl(PyModule<T>& self, const std::vector<T>& factors, int degree) {
  {
    nb::gil_scoped_release release;
    self.mod.rescale(factors, degree);
  }
  return self;
}

template <typename T>
PyModule<T>& translate_impl(PyModule<T>& self, const std::vector<T>& factors, int degree) {
  {
    nb::gil_scoped_release release;
    self.mod.translate(factors, degree);
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
    auto births = summand.compute_birth_list();
    auto deaths = summand.compute_death_list();
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
std::vector<T> module_dimensions(const Gudhi::multi_persistence::Module<T>& module) {
  std::vector<T> dims;
  dims.reserve(module.size());
  for (size_t i = 0; i < module.size(); ++i) dims.push_back((T)module.get_summand((unsigned int)i).get_dimension());
  return dims;
}

template <typename T>
struct PySummand {
  Gudhi::multi_persistence::Summand<T> sum;
};

template <typename T>
struct PyBox {
  Gudhi::multi_persistence::Box<T> box;
};

template <typename T>
struct PyModuleIterator {
  PyModule<T>* module = nullptr;
  size_t index = 0;
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
    using WrapperMod = PyModule<T>;

    cls.def(
           "_get_barcode_from_line",
           [](WrapperMod& self, nb::handle basepoint_handle, nb::handle direction_handle, int degree) -> nb::tuple {
             auto basepoint = vector_from_handle<T>(basepoint_handle);
             auto direction = direction_handle.is_none()
                                  ? std::optional<std::vector<T>>{}
                                  : std::optional<std::vector<T>>(vector_from_handle<T>(direction_handle));
             return barcode_from_line_impl<T>(self, basepoint, direction ? &*direction : nullptr, degree);
           },
           "basepoint"_a,
           "direction"_a = nb::none(),
           "degree"_a = -1)
        .def(
            "_get_barcode_from_line",
            [](WrapperMod& self,
               nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> basepoint,
               nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> direction,
               int degree) -> nb::tuple {
              auto basepoint_values = vector_from_array(basepoint);
              auto direction_values = vector_from_array(direction);
              return barcode_from_line_impl<T>(self, basepoint_values, &direction_values, degree);
            },
            "basepoint"_a,
            "direction"_a,
            "degree"_a = -1)
        .def(
            "evaluate_in_grid",
            [](WrapperMod& self, nb::handle grid_handle) -> WrapperMod& {
              return evaluate_in_grid_impl<T>(self, matrix_from_handle<T>(grid_handle));
            },
            nb::rv_policy::reference_internal)
        .def(
            "evaluate_in_grid",
            [](WrapperMod& self, nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> grid) -> WrapperMod& {
              return evaluate_in_grid_impl<T>(self, matrix_from_array(grid));
            },
            nb::rv_policy::reference_internal)
        .def(
            "_compute_landscapes_box",
            [](WrapperMod& self,
               int degree,
               nb::handle ks_handle,
               nb::handle box_handle,
               nb::handle resolution_handle,
               int n_jobs) {
              auto ks_in = vector_from_handle<int>(ks_handle);
              std::vector<unsigned int> ks;
              ks.reserve(ks_in.size());
              for (int k : ks_in) ks.push_back((unsigned int)k);
              auto resolution_in = vector_from_handle<int>(resolution_handle);
              std::vector<unsigned int> resolution;
              resolution.reserve(resolution_in.size());
              for (int r : resolution_in) resolution.push_back((unsigned int)r);
              return compute_landscapes_box_impl<T>(
                  self, degree, ks, box_from_handle<T>(box_handle), resolution, n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "box"_a,
            "resolution"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_box",
            [](WrapperMod& self,
               int degree,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> resolution,
               int n_jobs) {
              return compute_landscapes_box_impl<T>(self,
                                                    degree,
                                                    cast_vector_from_array<unsigned int>(ks),
                                                    box_from_rows<T>(matrix_from_array(box)),
                                                    cast_vector_from_array<unsigned int>(resolution),
                                                    n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "box"_a,
            "resolution"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_box",
            [](WrapperMod& self,
               int degree,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> resolution,
               int n_jobs) {
              return compute_landscapes_box_impl<T>(self,
                                                    degree,
                                                    cast_vector_from_array<unsigned int>(ks),
                                                    box_from_rows<T>(matrix_from_array(box)),
                                                    cast_vector_from_array<unsigned int>(resolution),
                                                    n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "box"_a,
            "resolution"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_grid",
            [](WrapperMod& self, int degree, nb::handle ks_handle, nb::handle grid_handle, int n_jobs) {
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
            [](WrapperMod& self,
               int degree,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> grid,
               int n_jobs) {
              return compute_landscapes_grid_impl<T>(
                  self, degree, cast_vector_from_array<unsigned int>(ks), matrix_from_array(grid), n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "grid"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_landscapes_grid",
            [](WrapperMod& self,
               int degree,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> ks,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> grid,
               int n_jobs) {
              return compute_landscapes_grid_impl<T>(
                  self, degree, cast_vector_from_array<unsigned int>(ks), matrix_from_array(grid), n_jobs);
            },
            "degree"_a,
            "ks"_a,
            "grid"_a,
            "n_jobs"_a = 0)
        .def(
            "_compute_pixels",
            [](WrapperMod& self,
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
                                            box_from_handle<T>(box_handle),
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
            [](WrapperMod& self,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> coordinates,
               nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> degrees,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               T delta,
               T p,
               bool normalize,
               int n_jobs) {
              return compute_pixels_impl<T>(self,
                                            matrix_from_array(coordinates),
                                            cast_vector_from_array<int>(degrees),
                                            box_from_rows<T>(matrix_from_array(box)),
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
            [](WrapperMod& self,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> coordinates,
               nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> degrees,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box,
               T delta,
               T p,
               bool normalize,
               int n_jobs) {
              return compute_pixels_impl<T>(self,
                                            matrix_from_array(coordinates),
                                            cast_vector_from_array<int>(degrees),
                                            box_from_rows<T>(matrix_from_array(box)),
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
            [](WrapperMod& self, nb::handle pts_handle, bool signed_distance, int n_jobs) {
              return distance_to_impl<T>(self, matrix_from_handle<T>(pts_handle), signed_distance, n_jobs);
            },
            "pts"_a,
            "signed"_a = false,
            "n_jobs"_a = 0)
        .def(
            "distance_to",
            [](WrapperMod& self,
               nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> pts,
               bool signed_distance,
               int n_jobs) { return distance_to_impl<T>(self, matrix_from_array(pts), signed_distance, n_jobs); },
            "pts"_a,
            "signed"_a = false,
            "n_jobs"_a = 0)
        .def(
            "get_interleavings",
            [](WrapperMod& self, nb::handle box_handle) {
              Box box;
              if (box_handle.is_none()) {
                box = self.mod.get_box();
              } else {
                box = box_from_handle<T>(box_handle);
              }
              std::vector<T> interleavings;
              {
                nb::gil_scoped_release release;
                interleavings = Gudhi::multi_persistence::compute_module_interleavings(self.mod, box);
              }
              return nb::cast(owned_array<T>(std::move(interleavings), {interleavings.size()}));
            },
            "box"_a = nb::none())
        .def(
            "get_interleavings",
            [](WrapperMod& self, nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box) {
              std::vector<T> interleavings;
              {
                nb::gil_scoped_release release;
                interleavings = Gudhi::multi_persistence::compute_module_interleavings(
                    self.mod, box_from_rows<T>(matrix_from_array(box)));
              }
              return nb::cast(owned_array<T>(std::move(interleavings), {interleavings.size()}));
            },
            "box"_a);
  }
}

template <typename Desc>
void bind_mma_type(nb::module_& m) {
  using T = typename Desc::value_type;
  using Summand = Gudhi::multi_persistence::Summand<T>;
  using Module = Gudhi::multi_persistence::Module<T>;
  using Box = Gudhi::multi_persistence::Box<T>;
  using WrapperSum = PySummand<T>;
  using WrapperBox = PyBox<T>;
  using WrapperMod = PyModule<T>;
  using WrapperIter = PyModuleIterator<T>;

  nb::class_<WrapperSum>(m, Desc::summand_name.data())
      .def(nb::init<>())
      .def("get_birth_list",
           [](WrapperSum& self) -> nb::object {
             std::vector<T> births;
             const size_t num_parameters = static_cast<size_t>(self.sum.get_number_of_parameters());
             const size_t num_birth_corners = static_cast<size_t>(self.sum.get_number_of_birth_corners());
             {
               nb::gil_scoped_release release;
               births = self.sum.compute_birth_list();
             }
             return corner_matrix_to_python<T>(std::move(births), num_birth_corners, num_parameters);
           })
      .def("get_death_list",
           [](WrapperSum& self) -> nb::object {
             std::vector<T> deaths;
             const size_t num_parameters = static_cast<size_t>(self.sum.get_number_of_parameters());
             const size_t num_death_corners = static_cast<size_t>(self.sum.get_number_of_death_corners());
             {
               nb::gil_scoped_release release;
               deaths = self.sum.compute_death_list();
             }
             return corner_matrix_to_python<T>(std::move(deaths), num_death_corners, num_parameters);
           })
      .def_prop_ro("degree", [](const WrapperSum& self) -> int { return self.sum.get_dimension(); })
      .def("get_bounds",
           [](WrapperSum& self) -> nb::tuple {
             std::pair<std::vector<T>, std::vector<T>> cbounds;
             {
               nb::gil_scoped_release release;
               auto cpp_bounds = self.sum.compute_bounds().get_bounding_corners();
               cbounds.first.assign(cpp_bounds.first.begin(), cpp_bounds.first.end());
               cbounds.second.assign(cpp_bounds.second.begin(), cpp_bounds.second.end());
             }
             return nb::make_tuple(nb::cast(owned_array<T>(std::vector<T>(cbounds.first.begin(), cbounds.first.end()),
                                                           {cbounds.first.size()})),
                                   nb::cast(owned_array<T>(std::vector<T>(cbounds.second.begin(), cbounds.second.end()),
                                                           {cbounds.second.size()})));
           })
      .def("num_parameters", [](WrapperSum& self) -> int { return self.sum.get_number_of_parameters(); })
      .def_prop_ro("_template_id", [](const WrapperSum&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const WrapperSum&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def("__eq__", [](WrapperSum& self, WrapperSum& other) { return self.sum == other.sum; });

  nb::class_<WrapperBox>(m, Desc::box_name.data())
      .def(nb::new_([](nb::handle bottom, nb::handle top) {
             WrapperBox out;
             auto lower = vector_from_handle<T>(bottom);
             auto upper = vector_from_handle<T>(top);
             out.box = Box(lower, upper);
             return out;
           }),
           "bottomCorner"_a,
           "topCorner"_a)
      .def(nb::new_([](nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> bottom,
                       nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> top) {
             WrapperBox out;
             out.box = Box(vector_from_array(bottom), vector_from_array(top));
             return out;
           }),
           "bottomCorner"_a,
           "topCorner"_a)
      .def_prop_ro("num_parameters", [](const WrapperBox& self) -> int { return self.box.get_lower_corner().size(); })
      .def("contains",
           [](WrapperBox& self, nb::handle x) {
             auto values = vector_from_handle<T>(x);
             return self.box.contains(values);
           })
      .def("contains",
           [](WrapperBox& self, nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> x) {
             return self.box.contains(vector_from_array(x));
           })
      .def("get",
           [](WrapperBox& self) -> nb::tuple {
             auto lower = std::vector<T>(self.box.get_lower_corner().begin(), self.box.get_lower_corner().end());
             auto upper = std::vector<T>(self.box.get_upper_corner().begin(), self.box.get_upper_corner().end());
             return nb::make_tuple(nb::cast(owned_array<T>(std::move(lower), {self.box.get_lower_corner().size()})),
                                   nb::cast(owned_array<T>(std::move(upper), {self.box.get_upper_corner().size()})));
           })
      .def("to_multipers",
           [](WrapperBox& self) -> nb::object {
             auto lower = std::vector<T>(self.box.get_lower_corner().begin(), self.box.get_lower_corner().end());
             auto upper = std::vector<T>(self.box.get_upper_corner().begin(), self.box.get_upper_corner().end());
             std::vector<T> flat;
             flat.reserve(lower.size() * 2);
             for (size_t i = 0; i < lower.size(); ++i) {
               flat.push_back(lower[i]);
               flat.push_back(upper[i]);
             }
             return nb::cast(owned_array<T>(std::move(flat), {size_t(2), lower.size()}));
           })
      .def_prop_ro("_template_id", [](const WrapperBox&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const WrapperBox&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); });

  std::string iterator_name = std::string("_PyModuleIterator_") + std::string(Desc::short_name);
  nb::class_<WrapperIter>(m, iterator_name.c_str())
      .def(
          "__iter__", [](WrapperIter& self) -> WrapperIter& { return self; }, nb::rv_policy::reference_internal)
      .def("__next__", [](WrapperIter& self) {
        if (self.module == nullptr || self.index >= self.module->mod.size()) {
          throw nb::stop_iteration();
        }
        WrapperSum out;
        out.sum = self.module->mod.get_summand((unsigned int)self.index++);
        return out;
      });

  auto module_cls =
      nb::class_<WrapperMod>(m, Desc::module_name.data())
          .def(nb::init<>())
          .def(
              "_from_ptr",
              [](WrapperMod& self, intptr_t ptr) -> WrapperMod& {
                auto* other = reinterpret_cast<Module*>(ptr);
                self.mod = std::move(*other);
                delete other;
                return self;
              },
              nb::rv_policy::reference_internal)
          .def("__len__", [](WrapperMod& self) -> int { return self.mod.size(); })
          .def("__eq__", [](WrapperMod& self, WrapperMod& other) { return self.mod == other.mod; })
          .def_prop_ro("_template_id", [](const WrapperMod&) -> int { return Desc::template_id; })
          .def("__getitem__",
               [](WrapperMod& self, nb::object key) -> nb::object {
                 if (PySlice_Check(key.ptr())) {
                   if (key.attr("start").is_none() && key.attr("stop").is_none() && key.attr("step").is_none()) {
                     return self_handle(self);
                   }
                   throw nb::index_error("Only [:] slices are supported.");
                 }
                 int index = nb::cast<int>(key);
                 size_t size = self.mod.size();
                 if (size == 0) {
                   throw nb::index_error("Module is empty.");
                 }
                 if (index < 0) {
                   index += (int)size;
                 }
                 if (index < 0 || index >= (int)size) {
                   throw nb::index_error("Summand index out of range.");
                 }
                 WrapperSum out;
                 out.sum = self.mod.get_summand((unsigned int)index);
                 return nb::cast(out);
               })
          .def(
              "__iter__",
              [](WrapperMod& self) {
                WrapperIter out;
                out.module = &self;
                out.index = 0;
                return out;
              },
              nb::keep_alive<0, 1>())
          .def(
              "merge",
              [](WrapperMod& self, WrapperMod& other, int dim) -> WrapperMod& {
                Module c_other = other.mod;
                {
                  nb::gil_scoped_release release;
                  for (auto summand : c_other) self.mod.add_summand(summand, dim);
                }
                return self;
              },
              "other"_a,
              "dim"_a = -1,
              nb::rv_policy::reference_internal)
          .def("permute_summands",
               [](WrapperMod& self, nb::handle permutation) {
                 WrapperMod out;
                 auto c_permutation = vector_from_handle<int>(permutation);
                 {
                   nb::gil_scoped_release release;
                   out.mod = Gudhi::multi_persistence::build_permuted_module(self.mod, c_permutation);
                 }
                 return out;
               })
          .def(
              "set_box",
              [](WrapperMod& self, nb::handle box_handle) -> WrapperMod& {
                return set_box_impl<T>(self, box_from_handle<T>(box_handle));
              },
              nb::rv_policy::reference_internal)
          .def(
              "set_box",
              [](WrapperMod& self, nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig> box) -> WrapperMod& {
                return set_box_impl<T>(self, box_from_rows<T>(matrix_from_array(box)));
              },
              nb::rv_policy::reference_internal)
          .def("get_module_of_degree",
               [](WrapperMod& self, int degree) {
                 WrapperMod out;
                 {
                   nb::gil_scoped_release release;
                   out.mod.set_box(self.mod.get_box());
                   for (auto summand : self.mod)
                     if (summand.get_dimension() == degree) out.mod.add_summand(summand);
                 }
                 return out;
               })
          .def("get_module_of_degrees",
               [](WrapperMod& self, nb::handle degrees_handle) {
                 auto degrees = vector_from_handle<int>(degrees_handle);
                 WrapperMod out;
                 {
                   nb::gil_scoped_release release;
                   out.mod.set_box(self.mod.get_box());
                   for (auto summand : self.mod) {
                     for (int degree : degrees) {
                       if (degree == summand.get_dimension()) {
                         out.mod.add_summand(summand);
                         break;
                       }
                     }
                   }
                 }
                 return out;
               })
          .def("_get_dump", [](WrapperMod& self) -> nb::tuple { return dump_module<T>(self.mod); })
          .def(
              "dump",
              [](WrapperMod& self, nb::object path) -> nb::tuple {
                nb::tuple dump = dump_module<T>(self.mod);
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
              [](WrapperMod& self, nb::handle dump) -> WrapperMod& {
                self.mod = module_from_dump<T>(dump);
                return self;
              },
              nb::rv_policy::reference_internal)
          .def("__getstate__", [](WrapperMod& self) -> nb::tuple { return dump_module<T>(self.mod); })
          .def("__setstate__",
               [](WrapperMod& self, nb::handle state) { new (&self) WrapperMod{module_from_dump<T>(state)}; })
          .def(
              "_add_mmas",
              [](WrapperMod& self, nb::iterable mmas) -> WrapperMod& {
                for (nb::handle item : mmas) {
                  Module c_other = nb::cast<WrapperMod&>(item).mod;
                  {
                    nb::gil_scoped_release release;
                    for (auto summand : c_other) self.mod.add_summand(summand);
                  }
                }
                return self;
              },
              nb::rv_policy::reference_internal)
          .def("_get_summand",
               [](WrapperMod& self, int index) {
                 WrapperSum out;
                 size_t size = self.mod.size();
                 out.sum = self.mod.get_summand((unsigned int)(index % (int)size));
                 return out;
               })
          .def("get_bottom",
               [](WrapperMod& self) -> nb::object {
                 return nb::cast(owned_array<T>(std::vector<T>(self.mod.get_box().get_lower_corner().begin(),
                                                               self.mod.get_box().get_lower_corner().end()),
                                                {self.mod.get_box().get_lower_corner().size()}));
               })
          .def("get_top",
               [](WrapperMod& self) -> nb::object {
                 return nb::cast(owned_array<T>(std::vector<T>(self.mod.get_box().get_upper_corner().begin(),
                                                               self.mod.get_box().get_upper_corner().end()),
                                                {self.mod.get_box().get_upper_corner().size()}));
               })
          .def("get_box",
               [](WrapperMod& self) -> nb::object {
                 auto lower = std::vector<T>(self.mod.get_box().get_lower_corner().begin(),
                                             self.mod.get_box().get_lower_corner().end());
                 auto upper = std::vector<T>(self.mod.get_box().get_upper_corner().begin(),
                                             self.mod.get_box().get_upper_corner().end());
                 std::vector<T> flat_box;
                 flat_box.reserve(lower.size() + upper.size());
                 flat_box.insert(flat_box.end(), lower.begin(), lower.end());
                 flat_box.insert(flat_box.end(), upper.begin(), upper.end());
                 return nb::cast(
                     owned_array<T>(std::move(flat_box), {size_t(2), self.mod.get_box().get_lower_corner().size()}));
               })
          .def_prop_ro("max_degree", [](const WrapperMod& self) -> int { return self.mod.get_max_dimension(); })
          .def_prop_ro("num_parameters",
                       [](const WrapperMod& self) -> int { return self.mod.get_box().get_lower_corner().size(); })
          .def("get_bounds",
               [](WrapperMod& self) -> nb::tuple {
                 std::pair<std::vector<T>, std::vector<T>> cbounds;
                 {
                   nb::gil_scoped_release release;
                   auto cpp_bounds = self.mod.compute_bounds().get_bounding_corners();
                   cbounds.first.assign(cpp_bounds.first.begin(), cpp_bounds.first.end());
                   cbounds.second.assign(cpp_bounds.second.begin(), cpp_bounds.second.end());
                 }
                 return nb::make_tuple(
                     nb::cast(owned_array<T>(std::vector<T>(cbounds.first.begin(), cbounds.first.end()),
                                             {cbounds.first.size()})),
                     nb::cast(owned_array<T>(std::vector<T>(cbounds.second.begin(), cbounds.second.end()),
                                             {cbounds.second.size()})));
               })
          .def(
              "rescale",
              [](WrapperMod& self, nb::handle factors, int degree) -> WrapperMod& {
                return rescale_impl<T>(self, vector_from_handle<T>(factors), degree);
              },
              "rescale_factors"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "rescale",
              [](WrapperMod& self,
                 nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> factors,
                 int degree) -> WrapperMod& { return rescale_impl<T>(self, vector_from_array(factors), degree); },
              "rescale_factors"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "translate",
              [](WrapperMod& self, nb::handle factors, int degree) -> WrapperMod& {
                return translate_impl<T>(self, vector_from_handle<T>(factors), degree);
              },
              "translation"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "translate",
              [](WrapperMod& self,
                 nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> factors,
                 int degree) -> WrapperMod& { return translate_impl<T>(self, vector_from_array(factors), degree); },
              "translation"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "get_filtration_values",
              [](WrapperMod& self, bool unique) -> std::vector<std::vector<T>> {
                return filtration_values_from_module<T>(self.mod, unique);
              },
              "unique"_a = true)
          .def("get_dimensions", [](WrapperMod& self) -> nb::object {
            std::vector<int32_t> dims;
            dims.reserve(self.mod.size());
            for (size_t i = 0; i < self.mod.size(); ++i)
              dims.push_back((int32_t)self.mod.get_summand((unsigned int)i).get_dimension());
            return nb::cast(owned_array<int32_t>(std::move(dims), {self.mod.size()}));
          });

  bind_float_module_methods<Desc>(module_cls);

  module_cls.def_prop_ro("dtype", [](const WrapperMod&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); });

  std::string from_dump_name = std::string(Desc::from_dump_name);
  m.def(
      from_dump_name.c_str(),
      [](nb::handle dump) {
        WrapperMod out;
        out.mod = module_from_dump<T>(dump);
        return out;
      },
      "dump"_a);
}

template <typename... Desc>
void bind_all_mma(type_list<Desc...>, nb::module_& m) {
  (bind_mma_type<Desc>(m), ...);
}

template <typename Desc>
bool is_mma_desc(nb::handle stuff) {
  using WrapperMod = module_wrapper_t<Desc>;
  return nb::isinstance<WrapperMod>(stuff);
}

inline bool is_mma(nb::handle stuff) { return is_mma_module_object(stuff); }

}  // namespace mpmma

NB_MODULE(_mma_nanobind, m) {
  m.doc() = "nanobind MMA bindings";
  mpmma::bind_all_mma(mpmma::MMADescriptorList{}, m);
  m.def("is_mma", [](nb::handle stuff) { return mpmma::is_mma(stuff); });
}
