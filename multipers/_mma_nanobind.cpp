#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include "Persistence_slices_interface.h"
#include "gudhi/Multi_persistence/Box.h"
#include "gudhi/Multi_persistence/Line.h"
#include "gudhi/Multi_persistence/Module.h"
#include "gudhi/Multi_persistence/Summand.h"
#include "gudhi/Multi_persistence/module_helpers.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpmma {

template <typename... Types>
struct type_list {};

#include "_mma_nanobind_registry.inc"

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
std::vector<T> vector_from_handle(nb::handle h) {
  nb::object np = nb::module_::import_("numpy");
  nb::object obj = nb::borrow(h);
  if (nb::hasattr(obj, "shape")) {
    obj = np.attr("asarray")(obj).attr("reshape")(-1);
  }
  return nb::cast<std::vector<T>>(obj);
}

template <typename T>
std::vector<std::vector<T>> matrix_from_handle(nb::handle h) {
  nb::object np = nb::module_::import_("numpy");
  nb::object obj = nb::borrow(h);
  if (nb::hasattr(obj, "shape")) {
    obj = np.attr("asarray")(obj);
  }
  return nb::cast<std::vector<std::vector<T>>>(obj);
}

template <typename T>
nb::object one_filtration_to_python(const multipers::tmp_interface::One_critical_filtration<T>& f) {
  size_t p = f.num_parameters();
  if (!f.is_finite()) {
    return nb::cast(owned_array<T>(std::vector<T>(p, f(0, 0)), {p}));
  }
  std::vector<T> out(p);
  for (size_t i = 0; i < p; ++i) {
    out[i] = f(0, i);
  }
  return nb::cast(owned_array<T>(std::move(out), {p}));
}

template <typename T>
nb::list filtration_list_to_python(const std::vector<multipers::tmp_interface::One_critical_filtration<T>>& values) {
  nb::list out;
  for (const auto& value : values) {
    out.append(one_filtration_to_python(value));
  }
  return out;
}

template <typename T>
nb::tuple dump_summand(const Gudhi::multi_persistence::Summand<T>& summand) {
  auto births = filtration_list_to_python<T>(summand.compute_birth_list());
  auto deaths = filtration_list_to_python<T>(summand.compute_death_list());
  nb::object births_arr = nb::module_::import_("numpy").attr("array")(births);
  nb::object deaths_arr = nb::module_::import_("numpy").attr("array")(deaths);
  return nb::make_tuple(births_arr, deaths_arr, summand.get_dimension());
}

template <typename T>
nb::tuple barcode_to_python(const std::vector<std::vector<std::pair<T, T>>>& barcode) {
  nb::tuple out = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)barcode.size()));
  for (size_t dim = 0; dim < barcode.size(); ++dim) {
    const auto& bars = barcode[dim];
    std::vector<T> flat;
    flat.reserve(bars.size() * 2);
    for (const auto& bar : bars) {
      flat.push_back(bar.first);
      flat.push_back(bar.second);
    }
    nb::object value = nb::cast(owned_array<T>(std::move(flat), {bars.size(), size_t(2)}));
    PyTuple_SET_ITEM(out.ptr(), (Py_ssize_t)dim, value.release().ptr());
  }
  return out;
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
  nb::object box_arr = nb::module_::import_("numpy").attr("array")(
      nb::make_tuple(nb::cast(owned_array<T>(std::move(lower), {box.get_lower_corner().size()})),
                     nb::cast(owned_array<T>(std::move(upper), {box.get_upper_corner().size()}))));

  nb::tuple summands = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)module.size()));
  for (size_t i = 0; i < module.size(); ++i) {
    nb::object value = dump_summand<T>(module.get_summand((unsigned int)i));
    PyTuple_SET_ITEM(summands.ptr(), (Py_ssize_t)i, value.release().ptr());
  }
  return nb::make_tuple(box_arr, summands);
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
    for (size_t p = 0; p < num_parameters; ++p) {
      for (const auto& fil : births) {
        if (fil.is_finite()) values[p].push_back(fil(0, p));
      }
      for (const auto& fil : deaths) {
        if (fil.is_finite()) values[p].push_back(fil(0, p));
      }
    }
  }
  if (unique) {
    for (auto& vals : values) {
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
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
struct PyModule {
  Gudhi::multi_persistence::Module<T> mod;
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

inline nb::object numpy_dtype_type(std::string_view name) {
  nb::object np = nb::module_::import_("numpy");
  return np.attr("dtype")(std::string(name)).attr("type");
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
             Gudhi::multi_persistence::Line<T> line =
                 direction_handle.is_none()
                     ? Gudhi::multi_persistence::Line<T>(basepoint)
                     : Gudhi::multi_persistence::Line<T>(basepoint, vector_from_handle<T>(direction_handle));
             decltype(self.mod.get_barcode_from_line(line, degree)) barcode;
             {
               nb::gil_scoped_release release;
               barcode = self.mod.get_barcode_from_line(line, degree);
             }
             return barcode_to_python<T>(barcode);
           },
           "basepoint"_a,
           "direction"_a = nb::none(),
           "degree"_a = -1)
        .def(
            "evaluate_in_grid",
            [](WrapperMod& self, nb::handle grid_handle) -> WrapperMod& {
              auto grid = matrix_from_handle<T>(grid_handle);
              {
                nb::gil_scoped_release release;
                self.mod.evaluate_in_grid(grid);
              }
              return self;
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
              auto box_values = matrix_from_handle<T>(box_handle);
              auto resolution_in = vector_from_handle<int>(resolution_handle);
              std::vector<unsigned int> resolution;
              resolution.reserve(resolution_in.size());
              for (int r : resolution_in) resolution.push_back((unsigned int)r);
              decltype(Gudhi::multi_persistence::compute_set_of_module_landscapes(
                  self.mod, degree, ks, Box(box_values[0], box_values[1]), resolution, n_jobs)) out;
              {
                nb::gil_scoped_release release;
                out = Gudhi::multi_persistence::compute_set_of_module_landscapes(
                    self.mod, degree, ks, Box(box_values[0], box_values[1]), resolution, n_jobs);
              }
              return nb::cast(out);
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
              auto grid = matrix_from_handle<T>(grid_handle);
              decltype(Gudhi::multi_persistence::compute_set_of_module_landscapes(
                  self.mod, degree, ks, grid, n_jobs)) out;
              {
                nb::gil_scoped_release release;
                out = Gudhi::multi_persistence::compute_set_of_module_landscapes(self.mod, degree, ks, grid, n_jobs);
              }
              return nb::cast(out);
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
              auto coordinates = matrix_from_handle<T>(coordinates_handle);
              auto degrees = vector_from_handle<int>(degrees_handle);
              auto box_values = matrix_from_handle<T>(box_handle);
              std::vector<std::vector<T>> out;
              {
                nb::gil_scoped_release release;
                out = Gudhi::multi_persistence::compute_module_pixels(
                    self.mod, coordinates, degrees, Box(box_values[0], box_values[1]), delta, p, normalize, n_jobs);
              }
              return nb::cast(out);
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
              auto pts = matrix_from_handle<T>(pts_handle);
              std::vector<T> out(pts.size() * self.mod.size());
              {
                nb::gil_scoped_release release;
                Gudhi::multi_persistence::compute_module_distances_to(
                    self.mod, out.data(), pts, signed_distance, n_jobs);
              }
              return nb::cast(owned_array<T>(std::move(out), {pts.size(), self.mod.size()}));
            },
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
                auto box_values = matrix_from_handle<T>(box_handle);
                box = Box(box_values[0], box_values[1]);
              }
              std::vector<T> interleavings;
              {
                nb::gil_scoped_release release;
                interleavings = Gudhi::multi_persistence::compute_module_interleavings(self.mod, box);
              }
              return nb::cast(owned_array<T>(std::move(interleavings), {interleavings.size()}));
            },
            "box"_a = nb::none());
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
           [](WrapperSum& self) -> nb::list {
             decltype(self.sum.compute_birth_list()) births;
             {
               nb::gil_scoped_release release;
               births = self.sum.compute_birth_list();
             }
             return filtration_list_to_python<T>(births);
           })
      .def("get_death_list",
           [](WrapperSum& self) -> nb::list {
             decltype(self.sum.compute_death_list()) deaths;
             {
               nb::gil_scoped_release release;
               deaths = self.sum.compute_death_list();
             }
             return filtration_list_to_python<T>(deaths);
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
      .def("num_parameters",
           [](WrapperSum& self) -> int {
             auto births = self.sum.compute_birth_list();
             if (!births.empty() && births[0].is_finite()) return births[0].num_parameters();
             auto deaths = self.sum.compute_death_list();
             return deaths.empty() ? 0 : deaths[0].num_parameters();
           })
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
      .def_prop_ro("num_parameters", [](const WrapperBox& self) -> int { return self.box.get_lower_corner().size(); })
      .def("contains",
           [](WrapperBox& self, nb::handle x) {
             auto values = vector_from_handle<T>(x);
             return self.box.contains(values);
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
                auto box = matrix_from_handle<T>(box_handle);
                {
                  nb::gil_scoped_release release;
                  self.mod.set_box(Box(box[0], box[1]));
                }
                return self;
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
          .def("__reduce__",
               [](WrapperMod& self) -> nb::tuple {
                 return nb::make_tuple(
                     nb::module_::import_("multipers.multiparameter_module_approximation").attr("_reconstruct_module"),
                     nb::make_tuple(nb::str(Desc::from_dump_name.data()), dump_module<T>(self.mod)));
               })
          .def("__reduce_ex__",
               [](WrapperMod& self, int) -> nb::tuple {
                 return nb::make_tuple(
                     nb::module_::import_("multipers.multiparameter_module_approximation").attr("_reconstruct_module"),
                     nb::make_tuple(nb::str(Desc::from_dump_name.data()), dump_module<T>(self.mod)));
               })
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
                 return nb::module_::import_("numpy").attr("array")(nb::make_tuple(
                     nb::cast(owned_array<T>(std::move(lower), {self.mod.get_box().get_lower_corner().size()})),
                     nb::cast(owned_array<T>(std::move(upper), {self.mod.get_box().get_upper_corner().size()}))));
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
                auto c_factors = vector_from_handle<T>(factors);
                {
                  nb::gil_scoped_release release;
                  self.mod.rescale(c_factors, degree);
                }
                return self;
              },
              "rescale_factors"_a,
              "degree"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "translate",
              [](WrapperMod& self, nb::handle factors, int degree) -> WrapperMod& {
                auto c_factors = vector_from_handle<T>(factors);
                {
                  nb::gil_scoped_release release;
                  self.mod.translate(c_factors, degree);
                }
                return self;
              },
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

}  // namespace mpmma

NB_MODULE(_mma_nanobind, m) {
  m.doc() = "nanobind MMA bindings";
  mpmma::bind_all_mma(mpmma::MMADescriptorList{}, m);
}
