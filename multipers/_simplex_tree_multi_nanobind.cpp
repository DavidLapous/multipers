#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ext_interface/nanobind_registry_helpers.hpp"
#include "simplextree_conversion_core.hpp"
#include "nanobind_array_utils.hpp"
#include "nanobind_object_utils.hpp"
#include "nanobind_simplextree_utils.hpp"
#include "multi_parameter_rank_invariant/euler_characteristic.h"
#include "multi_parameter_rank_invariant/hilbert_function.h"
#include "multi_parameter_rank_invariant/rank_invariant.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpst {

using tensor_dtype = int32_t;
using indices_type = int32_t;
using signed_measure_type = std::pair<std::vector<std::vector<indices_type>>, std::vector<tensor_dtype>>;

using multipers::core::SimplexTreeConversion;
using multipers::nanobind_helpers::dispatch_simplextree_by_template_id;
using multipers::nanobind_helpers::dispatch_slicer_by_template_id;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::is_slicer_object;
using multipers::nanobind_helpers::PySimplexTree;
using multipers::nanobind_helpers::simplextree_wrapper_t;
using multipers::nanobind_helpers::SimplexTreeDescriptorList;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_helpers::visit_const_simplextree_wrapper;
using multipers::nanobind_helpers::visit_const_slicer_wrapper;
using multipers::nanobind_simplextree_utils::flat_simplex_batch;
using multipers::nanobind_simplextree_utils::kcritical_filtration_from_array;
using multipers::nanobind_simplextree_utils::kcritical_filtrations_from_array;
using multipers::nanobind_simplextree_utils::one_critical_filtration_from_array;
using multipers::nanobind_simplextree_utils::one_critical_filtrations_from_array;
using multipers::nanobind_simplextree_utils::one_critical_filtrations_from_rows;
using multipers::nanobind_simplextree_utils::simplex_from_array;
using multipers::nanobind_simplextree_utils::simplices_from_vertex_array;
using multipers::nanobind_simplextree_utils::simplices_from_vertex_rows;
using multipers::nanobind_utils::has_template_id;
using multipers::nanobind_utils::lowercase_copy;
using multipers::nanobind_utils::matrix_from_handle;
using multipers::nanobind_utils::numpy_dtype_name;
using multipers::nanobind_utils::numpy_dtype_type;
using multipers::nanobind_utils::owned_array;
using multipers::nanobind_utils::template_id_of;
using multipers::nanobind_utils::vector_from_handle;
using multipers::nanobind_utils::view_array;

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

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical, bool SortRows, typename SimplexHandle>
nb::tuple simplex_entry_to_python(Wrapper& self, SimplexHandle sh);

template <typename... Ds>
nb::object get_simplextree_class(type_list<Ds...>,
                                 const nb::handle& dtype,
                                 bool kcritical,
                                 std::string filtration_container) {
  std::string dtype_name = numpy_dtype_name(dtype);
  filtration_container = lowercase_copy(std::move(filtration_container));
  std::string_view normalized_filtration_container = filtration_container;
  bool matched = false;
  nb::object result;
  (
      [&]<typename D>() {
        if (!matched && D::dtype_name == dtype_name && D::is_kcritical == kcritical &&
            D::filtration_container == normalized_filtration_container) {
          result = nb::borrow<nb::object>(nb::type<simplextree_wrapper_t<D>>());
          matched = true;
        }
      }.template operator()<Ds>(),
      ...);
  if (!matched) {
    throw nb::type_error("Unknown SimplexTreeMulti implementation.");
  }
  return result;
}

inline nb::object get_simplextree_class_from_template_id(int template_id) {
  return dispatch_simplextree_by_template_id(template_id, [&]<typename Desc>() -> nb::object {
    return nb::borrow<nb::object>(nb::type<simplextree_wrapper_t<Desc>>());
  });
}

inline bool is_simplextree_multi(nb::handle input) { return is_simplextree_object(input); }

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical, bool SortRows, typename Iterator>
class SimplexEntryIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = nb::tuple;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type;

  SimplexEntryIterator(Wrapper* owner, Iterator current, Iterator end) : owner_(owner), current_(current), end_(end) {}

  reference operator*() const {
    return simplex_entry_to_python<Wrapper, Filtration, T, IsKCritical, SortRows>(*owner_, *current_);
  }

  SimplexEntryIterator& operator++() {
    ++current_;
    return *this;
  }

  bool operator==(const SimplexEntryIterator& other) const { return current_ == other.current_; }

  bool operator!=(const SimplexEntryIterator& other) const { return !(*this == other); }

 private:
  Wrapper* owner_;
  Iterator current_;
  Iterator end_;
};

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical, bool SortRows, typename Iterator>
auto make_simplextree_python_iterator(Wrapper& self, const char* name, Iterator begin, Iterator end) {
  // The iterator borrows `self`; mutating the tree while iterating invalidates
  // the underlying native iterators.
  using PythonIterator = SimplexEntryIterator<Wrapper, Filtration, T, IsKCritical, SortRows, Iterator>;
  return nb::make_iterator(
      nb::type<Wrapper>(), name, PythonIterator(&self, begin, end), PythonIterator(nullptr, end, end));
}

template <typename Filtration, typename T, bool IsKCritical>
nb::object filtration_to_python(const Filtration& filtration, nb::handle owner = nb::handle());

template <typename Filtration, typename T, bool IsKCritical, bool SortRows>
nb::object normalized_filtration_to_python(const Filtration& filtration, nb::handle owner = nb::handle()) {
  if constexpr (!IsKCritical) {
    return filtration_to_python<Filtration, T, false>(filtration, owner);
  } else {
    std::vector<std::vector<T>> rows;
    rows.reserve(filtration.num_generators());
    for (size_t i = 0; i < filtration.num_generators(); ++i) {
      bool is_all_inf = true;
      std::vector<T> row(filtration.num_parameters());
      for (size_t j = 0; j < filtration.num_parameters(); ++j) {
        row[j] = filtration(i, j);
        is_all_inf = is_all_inf && std::isinf(static_cast<double>(row[j]));
      }
      if (!is_all_inf) {
        rows.push_back(std::move(row));
      }
    }
    if constexpr (SortRows) {
      std::sort(rows.begin(), rows.end());
    }
    nb::list out;
    for (auto& row : rows) {
      out.append(nb::cast(owned_array<T>(std::move(row), {static_cast<size_t>(filtration.num_parameters())})));
    }
    return out;
  }
}

template <typename TargetInterface, typename SourceSlicer>
void copy_simplicial_slicer_to_simplextree(TargetInterface& out, const SourceSlicer& slicer, int max_dim) {
  using TargetFiltration = typename TargetInterface::Filtration_value;
  using TargetVertex = typename TargetInterface::Vertex_handle;
  using TargetSimplex = typename TargetInterface::Simplex;
  using namespace Gudhi::multi_filtration;

  out.clear();
  out.set_num_parameters(slicer.get_number_of_parameters());

  const auto& dims = slicer.get_dimensions();
  const auto& boundaries = slicer.get_boundaries();
  const auto& filtrations = slicer.get_filtration_values();

  std::vector<std::vector<TargetVertex>> simplex_vertices(dims.size());
  std::vector<TargetSimplex> simplices;
  std::vector<TargetFiltration> converted_filtrations;
  simplices.reserve(dims.size());
  converted_filtrations.reserve(dims.size());

  int next_vertex = 0;
  int previous_dim = -1;
  for (size_t i = 0; i < dims.size(); ++i) {
    int dim = dims[i];
    if (dim < previous_dim) {
      throw std::invalid_argument("Dims is not sorted.");
    }
    previous_dim = dim;
    if (max_dim >= 0 && dim > max_dim) {
      break;
    }

    auto& vertices = simplex_vertices[i];
    if (dim == 0) {
      vertices.push_back(static_cast<TargetVertex>(next_vertex++));
    } else {
      for (auto face_idx : boundaries[i]) {
        if (static_cast<size_t>(face_idx) >= i) {
          throw std::invalid_argument("Invalid boundary in slicer.");
        }
        const auto& face_vertices = simplex_vertices[face_idx];
        vertices.insert(vertices.end(), face_vertices.begin(), face_vertices.end());
      }
      std::sort(vertices.begin(), vertices.end());
      vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
      if (vertices.size() != static_cast<size_t>(dim + 1)) {
        throw std::invalid_argument("Input slicer is not simplicial.");
      }
    }

    simplices.emplace_back(vertices.begin(), vertices.end());
    if constexpr (std::is_same_v<TargetFiltration, typename SourceSlicer::Filtration_value>) {
      converted_filtrations.push_back(filtrations[i]);
    } else {
      converted_filtrations.push_back(as_type<TargetFiltration>(filtrations[i]));
    }
    out.insert(simplices.back(), converted_filtrations.back());
  }

  for (size_t i = 0; i < simplices.size(); ++i) {
    out.assign_simplex_filtration(simplices[i], converted_filtrations[i]);
  }
}

template <typename Desc, typename Wrapper, typename Interface>
void build_from_slicer_desc(Wrapper& self, const typename Desc::wrapper& source, int max_dim) {
  {
    nb::gil_scoped_release release;
    copy_simplicial_slicer_to_simplextree<Interface>(self.tree, source.truc, max_dim);
  }
  self.filtration_grid = nb::list();
}

template <typename Wrapper, typename Interface>
bool try_build_from_slicer(Wrapper& self, nb::handle source, int max_dim) {
  if (!is_slicer_object(source)) {
    return false;
  }
  visit_const_slicer_wrapper(source, [&]<typename D>(const typename D::wrapper& wrapper) {
    build_from_slicer_desc<D, Wrapper, Interface>(self, wrapper, max_dim);
  });
  return true;
}

template <typename Filtration, typename T, bool IsKCritical>
Filtration filtration_from_handle(nb::handle filtration_handle, int num_parameters) {
  if (filtration_handle.is_none()) {
    return Filtration::minus_inf(num_parameters);
  }
  if constexpr (IsKCritical) {
    try {
      auto rows = matrix_from_handle<T>(filtration_handle);
      int p = rows.empty() ? num_parameters : static_cast<int>(rows[0].size());
      std::vector<T> flat;
      for (const auto& row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
      }
      return Filtration(flat.begin(), flat.end(), p);
    } catch (const std::exception&) {
      auto values = vector_from_handle<T>(filtration_handle);
      return Filtration(values.begin(), values.end(), num_parameters);
    }
  } else {
    auto values = vector_from_handle<T>(filtration_handle);
    return Filtration(values.begin(), values.end());
  }
}

template <typename Tree, typename Filtration>
Filtration first_possible_filtration(Tree& tree, const std::vector<int>& simplex) {
  auto simplex_handle = tree.find(simplex);
  Filtration out = Filtration::minus_inf(tree.num_parameters());
  if (simplex_handle == tree.null_simplex()) {
    return out;
  }
  for (auto boundary_handle : tree.boundary_simplex_range(simplex_handle)) {
    intersect_lifetimes(out, tree.filtration(boundary_handle));
  }
  return out;
}

template <typename Tree, typename Filtration>
bool insert_kcritical_simplex(Tree& tree, const std::vector<int>& simplex, const Filtration* filtration) {
  if (tree.find_simplex(simplex)) {
    auto updated = *tree.simplex_filtration(simplex);
    if (filtration != nullptr) {
      unify_lifetimes(updated, *filtration);
    } else {
      auto first_possible = first_possible_filtration<Tree, Filtration>(tree, simplex);
      unify_lifetimes(updated, first_possible);
    }
    tree.assign_simplex_filtration(simplex, updated);
    return true;
  }

  if (filtration != nullptr) {
    return tree.insert_force(simplex, *filtration);
  }

  using BaseTree = typename Tree::Base_tree;
  auto& base_tree = static_cast<BaseTree&>(tree);
  return base_tree
      .insert_simplex_and_subfaces(
          BaseTree::Filtration_maintenance::INCREASE_NEW, simplex, Filtration::minus_inf(tree.num_parameters()))
      .second;
}

template <typename Filtration, typename T>
Filtration default_filtration_from_handle(nb::handle filtration_handle, int num_parameters) {
  auto values = vector_from_handle<T>(filtration_handle);
  return Filtration(values.begin(), values.end(), num_parameters);
}

template <typename Wrapper, typename Filtration, typename Value, bool IsKCritical>
Wrapper& insert_batch_simplices(Wrapper& self,
                                const flat_simplex_batch& simplices,
                                const std::vector<Filtration>& filtrations,
                                bool empty_filtration) {
  if (simplices.empty()) {
    return self;
  }

  std::vector<int> simplex(simplices.simplex_size);

  if constexpr (!IsKCritical) {
    {
      nb::gil_scoped_release release;
      for (size_t i = 0; i < simplices.num_simplices; ++i) {
        const int* simplex_data = simplices.vertices.data() + i * simplices.simplex_size;
        std::copy_n(simplex_data, simplices.simplex_size, simplex.begin());
        if (empty_filtration) {
          self.tree.insert(simplex, Filtration::minus_inf(self.tree.num_parameters()));
        } else {
          self.tree.insert(simplex, filtrations[i]);
        }
      }
    }
    if (empty_filtration) {
      nb::gil_scoped_release release;
      self.tree.make_filtration_non_decreasing();
    }
  } else {
    nb::gil_scoped_release release;
    for (size_t i = 0; i < simplices.num_simplices; ++i) {
      const int* simplex_data = simplices.vertices.data() + i * simplices.simplex_size;
      std::copy_n(simplex_data, simplices.simplex_size, simplex.begin());
      insert_kcritical_simplex(self.tree, simplex, empty_filtration ? nullptr : &filtrations[i]);
    }
  }
  return self;
}

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical>
bool insert_single_simplex(Wrapper& self, const std::vector<int>& simplex, nb::handle filtration_handle, bool force);

template <typename Class, typename Wrapper, typename Filtration, typename Value, bool IsKCritical, typename Index>
void bind_insert_batch_overloads(Class& cls) {
  if constexpr (!IsKCritical) {
    cls.def(
        "_insert_batch",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<2>> vertex_array,
           nb::ndarray<nb::numpy, const Value, nb::ndim<2>> filtrations) -> Wrapper& {
          auto simplices = simplices_from_vertex_array(vertex_array);
          const bool empty_filtration = filtrations.shape(0) == 0 || filtrations.shape(1) == 0;
          if (!empty_filtration && filtrations.shape(0) != simplices.num_simplices) {
            throw std::runtime_error("Invalid filtration batch shape.");
          }
          auto dense_filtrations = empty_filtration
                                       ? std::vector<Filtration>{}
                                       : one_critical_filtrations_from_array<Filtration, Value>(filtrations);
          return insert_batch_simplices<Wrapper, Filtration, Value, false>(
              self, simplices, dense_filtrations, empty_filtration);
        },
        "vertex_array"_a,
        "filtrations"_a,
        nb::rv_policy::reference_internal);
  } else {
    cls.def(
        "_insert_batch",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<2>> vertex_array,
           nb::ndarray<nb::numpy, const Value, nb::ndim<3>> filtrations) -> Wrapper& {
          auto simplices = simplices_from_vertex_array(vertex_array);
          const bool empty_filtration =
              filtrations.shape(0) == 0 || filtrations.shape(1) == 0 || filtrations.shape(2) == 0;
          if (!empty_filtration && filtrations.shape(0) != simplices.num_simplices) {
            throw std::runtime_error("Invalid filtration batch shape.");
          }
          auto packed_filtrations = empty_filtration ? std::vector<Filtration>{}
                                                     : kcritical_filtrations_from_array<Filtration, Value>(
                                                           filtrations, self.tree.num_parameters());
          return insert_batch_simplices<Wrapper, Filtration, Value, true>(
              self, simplices, packed_filtrations, empty_filtration);
        },
        "vertex_array"_a,
        "filtrations"_a,
        nb::rv_policy::reference_internal);
  }
}

template <typename Class, typename Wrapper, typename Filtration, typename Value, bool IsKCritical, typename Index>
void bind_simplex_array_overloads(Class& cls) {
  if constexpr (!IsKCritical) {
    cls.def(
        "_insert_simplex",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<1>, nb::c_contig> filtration,
           bool force) {
          return insert_single_simplex<Wrapper, Filtration, Value, false>(
              self,
              simplex_from_array(simplex),
              nb::cast(one_critical_filtration_from_array<Filtration, Value>(filtration)),
              force);
        },
        "simplex"_a,
        "filtration"_a,
        "force"_a = false);

    cls.def(
        "_insert",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<1>, nb::c_contig> filtration) -> bool {
          return insert_single_simplex<Wrapper, Filtration, Value, false>(
              self,
              simplex_from_array(simplex),
              nb::cast(one_critical_filtration_from_array<Filtration, Value>(filtration)),
              false);
        },
        "simplex"_a,
        "filtration"_a);

    cls.def(
        "_assign_filtration",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<1>, nb::c_contig> filtration) -> Wrapper& {
          {
            nb::gil_scoped_release release;
            self.tree.assign_simplex_filtration(simplex_from_array(simplex),
                                                one_critical_filtration_from_array<Filtration, Value>(filtration));
          }
          return self;
        },
        nb::rv_policy::reference_internal);
  } else {
    cls.def(
        "_insert_simplex",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<1>, nb::c_contig> filtration,
           bool force) {
          return insert_single_simplex<Wrapper, Filtration, Value, true>(
              self,
              simplex_from_array(simplex),
              nb::cast(kcritical_filtration_from_array<Filtration, Value>(filtration, self.tree.num_parameters())),
              force);
        },
        "simplex"_a,
        "filtration"_a,
        "force"_a = false);

    cls.def(
        "_insert_simplex",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<2>, nb::c_contig> filtration,
           bool force) {
          return insert_single_simplex<Wrapper, Filtration, Value, true>(
              self,
              simplex_from_array(simplex),
              nb::cast(kcritical_filtration_from_array<Filtration, Value>(filtration, self.tree.num_parameters())),
              force);
        },
        "simplex"_a,
        "filtration"_a,
        "force"_a = false);

    cls.def(
        "_insert",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<1>, nb::c_contig> filtration) -> bool {
          return insert_single_simplex<Wrapper, Filtration, Value, true>(
              self,
              simplex_from_array(simplex),
              nb::cast(kcritical_filtration_from_array<Filtration, Value>(filtration, self.tree.num_parameters())),
              false);
        },
        "simplex"_a,
        "filtration"_a);

    cls.def(
        "_insert",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<2>, nb::c_contig> filtration) -> bool {
          return insert_single_simplex<Wrapper, Filtration, Value, true>(
              self,
              simplex_from_array(simplex),
              nb::cast(kcritical_filtration_from_array<Filtration, Value>(filtration, self.tree.num_parameters())),
              false);
        },
        "simplex"_a,
        "filtration"_a);

    cls.def(
        "_assign_filtration",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<1>, nb::c_contig> filtration) -> Wrapper& {
          {
            nb::gil_scoped_release release;
            self.tree.assign_simplex_filtration(
                simplex_from_array(simplex),
                kcritical_filtration_from_array<Filtration, Value>(filtration, self.tree.num_parameters()));
          }
          return self;
        },
        nb::rv_policy::reference_internal);

    cls.def(
        "_assign_filtration",
        [](Wrapper& self,
           nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex,
           nb::ndarray<nb::numpy, const Value, nb::ndim<2>, nb::c_contig> filtration) -> Wrapper& {
          {
            nb::gil_scoped_release release;
            self.tree.assign_simplex_filtration(
                simplex_from_array(simplex),
                kcritical_filtration_from_array<Filtration, Value>(filtration, self.tree.num_parameters()));
          }
          return self;
        },
        nb::rv_policy::reference_internal);
  }

  cls.def("_get_filtration", [](Wrapper& self, nb::ndarray<nb::numpy, const Index, nb::ndim<1>, nb::c_contig> simplex) {
    return filtration_to_python<Filtration, Value, IsKCritical>(
        *self.tree.simplex_filtration(simplex_from_array(simplex)), nb::find(self));
  });
}

template <typename Filtration, typename T, bool IsKCritical>
nb::object filtration_to_python(const Filtration& filtration, nb::handle owner) {
  if constexpr (IsKCritical) {
    nb::list out;
    const int k = static_cast<int>(filtration.num_generators());
    const int p = static_cast<int>(filtration.num_parameters());
    for (int i = 0; i < k; ++i) {
      std::vector<T> row(p);
      for (int j = 0; j < p; ++j) {
        row[j] = filtration(i, j);
      }
      out.append(nb::cast(owned_array<T>(std::move(row), {static_cast<size_t>(p)})));
    }
    return out;
  } else {
    const int p = static_cast<int>(filtration.num_parameters());
    return nb::cast(view_array(const_cast<T*>(&filtration(0, 0)), {static_cast<size_t>(p)}, owner));
  }
}

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical, bool SortRows, typename SimplexHandle>
nb::tuple simplex_entry_to_python(Wrapper& self, SimplexHandle sh) {
  auto pair = self.tree.get_simplex_and_filtration(sh);
  std::vector<int32_t> simplex(pair.first.begin(), pair.first.end());
  return nb::make_tuple(
      nb::cast(owned_array<int32_t>(std::move(simplex), {pair.first.size()})),
      normalized_filtration_to_python<Filtration, T, IsKCritical, SortRows>(*pair.second, nb::find(self)));
}

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical>
bool insert_single_simplex(Wrapper& self, const std::vector<int>& simplex, nb::handle filtration_handle, bool force) {
  if constexpr (IsKCritical) {
    (void)force;
    bool has_filtration = !filtration_handle.is_none();
    std::unique_ptr<Filtration> filtration;
    if (has_filtration) {
      filtration = std::make_unique<Filtration>(
          filtration_from_handle<Filtration, T, IsKCritical>(filtration_handle, self.tree.num_parameters()));
    }
    bool inserted = false;
    {
      nb::gil_scoped_release release;
      inserted = insert_kcritical_simplex(self.tree, simplex, filtration.get());
    }
    return inserted;
  } else {
    auto filtration = filtration_from_handle<Filtration, T, IsKCritical>(filtration_handle, self.tree.num_parameters());
    bool inserted = false;
    {
      nb::gil_scoped_release release;
      inserted = force ? self.tree.insert_force(simplex, filtration) : self.tree.insert(simplex, filtration);
    }
    return inserted;
  }
}

template <typename Desc, typename TargetWrapper, typename TargetInterface>
void copy_from_desc(TargetWrapper& self, const simplextree_wrapper_t<Desc>& source) {
  {
    nb::gil_scoped_release release;
    SimplexTreeConversion<TargetInterface, typename Desc::interface_type>::run(self.tree, source.tree);
  }
  self.filtration_grid = source.filtration_grid;
}

template <typename TargetWrapper, typename TargetInterface>
bool try_copy_from_any(TargetWrapper& self, nb::handle source) {
  if (!is_simplextree_object(source)) {
    return false;
  }
  visit_const_simplextree_wrapper(source, [&]<typename D>(const simplextree_wrapper_t<D>& wrapper) {
    copy_from_desc<D, TargetWrapper, TargetInterface>(self, wrapper);
  });
  return true;
}

template <typename TargetDesc, typename SourceDesc>
PySimplexTree<typename TargetDesc::interface_type, typename TargetDesc::value_type> construct_from_simplextree_wrapper(
    const simplextree_wrapper_t<SourceDesc>& source) {
  using Wrapper = PySimplexTree<typename TargetDesc::interface_type, typename TargetDesc::value_type>;
  using Interface = typename TargetDesc::interface_type;
  Wrapper out;
  copy_from_desc<SourceDesc, Wrapper, Interface>(out, source);
  return out;
}

template <typename TargetDesc, typename SourceDesc>
PySimplexTree<typename TargetDesc::interface_type, typename TargetDesc::value_type> construct_from_slicer_wrapper(
    const typename SourceDesc::wrapper& source,
    int max_dim) {
  using Wrapper = PySimplexTree<typename TargetDesc::interface_type, typename TargetDesc::value_type>;
  using Interface = typename TargetDesc::interface_type;
  Wrapper out;
  build_from_slicer_desc<SourceDesc, Wrapper, Interface>(out, source, max_dim);
  return out;
}

template <typename TargetDesc, typename Class, typename... SourceDesc>
void bind_simplextree_source_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(nb::new_([](const simplextree_wrapper_t<SourceDesc>& source) {
             return construct_from_simplextree_wrapper<TargetDesc, SourceDesc>(source);
           }),
           "source"_a),
   ...);
}

template <typename TargetDesc, typename Class, typename... SourceDesc>
void bind_slicer_source_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(nb::new_([](const typename SourceDesc::wrapper& source, int max_dim) {
             return construct_from_slicer_wrapper<TargetDesc, SourceDesc>(source, max_dim);
           }),
           "source"_a,
           "max_dim"_a = -1),
   ...);
}

template <typename TargetDesc, typename Class>
void bind_typed_source_constructors(Class& cls) {
  bind_simplextree_source_constructors<TargetDesc>(cls, SimplexTreeDescriptorList{});
  bind_slicer_source_constructors<TargetDesc>(cls, SlicerDescriptorList{});
}

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical>
nb::list simplices_to_python(Wrapper& self) {
  nb::list out;
  for (auto sh : self.tree.complex_simplex_range()) {
    auto pair = self.tree.get_simplex_and_filtration(sh);
    std::vector<int32_t> simplex(pair.first.begin(), pair.first.end());
    out.append(nb::make_tuple(nb::cast(owned_array<int32_t>(std::move(simplex), {pair.first.size()})),
                              filtration_to_python<Filtration, T, IsKCritical>(*pair.second, nb::find(self))));
  }
  return out;
}

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical>
nb::list skeleton_to_python(Wrapper& self, int dimension) {
  nb::list out;
  for (auto sh : self.tree.skeleton_simplex_range(dimension)) {
    auto pair = self.tree.get_simplex_and_filtration(sh);
    std::vector<int32_t> simplex(pair.first.begin(), pair.first.end());
    out.append(nb::make_tuple(nb::cast(owned_array<int32_t>(std::move(simplex), {pair.first.size()})),
                              filtration_to_python<Filtration, T, IsKCritical>(*pair.second, nb::find(self))));
  }
  return out;
}

template <typename Wrapper, typename Filtration, typename T, bool IsKCritical>
nb::list boundaries_to_python(Wrapper& self, const std::vector<int>& simplex) {
  nb::list out;
  auto it_pair = self.tree.get_boundary_iterators(simplex);
  while (it_pair.first != it_pair.second) {
    auto pair = self.tree.get_simplex_and_filtration(*it_pair.first);
    std::vector<int32_t> current(pair.first.begin(), pair.first.end());
    out.append(nb::make_tuple(nb::cast(owned_array<int32_t>(std::move(current), {pair.first.size()})),
                              filtration_to_python<Filtration, T, IsKCritical>(*pair.second, nb::find(self))));
    ++it_pair.first;
  }
  return out;
}

template <typename Wrapper>
nb::object serialized_state(Wrapper& self) {
  size_t buffer_size = 0;
  {
    nb::gil_scoped_release release;
    buffer_size = self.tree.get_serialization_size();
  }
  std::vector<uint8_t> buffer(buffer_size);
  if (buffer_size > 0) {
    nb::gil_scoped_release release;
    self.tree.serialize(reinterpret_cast<char*>(buffer.data()), buffer_size);
  }
  return nb::cast(owned_array<uint8_t>(std::move(buffer), {buffer_size}));
}

template <typename Wrapper>
void load_state(Wrapper& self, nb::handle state) {
  auto buffer = vector_from_handle<uint8_t>(state);
  int num_parameters = 0;
  {
    nb::gil_scoped_release release;
    self.tree.clear();
    if (!buffer.empty()) {
      self.tree.deserialize(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      auto it = self.tree.complex_simplex_range().begin();
      auto end = self.tree.complex_simplex_range().end();
      if (it != end) {
        auto pair = self.tree.get_simplex_and_filtration(*it);
        num_parameters = pair.second->num_parameters();
      }
    }
    self.tree.set_num_parameters(num_parameters);
  }
}

template <typename T>
nb::object edge_list_to_python(const std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>>& edges) {
  nb::list out;
  for (const auto& edge : edges) {
    out.append(nb::make_tuple(nb::make_tuple(edge.first.first, edge.first.second),
                              nb::make_tuple(edge.second.first, edge.second.second)));
  }
  return out;
}

template <typename Desc>
void bind_simplextree_class(nb::module_& m, nb::list& available_simplextrees) {
  using Filtration = typename Desc::filtration_type;
  using Interface = typename Desc::interface_type;
  using Value = typename Desc::value_type;
  using Wrapper = PySimplexTree<Interface, Value>;
  constexpr bool k_is_kcritical = Desc::is_kcritical;
  constexpr bool k_sort_rows = std::string_view(Desc::filtration_container_name) != std::string_view("Flat");

  auto adopt_ptr = [](Wrapper& self, intptr_t ptr) -> Wrapper& {
    Interface* other = reinterpret_cast<Interface*>(ptr);
    self.tree = *other;
    delete other;
    return self;
  };

  auto cls =
      nb::class_<Wrapper>(m, Desc::python_name.data())
          .def(nb::init<>())
          .def(nb::new_([](int num_parameters) {
                 Wrapper out;
                 int n = num_parameters <= 0 ? 2 : num_parameters;
                 out.tree.resize_all_filtrations(n);
                 out.tree.set_num_parameters(n);
                 return out;
               }),
               "num_parameters"_a = -1)
          .def_prop_rw(
              "filtration_grid",
              [](Wrapper& self) -> nb::object { return self.filtration_grid; },
              [](Wrapper& self, nb::object value) { self.filtration_grid = value.is_none() ? nb::list() : value; },
              nb::arg("value").none())
          .def_prop_rw(
              "thisptr",
              [](Wrapper& self) -> intptr_t { return reinterpret_cast<intptr_t>(&self.tree); },
              [adopt_ptr](Wrapper& self, intptr_t ptr) { adopt_ptr(self, ptr); })
          .def("_from_ptr", adopt_ptr, nb::rv_policy::reference_internal)
          .def("_from_interface_ptr", adopt_ptr, nb::rv_policy::reference_internal)
          .def(
              "_copy_from_any",
              [](Wrapper& self, nb::handle other) -> Wrapper& {
                if (!try_copy_from_any<Wrapper, Interface>(self, other)) {
                  throw std::runtime_error("Unsupported SimplexTreeMulti input type.");
                }
                return self;
              },
              nb::rv_policy::reference_internal)
          .def(
              "_from_slicer",
              [](Wrapper& self, nb::handle slicer, int max_dim) -> Wrapper& {
                if (!try_build_from_slicer<Wrapper, Interface>(self, slicer, max_dim)) {
                  throw std::runtime_error("Unsupported slicer input type.");
                }
                return self;
              },
              "slicer"_a,
              "max_dim"_a = -1,
              nb::rv_policy::reference_internal)
          .def(
              "_from_gudhi_state",
              [](Wrapper& self, nb::handle state, int num_parameters, nb::handle default_values) -> Wrapper& {
                auto buffer = vector_from_handle<uint8_t>(state);
                auto default_filtration =
                    default_filtration_from_handle<Filtration, Value>(default_values, num_parameters);
                {
                  nb::gil_scoped_release release;
                  if (!buffer.empty()) {
                    self.tree.from_std(reinterpret_cast<char*>(buffer.data()), buffer.size(), 0, default_filtration);
                  }
                  self.tree.resize_all_filtrations(num_parameters);
                  self.tree.set_num_parameters(num_parameters);
                }
                return self;
              },
              nb::rv_policy::reference_internal)
          .def("__getstate__",
               [](Wrapper& self) -> nb::tuple { return nb::make_tuple(serialized_state(self), self.filtration_grid); })
          .def("__reduce__",
               [](Wrapper& self) -> nb::tuple {
                 return nb::make_tuple(nb::borrow<nb::object>(nb::type<Wrapper>()),
                                       nb::make_tuple(),
                                       nb::make_tuple(serialized_state(self), self.filtration_grid));
               })
          .def("__reduce_ex__",
               [](Wrapper& self, int) -> nb::tuple {
                 return nb::make_tuple(nb::borrow<nb::object>(nb::type<Wrapper>()),
                                       nb::make_tuple(),
                                       nb::make_tuple(serialized_state(self), self.filtration_grid));
               })
          .def("_serialize_state", [](Wrapper& self) -> nb::object { return serialized_state(self); })
          .def(
              "_deserialize_state",
              [](Wrapper& self, nb::handle state) -> Wrapper& {
                load_state(self, state);
                return self;
              },
              nb::rv_policy::reference_internal)
          .def(
              "_insert_simplex",
              [](Wrapper& self, nb::handle simplex_handle, nb::handle filtration_handle, bool force) {
                auto simplex = vector_from_handle<int>(simplex_handle);
                return insert_single_simplex<Wrapper, Filtration, Value, k_is_kcritical>(
                    self, simplex, filtration_handle, force);
              },
              "simplex"_a,
              "filtration"_a = nb::none(),
              "force"_a = false)
          .def(
              "_insert",
              [](Wrapper& self, nb::handle simplex_handle, nb::handle filtration_handle) -> bool {
                auto simplex = vector_from_handle<int>(simplex_handle);
                return insert_single_simplex<Wrapper, Filtration, Value, k_is_kcritical>(
                    self, simplex, filtration_handle, false);
              },
              "simplex"_a,
              "filtration"_a = nb::none())
          .def(
              "_assign_filtration",
              [](Wrapper& self, nb::handle simplex_handle, nb::handle filtration_handle) -> Wrapper& {
                auto simplex = vector_from_handle<int>(simplex_handle);
                auto filtration = filtration_from_handle<Filtration, Value, k_is_kcritical>(filtration_handle,
                                                                                            self.tree.num_parameters());
                {
                  nb::gil_scoped_release release;
                  self.tree.assign_simplex_filtration(simplex, filtration);
                }
                return self;
              },
              nb::rv_policy::reference_internal);

  bind_insert_batch_overloads<decltype(cls), Wrapper, Filtration, Value, k_is_kcritical, int32_t>(cls);
  bind_insert_batch_overloads<decltype(cls), Wrapper, Filtration, Value, k_is_kcritical, int64_t>(cls);
  bind_simplex_array_overloads<decltype(cls), Wrapper, Filtration, Value, k_is_kcritical, int32_t>(cls);
  bind_simplex_array_overloads<decltype(cls), Wrapper, Filtration, Value, k_is_kcritical, int64_t>(cls);

  cls.def(
         "_insert_batch",
         [](Wrapper& self, nb::handle vertex_array_handle, nb::handle filtrations_handle) -> Wrapper& {
           auto vertex_array = matrix_from_handle<int>(vertex_array_handle);
           auto simplices = simplices_from_vertex_rows(vertex_array);
           if (simplices.empty()) {
             return self;
           }
           bool empty_filtration =
               filtrations_handle.is_none() ||
               (nb::hasattr(filtrations_handle, "size") && nb::cast<size_t>(filtrations_handle.attr("size")) == 0);

           if constexpr (!k_is_kcritical) {
             std::vector<Filtration> filtrations;
             if (!empty_filtration) {
               filtrations =
                   one_critical_filtrations_from_rows<Filtration, Value>(matrix_from_handle<Value>(filtrations_handle));
             }
             return insert_batch_simplices<Wrapper, Filtration, Value, false>(
                 self, simplices, filtrations, empty_filtration);
           } else {
             std::vector<Filtration> filtrations;
             if (!empty_filtration) {
               filtrations.reserve(simplices.num_simplices);
               for (nb::handle row_handle : nb::iter(filtrations_handle)) {
                 filtrations.push_back(
                     filtration_from_handle<Filtration, Value, k_is_kcritical>(row_handle, self.tree.num_parameters()));
               }
             }
             return insert_batch_simplices<Wrapper, Filtration, Value, true>(
                 self, simplices, filtrations, empty_filtration);
           }
         },
         nb::rv_policy::reference_internal)
      .def("_get_filtration",
           [](Wrapper& self, nb::handle simplex_handle) {
             auto simplex = vector_from_handle<int>(simplex_handle);
             return filtration_to_python<Filtration, Value, k_is_kcritical>(*self.tree.simplex_filtration(simplex),
                                                                            nb::find(self));
           })
      .def(
          "_iter_simplices",
          [](Wrapper& self) {
            return make_simplextree_python_iterator<Wrapper, Filtration, Value, k_is_kcritical, k_sort_rows>(
                self,
                "simplex_iterator",
                self.tree.get_simplices_iterator_begin(),
                self.tree.get_simplices_iterator_end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "get_simplices",
          [](Wrapper& self) {
            return make_simplextree_python_iterator<Wrapper, Filtration, Value, k_is_kcritical, k_sort_rows>(
                self,
                "simplex_iterator",
                self.tree.get_simplices_iterator_begin(),
                self.tree.get_simplices_iterator_end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "__iter__",
          [](Wrapper& self) {
            return make_simplextree_python_iterator<Wrapper, Filtration, Value, k_is_kcritical, k_sort_rows>(
                self,
                "simplex_iterator",
                self.tree.get_simplices_iterator_begin(),
                self.tree.get_simplices_iterator_end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "get_skeleton",
          [](Wrapper& self, int dimension) {
            return make_simplextree_python_iterator<Wrapper, Filtration, Value, k_is_kcritical, k_sort_rows>(
                self,
                "skeleton_iterator",
                self.tree.get_skeleton_iterator_begin(dimension),
                self.tree.get_skeleton_iterator_end(dimension));
          },
          "dimension"_a,
          nb::keep_alive<0, 1>())
      .def(
          "get_boundaries",
          [](Wrapper& self, nb::handle simplex_handle) {
            auto simplex = vector_from_handle<int>(simplex_handle);
            auto it_pair = self.tree.get_boundary_iterators(simplex);
            return make_simplextree_python_iterator<Wrapper, Filtration, Value, k_is_kcritical, k_sort_rows>(
                self, "boundary_iterator", it_pair.first, it_pair.second);
          },
          "simplex"_a,
          nb::keep_alive<0, 1>())
      .def("_get_skeleton",
           [](Wrapper& self, int dimension) {
             return skeleton_to_python<Wrapper, Filtration, Value, k_is_kcritical>(self, dimension);
           })
      .def("_get_boundaries",
           [](Wrapper& self, nb::handle simplex_handle) {
             return boundaries_to_python<Wrapper, Filtration, Value, k_is_kcritical>(
                 self, vector_from_handle<int>(simplex_handle));
           })
      .def("_get_filtration_values",
           [](Wrapper& self, nb::handle degrees_handle) {
             auto degrees = vector_from_handle<int>(degrees_handle);
             decltype(self.tree.get_filtration_values(degrees)) out;
             {
               nb::gil_scoped_release release;
               out = self.tree.get_filtration_values(degrees);
             }
             return nb::cast(out);
           })
      .def("_get_to_std_state",
           [](Wrapper& self, nb::handle basepoint_handle, nb::handle direction_handle, int parameter) {
             auto basepoint = vector_from_handle<double>(basepoint_handle);
             auto direction = vector_from_handle<double>(direction_handle);
             Gudhi::multi_persistence::Line<double> line(basepoint, direction);
             decltype(self.tree.get_to_std_state(line, parameter)) serialized;
             {
               nb::gil_scoped_release release;
               serialized = self.tree.get_to_std_state(line, parameter);
             }
             return nb::cast(owned_array<char>(std::move(serialized), {serialized.size()}));
           })
      .def("_get_to_std_linear_projection_state",
           [](Wrapper& self, nb::handle linear_form_handle) {
             auto linear_form = vector_from_handle<double>(linear_form_handle);
             decltype(self.tree.get_to_std_linear_projection_state(linear_form)) serialized;
             {
               nb::gil_scoped_release release;
               serialized = self.tree.get_to_std_linear_projection_state(linear_form);
             }
             return nb::cast(owned_array<char>(std::move(serialized), {serialized.size()}));
           })
      .def(
          "_squeeze_inplace",
          [](Wrapper& self, nb::handle grid_handle, bool coordinate_values) -> Wrapper& {
            auto grid = matrix_from_handle<double>(grid_handle);
            {
              nb::gil_scoped_release release;
              self.tree.squeeze_filtration_inplace(grid, coordinate_values);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("_squeeze_to",
           [](Wrapper& self, Wrapper& out, nb::handle grid_handle) {
             auto grid = matrix_from_handle<double>(grid_handle);
             {
               nb::gil_scoped_release release;
               self.tree.squeeze_filtration(reinterpret_cast<intptr_t>(&out.tree), grid);
             }
           })
      .def("_unsqueeze_to",
           [](Wrapper& self, Wrapper& out, nb::handle grid_handle) {
             auto grid = matrix_from_handle<double>(grid_handle);
             {
               nb::gil_scoped_release release;
               out.tree.unsqueeze_filtration(reinterpret_cast<intptr_t>(&self.tree), grid);
             }
           })
      .def("num_vertices", [](Wrapper& self) -> int { return self.tree.num_vertices(); })
      .def("num_simplices", [](Wrapper& self) -> int { return self.tree.num_simplices(); })
      .def("dimension", [](Wrapper& self) -> int { return self.tree.dimension(); })
      .def("upper_bound_dimension", [](Wrapper& self) -> int { return self.tree.upper_bound_dimension(); })
      .def("simplex_dimension",
           [](Wrapper& self, nb::handle simplex_handle) {
             return self.tree.simplex_dimension(vector_from_handle<int>(simplex_handle));
           })
      .def("find_simplex",
           [](Wrapper& self, nb::handle simplex_handle) {
             auto simplex = vector_from_handle<int>(simplex_handle);
             return self.tree.find_simplex(simplex);
           })
      .def(
          "remove_maximal_simplex",
          [](Wrapper& self, nb::handle simplex_handle) -> Wrapper& {
            auto simplex = vector_from_handle<int>(simplex_handle);
            {
              nb::gil_scoped_release release;
              self.tree.remove_maximal_simplex(simplex);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("prune_above_dimension",
           [](Wrapper& self, int dimension) {
             bool out;
             {
               nb::gil_scoped_release release;
               out = self.tree.prune_above_dimension(dimension);
             }
             return out;
           })
      .def(
          "expansion",
          [](Wrapper& self, int max_dim) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.tree.expansion(max_dim);
              self.tree.make_filtration_non_decreasing();
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("make_filtration_non_decreasing",
           [](Wrapper& self) -> bool {
             bool out;
             {
               nb::gil_scoped_release release;
               out = self.tree.make_filtration_non_decreasing();
             }
             return out;
           })
      .def(
          "_simplify_filtration_raw",
          [](Wrapper& self) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.tree.simplify_filtration_inplace();
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "reset_filtration",
          [](Wrapper& self, nb::handle filtration_handle, int min_dim) -> Wrapper& {
            auto filtration = filtration_from_handle<Filtration, Value, k_is_kcritical>(filtration_handle,
                                                                                        self.tree.num_parameters());
            {
              nb::gil_scoped_release release;
              self.tree.reset_filtration(filtration, min_dim);
            }
            return self;
          },
          "filtration"_a,
          "min_dim"_a = 0,
          nb::rv_policy::reference_internal)
      .def(
          "fill_lowerstar",
          [](Wrapper& self, nb::handle values_handle, int axis) -> Wrapper& {
            auto values = vector_from_handle<Value>(values_handle);
            {
              nb::gil_scoped_release release;
              self.tree.fill_lowerstar(values, axis);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("get_simplices_of_dimension",
           [](Wrapper& self, int dim) {
             decltype(self.tree.get_simplices_of_dimension(dim)) out;
             {
               nb::gil_scoped_release release;
               out = self.tree.get_simplices_of_dimension(dim);
             }
             std::vector<int32_t> values(out.begin(), out.end());
             return nb::cast(owned_array<int32_t>(
                 std::move(values), {out.size() / static_cast<size_t>(dim + 1), static_cast<size_t>(dim + 1)}));
           })
      .def("get_edge_list",
           [](Wrapper& self) -> nb::object {
             decltype(self.tree.get_edge_list()) edges;
             {
               nb::gil_scoped_release release;
               edges = self.tree.get_edge_list();
             }
             return edge_list_to_python<Value>(edges);
           })
      .def("pts_to_indices",
           [](Wrapper& self, nb::handle pts_handle, nb::handle dims_handle) {
             auto pts = matrix_from_handle<Value>(pts_handle);
             auto dims = vector_from_handle<int>(dims_handle);
             decltype(self.tree.pts_to_indices(pts, dims)) result;
             {
               nb::gil_scoped_release release;
               result = self.tree.pts_to_indices(pts, dims);
             }
             return nb::make_tuple(nb::cast(result.first), nb::cast(result.second));
           })
      .def(
          "set_dimension",
          [](Wrapper& self, int value) -> Wrapper& {
            self.tree.set_dimension(value);
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "set_key",
          [](Wrapper& self, nb::handle simplex_handle, int key) -> Wrapper& {
            auto simplex = vector_from_handle<int>(simplex_handle);
            {
              nb::gil_scoped_release release;
              self.tree.set_key(simplex, key);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("get_key",
           [](Wrapper& self, nb::handle simplex_handle) {
             return self.tree.get_key(vector_from_handle<int>(simplex_handle));
           })
      .def(
          "set_keys_to_enumerate",
          [](Wrapper& self) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.tree.set_keys_to_enumerate();
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "set_num_parameter",
          [](Wrapper& self, int num) -> Wrapper& {
            {
              nb::gil_scoped_release release;
              self.tree.resize_all_filtrations(num);
              self.tree.set_num_parameters(num);
            }
            return self;
          },
          nb::rv_policy::reference_internal)
      .def("__eq__", [](Wrapper& self, Wrapper& other) { return self.tree == other.tree; })
      .def_prop_ro("num_parameters", [](const Wrapper& self) -> int { return self.tree.num_parameters(); })
      .def_prop_ro("is_kcritical", [](const Wrapper&) -> bool { return k_is_kcritical; })
      .def_prop_ro("_template_id", [](const Wrapper&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const Wrapper&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def_prop_ro("ftype", [](const Wrapper&) -> std::string { return std::string(Desc::ftype_name); })
      .def_prop_ro("filtration_container",
                   [](const Wrapper&) -> std::string { return std::string(Desc::filtration_container_name); });

  bind_typed_source_constructors<Desc>(cls);

  available_simplextrees.append(cls);
}

template <typename... Desc>
void bind_all_simplextrees(type_list<Desc...>, nb::module_& m, nb::list& available_simplextrees) {
  (bind_simplextree_class<Desc>(m, available_simplextrees), ...);
}

template <typename... Desc>
nb::tuple compute_hilbert_signed_measure(type_list<Desc...>,
                                         nb::handle simplextree,
                                         std::vector<indices_type>& container,
                                         const std::vector<indices_type>& full_shape,
                                         const std::vector<indices_type>& degrees,
                                         size_t width,
                                         bool zero_pad,
                                         indices_type n_jobs,
                                         bool verbose,
                                         bool expand_collapse) {
  if (!is_simplextree_multi(simplextree)) {
    throw std::runtime_error("Unsupported SimplexTreeMulti type.");
  }
  return dispatch_simplextree_by_template_id(template_id_of(simplextree), [&]<typename D>() -> nb::tuple {
    using Wrapper = PySimplexTree<typename D::interface_type, typename D::value_type>;
    auto& st = nb::cast<Wrapper&>(simplextree).tree;
    signed_measure_type sm;
    {
      nb::gil_scoped_release release;
      sm = Gudhi::multiparameter::hilbert_function::get_hilbert_signed_measure(
          st, container.data(), full_shape, degrees, zero_pad, n_jobs, verbose, expand_collapse);
    }
    return signed_measure_to_python(sm, width);
  });
}

template <typename... Desc>
nb::tuple compute_euler_signed_measure(type_list<Desc...>,
                                       nb::handle simplextree,
                                       std::vector<tensor_dtype>& container,
                                       const std::vector<indices_type>& grid_shape,
                                       size_t width,
                                       bool zero_pad,
                                       bool verbose) {
  if (!is_simplextree_multi(simplextree)) {
    throw std::runtime_error("Unsupported SimplexTreeMulti type.");
  }
  return dispatch_simplextree_by_template_id(template_id_of(simplextree), [&]<typename D>() -> nb::tuple {
    if constexpr (D::is_kcritical) {
      throw std::runtime_error("Unsupported SimplexTreeMulti type.");
    } else {
      using Wrapper = PySimplexTree<typename D::interface_type, typename D::value_type>;
      auto& st = nb::cast<Wrapper&>(simplextree).tree;
      signed_measure_type sm;
      {
        nb::gil_scoped_release release;
        sm = Gudhi::multiparameter::euler_characteristic::get_euler_signed_measure(
            st, container.data(), grid_shape, zero_pad, verbose);
      }
      return signed_measure_to_python(sm, width);
    }
  });
}

template <typename... Desc>
nb::tuple compute_rank_tensor(type_list<Desc...>,
                              nb::handle simplextree,
                              std::vector<tensor_dtype>& container,
                              const std::vector<indices_type>& full_shape,
                              const std::vector<indices_type>& degrees,
                              size_t total,
                              indices_type n_jobs,
                              bool expand_collapse) {
  if (!is_simplextree_multi(simplextree)) {
    throw std::runtime_error("Unsupported SimplexTreeMulti type.");
  }
  return dispatch_simplextree_by_template_id(template_id_of(simplextree), [&]<typename D>() -> nb::tuple {
    using Wrapper = PySimplexTree<typename D::interface_type, typename D::value_type>;
    auto& st = nb::cast<Wrapper&>(simplextree).tree;
    {
      nb::gil_scoped_release release;
      Gudhi::multiparameter::rank_invariant::compute_rank_invariant_python(
          st, container.data(), full_shape, degrees, n_jobs, expand_collapse);
    }
    return nb::make_tuple(nb::cast(owned_array<tensor_dtype>(std::move(container), {total})), nb::cast(full_shape));
  });
}

}  // namespace mpst

NB_MODULE(_simplex_tree_multi_nanobind, m) {
  m.doc() = "nanobind SimplexTreeMulti bindings";
  nb::list available_simplextrees;

  mpst::bind_all_simplextrees(mpst::SimplexTreeDescriptorList{}, m, available_simplextrees);
  m.def(
      "_get_simplextree_class",
      [](nb::handle dtype, bool kcritical, std::string filtration_container) {
        return mpst::get_simplextree_class(
            mpst::SimplexTreeDescriptorList{}, dtype, kcritical, std::move(filtration_container));
      },
      "dtype"_a,
      "kcritical"_a = false,
      "filtration_container"_a = "Contiguous");
  m.def("_get_simplextree_class_from_template_id", &mpst::get_simplextree_class_from_template_id, "template_id"_a);
  m.def("is_simplextree_multi", [](nb::object input) { return mpst::is_simplextree_multi(input); });

  m.def(
      "_compute_hilbert_signed_measure",
      [](nb::handle simplextree,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         bool zero_pad,
         mpst::indices_type n_jobs,
         bool verbose,
         bool expand_collapse) {
        auto grid_shape = mpst::vector_from_handle<mpst::indices_type>(grid_shape_handle);
        auto degrees = mpst::vector_from_handle<mpst::indices_type>(degrees_handle);
        std::vector<mpst::indices_type> full_shape;
        full_shape.reserve(grid_shape.size() + 1);
        full_shape.push_back(static_cast<mpst::indices_type>(degrees.size()));
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        size_t width = grid_shape.size() + 1;
        size_t total = 1;
        for (mpst::indices_type value : full_shape) {
          total *= static_cast<size_t>(value);
        }
        std::vector<mpst::tensor_dtype> container(total, 0);
        return mpst::compute_hilbert_signed_measure(mpst::SimplexTreeDescriptorList{},
                                                    simplextree,
                                                    container,
                                                    full_shape,
                                                    degrees,
                                                    width,
                                                    zero_pad,
                                                    n_jobs,
                                                    verbose,
                                                    expand_collapse);
      },
      "simplextree"_a,
      "grid_shape"_a,
      "degrees"_a,
      "zero_pad"_a = false,
      "n_jobs"_a = 0,
      "verbose"_a = false,
      "expand_collapse"_a = false);

  m.def(
      "_compute_euler_signed_measure",
      [](nb::handle simplextree, nb::handle grid_shape_handle, bool zero_pad, bool verbose) {
        auto grid_shape = mpst::vector_from_handle<mpst::indices_type>(grid_shape_handle);
        size_t width = grid_shape.size();
        size_t total = 1;
        for (mpst::indices_type value : grid_shape) {
          total *= static_cast<size_t>(value);
        }
        std::vector<mpst::tensor_dtype> container(total, 0);
        return mpst::compute_euler_signed_measure(
            mpst::SimplexTreeDescriptorList{}, simplextree, container, grid_shape, width, zero_pad, verbose);
      },
      "simplextree"_a,
      "grid_shape"_a,
      "zero_pad"_a = false,
      "verbose"_a = false);

  m.def(
      "_compute_rank_tensor",
      [](nb::handle simplextree,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         mpst::indices_type n_jobs,
         bool expand_collapse) {
        auto grid_shape = mpst::vector_from_handle<mpst::indices_type>(grid_shape_handle);
        auto degrees = mpst::vector_from_handle<mpst::indices_type>(degrees_handle);
        std::vector<mpst::indices_type> full_shape;
        full_shape.reserve(1 + 2 * grid_shape.size());
        full_shape.push_back(static_cast<mpst::indices_type>(degrees.size()));
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        size_t total = 1;
        for (mpst::indices_type value : full_shape) {
          total *= static_cast<size_t>(value);
        }
        std::vector<mpst::tensor_dtype> container(total, 0);
        return mpst::compute_rank_tensor(mpst::SimplexTreeDescriptorList{},
                                         simplextree,
                                         container,
                                         full_shape,
                                         degrees,
                                         total,
                                         n_jobs,
                                         expand_collapse);
      },
      "simplextree"_a,
      "grid_shape"_a,
      "degrees"_a,
      "n_jobs"_a = 0,
      "expand_collapse"_a = false);

  m.attr("available_simplextrees") = available_simplextrees;
}
