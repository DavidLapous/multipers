#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <stdexcept>
// #include <type_traits>
#include <vector>

#include "ext_interface/function_delaunay_interface.hpp"
#include "nanobind_array_utils.hpp"
#include <map>
#include <set>

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace mpfd {

using F64Matrix = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_simplextree_wrapper;

// template <typename Wrapper>
// void build_function_delaunay_simplextree(Wrapper& wrapper,
//                                          const multipers::function_delaunay_interface_input<int>& input,
//                                          bool verbose) {
//   using Interface = multipers::function_delaunay_simplextree_interface_output;
//   Interface output = multipers::function_delaunay_simplextree_interface<int>(input, verbose);
//   {
//     // nb::gil_scoped_release release;
//     wrapper.tree.copy_from(output);
//   }
// }

inline multipers::function_delaunay_interface_input<int> build_input(const F64Matrix& point_cloud,
                                                                     const F64Matrix& function_values) {
  const std::size_t num_points = point_cloud.shape(0);
  const std::size_t num_point_coordinates = point_cloud.shape(1);
  const std::size_t num_rows = function_values.shape(0);
  const std::size_t num_function_parameters = function_values.shape(1);
  if (num_points != num_rows) {
    throw std::runtime_error("point_cloud and function_values row counts do not match.");
  }
  std::vector<double> points;
  std::vector<double> values;
  {
    nb::gil_scoped_release release;
    points.assign(point_cloud.data(), point_cloud.data() + num_points * num_point_coordinates);
    values.assign(function_values.data(), function_values.data() + num_rows * num_function_parameters);
  }
  multipers::function_delaunay_interface_input<int> input;
  input.points = std::move(points);
  input.num_points = num_points;
  input.num_point_coordinates = num_point_coordinates;
  input.function_values = std::move(values);
  input.num_function_parameters = num_function_parameters;
  return input;
}

nb::object function_delaunay_to_slicer_for_target(nb::object target,
                                                  const multipers::function_delaunay_interface_input<int>& input,
                                                  int degree,
                                                  bool multi_chunk,
                                                  bool verbose) {
  multipers::contiguous_f64_complex complex;
  {
    nb::gil_scoped_release release;
    complex = multipers::function_delaunay_interface_contiguous_slicer<int>(input, degree, multi_chunk, verbose);
  }
  return multipers::nanobind_helpers::build_canonical_contiguous_f64_slicer_object_from_complex(target, complex);
}

nb::tuple recover_delaunay_indices(const multipers::function_delaunay_support_records& records) {
  using multipers::nanobind_utils::owned_array;
  using Vertices = std::vector<int>;
  std::set<std::pair<std::size_t, Vertices>> unique;
  std::map<std::size_t, std::vector<const std::pair<Vertices, Vertices>*>> dimensions;
  for (const auto& row : records) {
    unique.emplace(row.second.size(), row.second);
    dimensions[row.first.size()].push_back(&row);
  }
  std::map<Vertices, int64_t> support_ids, simplex_ids;
  std::map<std::size_t, std::vector<int64_t>> support_groups;
  int64_t id = 0;
  for (const auto& [size, vertices] : unique) {
    support_ids.emplace(vertices, id++);
    auto& group = support_groups[size];
    group.insert(group.end(), vertices.begin(), vertices.end());
  }
  nb::list supports, simplices, owners, faces;
  for (auto& [size, group] : support_groups) {
    const auto rows = group.size() / size;
    supports.append(nb::cast(owned_array<int64_t>(std::move(group), {rows, size})));
  }
  for (const auto& [size, group] : dimensions) {
    for (std::size_t i = 0; i < group.size(); ++i) simplex_ids.emplace(group[i]->first, i);
    std::vector<int64_t> vertices, owner, boundary;
    vertices.reserve(group.size() * size);
    owner.reserve(group.size());
    if (size > 1) boundary.reserve(group.size() * size);
    for (const auto* row : group) {
      vertices.insert(vertices.end(), row->first.begin(), row->first.end());
      owner.push_back(support_ids.at(row->second));
      if (size > 1) {
        for (std::size_t i = 0; i < size; ++i) {
          auto face = row->first;
          face.erase(face.begin() + i);
          boundary.push_back(simplex_ids.at(face));
        }
      }
    }
    simplices.append(nb::cast(owned_array<int64_t>(std::move(vertices), {group.size(), size})));
    owners.append(nb::cast(owned_array<int64_t>(std::move(owner), {group.size()})));
    faces.append(nb::cast(owned_array<int64_t>(std::move(boundary), {group.size(), size > 1 ? size : 0})));
  }
  return nb::make_tuple(supports, simplices, owners, faces);
}

nb::object function_delaunay_to_simplextree_for_target(nb::object target,
                                                       const multipers::function_delaunay_interface_input<int>& input,
                                                       bool verbose,
                                                       bool recover_indices = false) {
  multipers::function_delaunay_support_records supports;
  multipers::function_delaunay_simplextree_interface_output output =
      multipers::function_delaunay_simplextree_interface<int>(input, verbose, recover_indices ? &supports : nullptr);

  nb::object out = target.type()();
  visit_simplextree_wrapper(out, [&]<typename Desc>(auto& wrapper) { wrapper.tree.copy_from(output); });
  if (recover_indices) return nb::make_tuple(out, recover_delaunay_indices(supports));
  return out;
}

#endif

}  // namespace mpfd

NB_MODULE(_function_delaunay_interface, m) {
  auto available = []() { return multipers::function_delaunay_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "function_delaunay interface is not available in this build. Rebuild multipers with function_delaunay "
          "support to enable this backend.");
    }
  });

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
  m.def("_assign_filtrations",
        [](nb::object tree,
           nb::ndarray<nb::numpy, const int64_t, nb::ndim<2>, nb::c_contig> vertices,
           mpfd::F64Matrix grades) {
          if (vertices.shape(0) != grades.shape(0)) throw nb::value_error("Simplex and grade row counts differ.");
          mpfd::visit_simplextree_wrapper(tree, [&]<typename Desc>(auto& wrapper) {
            if constexpr (Desc::is_kcritical) {
              throw nb::type_error("Delaunay grade assignment requires a one-critical tree.");
            } else {
              if (grades.shape(1) != static_cast<std::size_t>(wrapper.tree.num_parameters()))
                throw nb::value_error("Incorrect number of filtration parameters.");
              nb::gil_scoped_release release;
              std::vector<int> simplex(vertices.shape(1));
              // Invalidate once, including batches that fail after partial assignment.
              wrapper.tree.clear_filtration();
              for (std::size_t i = 0; i < vertices.shape(0); ++i) {
                for (std::size_t j = 0; j < simplex.size(); ++j) simplex[j] = vertices(i, j);
                const auto handle = wrapper.tree.find(simplex);
                if (handle == wrapper.tree.null_simplex())
                  throw std::invalid_argument("Cannot assign a missing Delaunay simplex.");
                const auto* row = grades.data() + i * grades.shape(1);
                wrapper.tree.assign_filtration(handle, typename Desc::filtration_type(row, row + grades.shape(1)));
              }
            }
          });
        });
#endif

  m.def(
      "function_delaunay_to_slicer",
      [](nb::object slicer,
         mpfd::F64Matrix point_cloud,
         mpfd::F64Matrix function_values,
         int degree,
         bool multi_chunk,
         bool verbose) {
#if MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
        (void)slicer;
        (void)point_cloud;
        (void)function_values;
        (void)degree;
        (void)multi_chunk;
        (void)verbose;
        throw std::runtime_error("function_delaunay interface is disabled at compile time.");
#else
        auto input = mpfd::build_input(point_cloud, function_values);
        return multipers::nanobind_helpers::run_with_canonical_contiguous_f64_slicer_output(
            slicer, [&](const nb::object& target) {
              return mpfd::function_delaunay_to_slicer_for_target(target, input, degree, multi_chunk, verbose);
            });
#endif
      },
      "slicer"_a,
      "point_cloud"_a,
      "function_values"_a,
      "degree"_a,
      "multi_chunk"_a,
      "verbose"_a = false);

  m.def(
      "function_delaunay_to_simplextree",
      [](nb::object simplextree,
         mpfd::F64Matrix point_cloud,
         mpfd::F64Matrix function_values,
         bool verbose,
         bool recover_indices) {
#if MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
        (void)simplextree;
        (void)point_cloud;
        (void)function_values;
        (void)verbose;
        (void)recover_indices;
        throw std::runtime_error("function_delaunay interface is disabled at compile time.");
#else
        if (!mpfd::is_simplextree_object(simplextree)) {
          throw nb::type_error("function_delaunay_to_simplextree expects a SimplexTreeMulti target.");
        }
        auto input = mpfd::build_input(point_cloud, function_values);
        return mpfd::function_delaunay_to_simplextree_for_target(simplextree, input, verbose, recover_indices);
#endif
      },
      "simplextree"_a,
      "point_cloud"_a,
      "function_values"_a,
      "verbose"_a = false,
      "recover_indices"_a = false);
}
