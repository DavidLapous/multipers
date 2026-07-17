#include "graph_mph0/nanobind_interface.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ext_interface/contiguous_slicer_bridge.hpp"
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "graph_mph0/graph_mph0.h"
#include "nanobind_array_utils.hpp"

namespace nb = nanobind;

namespace mpnb {
namespace {

using multipers::nanobind_helpers::has_nonempty_filtration_grid;
using multipers::nanobind_helpers::is_slicer_object;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_helpers::visit_const_slicer_wrapper;
using multipers::nanobind_utils::owned_array;

struct Graph_mph0_input {
  multipers::graph_mph0::Graph graph;
  std::vector<multipers::graph_mph0::Grade> zero_relations;
};

template <class DimensionAt, class BoundaryAt, class GradeAt>
Graph_mph0_input build_graph_mph0_input(std::size_t num_generators,
                                        std::int32_t degree,
                                        DimensionAt&& dimension_at,
                                        BoundaryAt&& boundary_at,
                                        GradeAt&& grade_at,
                                        bool collect_zero_relations = true) {
  if (degree < 0) throw std::invalid_argument("graph degree must be nonnegative");
  Graph_mph0_input out;
  std::vector<std::size_t> vertex_number(num_generators, std::numeric_limits<std::size_t>::max());
  std::size_t num_edges = 0;
  for (std::size_t generator = 0; generator < num_generators; ++generator) {
    const std::int32_t dimension = dimension_at(generator);
    if (dimension < 0) throw std::invalid_argument("graph generator dimensions must be nonnegative");
    if (dimension < degree || static_cast<std::int64_t>(dimension) > static_cast<std::int64_t>(degree) + 1) continue;
    const auto grade = grade_at(generator);
    if (!std::isfinite(grade[0]) || !std::isfinite(grade[1])) {
      throw std::invalid_argument("graph requires finite filtration values");
    }
    auto&& boundary = boundary_at(generator);
    if (dimension == degree) {
      if (!boundary.empty()) throw std::invalid_argument("Graph presentation generators must have empty boundaries");
      vertex_number[generator] = out.graph.vertices.size();
      out.graph.vertices.push_back(grade);
    } else if (boundary.empty()) {
      if (collect_zero_relations) out.zero_relations.push_back(grade);
    } else {
      ++num_edges;
    }
  }

  out.graph.edges.reserve(num_edges);
  for (std::size_t generator = 0; generator < num_generators; ++generator) {
    if (static_cast<std::int64_t>(dimension_at(generator)) != static_cast<std::int64_t>(degree) + 1) continue;
    auto&& boundary = boundary_at(generator);
    if (boundary.empty()) continue;
    if (boundary.size() != 2) {
      throw std::invalid_argument("Every nonempty graph relation must contain exactly two generators");
    }
    const std::size_t u = boundary[0];
    const std::size_t v = boundary[1];
    if (u >= num_generators || v >= num_generators) {
      throw std::invalid_argument("Graph relation endpoint is out of range");
    }
    if (dimension_at(u) != degree || dimension_at(v) != degree) {
      throw std::invalid_argument("Graph relations must reference generators in the requested degree");
    }
    if (u == v) throw std::invalid_argument("Graph relations must reference two distinct generators");
    const auto grade = grade_at(generator);
    const auto grade_u = grade_at(u);
    const auto grade_v = grade_at(v);
    if (grade[0] < grade_u[0] || grade[1] < grade_u[1] || grade[0] < grade_v[0] || grade[1] < grade_v[1]) {
      throw std::invalid_argument("Graph relation grade must dominate both endpoint grades");
    }
    out.graph.edges.push_back({out.graph.edges.size(), vertex_number[u], vertex_number[v], grade});
  }
  return out;
}

template <typename Desc>
inline constexpr bool is_contiguous_f64_graph_slicer_v =
    std::is_same_v<typename Desc::value_type, double> && Desc::is_vine && !Desc::is_kcritical &&
    !Desc::is_degree_rips && Desc::column_type == std::string_view("UNORDERED_SET") &&
    Desc::backend_type == std::string_view("Graph") && Desc::filtration_container == std::string_view("Contiguous");

template <typename List>
struct contiguous_f64_graph_slicer_desc_impl;

template <>
struct contiguous_f64_graph_slicer_desc_impl<type_list<>> {
  using type = void;
  static constexpr bool found = false;
  static constexpr int matches = 0;
};

template <typename Head, typename... Tail>
struct contiguous_f64_graph_slicer_desc_impl<type_list<Head, Tail...>> {
  using tail = contiguous_f64_graph_slicer_desc_impl<type_list<Tail...>>;
  static constexpr bool is_match = is_contiguous_f64_graph_slicer_v<Head>;
  static constexpr bool found = is_match || tail::found;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
  using type = std::conditional_t<is_match, Head, typename tail::type>;
};

using ContiguousF64GraphSlicerDesc = typename contiguous_f64_graph_slicer_desc_impl<SlicerDescriptorList>::type;
using ContiguousF64MatrixSlicerWrapper = multipers::nanobind_helpers::PySlicer<multipers::contiguous_f64_slicer>;

static_assert(!std::is_void_v<ContiguousF64GraphSlicerDesc>,
              "Expected exactly one one-critical contiguous float64 Graph slicer template.");
static_assert(contiguous_f64_graph_slicer_desc_impl<SlicerDescriptorList>::matches == 1,
              "One-critical contiguous float64 Graph slicer template must be unique.");

template <typename Wrapper>
nb::object graph_mph0_slicer_output(multipers::contiguous_f64_complex&& complex, std::int32_t degree, bool is_minres) {
  nb::object out = nb::type<Wrapper>()();
  auto& wrapper = nb::cast<Wrapper&>(out);
  {
    nb::gil_scoped_release release;
    multipers::build_slicer_from_complex(wrapper.truc, complex);
  }
  multipers::nanobind_helpers::mark_slicer_minpres(wrapper, degree, is_minres);
  return out;
}

}  // namespace

nb::tuple graph_mph0_raw(nb::ndarray<nb::numpy, const std::uint64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
                         nb::ndarray<nb::numpy, const std::uint32_t, nb::ndim<1>, nb::c_contig> boundary_indices,
                         nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig> dimensions,
                         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades,
                         std::int32_t degree) {
  multipers::graph_mph0::Result result;
  {
    nb::gil_scoped_release release;
    const std::size_t num_generators = dimensions.shape(0);
    if (boundary_indptr.shape(0) != num_generators + 1 || grades.shape(0) != num_generators || grades.shape(1) != 2) {
      throw std::invalid_argument("graph CSR dimensions and grades must describe the same generators");
    }
    if (boundary_indptr(0) != 0 || boundary_indptr(num_generators) != boundary_indices.shape(0)) {
      throw std::invalid_argument("graph boundary indptr does not span boundary indices");
    }
    for (std::size_t generator = 0; generator < num_generators; ++generator) {
      if (boundary_indptr(generator) > boundary_indptr(generator + 1) ||
          boundary_indptr(generator + 1) > boundary_indices.shape(0)) {
        throw std::invalid_argument("graph boundary indptr must be nondecreasing and in range");
      }
    }
    auto input = build_graph_mph0_input(
        num_generators,
        degree,
        [&](std::size_t generator) { return dimensions(generator); },
        [&](std::size_t generator) {
          const auto start = boundary_indptr(generator);
          return std::span<const std::uint32_t>(boundary_indices.data() + start,
                                                boundary_indptr(generator + 1) - start);
        },
        [&](std::size_t generator) {
          return multipers::graph_mph0::Grade{grades(generator, 0), grades(generator, 1)};
        });
    result = multipers::graph_mph0::compute(input.graph);
    result.beta_0_h1.insert(result.beta_0_h1.end(), input.zero_relations.begin(), input.zero_relations.end());
    std::sort(result.beta_0_h1.begin(), result.beta_0_h1.end());
  }
  auto grades_array = [](std::vector<multipers::graph_mph0::Grade>&& grades) {
    const std::size_t rows = grades.size();
    std::vector<double> flat;
    flat.reserve(2 * rows);
    for (const auto& grade : grades) {
      flat.push_back(grade[0]);
      flat.push_back(grade[1]);
    }
    return owned_array<double>(std::move(flat), {rows, std::size_t(2)});
  };
  std::vector<std::int64_t> relations;
  relations.reserve(2 * result.relations.size());
  for (const auto& relation : result.relations) {
    relations.push_back(static_cast<std::int64_t>(relation[0]));
    relations.push_back(static_cast<std::int64_t>(relation[1]));
  }
  const std::size_t relation_count = result.relations.size();
  return nb::make_tuple(grades_array(std::move(result.beta_0)),
                        grades_array(std::move(result.beta_1)),
                        grades_array(std::move(result.beta_2)),
                        grades_array(std::move(result.beta_0_h1)),
                        owned_array<std::int64_t>(std::move(relations), {relation_count, std::size_t(2)}));
}

nb::object graph_mph0_minimal_presentation(const nb::handle& slicer, std::int32_t degree, bool full_resolution) {
  if (!is_slicer_object(slicer)) throw nb::type_error("graph expects a Slicer input");
  const std::int32_t dimension_margin = full_resolution ? 2 : 1;
  if (degree < 0 || degree > std::numeric_limits<std::int32_t>::max() - dimension_margin) {
    throw std::invalid_argument("graph degree exceeds output dimension range");
  }
  return visit_const_slicer_wrapper(slicer, [&]<typename Desc>(const auto& wrapper) -> nb::object {
    if constexpr (Desc::is_kcritical) {
      throw std::invalid_argument("graph requires a one-critical filtration");
    } else {
      if (has_nonempty_filtration_grid(wrapper.filtration_grid)) {
        throw std::invalid_argument("graph expects unsqueezed filtration coordinates");
      }
      if (wrapper.truc.get_number_of_parameters() != 2) {
        throw std::invalid_argument("graph requires exactly two filtration parameters");
      }

      auto complex = [&] {
        nb::gil_scoped_release release;
        const auto& dimensions = wrapper.truc.get_dimensions();
        const auto& boundaries = wrapper.truc.get_boundaries();
        const auto& filtrations = wrapper.truc.get_filtration_values();
        auto input = build_graph_mph0_input(
            dimensions.size(),
            degree,
            [&](std::size_t generator) { return static_cast<std::int32_t>(dimensions[generator]); },
            [&](std::size_t generator) -> const auto& { return boundaries[generator]; },
            [&](std::size_t generator) {
              return multipers::graph_mph0::Grade{static_cast<double>(filtrations[generator](0, 0)),
                                                  static_cast<double>(filtrations[generator](0, 1))};
            },
            false);
        auto result =
            multipers::graph_mph0::compute(input.graph, multipers::graph_mph0::Compute_options{full_resolution, false});

        constexpr std::size_t max_output_generators = std::numeric_limits<std::uint32_t>::max();
        if (result.beta_0.size() > max_output_generators ||
            result.beta_1.size() > max_output_generators - result.beta_0.size() ||
            result.beta_2.size() > max_output_generators - result.beta_0.size() - result.beta_1.size()) {
          throw std::overflow_error("graph output exceeds uint32 generator capacity");
        }
        const std::size_t num_generators = result.beta_0.size() + result.beta_1.size() + result.beta_2.size();
        std::vector<double> grades;
        std::vector<std::vector<std::uint32_t>> output_boundaries;
        std::vector<int> output_dimensions;
        grades.reserve(2 * num_generators);
        output_boundaries.resize(result.beta_0.size());
        output_boundaries.reserve(num_generators);
        output_dimensions.reserve(num_generators);
        auto append_grades = [&](const auto& values, int dimension) {
          for (const auto& grade : values) {
            grades.push_back(grade[0]);
            grades.push_back(grade[1]);
            output_dimensions.push_back(dimension);
          }
        };
        append_grades(result.beta_0, degree);
        append_grades(result.beta_1, degree + 1);
        for (const auto& relation : result.relations) {
          if (relation[0] >= result.beta_0.size() || relation[1] >= result.beta_0.size()) {
            throw std::logic_error("graph relation endpoint is out of range");
          }
          output_boundaries.push_back(
              {static_cast<std::uint32_t>(relation[0]), static_cast<std::uint32_t>(relation[1])});
        }
        append_grades(result.beta_2, degree + 2);
        if (result.beta_2.size() != result.syzygies.size()) {
          throw std::logic_error("graph syzygy count does not match beta_2");
        }
        for (const auto& syzygy : result.syzygies) {
          auto& boundary = output_boundaries.emplace_back();
          boundary.reserve(syzygy.size());
          for (const std::size_t relation : syzygy) {
            if (relation >= result.beta_1.size() || relation > max_output_generators - result.beta_0.size()) {
              throw std::overflow_error("graph syzygy boundary index is out of range");
            }
            boundary.push_back(static_cast<std::uint32_t>(result.beta_0.size() + relation));
          }
        }
        return multipers::build_contiguous_f64_slicer_from_output(
            grades, std::size_t(2), output_boundaries, output_dimensions);
      }();

      if (full_resolution) {
        return graph_mph0_slicer_output<ContiguousF64MatrixSlicerWrapper>(std::move(complex), degree, true);
      }
      return graph_mph0_slicer_output<typename ContiguousF64GraphSlicerDesc::wrapper>(
          std::move(complex), degree, false);
    }
  });
}

}  // namespace mpnb
