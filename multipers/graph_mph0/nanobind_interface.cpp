#include "graph_mph0/nanobind_interface.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <nanobind/ndarray.h>

#include "ext_interface/contiguous_slicer_bridge.hpp"
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "graph_mph0/graph_input.h"

namespace nb = nanobind;

namespace mpnb {
namespace {

using multipers::nanobind_helpers::is_slicer_object;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_helpers::visit_const_slicer_wrapper;

template <typename Desc, typename Value>
inline constexpr bool is_contiguous_graph_slicer_v =
    std::is_same_v<typename Desc::value_type, Value> && Desc::is_vine && !Desc::is_kcritical && !Desc::is_degree_rips &&
    Desc::column_type == std::string_view("UNORDERED_SET") && Desc::backend_type == std::string_view("Graph") &&
    Desc::filtration_container == std::string_view("Contiguous");

template <typename Value, typename List>
struct contiguous_graph_slicer_desc_impl;

template <typename Value>
struct contiguous_graph_slicer_desc_impl<Value, type_list<>> {
  using type = void;
  static constexpr bool found = false;
  static constexpr int matches = 0;
};

template <typename Value, typename Head, typename... Tail>
struct contiguous_graph_slicer_desc_impl<Value, type_list<Head, Tail...>> {
  using tail = contiguous_graph_slicer_desc_impl<Value, type_list<Tail...>>;
  static constexpr bool is_match = is_contiguous_graph_slicer_v<Head, Value>;
  static constexpr bool found = is_match || tail::found;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
  using type = std::conditional_t<is_match, Head, typename tail::type>;
};

using ContiguousF64GraphSlicerDesc = typename contiguous_graph_slicer_desc_impl<double, SlicerDescriptorList>::type;
using ContiguousI32GraphSlicerDesc =
    typename contiguous_graph_slicer_desc_impl<std::int32_t, SlicerDescriptorList>::type;
using ContiguousF64MatrixSlicerWrapper = multipers::nanobind_helpers::PySlicer<multipers::contiguous_f64_slicer>;
using ContiguousI32MatrixSlicerWrapper = multipers::nanobind_helpers::PySlicer<multipers::contiguous_i32_slicer>;

static_assert(!std::is_void_v<ContiguousF64GraphSlicerDesc>,
              "Expected exactly one one-critical contiguous float64 Graph slicer template.");
static_assert(contiguous_graph_slicer_desc_impl<double, SlicerDescriptorList>::matches == 1,
              "One-critical contiguous float64 Graph slicer template must be unique.");
static_assert(!std::is_void_v<ContiguousI32GraphSlicerDesc>,
              "Expected exactly one one-critical contiguous int32 Graph slicer template.");
static_assert(contiguous_graph_slicer_desc_impl<std::int32_t, SlicerDescriptorList>::matches == 1,
              "One-critical contiguous int32 Graph slicer template must be unique.");

template <typename Wrapper, typename Complex>
nb::object graph_mph0_slicer_output(Complex&& complex, std::int32_t degree, bool is_minres) {
  nb::object out = nb::type<Wrapper>()();
  auto& wrapper = nb::cast<Wrapper&>(out);
  {
    nb::gil_scoped_release release;
    multipers::build_slicer_from_complex(wrapper.get_slicer(), complex);
  }
  wrapper.set_min_pres_degree(degree, is_minres);
  return out;
}

}  // namespace

nb::object graph_mph0_minimal_presentation(const nb::handle& slicer,
                                           std::int32_t degree,
                                           bool full_resolution,
                                           const nb::handle& finite_grid_masks) {
  if (!is_slicer_object(slicer)) throw nb::type_error("graph expects a Slicer input");
  const std::int32_t dimension_margin = full_resolution ? 2 : 1;
  if (degree < 0 || degree > std::numeric_limits<std::int32_t>::max() - dimension_margin) {
    throw std::invalid_argument("graph degree exceeds output dimension range");
  }
  using GridMask = nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  std::array<GridMask, 2> masks;
  const bool indexed = !finite_grid_masks.is_none();
  if (indexed) {
    auto axes = nb::cast<nb::sequence>(finite_grid_masks);
    if (nb::len(axes) != 2) throw std::invalid_argument("graph requires exactly two filtration grid axes");
    for (std::size_t p = 0; p < masks.size(); ++p) masks[p] = nb::cast<GridMask>(axes[p], false);
  }
  return visit_const_slicer_wrapper(slicer, [&]<typename Desc>(const auto& wrapper) -> nb::object {
    if constexpr (Desc::is_degree_rips) {
      throw std::invalid_argument("graph requires explicit lifetime corners, not implicit Degree-Rips storage");
    } else {
      const auto num_parameters = wrapper.get_slicer().get_number_of_parameters();
      if (num_parameters != 2 && !(Desc::is_kcritical && indexed && num_parameters == 0)) {
        throw std::invalid_argument("graph requires exactly two filtration parameters");
      }

      using Output_value =
          std::conditional_t<std::is_same_v<typename Desc::value_type, std::int32_t>, std::int32_t, double>;
      auto complex = [&] {
        nb::gil_scoped_release release;
        const auto& dimensions = wrapper.get_slicer().get_dimensions();
        const auto& boundaries = wrapper.get_slicer().get_boundaries();
        const auto& filtrations = wrapper.get_slicer().get_filtration_values();
        if (boundaries.size() != dimensions.size() || filtrations.size() != dimensions.size()) {
          throw std::invalid_argument("graph requires one boundary and lifetime row per source cell");
        }
        auto input = multipers::graph_mph0::build_graph_mph0_input(
            dimensions.size(),
            degree,
            [&](std::size_t generator) { return static_cast<std::int32_t>(dimensions[generator]); },
            [&](std::size_t generator) -> const auto& { return boundaries[generator]; },
            [&](std::size_t generator) -> std::size_t {
              const auto& filtration = filtrations[generator];
              if constexpr (Desc::is_kcritical) {
                // Legacy construction encodes an empty MC lifetime by canonical +infinity.
                if (filtration.is_plus_inf()) return 0;
              }
              return filtration.num_generators();
            },
            [&](std::size_t generator, std::size_t corner) -> std::optional<multipers::graph_mph0::Grade> {
              if (filtrations[generator].num_parameters() != 2) {
                throw std::invalid_argument("graph requires exactly two filtration parameters");
              }
              multipers::graph_mph0::Grade grade{
                  multipers::graph_mph0::graph_grade_coordinate(filtrations[generator](corner, 0)),
                  multipers::graph_mph0::graph_grade_coordinate(filtrations[generator](corner, 1))};
              if (indexed) {
                for (std::size_t p = 0; p < masks.size(); ++p) {
                  if (!std::isfinite(grade[p]) || grade[p] != std::trunc(grade[p]) || grade[p] < 0 ||
                      grade[p] >= masks[p].shape(0) || !masks[p].data()[static_cast<std::size_t>(grade[p])]) {
                    throw std::invalid_argument("graph requires finite filtration values");
                  }
                }
              }
              return grade;
            },
            /*canonical_lifetimes=*/!Desc::is_kcritical);
        auto result =
            multipers::graph_mph0::compute(input, multipers::graph_mph0::Compute_options{full_resolution, false});

        constexpr std::size_t max_output_generators = std::numeric_limits<std::uint32_t>::max();
        if (result.beta_0.size() > max_output_generators ||
            result.beta_1.size() > max_output_generators - result.beta_0.size() ||
            result.beta_2.size() > max_output_generators - result.beta_0.size() - result.beta_1.size()) {
          throw std::overflow_error("graph output exceeds uint32 generator capacity");
        }
        const std::size_t num_generators = result.beta_0.size() + result.beta_1.size() + result.beta_2.size();
        std::vector<Output_value> grades;
        std::vector<std::vector<std::uint32_t>> output_boundaries;
        std::vector<int> output_dimensions;
        grades.reserve(2 * num_generators);
        output_boundaries.resize(result.beta_0.size());
        output_boundaries.reserve(num_generators);
        output_dimensions.reserve(num_generators);
        auto append_grades = [&](const auto& values, int dimension) {
          for (const auto& grade : values) {
            grades.push_back(static_cast<Output_value>(grade[0]));
            grades.push_back(static_cast<Output_value>(grade[1]));
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
        return multipers::build_contiguous_slicer_from_owned_output<Output_value>(
            grades, std::size_t(2), std::move(output_boundaries), std::move(output_dimensions));
      }();

      if constexpr (std::is_same_v<Output_value, std::int32_t>) {
        if (full_resolution) {
          return graph_mph0_slicer_output<ContiguousI32MatrixSlicerWrapper>(std::move(complex), degree, true);
        }
        return graph_mph0_slicer_output<typename ContiguousI32GraphSlicerDesc::interface>(
            std::move(complex), degree, false);
      } else {
        if (full_resolution) {
          return graph_mph0_slicer_output<ContiguousF64MatrixSlicerWrapper>(std::move(complex), degree, true);
        }
        return graph_mph0_slicer_output<typename ContiguousF64GraphSlicerDesc::interface>(
            std::move(complex), degree, false);
      }
    }
  });
}

}  // namespace mpnb
