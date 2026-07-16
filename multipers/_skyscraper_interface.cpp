#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !MULTIPERS_DISABLE_SKYSCRAPER_INTERFACE
#include "ext_interface/nanobind_registry_runtime.hpp"
#include "nanobind_array_utils.hpp"
#include "skyscraper_core.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_SKYSCRAPER_INTERFACE
namespace {
using Wrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;
using F64Vector = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using U64Vector = nb::ndarray<nb::numpy, const std::uint64_t, nb::ndim<1>, nb::c_contig>;
using I64Vector = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>, nb::c_contig>;
using F64Matrix = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

enum MetadataField : std::uint16_t {
  MaxInducedRank = 1 << 0,
  RankCap = 1 << 1,
  CompleteSubspaceMode = 1 << 2,
  Algorithm = 1 << 3,
  Field = 1 << 4,
  CoordinateOrder = 1 << 5,
  BackendRevision = 1 << 6,
  SlopeTolerance = 1 << 7,
  SkyVersion = 1 << 8,
  GroupingLost = 1 << 9,
};

constexpr std::uint16_t computation_metadata = MaxInducedRank | RankCap | CompleteSubspaceMode | Algorithm | Field |
                                               CoordinateOrder | BackendRevision | SlopeTolerance;
constexpr const char* backend_revision = "cea2ef8fd7dcdba24bd3c53820b18287de1308fe";

struct SkyscraperInvariant {
  hnf::GridResult result;
  int degree = -1;
  std::string coordinates;
  std::string sky_version;
  bool grouping_lost = false;
  std::uint16_t metadata_fields = 0;
};

hnf::Presentation presentation_from_slicer(const Wrapper& wrapper) {
  if (wrapper.minpres_degree < 0 || wrapper.truc.get_number_of_parameters() != 2)
    throw std::invalid_argument("Each summand must be a one-critical 2D minimal presentation.");
  if (multipers::nanobind_helpers::has_nonempty_filtration_grid(wrapper.filtration_grid))
    throw std::invalid_argument("Summand coordinates must be unsqueezed physical coordinates.");
  auto block = multipers::nanobind_helpers::extract_bifiltration_minpres_degree_block(wrapper, wrapper.minpres_degree);
  const auto finite_grade = [](const auto& grade) { return std::isfinite(grade.first) && std::isfinite(grade.second); };
  if (!std::all_of(block.row_grades.begin(), block.row_grades.end(), finite_grade) ||
      !std::all_of(block.relation_grades.begin(), block.relation_grades.end(), finite_grade))
    throw std::invalid_argument("Summand grades must be finite.");
  auto boundaries = multipers::nanobind_helpers::localize_degree_block_relation_boundaries(block);
  hnf::Presentation out(static_cast<int>(block.relation_grades.size()), static_cast<int>(block.row_grades.size()));
  out.row_degrees = std::move(block.row_grades);
  out.col_degrees = std::move(block.relation_grades);
  out.data.reserve(boundaries.size());
  for (auto& boundary : boundaries) {
    std::sort(boundary.begin(), boundary.end());
    if (std::adjacent_find(boundary.begin(), boundary.end()) != boundary.end())
      throw std::invalid_argument("F2 relation boundaries must not contain duplicate indices.");
    out.data.push_back(std::move(boundary));
  }
  return out;
}

nb::object array(const nb::object& numpy, const nb::object& value, const char* dtype) {
  return numpy.attr("ascontiguousarray")(value, "dtype"_a = dtype);
}

std::vector<double> copy(const F64Vector& input) {
  return std::vector<double>(input.data(), input.data() + input.shape(0));
}

std::vector<std::size_t> copy(const U64Vector& input) {
  return std::vector<std::size_t>(input.data(), input.data() + input.shape(0));
}

std::vector<double> doubles(const nb::object& numpy, const nb::object& input, const char* name) {
  try {
    return copy(nb::cast<F64Vector>(array(numpy, input, "float64")));
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) + " must be 1D.");
  }
}

std::vector<std::size_t> indices(const nb::object& numpy, const nb::object& input, const char* name) {
  nb::object raw;
  try {
    raw = numpy.attr("asarray")(input);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(name) + " must be 1D.");
  }
  if (nb::cast<std::size_t>(raw.attr("ndim")) != 1) throw std::invalid_argument(std::string(name) + " must be 1D.");
  const auto kind = nb::cast<std::string>(raw.attr("dtype").attr("kind"));
  std::vector<std::size_t> out;
  if (kind == "u") {
    const auto values = nb::cast<U64Vector>(array(numpy, raw, "uint64"));
    out.reserve(values.shape(0));
    for (std::size_t i = 0; i < values.shape(0); ++i) {
      if (values.data()[i] > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(name) + " must contain nonnegative integers.");
      out.push_back(static_cast<std::size_t>(values.data()[i]));
    }
    return out;
  }
  if (kind == "i") {
    const auto values = nb::cast<I64Vector>(array(numpy, raw, "int64"));
    out.reserve(values.shape(0));
    for (std::size_t i = 0; i < values.shape(0); ++i) {
      if (values.data()[i] < 0 ||
          static_cast<std::uint64_t>(values.data()[i]) > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(name) + " must contain nonnegative integers.");
      out.push_back(static_cast<std::size_t>(values.data()[i]));
    }
    return out;
  }
  if (kind != "f" || nb::cast<std::size_t>(raw.attr("dtype").attr("itemsize")) > sizeof(double))
    throw std::invalid_argument(std::string(name) + " must contain nonnegative integers.");
  const auto values = nb::cast<F64Vector>(array(numpy, raw, "float64"));
  out.reserve(values.shape(0));
  const double size_limit = std::ldexp(1.0, std::numeric_limits<std::size_t>::digits);
  for (std::size_t i = 0; i < values.shape(0); ++i) {
    const double value = values.data()[i];
    if (!std::isfinite(value) || value < 0 || std::floor(value) != value || value >= size_limit)
      throw std::invalid_argument(std::string(name) + " must contain nonnegative integers.");
    out.push_back(static_cast<std::size_t>(value));
  }
  return out;
}

void validate(const hnf::GridResult& result) {
  const auto finite = [](double value) { return std::isfinite(value); };
  const auto valid_axis = [&](const std::vector<double>& axis) {
    return !axis.empty() && std::adjacent_find(axis.begin(), axis.end(), std::greater_equal<>()) == axis.end();
  };
  if (!std::all_of(result.x_grid.begin(), result.x_grid.end(), finite) ||
      !std::all_of(result.y_grid.begin(), result.y_grid.end(), finite))
    throw std::invalid_argument("Grids, box, slopes, and corners must be finite.");
  if (!valid_axis(result.x_grid) || !valid_axis(result.y_grid))
    throw std::invalid_argument("Grid axes must be nonempty and strictly increasing.");
  if (!finite(result.measure_lower.first) || !finite(result.measure_lower.second) ||
      !finite(result.measure_upper.first) || !finite(result.measure_upper.second) ||
      result.measure_lower.first >= result.measure_upper.first ||
      result.measure_lower.second >= result.measure_upper.second)
    throw std::invalid_argument("box must define a finite nonempty rectangle.");
  if (result.x_grid.size() > std::numeric_limits<std::size_t>::max() / result.y_grid.size() ||
      result.source_offsets.size() != result.x_grid.size() * result.y_grid.size() + 1)
    throw std::invalid_argument("source_offsets does not match grid shape.");
  const auto valid_offsets = [](const auto& offsets, std::size_t end) {
    return !offsets.empty() && offsets.front() == 0 && offsets.back() == end &&
           std::is_sorted(offsets.begin(), offsets.end());
  };
  if (!valid_offsets(result.source_offsets, result.slopes.size()))
    throw std::invalid_argument("Invalid source_offsets.");
  if (result.factor_ranks.size() != result.slopes.size() || result.factor_group_ids.size() != result.slopes.size())
    throw std::invalid_argument("Factor arrays must have equal lengths.");
  if (!valid_offsets(result.staircase_offsets, result.corner_offsets.size() - 1) ||
      result.staircase_offsets.size() != result.slopes.size() + 1)
    throw std::invalid_argument("Invalid staircase_offsets.");
  for (std::size_t i = 0; i < result.factor_ranks.size(); ++i)
    if (result.staircase_offsets[i + 1] - result.staircase_offsets[i] != result.factor_ranks[i])
      throw std::invalid_argument("Each factor must have one staircase per rank.");
  if (!valid_offsets(result.corner_offsets, result.corners.size()))
    throw std::invalid_argument("Invalid packed corners.");
  if (!std::all_of(result.slopes.begin(), result.slopes.end(), finite) ||
      !std::all_of(result.corners.begin(), result.corners.end(), [&](const auto& corner) {
        return finite(corner.first) && finite(corner.second);
      }))
    throw std::invalid_argument("Grids, box, slopes, and corners must be finite.");
}

hnf::GridResult result_from_arrays(const nb::object& x_grid,
                                   const nb::object& y_grid,
                                   const nb::object& box_object,
                                   const nb::object& source_offsets,
                                   const nb::object& slopes,
                                   const nb::object& factor_ranks,
                                   const nb::object& factor_group_ids,
                                   const nb::object& staircase_offsets,
                                   const nb::object& corner_offsets,
                                   const nb::object& corners_object) {
  auto numpy = nb::module_::import_("numpy");
  hnf::GridResult result;
  result.x_grid = doubles(numpy, x_grid, "x_grid");
  result.y_grid = doubles(numpy, y_grid, "y_grid");
  const auto box_array = array(numpy, box_object, "float64");
  if (nb::cast<std::size_t>(box_array.attr("ndim")) != 2) throw std::invalid_argument("box must have shape (2, 2)");
  const auto box = nb::cast<F64Matrix>(box_array);
  if (box.shape(0) != 2 || box.shape(1) != 2) throw std::invalid_argument("box must have shape (2, 2)");
  result.measure_lower = {box.data()[0], box.data()[1]};
  result.measure_upper = {box.data()[2], box.data()[3]};
  result.source_offsets = indices(numpy, source_offsets, "source_offsets");
  result.slopes = doubles(numpy, slopes, "slopes");
  result.factor_ranks = indices(numpy, factor_ranks, "factor_ranks");
  result.factor_group_ids = indices(numpy, factor_group_ids, "factor_group_ids");
  result.staircase_offsets = indices(numpy, staircase_offsets, "staircase_offsets");
  result.corner_offsets = indices(numpy, corner_offsets, "corner_offsets");
  const auto corners_array = array(numpy, corners_object, "float64");
  if (nb::cast<std::size_t>(corners_array.attr("ndim")) != 2)
    throw std::invalid_argument("corners must have shape (n, 2)");
  const auto corners = nb::cast<F64Matrix>(corners_array);
  if (corners.shape(1) != 2) throw std::invalid_argument("corners must have shape (n, 2)");
  result.corners.reserve(corners.shape(0));
  for (std::size_t i = 0; i < corners.shape(0); ++i) {
    result.corners.emplace_back(corners.data()[2 * i], corners.data()[2 * i + 1]);
  }
  validate(result);
  return result;
}

nb::object vector_array(const std::vector<double>& values) {
  return nb::cast(multipers::nanobind_utils::owned_array(std::vector<double>(values), {values.size()}));
}

nb::object vector_array(const std::vector<std::size_t>& values) {
  std::vector<std::uint64_t> copied(values.begin(), values.end());
  return nb::cast(multipers::nanobind_utils::owned_array(std::move(copied), {values.size()}));
}

nb::object box_array(const hnf::GridResult& result) {
  return vector_array(std::vector<double>{result.measure_lower.first,
                                          result.measure_lower.second,
                                          result.measure_upper.first,
                                          result.measure_upper.second})
      .attr("reshape")(2, 2);
}

nb::object corners_array(const hnf::GridResult& result) {
  std::vector<double> copied;
  copied.reserve(result.corners.size() * 2);
  for (const auto& corner : result.corners) {
    copied.push_back(corner.first);
    copied.push_back(corner.second);
  }
  return nb::cast(multipers::nanobind_utils::owned_array(std::move(copied), {result.corners.size(), std::size_t{2}}));
}

nb::dict metadata(const SkyscraperInvariant& invariant) {
  const auto& result = invariant.result;
  nb::dict out;
  if (invariant.metadata_fields & MaxInducedRank) out["max_induced_rank"] = result.max_induced_rank;
  if (invariant.metadata_fields & RankCap) out["rank_cap"] = result.rank_cap;
  if (invariant.metadata_fields & CompleteSubspaceMode) out["complete_subspace_mode"] = result.complete_subspace_mode;
  if (invariant.metadata_fields & Algorithm) out["algorithm"] = result.algorithm;
  if (invariant.metadata_fields & Field) out["field"] = result.field;
  if (invariant.metadata_fields & CoordinateOrder) out["coordinate_order"] = result.coordinate_order;
  if (invariant.metadata_fields & BackendRevision) out["backend_revision"] = result.backend_revision;
  if (invariant.metadata_fields & SlopeTolerance)
    out["slope_tie_relative_tolerance"] = result.slope_tie_relative_tolerance;
  if (invariant.degree >= 0) out["degree"] = invariant.degree;
  if (!invariant.coordinates.empty()) out["coordinates"] = invariant.coordinates;
  if (invariant.metadata_fields & SkyVersion) out["sky_version"] = invariant.sky_version;
  if (invariant.metadata_fields & GroupingLost) out["grouping_lost"] = invariant.grouping_lost;
  return out;
}

std::string path_string(const nb::object& path) {
  return nb::cast<std::string>(nb::module_::import_("os").attr("fspath")(path));
}

void apply_metadata(SkyscraperInvariant& invariant, const nb::dict& input) {
  const auto get = [&](const char* key) -> nb::handle {
    const nb::str name(key);
    return input.contains(name) ? input[name] : nb::handle();
  };
  if (auto value = get("degree"); value.is_valid()) invariant.degree = nb::cast<int>(value);
  if (auto value = get("coordinates"); value.is_valid()) invariant.coordinates = nb::cast<std::string>(value);
  const std::vector<std::string> known{"degree",
                                       "coordinates",
                                       "max_induced_rank",
                                       "rank_cap",
                                       "complete_subspace_mode",
                                       "algorithm",
                                       "field",
                                       "coordinate_order",
                                       "backend_revision",
                                       "slope_tie_relative_tolerance",
                                       "sky_version",
                                       "grouping_lost"};
  for (const auto& [key, value] : input)
    if (std::find(known.begin(), known.end(), nb::cast<std::string>(key)) == known.end())
      throw std::invalid_argument("Unknown Skyscraper metadata field.");
  if (auto value = get("max_induced_rank"); value.is_valid()) {
    invariant.result.max_induced_rank = nb::cast<std::size_t>(value);
    invariant.metadata_fields |= MaxInducedRank;
  }
  if (auto value = get("rank_cap"); value.is_valid()) {
    invariant.result.rank_cap = nb::cast<std::size_t>(value);
    invariant.metadata_fields |= RankCap;
  }
  if (auto value = get("complete_subspace_mode"); value.is_valid()) {
    invariant.result.complete_subspace_mode = nb::cast<bool>(value);
    invariant.metadata_fields |= CompleteSubspaceMode;
  }
  if (auto value = get("algorithm"); value.is_valid()) {
    invariant.result.algorithm = nb::cast<std::string>(value);
    invariant.metadata_fields |= Algorithm;
  }
  if (auto value = get("field"); value.is_valid()) {
    invariant.result.field = nb::cast<std::string>(value);
    invariant.metadata_fields |= Field;
  }
  if (auto value = get("coordinate_order"); value.is_valid()) {
    invariant.result.coordinate_order = nb::cast<std::string>(value);
    invariant.metadata_fields |= CoordinateOrder;
  }
  if (auto value = get("backend_revision"); value.is_valid()) {
    invariant.result.backend_revision = nb::cast<std::string>(value);
    invariant.metadata_fields |= BackendRevision;
  }
  if (auto value = get("slope_tie_relative_tolerance"); value.is_valid()) {
    invariant.result.slope_tie_relative_tolerance = nb::cast<double>(value);
    invariant.metadata_fields |= SlopeTolerance;
  }
  if (auto value = get("sky_version"); value.is_valid()) {
    invariant.sky_version = nb::cast<std::string>(value);
    invariant.metadata_fields |= SkyVersion;
  }
  if (auto value = get("grouping_lost"); value.is_valid()) {
    invariant.grouping_lost = nb::cast<bool>(value);
    invariant.metadata_fields |= GroupingLost;
  }
}

std::size_t cell_at(const hnf::GridResult& result, double x, double y) {
  if (!std::isfinite(x) || !std::isfinite(y)) throw std::invalid_argument("Coordinates must be finite.");
  const auto x_it = std::upper_bound(result.x_grid.begin(), result.x_grid.end(), x);
  const auto y_it = std::upper_bound(result.y_grid.begin(), result.y_grid.end(), y);
  if (x_it == result.x_grid.begin() || y_it == result.y_grid.begin()) return std::numeric_limits<std::size_t>::max();
  return static_cast<std::size_t>(y_it - result.y_grid.begin() - 1) * result.x_grid.size() +
         static_cast<std::size_t>(x_it - result.x_grid.begin() - 1);
}

std::pair<double, double> coordinate(const std::vector<double>& value, const char* name) {
  if (value.size() != 2 || !std::isfinite(value[0]) || !std::isfinite(value[1]))
    throw std::invalid_argument(std::string(name) + " must be a finite 2D coordinate.");
  return {value[0], value[1]};
}

std::size_t filtered_rank_in_cell(const hnf::GridResult& result,
                                  double theta,
                                  std::size_t cell,
                                  const std::pair<double, double>& target) {
  std::size_t total = 0;
  for (std::size_t factor = result.source_offsets[cell]; factor < result.source_offsets[cell + 1]; ++factor) {
    if (result.slopes[factor] < theta) continue;
    for (std::size_t piece = result.staircase_offsets[factor]; piece < result.staircase_offsets[factor + 1]; ++piece) {
      bool blocked = false;
      for (std::size_t corner = result.corner_offsets[piece]; corner < result.corner_offsets[piece + 1]; ++corner)
        blocked |= result.corners[corner].first <= target.first && result.corners[corner].second <= target.second;
      total += !blocked;
    }
  }
  return total;
}

std::size_t filtered_rank(const SkyscraperInvariant& invariant,
                          double theta,
                          const std::vector<double>& source_value,
                          const std::vector<double>& target_value) {
  if (!std::isfinite(theta)) throw std::invalid_argument("theta must be finite.");
  const auto source = coordinate(source_value, "source");
  const auto target = coordinate(target_value, "target");
  if (target.first < source.first || target.second < source.second) return 0;
  const auto& result = invariant.result;
  const auto cell = cell_at(result, source.first, source.second);
  if (cell == std::numeric_limits<std::size_t>::max()) return 0;
  return filtered_rank_in_cell(result, theta, cell, target);
}

nb::object landscape(const SkyscraperInvariant& invariant, double theta, std::size_t levels) {
  std::vector<double> values;
  {
    nb::gil_scoped_release release;
    values = hnf::filtered_landscape(invariant.result, theta, levels);
  }
  return nb::cast(multipers::nanobind_utils::owned_array(
      std::move(values), {levels, invariant.result.y_grid.size(), invariant.result.x_grid.size()}));
}
}  // namespace
#endif

NB_MODULE(_skyscraper_interface, m) {
#if MULTIPERS_DISABLE_SKYSCRAPER_INTERFACE
  m.def("available", [] { return false; });
  m.def("require", [] { throw std::runtime_error("Skyscraper interface is unavailable in this build."); });
  m.def("fixed_grid",
        [](nb::args, nb::kwargs) { throw std::runtime_error("Skyscraper interface is unavailable in this build."); });
#else
  m.def("available", [] { return true; });
  m.def("require", [] {});

  nb::class_<SkyscraperInvariant>(m, "SkyscraperInvariant")
      .def(
          "__init__",
          [](SkyscraperInvariant* self,
             const nb::object& x_grid,
             const nb::object& y_grid,
             const nb::object& box,
             const nb::object& source_offsets,
             const nb::object& slopes,
             const nb::object& factor_ranks,
             const nb::object& factor_group_ids,
             const nb::object& staircase_offsets,
             const nb::object& corner_offsets,
             const nb::object& corners,
             const nb::dict& input_metadata) {
            SkyscraperInvariant value{result_from_arrays(x_grid,
                                                         y_grid,
                                                         box,
                                                         source_offsets,
                                                         slopes,
                                                         factor_ranks,
                                                         factor_group_ids,
                                                         staircase_offsets,
                                                         corner_offsets,
                                                         corners)};
            apply_metadata(value, input_metadata);
            new (self) SkyscraperInvariant(std::move(value));
          },
          "x_grid"_a,
          "y_grid"_a,
          "box"_a,
          "source_offsets"_a,
          "slopes"_a,
          "factor_ranks"_a,
          "factor_group_ids"_a,
          "staircase_offsets"_a,
          "corner_offsets"_a,
          "corners"_a,
          "metadata"_a = nb::dict())
      .def_prop_ro("x_grid", [](const SkyscraperInvariant& self) { return vector_array(self.result.x_grid); })
      .def_prop_ro("y_grid", [](const SkyscraperInvariant& self) { return vector_array(self.result.y_grid); })
      .def_prop_ro("box", [](const SkyscraperInvariant& self) { return box_array(self.result); })
      .def_prop_ro("source_offsets",
                   [](const SkyscraperInvariant& self) { return vector_array(self.result.source_offsets); })
      .def_prop_ro("slopes", [](const SkyscraperInvariant& self) { return vector_array(self.result.slopes); })
      .def_prop_ro("factor_ranks",
                   [](const SkyscraperInvariant& self) { return vector_array(self.result.factor_ranks); })
      .def_prop_ro("factor_group_ids",
                   [](const SkyscraperInvariant& self) { return vector_array(self.result.factor_group_ids); })
      .def_prop_ro("staircase_offsets",
                   [](const SkyscraperInvariant& self) { return vector_array(self.result.staircase_offsets); })
      .def_prop_ro("corner_offsets",
                   [](const SkyscraperInvariant& self) { return vector_array(self.result.corner_offsets); })
      .def_prop_ro("corners", [](const SkyscraperInvariant& self) { return corners_array(self.result); })
      .def_prop_ro("metadata", &metadata)
      .def(
          "slopes_at",
          [](const SkyscraperInvariant& self, double x, double y) {
            const auto cell = cell_at(self.result, x, y);
            std::vector<double> values;
            if (cell != std::numeric_limits<std::size_t>::max())
              values.assign(self.result.slopes.begin() + self.result.source_offsets[cell],
                            self.result.slopes.begin() + self.result.source_offsets[cell + 1]);
            return vector_array(values);
          },
          "x"_a,
          "y"_a)
      .def(
          "at",
          [](const SkyscraperInvariant& self, double x, double y) {
            const auto cell = cell_at(self.result, x, y);
            const std::size_t begin =
                cell == std::numeric_limits<std::size_t>::max() ? 0 : self.result.source_offsets[cell];
            const std::size_t end =
                cell == std::numeric_limits<std::size_t>::max() ? 0 : self.result.source_offsets[cell + 1];
            nb::dict out;
            out["slopes"] =
                vector_array(std::vector<double>(self.result.slopes.begin() + begin, self.result.slopes.begin() + end));
            out["factor_ranks"] = vector_array(std::vector<std::size_t>(self.result.factor_ranks.begin() + begin,
                                                                        self.result.factor_ranks.begin() + end));
            out["factor_group_ids"] = vector_array(std::vector<std::size_t>(
                self.result.factor_group_ids.begin() + begin, self.result.factor_group_ids.begin() + end));
            nb::list factors;
            for (std::size_t factor = begin; factor < end; ++factor) {
              nb::list pieces;
              for (std::size_t piece = self.result.staircase_offsets[factor];
                   piece < self.result.staircase_offsets[factor + 1];
                   ++piece) {
                std::vector<hnf::Degree> corners(self.result.corners.begin() + self.result.corner_offsets[piece],
                                                 self.result.corners.begin() + self.result.corner_offsets[piece + 1]);
                hnf::GridResult piece_result;
                piece_result.corners = std::move(corners);
                pieces.append(corners_array(piece_result));
              }
              factors.append(nb::tuple(pieces));
            }
            out["staircases"] = nb::tuple(factors);
            return out;
          },
          "x"_a,
          "y"_a)
      .def("filtered_rank", &filtered_rank, "theta"_a, "source"_a, "target"_a)
      .def(
          "filtered_rank_on_grid",
          [](const SkyscraperInvariant& self, double theta, const std::vector<double>& target_value) {
            if (!std::isfinite(theta)) throw std::invalid_argument("theta must be finite.");
            const auto target = coordinate(target_value, "target");
            std::vector<std::uint64_t> ranks(self.result.x_grid.size() * self.result.y_grid.size());
            {
              nb::gil_scoped_release release;
              for (std::size_t iy = 0; iy < self.result.y_grid.size(); ++iy)
                for (std::size_t ix = 0; ix < self.result.x_grid.size(); ++ix)
                  if (self.result.x_grid[ix] <= target.first && self.result.y_grid[iy] <= target.second) {
                    const auto cell = iy * self.result.x_grid.size() + ix;
                    ranks[cell] = filtered_rank_in_cell(self.result, theta, cell, target);
                  }
            }
            return nb::cast(multipers::nanobind_utils::owned_array(
                std::move(ranks), {self.result.y_grid.size(), self.result.x_grid.size()}));
          },
          "theta"_a,
          "target"_a)
      .def("filtered_landscape", &landscape, "theta"_a, "k"_a = 1)
      .def("reference_landscape", &landscape, "theta"_a, "k"_a = 1)
      .def(
          "filtered_landscape_difference",
          [](const SkyscraperInvariant& self, double theta, double theta_prime, std::size_t k) {
            if (theta_prime > theta) throw std::invalid_argument("theta_prime must not exceed theta.");
            auto numpy = nb::module_::import_("numpy");
            return numpy.attr("subtract")(landscape(self, theta_prime, k), landscape(self, theta, k));
          },
          "theta"_a,
          "theta_prime"_a,
          "k"_a = 1)
      .def(
          "to_sky",
          [](const SkyscraperInvariant& self,
             const nb::object& path,
             const std::string& version,
             const std::string& orientation) {
            const auto filename = path_string(path);
            nb::gil_scoped_release release;
            hnf::write_sky(self.result, filename, version, orientation);
          },
          "path"_a,
          "version"_a = "HNF1",
          "orientation"_a = "xy")
      .def_static(
          "from_sky",
          [](const nb::object& path, const std::string& orientation) {
            const auto filename = path_string(path);
            hnf::SkyFile file;
            {
              nb::gil_scoped_release release;
              file = hnf::read_sky(filename, orientation);
            }
            SkyscraperInvariant invariant;
            invariant.result = std::move(file.result);
            invariant.sky_version = std::move(file.version);
            invariant.grouping_lost = file.grouping_lost;
            invariant.metadata_fields = SkyVersion | GroupingLost;
            return invariant;
          },
          "path"_a,
          "orientation"_a);

  m.def(
      "fixed_grid",
      [](const std::vector<nb::object>& summands,
         const std::vector<double>& x_grid,
         const std::vector<double>& y_grid,
         const F64Matrix& box,
         std::size_t max_rank,
         int degree) {
        if (box.shape(0) != 2 || box.shape(1) != 2) throw std::invalid_argument("box must have shape (2, 2).");
        const std::pair<double, double> lower{box.data()[0], box.data()[1]};
        const std::pair<double, double> upper{box.data()[2], box.data()[3]};
        const auto valid_axis = [](const std::vector<double>& axis) {
          return !axis.empty() &&
                 std::all_of(axis.begin(), axis.end(), [](double value) { return std::isfinite(value); }) &&
                 std::adjacent_find(axis.begin(), axis.end(), std::greater_equal<>()) == axis.end();
        };
        if (!valid_axis(x_grid) || !valid_axis(y_grid))
          throw std::invalid_argument("Grid axes must be nonempty and strictly increasing.");
        if (!std::isfinite(lower.first) || !std::isfinite(lower.second) || !std::isfinite(upper.first) ||
            !std::isfinite(upper.second) || lower.first >= upper.first || lower.second >= upper.second)
          throw std::invalid_argument("box must define a finite nonempty rectangle.");
        if (max_rank < 1) throw std::invalid_argument("max_rank must be positive.");
        std::vector<hnf::Presentation> input;
        input.reserve(summands.size());
        for (const auto& summand : summands) {
          auto canonical = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(summand);
          const auto& wrapper = nb::cast<const Wrapper&>(canonical);
          if (degree >= 0 && wrapper.minpres_degree != degree)
            throw std::invalid_argument("Summand degree does not match requested degree.");
          input.push_back(presentation_from_slicer(wrapper));
        }
        hnf::GridResult result;
        {
          nb::gil_scoped_release release;
          result = hnf::fixed_grid_invariant(input, x_grid, y_grid, lower, upper, max_rank);
        }
        result.backend_revision = backend_revision;
        validate(result);
        SkyscraperInvariant out;
        out.result = std::move(result);
        out.degree = degree;
        out.coordinates = "physical";
        out.metadata_fields = computation_metadata;
        return out;
      },
      "summand_slicers"_a,
      "x_grid"_a,
      "y_grid"_a,
      "box"_a,
      "max_rank"_a = 7,
      "degree"_a = -1);
  m.def("filtered_landscape", &landscape, "invariant"_a, "theta"_a, "levels"_a);
  m.def(
      "write_sky",
      [](const SkyscraperInvariant& invariant,
         const nb::object& path,
         const std::string& version,
         const std::string& orientation) {
        const auto filename = path_string(path);
        nb::gil_scoped_release release;
        hnf::write_sky(invariant.result, filename, version, orientation);
      },
      "invariant"_a,
      "filename"_a,
      "version"_a = "HNF1",
      "orientation"_a = "xy");
  m.def(
      "read_sky",
      [](const nb::object& path, const std::string& orientation) {
        const auto filename = path_string(path);
        hnf::SkyFile file;
        {
          nb::gil_scoped_release release;
          file = hnf::read_sky(filename, orientation);
        }
        SkyscraperInvariant out;
        out.result = std::move(file.result);
        out.sky_version = std::move(file.version);
        out.grouping_lost = file.grouping_lost;
        out.metadata_fields = SkyVersion | GroupingLost;
        return out;
      },
      "filename"_a,
      "orientation"_a);
#endif
}
