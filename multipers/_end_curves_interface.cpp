#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ext_interface/aida_interface.hpp"
#include "ext_interface/persistence_algebra_interface.hpp"

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
namespace mpendcurves {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;
using index_point = std::array<std::int64_t, 2>;
using index_curve = std::vector<index_point>;
using index_curves = std::vector<std::vector<std::pair<std::int64_t, std::int64_t>>>;

inline std::int64_t rounded_index(double value, std::size_t axis) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("Expected finite squeezed end-curve coordinate.");
  }
  const double rounded = std::round(value);
  if (std::fabs(value - rounded) > 1e-9) {
    throw std::runtime_error("Expected integer squeezed end-curve coordinate for parameter " + std::to_string(axis) +
                             ".");
  }
  return static_cast<std::int64_t>(rounded);
}

inline index_point point_from_degree(const std::pair<double, double>& degree) {
  return {rounded_index(degree.first, 0), rounded_index(degree.second, 1)};
}

inline void sort_unique_vertices(index_curve& points) {
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
}

inline void sort_spread_curve(index_curve& points) {
  std::sort(points.begin(), points.end(), [](const index_point& a, const index_point& b) {
    if (a[0] != b[0]) {
      return a[0] < b[0];
    }
    return a[1] > b[1];
  });
}

template <typename Summand>
index_curve spread_support_vertices(const Summand& summand,
                                    const std::array<std::int64_t, 2>& inf_indices,
                                    bool include_infinite) {
  std::vector<index_point> rows;
  rows.reserve(summand.row_degrees.size());
  for (const auto& degree : summand.row_degrees) {
    rows.push_back(point_from_degree(degree));
  }

  index_curve points = rows;
  std::vector<std::array<bool, 2>> terminated(rows.size(), {false, false});

  for (std::size_t relation_id = 0; relation_id < summand.matrix.size(); ++relation_id) {
    const auto& relation = summand.matrix[relation_id];
    if (relation.empty()) {
      continue;
    }
    const index_point relation_degree = point_from_degree(summand.col_degrees[relation_id]);
    if (relation.size() == 1) {
      const int generator = relation.front();
      if (generator < 0 || static_cast<std::size_t>(generator) >= rows.size()) {
        throw std::runtime_error("AIDA summand relation references an invalid generator.");
      }
      const auto& row = rows[static_cast<std::size_t>(generator)];
      int active_axis = -1;
      int active_count = 0;
      for (int axis = 0; axis < 2; ++axis) {
        if (relation_degree[axis] > row[axis]) {
          active_axis = axis;
          ++active_count;
        }
      }
      if (active_count == 1) {
        auto vertex = relation_degree;
        --vertex[active_axis];
        points.push_back(vertex);
        terminated[static_cast<std::size_t>(generator)][active_axis] = true;
      }
      continue;
    }

    index_point join = rows[static_cast<std::size_t>(relation.front())];
    for (const int generator : relation) {
      if (generator < 0 || static_cast<std::size_t>(generator) >= rows.size()) {
        throw std::runtime_error("AIDA summand relation references an invalid generator.");
      }
      const auto& row = rows[static_cast<std::size_t>(generator)];
      join[0] = std::max(join[0], row[0]);
      join[1] = std::max(join[1], row[1]);
    }
    points.push_back(join);
    for (const int generator : relation) {
      const auto& row = rows[static_cast<std::size_t>(generator)];
      int active_axis = -1;
      int active_count = 0;
      for (int axis = 0; axis < 2; ++axis) {
        if (join[axis] > row[axis]) {
          active_axis = axis;
          ++active_count;
        }
      }
      if (active_count == 1) {
        terminated[static_cast<std::size_t>(generator)][active_axis] = true;
      }
    }
  }

  const std::array<std::int64_t, 2> unbounded_limit = {
      include_infinite ? inf_indices[0] : std::max<std::int64_t>(inf_indices[0] - 1, 0),
      include_infinite ? inf_indices[1] : std::max<std::int64_t>(inf_indices[1] - 1, 0)};
  for (std::size_t generator = 0; generator < rows.size(); ++generator) {
    for (int axis = 0; axis < 2; ++axis) {
      if (!terminated[generator][axis]) {
        auto vertex = rows[generator];
        vertex[axis] = unbounded_limit[axis];
        points.push_back(vertex);
      }
    }
  }

  sort_unique_vertices(points);
  return points;
}

inline void append_curve(index_curves& out, index_curve curve, bool sort) {
  if (sort) {
    sort_spread_curve(curve);
  }
  if (curve.empty()) {
    return;
  }
  auto& py_curve = out.emplace_back();
  py_curve.reserve(curve.size());
  for (const auto& point : curve) {
    py_curve.emplace_back(point[0], point[1]);
  }
}

inline auto aida_decomposition(const CanonicalWrapper& wrapper,
                               int degree,
                               bool aida_sort,
                               bool verbose,
                               bool progress) {
  auto prepared = multipers::nanobind_helpers::extract_bifiltration_minpres_degree_block(wrapper, degree);
  aida::AIDA_functor functor;
  functor.config.show_info = verbose;
  functor.config.sort_output = false;
  functor.config.sort = aida_sort;
  functor.config.progress = progress;
  auto input =
      aida::multipers_interface_input<int>(prepared.relation_grades, prepared.row_grades, prepared.relation_boundaries);
  return functor.multipers_interface(input);
}

inline index_curves birth_curve_indices(const CanonicalWrapper& wrapper,
                                        const std::vector<std::int64_t>& inf_indices,
                                        bool include_infinite,
                                        bool sort,
                                        bool aida_sort,
                                        bool verbose,
                                        bool progress) {
  if (inf_indices.size() != 2) {
    throw std::invalid_argument("birth_curves expects two infinity sentinel indices.");
  }
  const int degree = wrapper.get_min_pres_degree();
  if (degree < 0) {
    throw std::runtime_error("birth_curves expects a minimal presentation.");
  }
  if (wrapper.get_number_of_parameters() != 2) {
    throw std::runtime_error("birth_curves is only compatible with 2-parameter minimal presentations.");
  }

  const auto output = aida_decomposition(wrapper, degree, aida_sort, verbose, progress);
  const std::array<std::int64_t, 2> sentinel = {inf_indices[0], inf_indices[1]};
  index_curves out;
  out.reserve(output.summands.size());
  for (const auto& summand : output.summands) {
    append_curve(out, spread_support_vertices(summand, sentinel, include_infinite), sort);
  }
  return out;
}

inline index_curve death_boundary_vertices(index_curve points,
                                           const std::array<std::int64_t, 2>& inf_indices,
                                           bool include_infinite) {
  if (points.empty()) {
    return points;
  }
  sort_spread_curve(points);
  const std::array<std::int64_t, 2> finite_limit = {std::max<std::int64_t>(inf_indices[0] - 1, 0),
                                                    std::max<std::int64_t>(inf_indices[1] - 1, 0)};
  index_curve out;

  auto bump = [&inf_indices](std::int64_t value, int axis) {
    return value >= inf_indices[axis] ? value : std::min<std::int64_t>(value + 1, inf_indices[axis]);
  };
  auto append = [&](index_point point) {
    point[0] = std::min(point[0], inf_indices[0]);
    point[1] = std::min(point[1], inf_indices[1]);
    if (!include_infinite) {
      point[0] = std::min(point[0], finite_limit[0]);
      point[1] = std::min(point[1], finite_limit[1]);
    }
    if (out.empty() || out.back() != point) {
      out.push_back(point);
    }
  };

  if (points.size() == 1) {
    const auto [x, y] = points.front();
    append({x, bump(y, 1)});
    append({bump(x, 0), bump(y, 1)});
    append({bump(x, 0), y});
    return out;
  }

  const std::size_t last_segment = points.size() - 2;
  for (std::size_t index = 0; index + 1 < points.size(); ++index) {
    const auto start = points[index];
    const auto stop = points[index + 1];
    const bool is_last = index == last_segment;
    if (start[1] == stop[1]) {
      if (index == 0) {
        append({start[0], bump(start[1], 1)});
      }
      append({bump(stop[0], 0), bump(start[1], 1)});
      if (is_last && stop[0] < inf_indices[0]) {
        append({bump(stop[0], 0), stop[1]});
      }
      continue;
    }
    if (start[0] == stop[0]) {
      if (index == 0 && start[1] < inf_indices[1]) {
        append({start[0], bump(start[1], 1)});
      }
      append({bump(start[0], 0), bump(start[1], 1)});
      append({bump(start[0], 0), is_last ? stop[1] : bump(stop[1], 1)});
      continue;
    }
    throw std::runtime_error("Death-curve support must be axis-aligned.");
  }
  return out;
}

#if !MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE
inline index_curves death_curve_indices(const CanonicalWrapper& wrapper,
                                        int degree,
                                        const std::vector<std::int64_t>& inf_indices,
                                        bool include_infinite,
                                        bool sort,
                                        bool aida_sort,
                                        bool verbose,
                                        bool progress) {
  if (inf_indices.size() != 2) {
    throw std::invalid_argument("death_curves expects two infinity sentinel indices.");
  }
  auto complex = multipers::persistence_algebra_death_curve_contiguous_interface(wrapper.get_slicer(), degree);
  CanonicalWrapper death_wrapper;
  multipers::build_slicer_from_complex(death_wrapper.get_slicer(), complex);
  death_wrapper.set_min_pres_degree(degree);
  death_wrapper.set_filtration_grid(wrapper.get_filtration_grid());
  if (aida_sort) {
    death_wrapper.sort_slicer_co_lexically();
  }

  const auto& dimensions = death_wrapper.get_slicer().get_dimensions();
  if (std::find(dimensions.begin(), dimensions.end(), degree) == dimensions.end()) {
    return {};
  }

  const auto output = aida_decomposition(death_wrapper, degree, aida_sort, verbose, progress);
  const std::array<std::int64_t, 2> sentinel = {inf_indices[0], inf_indices[1]};
  index_curves out;
  out.reserve(output.summands.size());
  for (const auto& summand : output.summands) {
    auto curve = death_boundary_vertices(spread_support_vertices(summand, sentinel, true), sentinel, include_infinite);
    append_curve(out, std::move(curve), sort);
  }
  return out;
}
#endif

}  // namespace mpendcurves
#endif

NB_MODULE(_end_curves_interface, m) {
  m.def("birth_available", []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
    return false;
#else
    return true;
#endif
  });
  m.def("death_available", []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE || MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE
    return false;
#else
    return multipers::persistence_algebra_interface_available();
#endif
  });
  m.def("require_birth", []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
    throw std::runtime_error("Birth curves require AIDA support.");
#endif
  });
  m.def("require_death", []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
    throw std::runtime_error("Death curves require AIDA support.");
#elif MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE
    throw std::runtime_error("Death curves require Persistence-Algebra support.");
#else
    if (!multipers::persistence_algebra_interface_available()) {
      throw std::runtime_error("Death curves require Persistence-Algebra support.");
    }
#endif
  });

  m.def(
      "birth_curve_indices",
      [](nb::object slicer,
         const std::vector<std::int64_t>& inf_indices,
         bool include_infinite,
         bool sort,
         bool aida_sort,
         bool verbose,
         bool progress) {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
        (void)slicer;
        (void)inf_indices;
        (void)include_infinite;
        (void)sort;
        (void)aida_sort;
        (void)verbose;
        (void)progress;
        throw std::runtime_error("Birth curves require AIDA support.");
#else
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        if (aida_sort) {
          nb::cast<mpendcurves::CanonicalWrapper&>(target).sort_slicer_co_lexically();
        }
        return mpendcurves::birth_curve_indices(nb::cast<const mpendcurves::CanonicalWrapper&>(target),
                                                inf_indices,
                                                include_infinite,
                                                sort,
                                                aida_sort,
                                                verbose,
                                                progress);
#endif
      },
      "slicer"_a,
      "inf_indices"_a,
      "include_infinite"_a = true,
      "sort"_a = true,
      "aida_sort"_a = true,
      "verbose"_a = false,
      "progress"_a = false);

  m.def(
      "death_curve_indices",
      [](nb::object slicer,
         int degree,
         const std::vector<std::int64_t>& inf_indices,
         bool include_infinite,
         bool sort,
         bool aida_sort,
         bool verbose,
         bool progress) {
#if MULTIPERS_DISABLE_AIDA_INTERFACE || MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE
        (void)slicer;
        (void)degree;
        (void)inf_indices;
        (void)include_infinite;
        (void)sort;
        (void)aida_sort;
        (void)verbose;
        (void)progress;
        throw std::runtime_error("Death curves require AIDA and Persistence-Algebra support.");
#else
        if (!multipers::persistence_algebra_interface_available()) {
          throw std::runtime_error("Death curves require Persistence-Algebra support.");
        }
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        return mpendcurves::death_curve_indices(nb::cast<const mpendcurves::CanonicalWrapper&>(target),
                                                degree,
                                                inf_indices,
                                                include_infinite,
                                                sort,
                                                aida_sort,
                                                verbose,
                                                progress);
#endif
      },
      "slicer"_a,
      "degree"_a,
      "inf_indices"_a,
      "include_infinite"_a = true,
      "sort"_a = true,
      "aida_sort"_a = true,
      "verbose"_a = false,
      "progress"_a = false);
}
