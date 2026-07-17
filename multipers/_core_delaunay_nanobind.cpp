#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/number_utils.h>
#include <tbb/parallel_for.h>

#include "Simplex_tree_multi_interface.h"
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "nanobind_dense_array_utils.hpp"

#include <gudhi/Alpha_complex.h>
#include <gudhi/Alpha_complex_3d.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace mpcd {

using SafeKernel = CGAL::Epeck_d<CGAL::Dynamic_dimension_tag>;
using FastKernel = CGAL::Epick_d<CGAL::Dynamic_dimension_tag>;
using SearchKernel = CGAL::Epick_d<CGAL::Dynamic_dimension_tag>;
using AlphaTree = Gudhi::multiparameter::python_interface::interface_std;

using multipers::nanobind_dense_utils::cast_vector_from_array;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_simplextree_wrapper;

template <typename Kernel>
using AlphaComplex = Gudhi::alpha_complex::Alpha_complex<Kernel>;

template <Gudhi::alpha_complex::complexity Complexity>
using PeriodicAlphaComplex =
    Gudhi::alpha_complex::Alpha_complex_3d<Complexity, false, true>;

template <typename Kernel>
std::vector<typename AlphaComplex<Kernel>::Point_d> point_cloud_from_array(
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& points) {
  using Point = typename AlphaComplex<Kernel>::Point_d;

  const size_t num_points = points.shape(0);
  const size_t num_dimensions = points.shape(1);
  const double* data = points.data();

  std::vector<Point> point_cloud;
  point_cloud.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const double* row = data + i * num_dimensions;
    point_cloud.emplace_back(static_cast<int>(num_dimensions), row, row + num_dimensions);
  }
  return point_cloud;
}

template <typename Kernel>
AlphaTree build_alpha_tree(const std::vector<typename AlphaComplex<Kernel>::Point_d>& point_cloud,
                           double max_alpha_square,
                           bool exact) {
  AlphaComplex<Kernel> alpha_complex(point_cloud);
  AlphaTree alpha_tree;
  if (!alpha_complex.create_complex(alpha_tree, max_alpha_square, exact)) {
    throw std::runtime_error("Failed to build Gudhi alpha complex.");
  }
  return alpha_tree;
}

std::vector<double> compute_knn_selected(const std::vector<typename AlphaComplex<SearchKernel>::Point_d>& point_cloud,
                                         const std::vector<int64_t>& ks) {
  using SearchTraits = CGAL::Search_traits_d<SearchKernel>;
  using NeighborSearch = CGAL::Orthogonal_k_neighbor_search<SearchTraits>;
  using KdTree = typename NeighborSearch::Tree;

  const size_t num_points = point_cloud.size();
  const size_t num_ks = ks.size();
  const unsigned int max_k = static_cast<unsigned int>(ks.back());

  KdTree tree(point_cloud.begin(), point_cloud.end());
  tree.build();

  std::vector<double> out(num_points * num_ks, 0.0);
  tbb::parallel_for(size_t{0}, num_points, [&](size_t i) {
    NeighborSearch search(tree, point_cloud[i], max_k, 0.0, true, typename NeighborSearch::Distance(), true);
    size_t next_k_index = 0;
    size_t neighbor_rank = 0;
    for (auto it = search.begin(); it != search.end() && next_k_index < num_ks; ++it, ++neighbor_rank) {
      if (neighbor_rank + 1 == static_cast<size_t>(ks[next_k_index])) {
        out[i * num_ks + next_k_index] = std::sqrt(CGAL::to_double((*it).second));
        ++next_k_index;
      }
    }
    if (next_k_index != num_ks) {
      throw std::runtime_error("CGAL k-neighbor search returned fewer neighbors than requested.");
    }
  });
  return out;
}

std::vector<double> compute_periodic_knn_selected(
    const std::vector<std::array<double, 3>>& point_cloud,
    const std::vector<int64_t>& ks,
    const std::array<double, 6>& domain) {
  const size_t num_points = point_cloud.size();
  const size_t num_ks = ks.size();
  const size_t max_k = static_cast<size_t>(ks.back());
  const std::array<double, 3> side_lengths = {
      domain[3] - domain[0], domain[4] - domain[1], domain[5] - domain[2]};

  std::vector<double> out(num_points * num_ks, 0.0);
  tbb::parallel_for(size_t{0}, num_points, [&](size_t i) {
    std::vector<double> squared_distances(num_points);
    for (size_t j = 0; j < num_points; ++j) {
      double squared_distance = 0.0;
      for (size_t axis = 0; axis < 3; ++axis) {
        double delta = std::abs(point_cloud[i][axis] - point_cloud[j][axis]);
        delta = std::min(delta, side_lengths[axis] - delta);
        squared_distance += delta * delta;
      }
      squared_distances[j] = squared_distance;
    }
    std::partial_sort(squared_distances.begin(),
                      squared_distances.begin() + max_k,
                      squared_distances.end());
    for (size_t k_index = 0; k_index < num_ks; ++k_index) {
      out[i * num_ks + k_index] =
          std::sqrt(squared_distances[static_cast<size_t>(ks[k_index] - 1)]);
    }
  });
  return out;
}

void validate_periodic_input(
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& points,
    const std::array<double, 6>& domain) {
  if (points.shape(1) != 3) {
    throw nb::value_error("Periodic CoreDelaunay requires three-dimensional points.");
  }
  const double side_length = domain[3] - domain[0];
  if (!std::isfinite(side_length) || side_length <= 0.0) {
    throw nb::value_error("periodic_domain must contain finite positive side lengths.");
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(domain[axis]) || !std::isfinite(domain[axis + 3]) ||
        domain[axis + 3] - domain[axis] != side_length) {
      throw nb::value_error("periodic_domain must define a finite cubic domain.");
    }
  }
  for (size_t point = 0; point < points.shape(0); ++point) {
    for (size_t axis = 0; axis < 3; ++axis) {
      const double coordinate = points.data()[3 * point + axis];
      if (!std::isfinite(coordinate) || coordinate < domain[axis] ||
          coordinate >= domain[axis + 3]) {
        throw nb::value_error(
            "Periodic points must be finite and lie in the half-open periodic domain.");
      }
    }
  }
}

template <typename Wrapper>
void fill_core_delaunay_simplextree(Wrapper& wrapper,
                                    const AlphaTree& alpha_tree,
                                    const std::vector<double>& knn_distances,
                                    const std::vector<int64_t>& ks,
                                    double beta,
                                    bool positive_degree) {
  using Tree = std::remove_reference_t<decltype(wrapper.tree)>;
  using Filtration = typename Tree::Filtration_value;
  using Value = typename Filtration::value_type;

  const size_t num_ks = ks.size();
  const Value beta_value = static_cast<Value>(beta);
  std::vector<Value> filtration_values(num_ks * 2);
  std::vector<Value> second_parameter_values(num_ks);
  std::vector<int> simplex;
  std::vector<const double*> knn_rows;

  wrapper.tree.clear();
  wrapper.tree.copy_from(alpha_tree, [](const auto&) { return Filtration(); });
  wrapper.tree.set_num_parameters(2);

  const Value top_degree = static_cast<Value>(ks.back());
  for (size_t k_index = 0; k_index < num_ks; ++k_index) {
    second_parameter_values[k_index] = positive_degree
                                         ? top_degree - static_cast<Value>(ks[k_index])
                                         : -static_cast<Value>(ks[k_index]);
  }
  auto source_it = alpha_tree.complex_simplex_range().begin();
  auto source_end = alpha_tree.complex_simplex_range().end();
  auto target_it = wrapper.tree.complex_simplex_range().begin();
  for (; source_it != source_end; ++source_it, ++target_it) {
    simplex.clear();
    knn_rows.clear();
    for (auto vertex : alpha_tree.simplex_vertex_range(*source_it)) {
      simplex.push_back(static_cast<int>(vertex));
    }
    simplex.reserve(simplex.size());
    std::reverse(simplex.begin(), simplex.end());

    const Value alpha = static_cast<Value>(std::sqrt(alpha_tree.filtration(*source_it)));
    knn_rows.reserve(simplex.size());
    for (int vertex : simplex) {
      knn_rows.push_back(knn_distances.data() + static_cast<size_t>(vertex) * num_ks);
    }
    for (size_t k_index = 0; k_index < num_ks; ++k_index) {
      Value max_knn_distance = static_cast<Value>(0);
      for (const double* row : knn_rows) {
        max_knn_distance = std::max(max_knn_distance, static_cast<Value>(row[k_index]));
      }
      filtration_values[2 * k_index] = std::max(alpha, beta_value * max_knn_distance);
      filtration_values[2 * k_index + 1] = second_parameter_values[k_index];
    }
    wrapper.tree.get_filtration_value(*target_it) = Filtration(filtration_values.begin(), filtration_values.end(), 2);
  }
  wrapper.tree.clear_filtration();
}

template <typename Kernel>
void build_core_delaunay_dispatch(nb::object& out,
                                  const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& points,
  const std::vector<int64_t>& ks,
  double beta,
  double max_alpha_square,
  bool positive_degree,
  bool exact) {
  auto point_cloud = point_cloud_from_array<Kernel>(points);
  auto knn_point_cloud = point_cloud_from_array<SearchKernel>(points);
  auto alpha_tree = build_alpha_tree<Kernel>(point_cloud, max_alpha_square, exact);
  auto knn_distances = compute_knn_selected(knn_point_cloud, ks);
  visit_simplextree_wrapper(out, [&]<typename Desc>(auto& wrapper) {
    if constexpr (Desc::is_kcritical && std::is_same_v<typename Desc::value_type, double>) {
      nb::gil_scoped_release release;
      fill_core_delaunay_simplextree(wrapper, alpha_tree, knn_distances, ks, beta, positive_degree);
    } else {
      throw nb::type_error(
          "build_core_delaunay_simplextree expects a float64 k-critical SimplexTreeMulti target.");
    }
  });
}

template <Gudhi::alpha_complex::complexity Complexity>
void build_periodic_core_delaunay_dispatch(
    nb::object& out,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& points,
    const std::vector<int64_t>& ks,
    double beta,
    double max_alpha_square,
    bool positive_degree,
    const std::array<double, 6>& domain) {
  using Complex = PeriodicAlphaComplex<Complexity>;
  using Point = typename Complex::Bare_point_3;

  std::vector<Point> point_cloud;
  point_cloud.reserve(points.shape(0));
  for (size_t i = 0; i < points.shape(0); ++i) {
    const double* row = points.data() + 3 * i;
    point_cloud.emplace_back(row[0], row[1], row[2]);
  }

  AlphaTree alpha_tree;
  std::vector<double> knn_distances;
  {
    nb::gil_scoped_release release;
    Complex alpha_complex(point_cloud,
                          domain[0], domain[1], domain[2],
                          domain[3], domain[4], domain[5]);
    if (!alpha_complex.create_complex(alpha_tree, max_alpha_square)) {
      throw std::runtime_error("Failed to build Gudhi periodic alpha complex.");
    }
    if (alpha_tree.num_vertices() != points.shape(0)) {
      throw std::invalid_argument(
          "Periodic points must be unique modulo the periodic domain.");
    }

    // Alpha_complex_3d assigns its own vertex ids. Compute k-neighbor radii in
    // that same order so simplex vertices and scalar rows remain aligned.
    std::vector<std::array<double, 3>> ordered_points(alpha_tree.num_vertices());
    for (size_t vertex = 0; vertex < ordered_points.size(); ++vertex) {
      const auto& point = alpha_complex.get_point(vertex);
      ordered_points[vertex] = {
          CGAL::to_double(point.x()),
          CGAL::to_double(point.y()),
          CGAL::to_double(point.z())};
    }
    knn_distances = compute_periodic_knn_selected(ordered_points, ks, domain);
  }
  visit_simplextree_wrapper(out, [&]<typename Desc>(auto& wrapper) {
    if constexpr (Desc::is_kcritical && std::is_same_v<typename Desc::value_type, double>) {
      nb::gil_scoped_release release;
      fill_core_delaunay_simplextree(
          wrapper, alpha_tree, knn_distances, ks, beta, positive_degree);
    } else {
      throw nb::type_error(
          "build_core_delaunay_simplextree expects a float64 k-critical SimplexTreeMulti target.");
    }
  });
}

}  // namespace mpcd

NB_MODULE(_core_delaunay_nanobind, m) {
  m.def(
      "build_core_delaunay_simplextree",
      [](nb::object target,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> ks,
         double beta,
         std::string precision,
         double max_alpha_square,
         bool positive_degree,
         nb::object periodic_domain_obj) -> nb::object {
        if (!mpcd::is_simplextree_object(target)) {
          throw nb::type_error("build_core_delaunay_simplextree expects a SimplexTreeMulti target.");
        }
        if (points.shape(0) == 0) {
          throw nb::value_error("The point cloud must contain at least one point.");
        }
        if (ks.shape(0) == 0) {
          throw nb::value_error("The parameter ks must contain at least one value.");
        }

        auto degree_values = mpcd::cast_vector_from_array<int64_t>(ks);
        for (size_t index = 0; index < degree_values.size(); ++index) {
          if (degree_values[index] <= 0 ||
              static_cast<size_t>(degree_values[index]) > points.shape(0)) {
            throw nb::value_error(
                "All values in ks must lie between one and the number of points.");
          }
          if (index > 0 && degree_values[index] <= degree_values[index - 1]) {
            throw nb::value_error("ks must be strictly increasing.");
          }
        }

        nb::object out = target.type()();
        if (!periodic_domain_obj.is_none()) {
          auto periodic_domain = nb::cast<
              nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>>(
              periodic_domain_obj);
          if (periodic_domain.shape(0) != 6) {
            throw nb::value_error("periodic_domain must contain six bounds.");
          }
          std::array<double, 6> domain;
          std::copy(periodic_domain.data(), periodic_domain.data() + 6, domain.begin());
          mpcd::validate_periodic_input(points, domain);
          if (precision == "fast") {
            mpcd::build_periodic_core_delaunay_dispatch<Gudhi::alpha_complex::complexity::FAST>(
                out, points, degree_values, beta, max_alpha_square, positive_degree, domain);
          } else if (precision == "safe") {
            mpcd::build_periodic_core_delaunay_dispatch<Gudhi::alpha_complex::complexity::SAFE>(
                out, points, degree_values, beta, max_alpha_square, positive_degree, domain);
          } else if (precision == "exact") {
            mpcd::build_periodic_core_delaunay_dispatch<Gudhi::alpha_complex::complexity::EXACT>(
                out, points, degree_values, beta, max_alpha_square, positive_degree, domain);
          } else {
            throw nb::value_error("precision must be one of {'safe', 'exact', 'fast'}.");
          }
        } else if (precision == "fast") {
          mpcd::build_core_delaunay_dispatch<mpcd::FastKernel>(
              out, points, degree_values, beta, max_alpha_square, positive_degree, false);
        } else if (precision == "safe") {
          mpcd::build_core_delaunay_dispatch<mpcd::SafeKernel>(
              out, points, degree_values, beta, max_alpha_square, positive_degree, false);
        } else if (precision == "exact") {
          mpcd::build_core_delaunay_dispatch<mpcd::SafeKernel>(
              out, points, degree_values, beta, max_alpha_square, positive_degree, true);
        } else {
          throw nb::value_error("precision must be one of {'safe', 'exact', 'fast'}.");
        }
        return out;
      },
      "target"_a,
      "points"_a,
      "ks"_a,
      "beta"_a,
      "precision"_a,
      "max_alpha_square"_a,
      "positive_degree"_a = false,
      "periodic_domain"_a = nb::none());
}
