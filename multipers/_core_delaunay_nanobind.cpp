#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
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
         bool positive_degree) -> nb::object {
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
        if (static_cast<size_t>(degree_values.back()) > points.shape(0)) {
          throw nb::value_error(
              "All values in ks must be less than or equal to the number of points in the point cloud.");
        }

        nb::object out = target.type()();
        if (precision == "fast") {
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
      "positive_degree"_a = false);
}
