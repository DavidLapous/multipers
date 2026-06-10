#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <oneapi/tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Simplex_tree_multi_interface.h"
#include "ext_interface/nanobind_registry_helpers.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpmcbif {

using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_simplextree_wrapper;

using Simplex = std::vector<int>;
using Cluster = std::vector<int>;

struct SimplexData {
  Simplex simplex;
  std::vector<size_t> occurrence_indices;
};

struct NerveSimplexInfo {
  Simplex simplex;
  Cluster intersection;
};

inline Cluster intersect_sorted(const Cluster& left, const Cluster& right) {
  Cluster out;
  out.reserve(std::min(left.size(), right.size()));
  std::set_intersection(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(out));
  return out;
}

inline std::vector<Cluster> clusters_for_partition(const int64_t* row, size_t num_points) {
  std::map<int64_t, Cluster> by_label;
  for (size_t vertex = 0; vertex < num_points; ++vertex) {
    by_label[row[vertex]].push_back(static_cast<int>(vertex));
  }

  std::vector<Cluster> clusters;
  clusters.reserve(by_label.size());
  for (auto& [label, cluster] : by_label) {
    (void)label;
    clusters.push_back(std::move(cluster));
  }
  return clusters;
}

inline SimplexData build_nerve_simplex_data(const Simplex& simplex,
                                            const std::vector<std::vector<size_t>>& cluster_occurrences,
                                            size_t num_partitions) {
  std::vector<size_t> rows;
  rows.reserve(num_partitions);
  for (size_t start_index = 0; start_index < num_partitions; ++start_index) {
    bool found = true;
    size_t end_index = 0;
    for (int cluster_id : simplex) {
      const auto& occurrences = cluster_occurrences[static_cast<size_t>(cluster_id)];
      auto occurrence_it = std::lower_bound(occurrences.begin(), occurrences.end(), start_index);
      if (occurrence_it == occurrences.end()) {
        found = false;
        break;
      }
      end_index = std::max(end_index, *occurrence_it);
    }
    if (!found) {
      break;
    }
    rows.push_back(end_index);
  }
  return {simplex, std::move(rows)};
}

inline void append_occurrence(std::vector<size_t>& occurrences, size_t partition_index) {
  if (occurrences.empty() || occurrences.back() != partition_index) {
    occurrences.push_back(partition_index);
  }
}

inline void append_standard_combinations(std::map<Simplex, std::vector<size_t>>& simplex_occurrences,
                                         const Cluster& cluster,
                                         size_t partition_index,
                                         size_t target_size,
                                         size_t start,
                                         Simplex& simplex) {
  if (simplex.size() == target_size) {
    append_occurrence(simplex_occurrences[simplex], partition_index);
    return;
  }

  const size_t remaining = target_size - simplex.size();
  for (size_t i = start; i + remaining <= cluster.size(); ++i) {
    simplex.push_back(cluster[i]);
    append_standard_combinations(simplex_occurrences, cluster, partition_index, target_size, i + 1, simplex);
    simplex.pop_back();
  }
}

inline void append_standard_cluster_simplices(std::map<Simplex, std::vector<size_t>>& simplex_occurrences,
                                              const Cluster& cluster,
                                              size_t partition_index,
                                              int max_dim) {
  Simplex simplex;
  const size_t max_size = std::min(cluster.size(), static_cast<size_t>(max_dim) + 1);
  for (size_t target_size = 1; target_size <= max_size; ++target_size) {
    append_standard_combinations(simplex_occurrences, cluster, partition_index, target_size, 0, simplex);
  }
}

inline std::vector<SimplexData> build_standard_simplices(const int64_t* partitions,
                                                         size_t num_partitions,
                                                         size_t num_points,
                                                         int max_dim) {
  std::map<Simplex, std::vector<size_t>> simplex_occurrences;
  for (size_t partition_index = 0; partition_index < num_partitions; ++partition_index) {
    const int64_t* row = partitions + partition_index * num_points;
    auto clusters = clusters_for_partition(row, num_points);
    for (const auto& cluster : clusters) {
      append_standard_cluster_simplices(simplex_occurrences, cluster, partition_index, max_dim);
    }
  }

  std::vector<SimplexData> out;
  out.reserve(simplex_occurrences.size());
  for (auto& [simplex, occurrences] : simplex_occurrences) {
    out.push_back({std::move(simplex), std::move(occurrences)});
  }
  return out;
}

inline void add_nerve_simplex(std::map<Simplex, std::vector<size_t>>& nerve_simplices, Simplex simplex) {
  std::sort(simplex.begin(), simplex.end());
  nerve_simplices.try_emplace(std::move(simplex), std::vector<size_t>{});
}

inline std::vector<SimplexData> build_nerve_simplices(const int64_t* partitions,
                                                      size_t num_partitions,
                                                      size_t num_points,
                                                      int max_dim) {
  std::map<Cluster, int> cluster_id_by_vertices;
  std::vector<Cluster> cluster_vertices;
  std::vector<std::vector<size_t>> cluster_occurrences;
  std::vector<std::vector<int>> new_cluster_ids_by_partition;
  new_cluster_ids_by_partition.reserve(num_partitions);

  for (size_t partition_index = 0; partition_index < num_partitions; ++partition_index) {
    const int64_t* row = partitions + partition_index * num_points;
    auto clusters = clusters_for_partition(row, num_points);
    std::vector<int> new_cluster_ids;
    for (auto& cluster : clusters) {
      auto [it, inserted] = cluster_id_by_vertices.emplace(cluster, static_cast<int>(cluster_vertices.size()));
      const int cluster_id = it->second;
      if (inserted) {
        cluster_vertices.push_back(std::move(cluster));
        cluster_occurrences.emplace_back();
        new_cluster_ids.push_back(cluster_id);
      }
      append_occurrence(cluster_occurrences[static_cast<size_t>(cluster_id)], partition_index);
    }
    new_cluster_ids_by_partition.push_back(std::move(new_cluster_ids));
  }

  std::map<Simplex, std::vector<size_t>> nerve_simplices;
  const size_t requested_max_simplex_size = static_cast<size_t>(max_dim) + 1;
  const size_t max_simplex_size =
      std::min(std::min(requested_max_simplex_size, cluster_vertices.size()), num_partitions);
  std::vector<std::vector<NerveSimplexInfo>> simplices_by_size(max_simplex_size + 1);

  for (const auto& new_cluster_ids : new_cluster_ids_by_partition) {
    for (int cluster_id : new_cluster_ids) {
      const Cluster& cluster = cluster_vertices[static_cast<size_t>(cluster_id)];

      for (size_t size = max_simplex_size - 1; size >= 1; --size) {
        const size_t simplex_count = simplices_by_size[size].size();
        for (size_t simplex_index = 0; simplex_index < simplex_count; ++simplex_index) {
          const auto& previous = simplices_by_size[size][simplex_index];
          auto intersection = intersect_sorted(cluster, previous.intersection);
          if (!intersection.empty()) {
            Simplex simplex = previous.simplex;
            simplex.push_back(cluster_id);
            add_nerve_simplex(nerve_simplices, simplex);
            simplices_by_size[size + 1].push_back({std::move(simplex), std::move(intersection)});
          }
        }
        if (size == 1) {
          break;
        }
      }

      Simplex vertex = {cluster_id};
      add_nerve_simplex(nerve_simplices, vertex);
      simplices_by_size[1].push_back({std::move(vertex), cluster});
    }
  }

  std::vector<Simplex> simplices;
  simplices.reserve(nerve_simplices.size());
  for (const auto& [simplex, ignored] : nerve_simplices) {
    (void)ignored;
    simplices.push_back(simplex);
  }

  std::vector<SimplexData> out(simplices.size());
  const auto build_one = [&](size_t simplex_index) {
    out[simplex_index] = build_nerve_simplex_data(simplices[simplex_index], cluster_occurrences, num_partitions);
  };
  if (simplices.size() < 1024) {
    for (size_t simplex_index = 0; simplex_index < simplices.size(); ++simplex_index) {
      build_one(simplex_index);
    }
  } else {
    tbb::parallel_for(size_t{0}, simplices.size(), build_one);
  }
  return out;
}

inline std::vector<double> rows_from_standard_occurrences(const std::vector<size_t>& occurrences,
                                                          const double* filtration_indices) {
  std::vector<double> flat_rows;
  if (occurrences.empty()) {
    return flat_rows;
  }

  flat_rows.reserve(occurrences.size() * 2);
  for (auto occurrence_it = occurrences.rbegin(); occurrence_it != occurrences.rend(); ++occurrence_it) {
    const size_t occurrence = *occurrence_it;
    flat_rows.push_back(-filtration_indices[occurrence]);
    flat_rows.push_back(filtration_indices[occurrence]);
  }
  return flat_rows;
}

inline std::vector<double> rows_from_nerve_end_indices(const std::vector<size_t>& end_indices,
                                                       const double* filtration_indices) {
  std::vector<double> flat_rows;
  flat_rows.reserve(end_indices.size() * 2);
  size_t best_end_index = std::numeric_limits<size_t>::max();
  for (size_t offset = end_indices.size(); offset > 0; --offset) {
    const size_t start_index = offset - 1;
    const size_t end_index = end_indices[start_index];
    if (end_index >= best_end_index) {
      continue;
    }
    flat_rows.push_back(-filtration_indices[start_index]);
    flat_rows.push_back(filtration_indices[end_index]);
    best_end_index = end_index;
  }
  return flat_rows;
}

template <typename Tree, typename Filtration>
bool insert_kcritical_simplex(Tree& tree, const Simplex& simplex, const Filtration& filtration) {
  using BaseTree = typename Tree::Base_tree;
  auto& base_tree = static_cast<BaseTree&>(tree);
  auto result =
      base_tree.insert_simplex_and_subfaces(BaseTree::Filtration_maintenance::LOWER_EXISTING, simplex, filtration);
  return result.first != tree.null_simplex();
}

template <typename Wrapper>
void fill_mcbif_simplextree(Wrapper& wrapper,
                            const int64_t* partitions,
                            size_t num_partitions,
                            size_t num_points,
                            const double* filtration_indices,
                            int max_dim,
                            const std::string& method) {
  using Tree = std::remove_reference_t<decltype(wrapper.tree)>;
  using Filtration = typename Tree::Filtration_value;

  std::vector<SimplexData> simplex_data;
  if (method == "standard") {
    simplex_data = build_standard_simplices(partitions, num_partitions, num_points, max_dim);
  } else if (method == "nerve") {
    simplex_data = build_nerve_simplices(partitions, num_partitions, num_points, max_dim);
  } else {
    throw nb::value_error("method must be 'standard' or 'nerve'.");
  }

  tbb::parallel_sort(simplex_data.begin(), simplex_data.end(), [](const SimplexData& left, const SimplexData& right) {
    if (left.simplex.size() != right.simplex.size()) {
      return left.simplex.size() < right.simplex.size();
    }
    return left.simplex < right.simplex;
  });

  wrapper.tree.clear();
  wrapper.tree.set_num_parameters(2);

  bool inserted = false;
  for (const auto& data : simplex_data) {
    std::vector<double> flat_rows;
    if (method == "standard") {
      flat_rows = rows_from_standard_occurrences(data.occurrence_indices, filtration_indices);
    } else if (method == "nerve") {
      flat_rows = rows_from_nerve_end_indices(data.occurrence_indices, filtration_indices);
    } else {
      throw nb::value_error("method must be 'standard' or 'nerve'.");
    }
    if (flat_rows.empty()) {
      continue;
    }
    Filtration filtration(flat_rows.begin(), flat_rows.end(), 2);
    inserted |= insert_kcritical_simplex(wrapper.tree, data.simplex, filtration);
  }
  if (inserted) {
    wrapper.tree.clear_filtration();
  }
}

inline void validate_filtration_indices(const double* filtration_indices, size_t num_partitions) {
  for (size_t i = 0; i < num_partitions; ++i) {
    if (!std::isfinite(filtration_indices[i])) {
      throw nb::value_error("filtration_indices must be finite.");
    }
    if (i > 0 && !(filtration_indices[i - 1] < filtration_indices[i])) {
      throw nb::value_error("filtration_indices must be strictly increasing.");
    }
  }
}

inline void build_mcbif_dispatch(
    nb::object& out,
    const nb::ndarray<nb::numpy, const int64_t, nb::ndim<2>, nb::c_contig>& partitions,
    const nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>& filtration_indices,
    int max_dim,
    const std::string& method) {
  const size_t num_partitions = partitions.shape(0);
  const size_t num_points = partitions.shape(1);
  visit_simplextree_wrapper(out, [&]<typename Desc>(auto& wrapper) {
    if constexpr (Desc::is_kcritical && std::is_same_v<typename Desc::value_type, double>) {
      nb::gil_scoped_release release;
      fill_mcbif_simplextree(
          wrapper, partitions.data(), num_partitions, num_points, filtration_indices.data(), max_dim, method);
    } else {
      throw nb::type_error("build_mcbif_simplextree expects a float64 k-critical SimplexTreeMulti target.");
    }
  });
}

}  // namespace mpmcbif

NB_MODULE(_mcbif_nanobind, m) {
  m.def(
      "build_mcbif_simplextree",
      [](nb::object target,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<2>, nb::c_contig> partitions,
         nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> filtration_indices,
         int max_dim,
         std::string method) -> nb::object {
        if (!mpmcbif::is_simplextree_object(target)) {
          throw nb::type_error("build_mcbif_simplextree expects a SimplexTreeMulti target.");
        }
        if (partitions.shape(0) == 0) {
          throw nb::value_error("partitions must contain at least one partition.");
        }
        if (partitions.shape(1) == 0) {
          throw nb::value_error("partitions must contain at least one point.");
        }
        if (filtration_indices.shape(0) != partitions.shape(0)) {
          throw nb::value_error("filtration_indices length must match the number of partitions.");
        }
        if (max_dim < 0) {
          throw nb::value_error("max_dim must be nonnegative.");
        }
        if (method != "standard" && method != "nerve") {
          throw nb::value_error("method must be 'standard' or 'nerve'.");
        }
        mpmcbif::validate_filtration_indices(filtration_indices.data(), filtration_indices.shape(0));

        nb::object out = target.type()();
        mpmcbif::build_mcbif_dispatch(out, partitions, filtration_indices, max_dim, method);
        return out;
      },
      "target"_a,
      "partitions"_a,
      "filtration_indices"_a,
      "max_dim"_a,
      "method"_a);
}
