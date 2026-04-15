#pragma once

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

#include <cstddef>
#include <vector>

namespace multipers::nanobind_simplextree_utils {

struct flat_simplex_batch {
  std::vector<int> vertices;
  size_t simplex_size = 0;
  size_t num_simplices = 0;

  bool empty() const { return simplex_size == 0 || num_simplices == 0; }
};

template <typename Index>
std::vector<int> simplex_from_array(
    const nanobind::ndarray<nanobind::numpy, const Index, nanobind::ndim<1>, nanobind::c_contig>& simplex) {
  std::vector<int> out;
  out.reserve(simplex.shape(0));
  for (size_t i = 0; i < simplex.shape(0); ++i) {
    out.push_back(static_cast<int>(simplex(i)));
  }
  return out;
}

inline flat_simplex_batch simplices_from_vertex_rows(const std::vector<std::vector<int>>& vertex_array) {
  if (vertex_array.empty() || vertex_array[0].empty()) {
    return {};
  }
  flat_simplex_batch simplices;
  simplices.simplex_size = vertex_array.size();
  simplices.num_simplices = vertex_array[0].size();
  simplices.vertices.resize(simplices.simplex_size * simplices.num_simplices);
  for (size_t i = 0; i < simplices.num_simplices; ++i) {
    for (size_t j = 0; j < simplices.simplex_size; ++j) {
      simplices.vertices[i * simplices.simplex_size + j] = vertex_array[j][i];
    }
  }
  return simplices;
}

template <typename Index>
flat_simplex_batch simplices_from_vertex_array(
    const nanobind::ndarray<nanobind::numpy, const Index, nanobind::ndim<2>>& vertex_array) {
  if (vertex_array.shape(0) == 0 || vertex_array.shape(1) == 0) {
    return {};
  }
  const auto view = vertex_array.view();
  flat_simplex_batch simplices;
  simplices.simplex_size = vertex_array.shape(0);
  simplices.num_simplices = vertex_array.shape(1);
  simplices.vertices.resize(simplices.simplex_size * simplices.num_simplices);
  for (size_t i = 0; i < simplices.num_simplices; ++i) {
    for (size_t j = 0; j < simplices.simplex_size; ++j) {
      simplices.vertices[i * simplices.simplex_size + j] = static_cast<int>(view(j, i));
    }
  }
  return simplices;
}

template <typename Filtration, typename Value>
std::vector<Filtration> one_critical_filtrations_from_rows(const std::vector<std::vector<Value>>& rows) {
  std::vector<Filtration> out;
  out.reserve(rows.size());
  for (const auto& row : rows) {
    out.emplace_back(row.begin(), row.end());
  }
  return out;
}

template <typename Filtration, typename Value>
Filtration one_critical_filtration_from_array(
    const nanobind::ndarray<nanobind::numpy, const Value, nanobind::ndim<1>, nanobind::c_contig>& filtration) {
  std::vector<Value> values;
  values.reserve(filtration.shape(0));
  for (size_t i = 0; i < filtration.shape(0); ++i) {
    values.push_back(filtration(i));
  }
  return Filtration(values.begin(), values.end());
}

template <typename Filtration, typename Value>
std::vector<Filtration> one_critical_filtrations_from_array(
    const nanobind::ndarray<nanobind::numpy, const Value, nanobind::ndim<2>>& filtrations) {
  std::vector<Filtration> out;
  if (filtrations.shape(0) == 0 || filtrations.shape(1) == 0) {
    return out;
  }
  const auto view = filtrations.view();
  const size_t num_simplices = filtrations.shape(0);
  const size_t num_parameters = filtrations.shape(1);
  out.reserve(num_simplices);
  for (size_t i = 0; i < num_simplices; ++i) {
    std::vector<Value> row(num_parameters);
    for (size_t p = 0; p < num_parameters; ++p) {
      row[p] = view(i, p);
    }
    out.emplace_back(row.begin(), row.end());
  }
  return out;
}

template <typename Filtration, typename Value>
std::vector<Filtration> kcritical_filtrations_from_array(
    const nanobind::ndarray<nanobind::numpy, const Value, nanobind::ndim<3>>& filtrations,
    int default_num_parameters) {
  std::vector<Filtration> out;
  if (filtrations.shape(0) == 0 || filtrations.shape(1) == 0 || filtrations.shape(2) == 0) {
    return out;
  }
  const auto view = filtrations.view();
  const size_t num_simplices = filtrations.shape(0);
  const size_t max_generators = filtrations.shape(1);
  const int num_parameters =
      filtrations.shape(2) == 0 ? default_num_parameters : static_cast<int>(filtrations.shape(2));
  out.reserve(num_simplices);
  for (size_t i = 0; i < num_simplices; ++i) {
    std::vector<Value> flat;
    flat.reserve(max_generators * static_cast<size_t>(num_parameters));
    for (size_t g = 0; g < max_generators; ++g) {
      for (int p = 0; p < num_parameters; ++p) {
        flat.push_back(view(i, g, static_cast<size_t>(p)));
      }
    }
    out.emplace_back(flat.begin(), flat.end(), num_parameters);
  }
  return out;
}

template <typename Filtration, typename Value>
Filtration kcritical_filtration_from_array(
    const nanobind::ndarray<nanobind::numpy, const Value, nanobind::ndim<1>, nanobind::c_contig>& filtration,
    int default_num_parameters) {
  const int num_parameters = filtration.shape(0) == 0 ? default_num_parameters : static_cast<int>(filtration.shape(0));
  std::vector<Value> values;
  values.reserve(filtration.shape(0));
  for (size_t i = 0; i < filtration.shape(0); ++i) {
    values.push_back(filtration(i));
  }
  return Filtration(values.begin(), values.end(), num_parameters);
}

template <typename Filtration, typename Value>
Filtration kcritical_filtration_from_array(
    const nanobind::ndarray<nanobind::numpy, const Value, nanobind::ndim<2>, nanobind::c_contig>& filtration,
    int default_num_parameters) {
  const auto view = filtration.view();
  const int num_parameters = filtration.shape(1) == 0 ? default_num_parameters : static_cast<int>(filtration.shape(1));
  std::vector<Value> flat;
  flat.reserve(filtration.shape(0) * filtration.shape(1));
  for (size_t i = 0; i < filtration.shape(0); ++i) {
    for (size_t j = 0; j < filtration.shape(1); ++j) {
      flat.push_back(view(i, j));
    }
  }
  return Filtration(flat.begin(), flat.end(), num_parameters);
}

}  // namespace multipers::nanobind_simplextree_utils
