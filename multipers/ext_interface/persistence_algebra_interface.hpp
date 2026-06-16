#pragma once

#ifndef MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE
#define MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE 0
#endif

#if MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE

#define MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE 0

namespace multipers {

inline bool persistence_algebra_interface_available() { return false; }

}  // namespace multipers

#else

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "contiguous_slicer_bridge.hpp"

namespace multipers {

template <typename index_type>
struct persistence_algebra_interface_output {
  std::vector<std::pair<double, double>> filtration_values;
  std::vector<std::vector<index_type>> boundaries;
  std::vector<int> dimensions;
};

inline bool persistence_algebra_interface_available();

template <typename contiguous_slicer_type>
contiguous_f64_complex persistence_algebra_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                                        int degree,
                                                                        bool full_resolution = true);

template <typename contiguous_slicer_type>
contiguous_f64_complex persistence_algebra_death_curve_contiguous_interface(contiguous_slicer_type& input, int degree);

template <typename contiguous_slicer_type>
contiguous_f64_complex persistence_algebra_kernel_contiguous_interface(contiguous_slicer_type& source,
                                                                       contiguous_slicer_type& target,
                                                                       const packed_morphism_columns& columns,
                                                                       int degree);

template <typename contiguous_slicer_type>
contiguous_f64_complex persistence_algebra_image_contiguous_interface(contiguous_slicer_type& source,
                                                                      contiguous_slicer_type& target,
                                                                      const packed_morphism_columns& columns,
                                                                      int degree);

template <typename contiguous_slicer_type>
contiguous_f64_complex persistence_algebra_cokernel_contiguous_interface(contiguous_slicer_type& source,
                                                                         contiguous_slicer_type& target,
                                                                         const packed_morphism_columns& columns,
                                                                         int degree);

template <typename contiguous_slicer_type>
contiguous_f64_complex persistence_algebra_coimage_contiguous_interface(contiguous_slicer_type& source,
                                                                        contiguous_slicer_type& target,
                                                                        const packed_morphism_columns& columns,
                                                                        int degree);

}  // namespace multipers

#if !MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE && __has_include(<grlina/r2graded_matrix.hpp>)
#define MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE 1
#include <grlina/r2graded_matrix.hpp>
#else
#define MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE 0
#endif

namespace multipers {

inline bool persistence_algebra_interface_available() { return MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE; }

#if MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE

namespace persistence_algebra_detail {

using pa_index = int;
using pa_degree = graded_linalg::r2degree;
using pa_matrix = graded_linalg::R2GradedSparseMatrix<pa_index>;
using pa_resolution = graded_linalg::R2Resolution<pa_index>;

inline std::size_t first_index_of_dimension(const std::vector<int>& dimensions, int dim) {
  return static_cast<std::size_t>(std::lower_bound(dimensions.begin(), dimensions.end(), dim) - dimensions.begin());
}

template <typename filtration_type>
inline pa_degree to_pa_degree(const filtration_type& filtration) {
  if (filtration.num_parameters() != 2 || filtration.num_generators() != 1) {
    throw std::invalid_argument(
        "Persistence-Algebra bridge expects 1-critical contiguous slicers with exactly 2 filtration parameters.");
  }
  return {filtration(0, 0), filtration(0, 1)};
}

template <typename source_index_type>
inline std::vector<pa_index> local_boundary(const std::vector<source_index_type>& boundary,
                                            std::size_t row_begin,
                                            std::size_t row_end,
                                            std::size_t column) {
  std::vector<pa_index> out;
  out.reserve(boundary.size());
  for (const auto raw_idx : boundary) {
    if (raw_idx < 0) {
      throw std::invalid_argument("Persistence-Algebra bridge received a negative boundary index.");
    }
    const auto idx = static_cast<std::size_t>(raw_idx);
    if (idx < row_begin || idx >= row_end) {
      throw std::invalid_argument(
          "Persistence-Algebra bridge received a boundary index outside the previous dimension "
          "block at column " +
          std::to_string(column) + ".");
    }
    out.push_back(static_cast<pa_index>(idx - row_begin));
  }
  std::sort(out.begin(), out.end());
  return out;
}

template <typename contiguous_slicer_type>
inline pa_matrix build_boundary_matrix(contiguous_slicer_type& slicer, int col_dimension) {
  const auto dimensions = slicer.get_dimensions();
  const auto& boundaries = slicer.get_boundaries();
  auto& filtrations = slicer.get_filtration_values();

  const std::size_t row_begin = first_index_of_dimension(dimensions, col_dimension - 1);
  const std::size_t row_end = first_index_of_dimension(dimensions, col_dimension);
  const std::size_t col_begin = row_end;
  const std::size_t col_end = first_index_of_dimension(dimensions, col_dimension + 1);

  graded_linalg::array<pa_index> data(col_end - col_begin);
  std::vector<pa_degree> col_degrees;
  std::vector<pa_degree> row_degrees;
  col_degrees.reserve(col_end - col_begin);
  row_degrees.reserve(row_end - row_begin);

  for (std::size_t i = row_begin; i < row_end; ++i) {
    row_degrees.push_back(to_pa_degree(filtrations[i]));
  }
  for (std::size_t i = col_begin; i < col_end; ++i) {
    data[i - col_begin] = local_boundary(boundaries[i], row_begin, row_end, i - col_begin);
    col_degrees.push_back(to_pa_degree(filtrations[i]));
  }

  return pa_matrix(static_cast<pa_index>(col_end - col_begin),
                   static_cast<pa_index>(row_end - row_begin),
                   data,
                   std::move(col_degrees),
                   std::move(row_degrees));
}

template <typename index_type>
inline void append_rows_as_generators(const pa_matrix& matrix,
                                      int degree,
                                      persistence_algebra_interface_output<index_type>& out) {
  for (const auto& row_degree : matrix.row_degrees) {
    out.filtration_values.emplace_back(row_degree.first, row_degree.second);
    out.boundaries.emplace_back();
    out.dimensions.push_back(degree);
  }
}

template <typename index_type>
inline void append_columns_as_generators(const pa_matrix& matrix,
                                         int degree,
                                         index_type row_shift,
                                         persistence_algebra_interface_output<index_type>& out) {
  for (pa_index col = 0; col < matrix.get_num_cols(); ++col) {
    out.filtration_values.emplace_back(matrix.col_degrees[col].first, matrix.col_degrees[col].second);
    std::vector<index_type> boundary;
    boundary.reserve(matrix.data[col].size());
    for (const auto row_idx : matrix.data[col]) {
      boundary.push_back(static_cast<index_type>(row_idx) + row_shift);
    }
    out.boundaries.push_back(std::move(boundary));
    out.dimensions.push_back(degree);
  }
}

inline void colexify_minpres(pa_matrix& matrix) {
  matrix.sort_rows_colexicographically();
  matrix.sort_columns_colexicographically();
}

inline std::vector<pa_index> invert_permutation(const std::vector<pa_index>& new_to_old) {
  std::vector<pa_index> old_to_new(new_to_old.size());
  for (pa_index new_idx = 0; new_idx < static_cast<pa_index>(new_to_old.size()); ++new_idx) {
    old_to_new[static_cast<std::size_t>(new_to_old[static_cast<std::size_t>(new_idx)])] = new_idx;
  }
  return old_to_new;
}

inline void colexify_resolution(pa_resolution& resolution) {
  resolution.d1.sort_rows_colexicographically();
  const auto relation_new_to_old = resolution.d1.sort_columns_colexicographically_with_output();
  resolution.d2.permute_rows_graded(invert_permutation(relation_new_to_old));
  resolution.d2.sort_columns_colexicographically();
}

inline bool degree_leq(const pa_degree& left, const pa_degree& right) {
  return graded_linalg::Degree_traits<pa_degree>::smaller_equal(left, right);
}

inline std::vector<pa_index> normalize_f2_column(std::vector<pa_index> column) {
  std::sort(column.begin(), column.end());
  std::vector<pa_index> out;
  out.reserve(column.size());
  for (std::size_t i = 0; i < column.size();) {
    const auto row = column[i];
    std::size_t count = 1;
    while (i + count < column.size() && column[i + count] == row) {
      ++count;
    }
    if (count % 2 == 1) {
      out.push_back(row);
    }
    i += count;
  }
  return out;
}

inline std::vector<pa_index> xor_sorted_columns(const std::vector<pa_index>& left,
                                                const std::vector<pa_index>& right) {
  std::vector<pa_index> out;
  out.reserve(left.size() + right.size());
  auto l = left.begin();
  auto r = right.begin();
  while (l != left.end() || r != right.end()) {
    if (r == right.end() || (l != left.end() && *l < *r)) {
      out.push_back(*l++);
    } else if (l == left.end() || *r < *l) {
      out.push_back(*r++);
    } else {
      ++l;
      ++r;
    }
  }
  return out;
}

using f2_sparse_basis = std::map<pa_index, std::vector<pa_index>>;

inline void add_to_f2_basis(std::vector<pa_index> column, f2_sparse_basis& basis) {
  while (!column.empty()) {
    const auto pivot = column.front();
    auto [it, inserted] = basis.emplace(pivot, std::vector<pa_index>{});
    if (inserted) {
      it->second = std::move(column);
      return;
    }
    column = xor_sorted_columns(column, it->second);
  }
}

inline f2_sparse_basis build_relation_basis(const pa_matrix& presentation, const pa_degree& max_degree) {
  f2_sparse_basis basis;
  for (pa_index col = 0; col < presentation.get_num_cols(); ++col) {
    if (degree_leq(presentation.col_degrees[col], max_degree)) {
      add_to_f2_basis(normalize_f2_column(presentation.data[col]), basis);
    }
  }
  return basis;
}

inline std::vector<pa_index> reduce_by_basis(std::vector<pa_index> column, const f2_sparse_basis& basis) {
  for (const auto& [pivot, basis_column] : basis) {
    if (std::binary_search(column.begin(), column.end(), pivot)) {
      column = xor_sorted_columns(column, basis_column);
    }
  }
  return column;
}

inline std::vector<pa_index> reduce_by_target_relations(std::vector<pa_index> column,
                                                        const pa_matrix& target_presentation,
                                                        const pa_degree& max_degree) {
  return reduce_by_basis(std::move(column), build_relation_basis(target_presentation, max_degree));
}

template <typename index_type>
inline persistence_algebra_interface_output<index_type> convert_minpres_to_output(pa_matrix matrix,
                                                                                  int degree,
                                                                                  bool full_resolution) {
  persistence_algebra_interface_output<index_type> out;

  if (!full_resolution) {
    colexify_minpres(matrix);
    append_rows_as_generators(matrix, degree, out);
    append_columns_as_generators(matrix, degree + 1, 0, out);
    return out;
  }

  pa_resolution resolution(matrix, false);
  colexify_resolution(resolution);
  append_rows_as_generators(resolution.d1, degree, out);
  append_columns_as_generators(resolution.d1, degree + 1, 0, out);
  append_columns_as_generators(resolution.d2, degree + 2, static_cast<index_type>(resolution.d1.get_num_rows()), out);
  return out;
}

inline void validate_packed_morphism(const packed_morphism_columns& columns, std::size_t num_columns) {
  if (columns.indptr == nullptr || (columns.indices == nullptr && columns.indices_size != 0)) {
    throw std::invalid_argument("Persistence-Algebra morphism columns must be packed CSR arrays.");
  }
  if (columns.indptr_size != num_columns + 1) {
    throw std::invalid_argument("Persistence-Algebra morphism needs one column per source generator.");
  }
  if (columns.indptr[0] != 0 || columns.indptr[num_columns] != columns.indices_size) {
    throw std::invalid_argument("Persistence-Algebra morphism CSR indptr is inconsistent with indices.");
  }
  for (std::size_t col = 0; col < num_columns; ++col) {
    if (columns.indptr[col] > columns.indptr[col + 1]) {
      throw std::invalid_argument("Persistence-Algebra morphism CSR indptr must be nondecreasing.");
    }
  }
}

inline pa_matrix build_morphism_matrix(const pa_matrix& source_presentation,
                                       const pa_matrix& target_presentation,
                                       const packed_morphism_columns& columns) {
  const auto num_columns = static_cast<std::size_t>(source_presentation.get_num_rows());
  validate_packed_morphism(columns, num_columns);
  graded_linalg::array<pa_index> data(num_columns);
  for (std::size_t col = 0; col < num_columns; ++col) {
    const auto begin = columns.indptr[col];
    const auto end = columns.indptr[col + 1];
    data[col].reserve(static_cast<std::size_t>(end - begin));
    for (std::uint64_t idx = begin; idx < end; ++idx) {
      const auto row = columns.indices[idx];
      if (static_cast<std::uint64_t>(row) >= static_cast<std::uint64_t>(target_presentation.get_num_rows())) {
        throw std::invalid_argument("Persistence-Algebra morphism row indices are outside target generators.");
      }
      data[col].push_back(static_cast<pa_index>(row));
    }
    data[col] = normalize_f2_column(std::move(data[col]));
    for (const auto row : data[col]) {
      if (!degree_leq(target_presentation.row_degrees[row], source_presentation.row_degrees[col])) {
        throw std::invalid_argument("Persistence-Algebra morphism entry is not coordinatewise grade-compatible.");
      }
    }
    data[col] = reduce_by_target_relations(
        std::move(data[col]), target_presentation, source_presentation.row_degrees[col]);
  }
  for (pa_index rel = 0; rel < source_presentation.get_num_cols(); ++rel) {
    std::vector<pa_index> image;
    for (const auto source_row : source_presentation.data[rel]) {
      image = xor_sorted_columns(image, data[source_row]);
    }
    if (!reduce_by_target_relations(std::move(image), target_presentation, source_presentation.col_degrees[rel])
             .empty()) {
      throw std::invalid_argument(
          "Persistence-Algebra morphism does not send source relations into the target relation submodule.");
    }
  }
  return pa_matrix(static_cast<pa_index>(num_columns),
                   target_presentation.get_num_rows(),
                   data,
                   source_presentation.row_degrees,
                   target_presentation.row_degrees);
}

inline pa_matrix empty_submodule(const pa_matrix& presentation) {
  pa_matrix out(0, presentation.get_num_rows());
  out.row_degrees = presentation.row_degrees;
  return out;
}

inline pa_matrix finalize_minimize(pa_matrix matrix) {
  matrix.sort_columns_lexicographically();
  matrix.sort_rows_lexicographically();
  matrix.minimize();
  return matrix;
}

template <typename contiguous_slicer_type>
inline pa_matrix build_minimal_presentation(contiguous_slicer_type& slicer, int degree) {
  if (degree < 0) {
    throw std::invalid_argument("Persistence-Algebra interface expects a non-negative homological degree.");
  }

  auto first = build_boundary_matrix(slicer, degree);
  auto second = build_boundary_matrix(slicer, degree + 1);

  first.sort_rows_lexicographically();
  const auto relation_row_permutation = first.sort_columns_lexicographically_with_output();
  second.permute_rows_graded(relation_row_permutation);
  second.sort_columns_lexicographically();
  second.sort_rows_lexicographically();

  auto cycles = first.graded_kernel();
  pa_matrix ambient(0, cycles.get_num_rows());
  ambient.row_degrees = cycles.row_degrees;

  auto cycles_for_presentation = cycles;
  auto kernel_presentation = cycles_for_presentation.presentation_of_submodule(ambient);
  kernel_presentation.sort_columns_lexicographically();
  kernel_presentation.sort_rows_lexicographically();
  kernel_presentation.minimize();

  auto cycles_copy = cycles;
  auto image_in_kernel = cycles_copy.inverse_image(ambient, second);

  kernel_presentation.quotient_by(image_in_kernel);
  kernel_presentation.minimize_variant();
  return kernel_presentation;
}

template <typename contiguous_slicer_type>
inline pa_matrix build_module_presentation(contiguous_slicer_type& slicer, int degree) {
  if (degree < 0) {
    throw std::invalid_argument("Persistence-Algebra interface expects a non-negative degree.");
  }
  auto presentation = build_boundary_matrix(slicer, degree + 1);
  presentation.sort_columns_lexicographically();
  return presentation;
}

template <typename contiguous_slicer_type, typename Fn>
inline contiguous_f64_complex algebra_operation(contiguous_slicer_type& source,
                                                contiguous_slicer_type& target,
                                                const packed_morphism_columns& columns,
                                                int degree,
                                                Fn&& fn) {
  auto source_presentation = build_module_presentation(source, degree);
  auto target_presentation = build_module_presentation(target, degree);
  auto morphism = build_morphism_matrix(source_presentation, target_presentation, columns);
  auto result = finalize_minimize(fn(source_presentation, target_presentation, morphism));
  auto out = convert_minpres_to_output<int>(std::move(result), degree, false);
  return build_contiguous_f64_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}

}  // namespace persistence_algebra_detail

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                                               int degree,
                                                                               bool full_resolution) {
  auto presentation = persistence_algebra_detail::build_minimal_presentation(input, degree);
  auto out =
      persistence_algebra_detail::convert_minpres_to_output<int>(std::move(presentation), degree, full_resolution);
  return build_contiguous_f64_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_death_curve_contiguous_interface(contiguous_slicer_type& input,
                                                                                   int degree) {
  auto presentation = persistence_algebra_detail::build_boundary_matrix(input, degree + 1);
  presentation.sort_columns_lexicographically();
  presentation.sort_rows_lexicographically();
  presentation.minimize();

  const persistence_algebra_detail::pa_degree step{1, 1};
  auto original = presentation;
  presentation.shift(step);
  persistence_algebra_detail::pa_matrix zero(0, presentation.get_num_rows());
  zero.col_degrees = {};
  zero.row_degrees = presentation.row_degrees;
  auto shifted =
      graded_linalg::shifted_identity<persistence_algebra_detail::pa_degree, persistence_algebra_detail::pa_matrix>(
          presentation.row_degrees, step);
  auto ker_epsilon = shifted.inverse_image(presentation, zero);
  auto death = ker_epsilon.presentation_of_submodule(original);
  death.sort_columns_lexicographically();
  death.sort_rows_lexicographically();
  death.minimize();

  auto out = persistence_algebra_detail::convert_minpres_to_output<int>(std::move(death), degree, false);
  return build_contiguous_f64_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_kernel_contiguous_interface(
    contiguous_slicer_type& source,
    contiguous_slicer_type& target,
    const packed_morphism_columns& columns,
    int degree) {
  return persistence_algebra_detail::algebra_operation(
      source, target, columns, degree, [](auto& source_presentation, auto& target_presentation, auto& morphism) {
        auto zero = persistence_algebra_detail::empty_submodule(target_presentation);
        auto kernel_in_source = morphism.inverse_image(target_presentation, zero);
        return kernel_in_source.presentation_of_submodule(source_presentation);
      });
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_image_contiguous_interface(
    contiguous_slicer_type& source,
    contiguous_slicer_type& target,
    const packed_morphism_columns& columns,
    int degree) {
  return persistence_algebra_detail::algebra_operation(
      source, target, columns, degree, [](auto&, auto& target_presentation, auto& morphism) {
        return morphism.presentation_of_submodule(target_presentation);
      });
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_cokernel_contiguous_interface(
    contiguous_slicer_type& source,
    contiguous_slicer_type& target,
    const packed_morphism_columns& columns,
    int degree) {
  return persistence_algebra_detail::algebra_operation(
      source, target, columns, degree, [](auto&, auto& target_presentation, auto& morphism) {
        morphism.column_reduction_graded_w_deletion();
        target_presentation.quotient_by(morphism);
        return target_presentation;
      });
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_coimage_contiguous_interface(
    contiguous_slicer_type& source,
    contiguous_slicer_type& target,
    const packed_morphism_columns& columns,
    int degree) {
  return persistence_algebra_detail::algebra_operation(
      source, target, columns, degree, [](auto& source_presentation, auto& target_presentation, auto& morphism) {
        auto zero = persistence_algebra_detail::empty_submodule(target_presentation);
        auto kernel_in_source = morphism.inverse_image(target_presentation, zero);
        source_presentation.quotient_by(kernel_in_source);
        return source_presentation;
      });
}

#else

namespace persistence_algebra_detail {

[[noreturn]] inline void unavailable() {
  throw std::runtime_error(
      "Persistence-Algebra interface is not available at compile time. Initialize ext/Persistence-Algebra and "
      "rebuild.");
}

}  // namespace persistence_algebra_detail

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_minpres_contiguous_interface(contiguous_slicer_type&, int, bool) {
  persistence_algebra_detail::unavailable();
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_death_curve_contiguous_interface(contiguous_slicer_type&, int) {
  persistence_algebra_detail::unavailable();
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_kernel_contiguous_interface(contiguous_slicer_type&,
                                                                              contiguous_slicer_type&,
                                                                              const packed_morphism_columns&,
                                                                              int) {
  persistence_algebra_detail::unavailable();
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_image_contiguous_interface(contiguous_slicer_type&,
                                                                             contiguous_slicer_type&,
                                                                             const packed_morphism_columns&,
                                                                             int) {
  persistence_algebra_detail::unavailable();
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_cokernel_contiguous_interface(contiguous_slicer_type&,
                                                                                contiguous_slicer_type&,
                                                                                const packed_morphism_columns&,
                                                                                int) {
  persistence_algebra_detail::unavailable();
}

template <typename contiguous_slicer_type>
inline contiguous_f64_complex persistence_algebra_coimage_contiguous_interface(contiguous_slicer_type&,
                                                                               contiguous_slicer_type&,
                                                                               const packed_morphism_columns&,
                                                                               int) {
  persistence_algebra_detail::unavailable();
}

#endif

}  // namespace multipers

#endif  // MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE
