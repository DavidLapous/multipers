#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ext_interface/packed_multi_critical_bridge.hpp"
#include "ext_interface/multi_critical_interface.hpp"

#if !MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
#include "multi_critical/basic.h"
#include "multi_chunk/basic.h"
#include "mpp_utils/basic.h"
#include "mpfree/global.h"
#include "scc/basic.h"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
namespace mpmc {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ptr_bridge_stats {
  double convert_s = 0.0;
  double free_resolution_s = 0.0;
  double minpres_s = 0.0;
  double output_pack_s = 0.0;
};

nb::dict stats_to_dict(const ptr_bridge_stats& stats) {
  nb::dict out;
  out["convert_s"] = nb::float_(stats.convert_s);
  out["free_resolution_s"] = nb::float_(stats.free_resolution_s);
  out["minpres_s"] = nb::float_(stats.minpres_s);
  out["output_pack_s"] = nb::float_(stats.output_pack_s);
  out["total_s"] = nb::float_(stats.convert_s + stats.free_resolution_s + stats.minpres_s + stats.output_pack_s);
  return out;
}

template <typename T>
std::vector<T> cast_vector(nb::handle h) {
  return nb::cast<std::vector<T>>(h);
}

template <typename T>
void delete_vector_capsule(void* ptr) noexcept {
  delete static_cast<std::vector<T>*>(ptr);
}

template <typename T>
nb::ndarray<nb::numpy, T> owned_array(std::vector<T>&& values, std::initializer_list<size_t> shape) {
  auto* storage = new std::vector<T>(std::move(values));
  nb::capsule owner(storage, &delete_vector_capsule<T>);
  return nb::ndarray<nb::numpy, T>(storage->data(), shape, owner);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(nb::handle h) {
  return nb::cast<std::vector<std::vector<T>>>(h);
}

inline void set_backend_stdout(bool enabled) {
  multi_critical::verbose = enabled;
  multi_critical::very_verbose = enabled;
  multi_chunk::verbose = enabled;
  mpp_utils::verbose = enabled;
  mpfree::verbose = enabled;
  scc::verbose = enabled;
}

std::vector<std::vector<int>> boundaries_from_packed(
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
    nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat) {
  if (boundary_indptr.shape(0) == 0) {
    return {};
  }
  std::vector<std::vector<int>> boundaries((size_t)boundary_indptr.shape(0) - 1);
  const int64_t* indptr = boundary_indptr.data();
  const int32_t* flat = boundary_flat.data();
  for (size_t cell = 0; cell + 1 < (size_t)boundary_indptr.shape(0); ++cell) {
    const int64_t begin = indptr[cell];
    const int64_t end = indptr[cell + 1];
    auto& boundary = boundaries[cell];
    boundary.reserve((size_t)std::max<int64_t>(end - begin, 0));
    for (int64_t idx = begin; idx < end; ++idx) {
      boundary.push_back((int)flat[idx]);
    }
  }
  return boundaries;
}

multipers::multi_critical_interface_input<int> input_from_packed(
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
    nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat,
    nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> dimensions,
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> grade_indptr,
    nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades_flat) {
  auto boundaries = boundaries_from_packed(boundary_indptr, boundary_flat);
  multipers::multi_critical_interface_input<int> input;
  input.dimensions.reserve((size_t)dimensions.shape(0));
  input.boundaries = std::move(boundaries);
  input.filtration_values.resize(input.boundaries.size());
  for (size_t i = 0; i < (size_t)dimensions.shape(0); ++i) {
    input.dimensions.push_back((int)dimensions(i));
  }
  const int64_t* indptr = grade_indptr.data();
  for (size_t cell = 0; cell < input.boundaries.size(); ++cell) {
    auto& out = input.filtration_values[cell];
    const int64_t begin = indptr[cell];
    const int64_t end = indptr[cell + 1];
    out.reserve((size_t)std::max<int64_t>(end - begin, 0));
    for (int64_t row = begin; row < end; ++row) {
      out.emplace_back(grades_flat((size_t)row, 0), grades_flat((size_t)row, 1));
    }
  }
  return input;
}

multipers::multi_critical_interface_input<int> input_from_bridge_ptr(
    const multipers::packed_multi_critical_bridge_input& input_bridge) {
  multipers::multi_critical_interface_input<int> input;
  const size_t num_cells = input_bridge.dimensions.size();
  input.dimensions.reserve(num_cells);
  input.boundaries.resize(num_cells);
  input.filtration_values.resize(num_cells);
  for (size_t i = 0; i < num_cells; ++i) {
    input.dimensions.push_back((int)input_bridge.dimensions[i]);
    auto& out_boundary = input.boundaries[i];
    out_boundary.reserve(input_bridge.boundaries[i].size());
    for (uint32_t face : input_bridge.boundaries[i]) {
      out_boundary.push_back((int)face);
    }
    auto& out = input.filtration_values[i];
    const int64_t begin = input_bridge.grade_indptr[i];
    const int64_t end = input_bridge.grade_indptr[i + 1];
    out.reserve((size_t)std::max<int64_t>(end - begin, 0));
    for (int64_t row = begin; row < end; ++row) {
      out.emplace_back(input_bridge.grade_values[2 * (size_t)row], input_bridge.grade_values[2 * (size_t)row + 1]);
    }
  }
  return input;
}

class packed_bridge_parser {
 public:
  explicit packed_bridge_parser(const multipers::packed_multi_critical_bridge_input& input) : input_(input) {
    const std::size_t num_generators = input_.dimensions.size();
    if (input_.boundaries.size() != num_generators || input_.grade_indptr.size() != num_generators + 1) {
      throw std::invalid_argument("Invalid packed bridge input: sizes of grades, boundaries and dimensions differ.");
    }
    if (num_generators == 0) return;
    if (!std::is_sorted(input_.dimensions.begin(), input_.dimensions.end())) {
      throw std::invalid_argument("Dimensions are expected to be sorted in non-decreasing order.");
    }
    const int max_dim = input_.dimensions.back();
    if (max_dim < 0) {
      throw std::invalid_argument("Dimensions must be non-negative.");
    }
    indices_by_level_.resize((std::size_t)max_dim + 1);
    shifted_indices_.assign(num_generators, -1);
    for (std::size_t i = 0; i < num_generators; ++i) {
      const int dim = input_.dimensions[i];
      const std::size_t level_idx = static_cast<std::size_t>(max_dim - dim);
      shifted_indices_[i] = static_cast<long>(indices_by_level_[level_idx].size());
      indices_by_level_[level_idx].push_back(i);
    }
    offsets_.assign(indices_by_level_.size(), 0);
  }

  int number_of_parameters() { return 2; }

  int number_of_levels() { return static_cast<int>(indices_by_level_.size() + 1); }

  bool has_next_column(int level) {
    validate_level(level);
    if (static_cast<std::size_t>(level - 1) >= indices_by_level_.size()) {
      return false;
    }
    return offsets_[level - 1] < indices_by_level_[level - 1].size();
  }

  bool has_grades_on_last_level() { return false; }

  template <typename OutputIterator1, typename OutputIterator2>
  void next_column(int level, OutputIterator1 out1, OutputIterator2 out2) {
    validate_level(level);
    if (!has_next_column(level)) {
      throw std::out_of_range("No more columns on requested level.");
    }
    const std::size_t global_idx = indices_by_level_[level - 1][offsets_[level - 1]++];
    const int64_t begin = input_.grade_indptr[global_idx];
    const int64_t end = input_.grade_indptr[global_idx + 1];
    if (begin >= end) {
      throw std::invalid_argument("Each generator must have at least one filtration grade.");
    }
    for (int64_t row = begin; row < end; ++row) {
      *out1++ = input_.grade_values[2 * (std::size_t)row];
      *out1++ = input_.grade_values[2 * (std::size_t)row + 1];
    }
    const int simplex_dim = input_.dimensions[global_idx];
    const auto& boundary = input_.boundaries[global_idx];
    if (simplex_dim == 0 && !boundary.empty()) {
      throw std::invalid_argument("Dimension-0 generators must have empty boundaries.");
    }
    for (uint32_t bd_idx_typed : boundary) {
      const std::size_t bd_idx = static_cast<std::size_t>(bd_idx_typed);
      if (bd_idx >= input_.dimensions.size()) {
        throw std::invalid_argument("Boundary index out of range.");
      }
      if (simplex_dim > 0 && input_.dimensions[bd_idx] != simplex_dim - 1) {
        throw std::invalid_argument("Boundary index does not point to previous dimension.");
      }
      const long shifted = shifted_indices_[bd_idx];
      if (shifted < 0) {
        throw std::invalid_argument("Internal index conversion failed.");
      }
      *out2++ = std::make_pair(shifted, 1);
    }
  }

  int number_of_generators(int level) {
    validate_level(level);
    if (static_cast<std::size_t>(level - 1) >= indices_by_level_.size()) {
      return 0;
    }
    return static_cast<int>(indices_by_level_[level - 1].size());
  }

  void reset(int level) {
    validate_level(level);
    if (static_cast<std::size_t>(level - 1) >= indices_by_level_.size()) {
      return;
    }
    offsets_[level - 1] = 0;
  }

  void reset_all() { std::fill(offsets_.begin(), offsets_.end(), 0); }

 private:
  const multipers::packed_multi_critical_bridge_input& input_;
  std::vector<std::vector<std::size_t>> indices_by_level_;
  std::vector<std::size_t> offsets_;
  std::vector<long> shifted_indices_;

  void validate_level(int level) const {
    if (level < 1 || static_cast<std::size_t>(level) > indices_by_level_.size() + 1) {
      throw std::out_of_range("Requested level is out of parser range.");
    }
  }
};

std::vector<multipers::multi_critical_detail::Graded_matrix> compute_free_resolution_matrices_from_bridge(
    const multipers::packed_multi_critical_bridge_input& input,
    bool use_logpath,
    bool use_multi_chunk,
    bool verbose_output,
    ptr_bridge_stats* stats = nullptr) {
  if (input.dimensions.empty()) {
    return {};
  }
  auto t_convert = Clock::now();
  packed_bridge_parser parser(input);
  if (stats != nullptr) {
    stats->convert_s += elapsed_seconds(t_convert);
  }
  const bool old_mc_verbose = multi_critical::verbose;
  const bool old_mc_very_verbose = multi_critical::very_verbose;
  const bool old_chunk_verbose = multi_chunk::verbose;
  multi_critical::verbose = verbose_output;
  multi_critical::very_verbose = false;
  multi_chunk::verbose = verbose_output;
  auto t_free = Clock::now();
  std::vector<multipers::multi_critical_detail::Graded_matrix> matrices;
  multi_critical::free_resolution(parser, matrices, use_logpath);
  for (std::size_t i = 0; i < matrices.size(); ++i) {
    auto& mat = matrices[i];
    if (!mpp_utils::is_lex_sorted(mat)) {
      mpp_utils::to_lex_order(mat, i + 1 < matrices.size(), false);
    }
  }
  if (use_multi_chunk) {
    multi_chunk::compress(matrices);
  }
  multi_critical::verbose = old_mc_verbose;
  multi_critical::very_verbose = old_mc_very_verbose;
  multi_chunk::verbose = old_chunk_verbose;
  if (stats != nullptr) {
    stats->free_resolution_s += elapsed_seconds(t_free);
  }
  return matrices;
}

nb::tuple output_to_raw_arrays(const multipers::multi_critical_interface_output<int>& output,
                               bool mark_minpres = false) {
  std::vector<int64_t> boundary_indptr(output.boundaries.size() + 1, 0);
  size_t boundary_nnz = 0;
  for (size_t i = 0; i < output.boundaries.size(); ++i) {
    boundary_nnz += output.boundaries[i].size();
    boundary_indptr[i + 1] = (int64_t)boundary_nnz;
  }
  std::vector<int32_t> boundary_flat;
  boundary_flat.reserve(boundary_nnz);
  for (const auto& row : output.boundaries) {
    for (int value : row) {
      boundary_flat.push_back((int32_t)value);
    }
  }
  std::vector<int32_t> dimensions(output.dimensions.begin(), output.dimensions.end());
  if (mark_minpres) {
    for (auto& d : dimensions) {
      d -= 1;
    }
  }
  std::vector<double> grades;
  grades.reserve(2 * output.filtration_values.size());
  for (const auto& grade : output.filtration_values) {
    grades.push_back(grade.first);
    grades.push_back(grade.second);
  }
  return nb::make_tuple(owned_array<int64_t>(std::move(boundary_indptr), {output.boundaries.size() + 1}),
                        owned_array<int32_t>(std::move(boundary_flat), {boundary_nnz}),
                        owned_array<int32_t>(std::move(dimensions), {output.dimensions.size()}),
                        owned_array<double>(std::move(grades), {output.filtration_values.size(), size_t(2)}));
}

multipers::multi_critical_interface_input<int> input_from_kcritical_slicer(nb::object slicer) {
  auto dimensions = cast_vector<int>(slicer.attr("get_dimensions")());
  auto packed_boundaries = nb::cast<nb::tuple>(slicer.attr("get_boundaries")("packed"_a = true));
  auto boundaries_indptr = cast_vector<int64_t>(packed_boundaries[0]);
  auto boundaries_indices = cast_vector<int>(packed_boundaries[1]);
  auto packed_filtrations = nb::cast<nb::tuple>(slicer.attr("get_filtrations")("packed"_a = true));
  auto indptr = cast_vector<int64_t>(packed_filtrations[0]);
  auto grades = cast_matrix<double>(packed_filtrations[1]);

  multipers::multi_critical_interface_input<int> input;
  input.dimensions = dimensions;
  input.boundaries.resize(dimensions.size());
  input.filtration_values.resize(dimensions.size());
  for (size_t i = 0; i < dimensions.size(); ++i) {
    for (int64_t j = boundaries_indptr[i]; j < boundaries_indptr[i + 1]; ++j)
      input.boundaries[i].push_back(boundaries_indices[(size_t)j]);
    for (int64_t j = indptr[i]; j < indptr[i + 1]; ++j)
      input.filtration_values[i].push_back({grades[(size_t)j][0], grades[(size_t)j][1]});
  }
  return input;
}

nb::object output_to_slicer(nb::object slicer_type,
                            const multipers::multi_critical_interface_output<int>& output,
                            bool mark_minpres = false,
                            int degree = -1) {
  std::vector<int> dims(output.dimensions.begin(), output.dimensions.end());
  if (mark_minpres) {
    for (auto& d : dims) d -= 1;
  }
  nb::list boundaries;
  for (const auto& row : output.boundaries) boundaries.append(nb::cast(row));
  std::vector<std::vector<double>> filtrations;
  filtrations.reserve(output.filtration_values.size());
  for (const auto& p : output.filtration_values) filtrations.push_back({p.first, p.second});
  nb::object out = slicer_type(boundaries, nb::cast(dims), nb::cast(filtrations));
  if (mark_minpres) out.attr("minpres_degree") = degree;
  return out;
}

}  // namespace mpmc
#endif

NB_MODULE(_multi_critical_interface, m) {
#if MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
  auto unavailable = [](nb::args, nb::kwargs) {
    throw std::runtime_error("multi_critical in-memory interface is disabled at compile time.");
  };
  m.def("_is_available", []() { return false; });
  m.def("_set_backend_stdout", [](bool) {}, "enabled"_a);
  m.def("one_criticalify", unavailable);
  m.def("resolution_from_packed", unavailable);
  m.def("minpres_from_packed", unavailable);
  m.def("minpres_all_from_packed", unavailable);
  m.def("resolution_from_ptr", unavailable);
  m.def("resolution_from_ptr_with_stats", unavailable);
  m.def("minpres_from_ptr", unavailable);
  m.def("minpres_from_ptr_with_stats", unavailable);
  m.def("minpres_all_from_ptr", unavailable);
  m.def("minpres_all_from_ptr_with_stats", unavailable);
#else
  bool ext_log_enabled = false;
  try {
    ext_log_enabled = nb::cast<bool>(nb::module_::import_("multipers.logs").attr("ext_log_enabled")());
  } catch (...) {
    ext_log_enabled = false;
  }
  mpmc::set_backend_stdout(ext_log_enabled);

  m.def("_is_available", []() {
    if (!multipers::multi_critical_interface_available()) return false;
    auto slicer_module = nb::module_::import_("multipers.slicer");
    for (nb::handle container : nb::iter(slicer_module.attr("available_filtration_container"))) {
      if (nb::cast<std::string>(container) == "Flat") return true;
    }
    return false;
  });

  m.def("_set_backend_stdout", [](bool enabled) { mpmc::set_backend_stdout(enabled); }, "enabled"_a);

  m.def(
      "one_criticalify",
      [](nb::object slicer,
         bool reduce,
         std::string algo,
         nb::object degree_obj,
         nb::object swedish_obj,
         bool verbose,
         bool backend_stdout,
         bool kcritical,
         std::string filtration_container) {
        if (!multipers::multi_critical_interface_available()) {
          throw std::runtime_error("multi_critical in-memory interface is not available in this build.");
        }

        bool use_logpath = algo != "path";
        if (use_logpath && algo != "tree") {
          throw std::runtime_error("Algo should be path or tree.");
        }
        bool swedish = swedish_obj.is_none() ? !degree_obj.is_none() : nb::cast<bool>(swedish_obj);
        auto mp_module = nb::module_::import_("multipers");
        nb::object slicer_factory = mp_module.attr("Slicer");
        nb::object new_slicer_type = slicer_factory(slicer,
                                                    "return_type_only"_a = true,
                                                    "kcritical"_a = kcritical,
                                                    "filtration_container"_a = filtration_container);
        nb::object input_slicer = slicer;
        input_slicer = slicer.attr("astype")("vineyard"_a = false,
                                             "kcritical"_a = true,
                                             "dtype"_a = nb::module_::import_("numpy").attr("float64"),
                                             "col"_a = slicer.attr("col_type"),
                                             "pers_backend"_a = "matrix",
                                             "filtration_container"_a = "contiguous");

        auto input = mpmc::input_from_kcritical_slicer(input_slicer);
        if (!reduce) {
          auto out = multipers::multi_critical_resolution_interface<int>(input, use_logpath, true, backend_stdout);
          return mpmc::output_to_slicer(new_slicer_type, out, false, -1);
        }
        if (degree_obj.is_none()) {
          auto outs =
              multipers::multi_critical_minpres_all_interface<int>(input, use_logpath, true, backend_stdout, swedish);
          nb::tuple result = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)(outs.size() > 0 ? outs.size() - 1 : 0)));
          for (size_t i = 1; i < outs.size(); ++i) {
            nb::object value = mpmc::output_to_slicer(new_slicer_type, outs[i], true, (int)(i - 1));
            PyTuple_SET_ITEM(result.ptr(), (Py_ssize_t)(i - 1), value.release().ptr());
          }
          return nb::object(result);
        }
        int degree = nb::cast<int>(degree_obj);
        auto out = multipers::multi_critical_minpres_interface<int>(
            input, degree + 1, use_logpath, true, backend_stdout, swedish);
        return mpmc::output_to_slicer(new_slicer_type, out, true, degree);
      },
      "slicer"_a,
      "reduce"_a = false,
      "algo"_a = "path",
      "degree"_a = nb::none(),
      "swedish"_a = nb::none(),
      "verbose"_a = false,
      "_backend_stdout"_a = false,
      "kcritical"_a = false,
      "filtration_container"_a = "contiguous");

  m.def(
      "resolution_from_packed",
      [](nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> dimensions,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> grade_indptr,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades_flat,
         bool use_tree,
         bool use_multi_chunk,
         bool verbose,
         bool backend_stdout) {
        if (!multipers::multi_critical_interface_available()) {
          throw std::runtime_error("multi_critical in-memory interface is not available in this build.");
        }
        auto input = mpmc::input_from_packed(boundary_indptr, boundary_flat, dimensions, grade_indptr, grades_flat);
        multipers::multi_critical_interface_output<int> output;
        {
          nb::gil_scoped_release release;
          output =
              multipers::multi_critical_resolution_interface<int>(input, use_tree, use_multi_chunk, backend_stdout);
        }
        return mpmc::output_to_raw_arrays(output);
      },
      "boundary_indptr"_a,
      "boundary_flat"_a,
      "dimensions"_a,
      "grade_indptr"_a,
      "grades_flat"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "_backend_stdout"_a = false);

  m.def(
      "minpres_from_packed",
      [](nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> dimensions,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> grade_indptr,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades_flat,
         int degree,
         bool use_tree,
         bool use_multi_chunk,
         bool verbose,
         bool swedish,
         bool backend_stdout) {
        if (!multipers::multi_critical_interface_available()) {
          throw std::runtime_error("multi_critical in-memory interface is not available in this build.");
        }
        auto input = mpmc::input_from_packed(boundary_indptr, boundary_flat, dimensions, grade_indptr, grades_flat);
        multipers::multi_critical_interface_output<int> output;
        {
          nb::gil_scoped_release release;
          output = multipers::multi_critical_minpres_interface<int>(
              input, degree + 1, use_tree, use_multi_chunk, backend_stdout, swedish);
        }
        return mpmc::output_to_raw_arrays(output, true);
      },
      "boundary_indptr"_a,
      "boundary_flat"_a,
      "dimensions"_a,
      "grade_indptr"_a,
      "grades_flat"_a,
      "degree"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "swedish"_a = true,
      "_backend_stdout"_a = false);

  m.def(
      "minpres_all_from_packed",
      [](nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> dimensions,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> grade_indptr,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> grades_flat,
         bool use_tree,
         bool use_multi_chunk,
         bool verbose,
         bool swedish,
         bool backend_stdout) {
        if (!multipers::multi_critical_interface_available()) {
          throw std::runtime_error("multi_critical in-memory interface is not available in this build.");
        }
        auto input = mpmc::input_from_packed(boundary_indptr, boundary_flat, dimensions, grade_indptr, grades_flat);
        std::vector<multipers::multi_critical_interface_output<int>> outputs;
        {
          nb::gil_scoped_release release;
          outputs = multipers::multi_critical_minpres_all_interface<int>(
              input, use_tree, use_multi_chunk, backend_stdout, swedish);
        }
        nb::tuple result = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)(outputs.size() > 0 ? outputs.size() - 1 : 0)));
        for (size_t i = 1; i < outputs.size(); ++i) {
          nb::object value = mpmc::output_to_raw_arrays(outputs[i], true);
          PyTuple_SET_ITEM(result.ptr(), (Py_ssize_t)(i - 1), value.release().ptr());
        }
        return result;
      },
      "boundary_indptr"_a,
      "boundary_flat"_a,
      "dimensions"_a,
      "grade_indptr"_a,
      "grades_flat"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "swedish"_a = true,
      "_backend_stdout"_a = false);

  m.def(
      "resolution_from_ptr",
      [](intptr_t input_ptr, bool use_tree, bool use_multi_chunk, bool verbose, bool backend_stdout) {
        std::unique_ptr<multipers::packed_multi_critical_bridge_input> input_bridge(
            reinterpret_cast<multipers::packed_multi_critical_bridge_input*>(input_ptr));
        multipers::multi_critical_interface_output<int> output;
        {
          nb::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(multipers::multi_critical_detail::multi_critical_interface_mutex());
          auto matrices = mpmc::compute_free_resolution_matrices_from_bridge(
              *input_bridge, use_tree, use_multi_chunk, backend_stdout);
          if (matrices.size() > 1) {
            std::vector<multipers::multi_critical_detail::Graded_matrix> shifted_matrices(matrices.begin(),
                                                                                          matrices.end() - 1);
            output = multipers::multi_critical_detail::convert_chain_complex<int>(shifted_matrices);
          }
        }
        return mpmc::output_to_raw_arrays(output);
      },
      "input_ptr"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "_backend_stdout"_a = false);

  m.def(
      "resolution_from_ptr_with_stats",
      [](intptr_t input_ptr, bool use_tree, bool use_multi_chunk, bool verbose, bool backend_stdout) {
        std::unique_ptr<multipers::packed_multi_critical_bridge_input> input_bridge(
            reinterpret_cast<multipers::packed_multi_critical_bridge_input*>(input_ptr));
        multipers::multi_critical_interface_output<int> output;
        mpmc::ptr_bridge_stats stats;
        {
          nb::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(multipers::multi_critical_detail::multi_critical_interface_mutex());
          auto matrices = mpmc::compute_free_resolution_matrices_from_bridge(
              *input_bridge, use_tree, use_multi_chunk, backend_stdout, &stats);
          if (matrices.size() > 1) {
            std::vector<multipers::multi_critical_detail::Graded_matrix> shifted_matrices(matrices.begin(),
                                                                                          matrices.end() - 1);
            output = multipers::multi_critical_detail::convert_chain_complex<int>(shifted_matrices);
          }
        }
        auto t_output = mpmc::Clock::now();
        nb::tuple raw = mpmc::output_to_raw_arrays(output);
        stats.output_pack_s += mpmc::elapsed_seconds(t_output);
        return nb::make_tuple(std::move(raw), mpmc::stats_to_dict(stats));
      },
      "input_ptr"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "_backend_stdout"_a = false);

  m.def(
      "minpres_from_ptr",
      [](intptr_t input_ptr,
         int degree,
         bool use_tree,
         bool use_multi_chunk,
         bool verbose,
         bool swedish,
         bool backend_stdout) {
        std::unique_ptr<multipers::packed_multi_critical_bridge_input> input_bridge(
            reinterpret_cast<multipers::packed_multi_critical_bridge_input*>(input_ptr));
        multipers::multi_critical_interface_output<int> output;
        {
          nb::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(multipers::multi_critical_detail::multi_critical_interface_mutex());
          auto matrices = mpmc::compute_free_resolution_matrices_from_bridge(
              *input_bridge, use_tree, use_multi_chunk, backend_stdout);
          if (matrices.size() >= 2) {
            const int matrix_index = static_cast<int>(matrices.size()) - 1 - degree;
            if (matrix_index >= 1 && matrix_index < static_cast<int>(matrices.size())) {
              auto first = matrices[(size_t)matrix_index - 1];
              auto second = matrices[(size_t)matrix_index];
              multipers::multi_critical_detail::Graded_matrix min_rep;
              const bool old_verbose = mpfree::verbose;
              mpfree::verbose = backend_stdout;
              mpfree::compute_minimal_presentation(first, second, min_rep, false, false);
              mpfree::verbose = old_verbose;
              output = multipers::multi_critical_detail::convert_minpres<int>(min_rep, degree);
            }
          }
        }
        return mpmc::output_to_raw_arrays(output, true);
      },
      "input_ptr"_a,
      "degree"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "swedish"_a = true,
      "_backend_stdout"_a = false);

  m.def(
      "minpres_from_ptr_with_stats",
      [](intptr_t input_ptr,
         int degree,
         bool use_tree,
         bool use_multi_chunk,
         bool verbose,
         bool swedish,
         bool backend_stdout) {
        std::unique_ptr<multipers::packed_multi_critical_bridge_input> input_bridge(
            reinterpret_cast<multipers::packed_multi_critical_bridge_input*>(input_ptr));
        multipers::multi_critical_interface_output<int> output;
        mpmc::ptr_bridge_stats stats;
        {
          nb::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(multipers::multi_critical_detail::multi_critical_interface_mutex());
          auto matrices = mpmc::compute_free_resolution_matrices_from_bridge(
              *input_bridge, use_tree, use_multi_chunk, backend_stdout, &stats);
          if (matrices.size() >= 2) {
            const int matrix_index = static_cast<int>(matrices.size()) - 1 - degree;
            if (matrix_index >= 1 && matrix_index < static_cast<int>(matrices.size())) {
              auto first = matrices[(size_t)matrix_index - 1];
              auto second = matrices[(size_t)matrix_index];
              multipers::multi_critical_detail::Graded_matrix min_rep;
              const bool old_verbose = mpfree::verbose;
              mpfree::verbose = backend_stdout;
              auto t_minpres = mpmc::Clock::now();
              mpfree::compute_minimal_presentation(first, second, min_rep, false, false);
              stats.minpres_s += mpmc::elapsed_seconds(t_minpres);
              mpfree::verbose = old_verbose;
              output = multipers::multi_critical_detail::convert_minpres<int>(min_rep, degree);
            }
          }
        }
        auto t_output = mpmc::Clock::now();
        nb::tuple raw = mpmc::output_to_raw_arrays(output, true);
        stats.output_pack_s += mpmc::elapsed_seconds(t_output);
        return nb::make_tuple(std::move(raw), mpmc::stats_to_dict(stats));
      },
      "input_ptr"_a,
      "degree"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "swedish"_a = true,
      "_backend_stdout"_a = false);

  m.def(
      "minpres_all_from_ptr",
      [](intptr_t input_ptr, bool use_tree, bool use_multi_chunk, bool verbose, bool swedish, bool backend_stdout) {
        std::unique_ptr<multipers::packed_multi_critical_bridge_input> input_bridge(
            reinterpret_cast<multipers::packed_multi_critical_bridge_input*>(input_ptr));
        std::vector<multipers::multi_critical_interface_output<int>> outputs;
        {
          nb::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(multipers::multi_critical_detail::multi_critical_interface_mutex());
          auto matrices = mpmc::compute_free_resolution_matrices_from_bridge(
              *input_bridge, use_tree, use_multi_chunk, backend_stdout);
          if (matrices.size() >= 2) {
            outputs.reserve(matrices.size() - 1);
            const bool old_verbose = mpfree::verbose;
            mpfree::verbose = backend_stdout;
            for (std::size_t i = 0; i + 1 < matrices.size(); ++i) {
              auto first = matrices[i];
              auto second = matrices[i + 1];
              multipers::multi_critical_detail::Graded_matrix min_rep;
              mpfree::compute_minimal_presentation(first, second, min_rep, false, false);
              outputs.push_back(multipers::multi_critical_detail::convert_minpres<int>(min_rep, static_cast<int>(i)));
            }
            mpfree::verbose = old_verbose;
          }
        }
        nb::tuple result = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)(outputs.size() > 0 ? outputs.size() - 1 : 0)));
        for (size_t i = 1; i < outputs.size(); ++i) {
          nb::object value = mpmc::output_to_raw_arrays(outputs[i], true);
          PyTuple_SET_ITEM(result.ptr(), (Py_ssize_t)(i - 1), value.release().ptr());
        }
        return result;
      },
      "input_ptr"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "swedish"_a = true,
      "_backend_stdout"_a = false);

  m.def(
      "minpres_all_from_ptr_with_stats",
      [](intptr_t input_ptr, bool use_tree, bool use_multi_chunk, bool verbose, bool swedish, bool backend_stdout) {
        std::unique_ptr<multipers::packed_multi_critical_bridge_input> input_bridge(
            reinterpret_cast<multipers::packed_multi_critical_bridge_input*>(input_ptr));
        std::vector<multipers::multi_critical_interface_output<int>> outputs;
        mpmc::ptr_bridge_stats stats;
        {
          nb::gil_scoped_release release;
          std::lock_guard<std::mutex> lock(multipers::multi_critical_detail::multi_critical_interface_mutex());
          auto matrices = mpmc::compute_free_resolution_matrices_from_bridge(
              *input_bridge, use_tree, use_multi_chunk, backend_stdout, &stats);
          if (matrices.size() >= 2) {
            outputs.reserve(matrices.size() - 1);
            const bool old_verbose = mpfree::verbose;
            mpfree::verbose = backend_stdout;
            auto t_minpres = mpmc::Clock::now();
            for (std::size_t i = 0; i + 1 < matrices.size(); ++i) {
              auto first = matrices[i];
              auto second = matrices[i + 1];
              multipers::multi_critical_detail::Graded_matrix min_rep;
              mpfree::compute_minimal_presentation(first, second, min_rep, false, false);
              outputs.push_back(multipers::multi_critical_detail::convert_minpres<int>(min_rep, static_cast<int>(i)));
            }
            stats.minpres_s += mpmc::elapsed_seconds(t_minpres);
            mpfree::verbose = old_verbose;
          }
        }
        auto t_output = mpmc::Clock::now();
        nb::tuple result = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)(outputs.size() > 0 ? outputs.size() - 1 : 0)));
        for (size_t i = 1; i < outputs.size(); ++i) {
          nb::object value = mpmc::output_to_raw_arrays(outputs[i], true);
          PyTuple_SET_ITEM(result.ptr(), (Py_ssize_t)(i - 1), value.release().ptr());
        }
        stats.output_pack_s += mpmc::elapsed_seconds(t_output);
        return nb::make_tuple(std::move(result), mpmc::stats_to_dict(stats));
      },
      "input_ptr"_a,
      "use_tree"_a = false,
      "use_multi_chunk"_a = true,
      "verbose"_a = false,
      "swedish"_a = true,
      "_backend_stdout"_a = false);
#endif
}
