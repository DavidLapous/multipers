#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>
#include <utility>
#include <vector>

#include "ext_interface/multi_critical_interface.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpmc {

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

struct ScopedStdoutSilence {
  bool active = false;
  int saved_stdout = -1;
  int devnull = -1;
  std::unique_lock<std::mutex> lock;

  static std::mutex& mutex() {
    static std::mutex m;
    return m;
  }

  explicit ScopedStdoutSilence(bool silence)
      : active(silence), lock(mutex(), std::defer_lock) {
    if (!active) return;
    lock.lock();
    std::fflush(stdout);
    devnull = open("/dev/null", O_WRONLY);
    saved_stdout = dup(STDOUT_FILENO);
    if (devnull >= 0 && saved_stdout >= 0) {
      dup2(devnull, STDOUT_FILENO);
    } else {
      active = false;
      if (saved_stdout >= 0) close(saved_stdout);
      if (devnull >= 0) close(devnull);
    }
  }

  ~ScopedStdoutSilence() {
    if (!lock.owns_lock()) return;
    if (active) {
      std::fflush(stdout);
      dup2(saved_stdout, STDOUT_FILENO);
      close(saved_stdout);
      close(devnull);
    }
  }
};

std::vector<std::vector<int>> boundaries_from_packed(
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> boundary_indptr,
    nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> boundary_flat) {
  if (boundary_indptr.shape(0) == 0) {
    return {};
  }
  std::vector<std::vector<int>> boundaries((size_t) boundary_indptr.shape(0) - 1);
  const int64_t* indptr = boundary_indptr.data();
  const int32_t* flat = boundary_flat.data();
  for (size_t cell = 0; cell + 1 < (size_t) boundary_indptr.shape(0); ++cell) {
    const int64_t begin = indptr[cell];
    const int64_t end = indptr[cell + 1];
    auto& boundary = boundaries[cell];
    boundary.reserve((size_t) std::max<int64_t>(end - begin, 0));
    for (int64_t idx = begin; idx < end; ++idx) {
      boundary.push_back((int) flat[idx]);
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
  input.dimensions.reserve((size_t) dimensions.shape(0));
  input.boundaries = std::move(boundaries);
  input.filtration_values.resize(input.boundaries.size());
  for (size_t i = 0; i < (size_t) dimensions.shape(0); ++i) {
    input.dimensions.push_back((int) dimensions(i));
  }
  const int64_t* indptr = grade_indptr.data();
  for (size_t cell = 0; cell < input.boundaries.size(); ++cell) {
    auto& out = input.filtration_values[cell];
    const int64_t begin = indptr[cell];
    const int64_t end = indptr[cell + 1];
    out.reserve((size_t) std::max<int64_t>(end - begin, 0));
    for (int64_t row = begin; row < end; ++row) {
      out.emplace_back(grades_flat((size_t) row, 0), grades_flat((size_t) row, 1));
    }
  }
  return input;
}

nb::tuple output_to_raw_arrays(const multipers::multi_critical_interface_output<int>& output, bool mark_minpres = false) {
  std::vector<int64_t> boundary_indptr(output.boundaries.size() + 1, 0);
  size_t boundary_nnz = 0;
  for (size_t i = 0; i < output.boundaries.size(); ++i) {
    boundary_nnz += output.boundaries[i].size();
    boundary_indptr[i + 1] = (int64_t) boundary_nnz;
  }
  std::vector<int32_t> boundary_flat;
  boundary_flat.reserve(boundary_nnz);
  for (const auto& row : output.boundaries) {
    for (int value : row) {
      boundary_flat.push_back((int32_t) value);
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
  return nb::make_tuple(
      owned_array<int64_t>(std::move(boundary_indptr), {output.boundaries.size() + 1}),
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

NB_MODULE(_multi_critical_interface, m) {
  m.def("_is_available", []() {
    if (!multipers::multi_critical_interface_available()) return false;
    auto slicer_module = nb::module_::import_("multipers.slicer");
    for (nb::handle container : nb::iter(slicer_module.attr("available_filtration_container"))) {
      if (nb::cast<std::string>(container) == "Flat") return true;
    }
    return false;
  });

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
          mpmc::ScopedStdoutSilence silence(!backend_stdout);
          auto out = multipers::multi_critical_resolution_interface<int>(input, use_logpath, true, verbose);
          return mpmc::output_to_slicer(new_slicer_type, out, false, -1);
        }
        if (degree_obj.is_none()) {
          mpmc::ScopedStdoutSilence silence(!backend_stdout);
          auto outs = multipers::multi_critical_minpres_all_interface<int>(input, use_logpath, true, verbose, swedish);
          nb::tuple result = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)(outs.size() > 0 ? outs.size() - 1 : 0)));
          for (size_t i = 1; i < outs.size(); ++i) {
            nb::object value = mpmc::output_to_slicer(new_slicer_type, outs[i], true, (int)(i - 1));
            PyTuple_SET_ITEM(result.ptr(), (Py_ssize_t)(i - 1), value.release().ptr());
          }
          return nb::object(result);
        }
        int degree = nb::cast<int>(degree_obj);
        mpmc::ScopedStdoutSilence silence(!backend_stdout);
        auto out =
            multipers::multi_critical_minpres_interface<int>(input, degree + 1, use_logpath, true, verbose, swedish);
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
        auto input = mpmc::input_from_packed(
            boundary_indptr,
            boundary_flat,
            dimensions,
            grade_indptr,
            grades_flat);
        multipers::multi_critical_interface_output<int> output;
        {
          mpmc::ScopedStdoutSilence silence(!backend_stdout);
          nb::gil_scoped_release release;
          output = multipers::multi_critical_resolution_interface<int>(
              input,
              use_tree,
              use_multi_chunk,
              verbose);
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
        auto input = mpmc::input_from_packed(
            boundary_indptr,
            boundary_flat,
            dimensions,
            grade_indptr,
            grades_flat);
        multipers::multi_critical_interface_output<int> output;
        {
          mpmc::ScopedStdoutSilence silence(!backend_stdout);
          nb::gil_scoped_release release;
          output = multipers::multi_critical_minpres_interface<int>(
              input,
              degree + 1,
              use_tree,
              use_multi_chunk,
              verbose,
              swedish);
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
        auto input = mpmc::input_from_packed(
            boundary_indptr,
            boundary_flat,
            dimensions,
            grade_indptr,
            grades_flat);
        std::vector<multipers::multi_critical_interface_output<int>> outputs;
        {
          mpmc::ScopedStdoutSilence silence(!backend_stdout);
          nb::gil_scoped_release release;
          outputs = multipers::multi_critical_minpres_all_interface<int>(
              input,
              use_tree,
              use_multi_chunk,
              verbose,
              swedish);
        }
        nb::tuple result = nb::steal<nb::tuple>(
            PyTuple_New((Py_ssize_t)(outputs.size() > 0 ? outputs.size() - 1 : 0)));
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
}
