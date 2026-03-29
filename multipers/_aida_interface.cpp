#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vector>

#include "ext_interface/aida_interface.hpp"

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
namespace mpaida {

template <typename T>
std::vector<T> cast_vector(nb::handle h) {
  return nb::cast<std::vector<T>>(h);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(nb::handle h) {
  return nb::cast<std::vector<std::vector<T>>>(h);
}

std::vector<std::pair<double, double>> cast_pair_vector(nb::handle h) {
  std::vector<std::pair<double, double>> out;
  for (const auto& row : cast_matrix<double>(h)) {
    out.emplace_back(row[0], row[1]);
  }
  return out;
}

}  // namespace mpaida
#endif

NB_MODULE(_aida_interface, m) {
  m.def("_is_available", []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
    return false;
#else
    return true;
#endif
  });

  m.def(
      "aida",
      [](nb::object s, bool sort, bool verbose, bool progress) {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
        throw std::runtime_error("AIDA in-memory interface is disabled at compile time.");
#else
        auto slicer_module = nb::module_::import_("multipers.slicer");
        if (!nb::cast<bool>(slicer_module.attr("is_slicer")(s))) {
          throw std::runtime_error("Input has to be a slicer.");
        }
        if (!nb::cast<bool>(s.attr("is_minpres"))) {
          throw std::runtime_error("AIDA takes a minimal presentation as an input.");
        }
        if (nb::cast<int>(s.attr("num_parameters")) != 2) {
          throw std::runtime_error("AIDA is only compatible with 2-parameter minimal presentations.");
        }

        bool is_squeezed = nb::cast<bool>(s.attr("is_squeezed"));
        int degree = nb::cast<int>(s.attr("minpres_degree"));
        if (sort) {
          s = s.attr("to_colexical")();
        }

        auto F = mpaida::cast_matrix<double>(
            nb::module_::import_("numpy").attr("asarray")(s.attr("get_filtrations")("view"_a = false)));
        auto D = mpaida::cast_vector<int>(s.attr("get_dimensions")());

        std::vector<std::pair<double, double>> row_degree, col_degree;
        for (size_t i = 0; i < D.size(); ++i) {
          if (D[i] == degree)
            row_degree.emplace_back(F[i][0], F[i][1]);
          else if (D[i] == degree + 1)
            col_degree.emplace_back(F[i][0], F[i][1]);
        }

        auto boundaries = nb::cast<std::vector<nb::object>>(s.attr("get_boundaries")());
        std::vector<std::vector<int>> matrix;
        size_t start = std::lower_bound(D.begin(), D.end(), degree + 1) - D.begin();
        size_t stop = std::lower_bound(D.begin(), D.end(), degree + 2) - D.begin();
        matrix.reserve(stop - start);
        for (size_t i = start; i < stop; ++i) {
          matrix.push_back(mpaida::cast_vector<int>(boundaries[i]));
        }

        aida::AIDA_functor functor;
        functor.config.show_info = verbose;
        functor.config.sort_output = false;
        functor.config.sort = sort;
        functor.config.progress = progress;
        auto input = aida::multipers_interface_input<int>(col_degree, row_degree, matrix);
        auto output = functor.multipers_interface(input);

        auto mp_module = nb::module_::import_("multipers");
        nb::object slicer_type = mp_module.attr("Slicer")(
            s, "return_type_only"_a = true, "dtype"_a = nb::module_::import_("numpy").attr("float64"));
        nb::list out;

        for (size_t i = 0; i < output.summands.size(); ++i) {
          const auto& summand = output.summands[i];
          std::vector<int> dims(summand.row_degrees.size() + summand.col_degrees.size());
          for (size_t j = 0; j < summand.row_degrees.size(); ++j) dims[j] = degree;
          for (size_t j = 0; j < summand.col_degrees.size(); ++j) dims[summand.row_degrees.size() + j] = degree + 1;
          nb::list boundary_container;
          for (size_t j = 0; j < summand.row_degrees.size(); ++j) boundary_container.append(nb::list());
          for (const auto& row : summand.matrix) boundary_container.append(nb::cast(row));
          std::vector<std::vector<double>> filtration_values;
          filtration_values.reserve(summand.row_degrees.size() + summand.col_degrees.size());
          for (const auto& p : summand.row_degrees) filtration_values.push_back({p.first, p.second});
          for (const auto& p : summand.col_degrees) filtration_values.push_back({p.first, p.second});
          nb::object slicer = slicer_type(boundary_container, nb::cast(dims), nb::cast(filtration_values));
          slicer.attr("minpres_degree") = degree;
          if (is_squeezed) {
            slicer.attr("filtration_grid") = s.attr("filtration_grid");
            slicer.attr("_clean_filtration_grid")();
          }
          out.append(slicer);
        }
        return out;
#endif
      },
      "s"_a,
      "sort"_a = true,
      "verbose"_a = false,
      "progress"_a = false);
}
