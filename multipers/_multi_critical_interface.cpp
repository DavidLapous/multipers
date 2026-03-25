#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
std::vector<std::vector<T>> cast_matrix(nb::handle h) {
  return nb::cast<std::vector<std::vector<T>>>(h);
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
          auto out = multipers::multi_critical_resolution_interface<int>(input, use_logpath, true, verbose);
          return mpmc::output_to_slicer(new_slicer_type, out, false, -1);
        }
        if (degree_obj.is_none()) {
          auto outs = multipers::multi_critical_minpres_all_interface<int>(input, use_logpath, true, verbose, swedish);
          nb::tuple result = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)(outs.size() > 0 ? outs.size() - 1 : 0)));
          for (size_t i = 1; i < outs.size(); ++i) {
            nb::object value = mpmc::output_to_slicer(new_slicer_type, outs[i], true, (int)(i - 1));
            PyTuple_SET_ITEM(result.ptr(), (Py_ssize_t)(i - 1), value.release().ptr());
          }
          return nb::object(result);
        }
        int degree = nb::cast<int>(degree_obj);
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
      "kcritical"_a = false,
      "filtration_container"_a = "contiguous");
}
