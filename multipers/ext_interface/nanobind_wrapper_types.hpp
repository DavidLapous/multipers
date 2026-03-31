#pragma once

#include <nanobind/nanobind.h>

namespace multipers::nanobind_helpers {

template <typename Slicer>
struct PySlicer {
  Slicer truc;
  nanobind::object filtration_grid;
  nanobind::object generator_basis;
  int minpres_degree;

  PySlicer() : filtration_grid(nanobind::none()), generator_basis(nanobind::none()), minpres_degree(-1) {}
};

template <typename Interface, typename T>
struct PySimplexTree {
  Interface tree;
  nanobind::object filtration_grid;

  PySimplexTree() : filtration_grid(nanobind::list()) {}
};

}  // namespace multipers::nanobind_helpers
