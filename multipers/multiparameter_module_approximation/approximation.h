
/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence class, renaming and cleanup.
 *      - 2022/05 Hannah Schreiber: Addition of Summand class and Module class.
 */

/**
 * @file approximation.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the functions related to the approximation of n-modules.
 */

#ifndef APPROXIMATION_H_INCLUDED
#define APPROXIMATION_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <ranges>

#include "../gudhi/Persistence_slices_interface.h"
// #include "../gudhi/gudhi/Dynamic_multi_parameter_filtration.h"
#include "../gudhi/gudhi/Multi_persistence/Line.h"
#include "../gudhi/gudhi/Multi_persistence/Box.h"
#include "../gudhi/gudhi/Multi_persistence/Module.h"
// #include "../tensor/tensor.h"
// #include "gudhi/Multi_filtration/multi_filtration_utils.h"
// #include "utilities.h"
#include "debug.h"

namespace Gudhi {
namespace multiparameter {
namespace mma {

using Debug::Timer;
using Gudhi::multi_persistence::Box;
using Gudhi::multi_persistence::Line;
using Gudhi::multi_persistence::Module;

using value_type = double;
using filtration_type = multipers::tmp_interface::Filtration_value<value_type>;

inline void threshold_filters_list(std::vector<filtration_type> &filtersList, const Box<value_type> &box) {
  return;
  for (unsigned int i = 0; i < filtersList.size(); i++) {
    for (unsigned int p = 0; p < filtersList[i].num_parameters(); ++p) {
      value_type &value = filtersList[i](0, p);
      value = std::min(std::max(value, box.get_lower_corner()[i]), box.get_upper_corner()[i]);
    }
  }
}

template <class Filtration_value, int axis = 0, bool sign = true>
class LineIterator {
 public:
  using value_type = typename Filtration_value::value_type;
  LineIterator(const typename Line<value_type>::Point_t &basepoint,
               const typename Line<value_type>::Point_t &direction,
               value_type precision,
               int num_iterations)
      : precision(precision), remaining_iterations(num_iterations), current_line(std::move(basepoint), direction) {};

  LineIterator<Filtration_value, axis, sign> &operator++() {
    //
    auto &basepoint = current_line.base_point();
    if (this->is_finished()) return *this;
    // If we didn't reached the end, go to the next line
    basepoint[axis] += sign ? precision : -precision;
    --remaining_iterations;
    return *this;
  }

  const Line<value_type> &operator*() const { return current_line; }

  LineIterator<Filtration_value, axis, sign> &next(std::size_t i) {
    auto &basepoint = current_line.base_point();
    if (this->is_finished()) return *this;
    // If we didn't reached the end, go to the next line
    basepoint[i] += sign ? precision : -precision;
    --remaining_iterations;
    return *this;
  }

  bool is_finished() const { return remaining_iterations <= 0; }

 private:
  const value_type precision;
  int remaining_iterations;
  Line<value_type> current_line;
};

template <class Filtration_value, int axis_ = 0, bool sign = true, class Slicer>
inline void __add_vineyard_trajectory_to_module(Module<typename Filtration_value::value_type> &module,
                                                Slicer &&slicer,
                                                LineIterator<Filtration_value, axis_, sign> &line_iterator,
                                                const bool threshold,
                                                int axis = 0) {
  static_assert(std::is_same_v<typename Filtration_value::value_type, typename Slicer::Filtration_value::value_type>);
  using value_type = typename Filtration_value::value_type;
  // Line iterator should be on the biggest axis
  constexpr const bool verbose = false;
  constexpr const bool verbose2 = false;
  while (!line_iterator.is_finished()) {
    const Line<value_type> &new_line = (axis_ >= 0) ? *(++line_iterator) : *line_iterator.next(axis);
    if constexpr (verbose) std::cout << "----------------------------------------------" << std::endl;
    if constexpr (verbose) std::cout << "Line basepoint " << new_line.base_point() << std::endl;
    slicer.push_to(new_line);

    slicer.update_persistence_computation();
    if constexpr (verbose2) std::cout << slicer << std::endl;
    module.add_barcode(new_line, slicer.template get_flat_barcode<true>(), threshold);
  };
};

template <class Filtration_value, class Slicer = multipers::tmp_interface::SimplicialVineMatrixTruc<>>
void _rec_mma(Module<typename Filtration_value::value_type> &module,
              typename Line<value_type>::Point_t &basepoint,
              const std::vector<int> &grid_size,
              int dim_to_iterate,
              Slicer &&current_persistence,
              const value_type precision,
              bool threshold) {
  if (dim_to_iterate <= 0) {
    LineIterator<Filtration_value, 0> line_iterator(std::move(basepoint), precision, grid_size[0]);
    __add_vineyard_trajectory_to_module<Filtration_value, 0, Slicer>(
        module, std::move(current_persistence), line_iterator, threshold);
    return;
  }
  Slicer pers_copy;
  typename Line<value_type>::Point_t basepoint_copy;
  for (int i = 0; i < grid_size[dim_to_iterate]; ++i) {
    // TODO : multithread, but needs matrix to be thread safe + put mutex on
    // module
    pers_copy = current_persistence;
    basepoint_copy = basepoint;
    _rec_mma(module, basepoint_copy, grid_size, dim_to_iterate - 1, pers_copy, precision, threshold);
    basepoint[dim_to_iterate] += precision;
    // current_persistence.push_to(Line(basepoint));
    // current_persistence.update_persistence_computation();
  }
}

template <int axis, class Filtration_value, class Slicer>
void _rec_mma2(Module<typename Filtration_value::value_type> &module,
               typename Line<typename Filtration_value::value_type>::Point_t &&basepoint,
               const Filtration_value &direction,
               const std::vector<int> &grid_size,
               const std::vector<bool> &signs,
               int dim_to_iterate,
               Slicer &&current_persistence,
               const value_type precision,
               bool threshold) {
  static_assert(std::is_same_v<typename Filtration_value::value_type, typename Slicer::T>);

  if (dim_to_iterate <= axis) {
    if (signs[axis]) {
      LineIterator<Filtration_value, axis, true> line_iterator(
          std::move(basepoint), direction, precision, grid_size[axis]);
      __add_vineyard_trajectory_to_module<Filtration_value, axis, true, Slicer>(
          module, std::move(current_persistence), line_iterator, threshold);
    } else {
      LineIterator<Filtration_value, axis, false> line_iterator(
          std::move(basepoint), direction, precision, grid_size[axis]);
      __add_vineyard_trajectory_to_module<Filtration_value, axis, false, Slicer>(
          module, std::move(current_persistence), line_iterator, threshold);
    }

    return;
  }
  if (grid_size[dim_to_iterate] == 0) {
    // no need to copy basepoint, we just skip the dim here
    _rec_mma2<axis, Filtration_value, Slicer>(module,
                                              std::move(basepoint),
                                              direction,
                                              grid_size,
                                              signs,
                                              dim_to_iterate - 1,
                                              std::move(current_persistence),
                                              precision,
                                              threshold);
    return;
  }
  for (int i = 0; i < grid_size[dim_to_iterate]; ++i) {
    // TODO : multithread, but needs matrix to be thread safe + put mutex on
    // module
    const bool is_last = i + 1 == grid_size[dim_to_iterate];
    if (is_last) {
      _rec_mma2<axis, Filtration_value, Slicer>(module,
                                                std::move(basepoint),
                                                direction,
                                                grid_size,
                                                signs,
                                                dim_to_iterate - 1,
                                                std::move(current_persistence),
                                                precision,
                                                threshold,
                                                line_step);
    } else {
      _rec_mma2<axis, Filtration_value, typename Slicer::Thread_safe>(
          module,
          typename Line<typename Filtration_value::value_type>::Point_t(basepoint),
          direction,
          grid_size,
          signs,
          dim_to_iterate - 1,
          current_persistence.weak_copy(),
          precision,
          threshold,
          line_step);
      basepoint[dim_to_iterate] += signs[dim_to_iterate] ? precision : -precision;
    }
    // current_persistence.push_to(Line(basepoint));
    // current_persistence.update_persistence_computation();
  }
}

template <class Slicer, typename value_type>
Module<value_type> multiparameter_module_approximation(Slicer &slicer,
                                                       const typename Line<value_type>::Point_t &direction,
                                                       const value_type precision,
                                                       Box<value_type> &box,
                                                       const bool threshold,
                                                       const bool complete,
                                                       const bool verbose,
                                                       const int n_jobs) {
  static_assert(std::is_same_v<typename Slicer::Filtration_value::value_type,
                               value_type>);  // Value type can be exposed to python interface.
  if (verbose) std::cout << "Starting Module Approximation" << std::endl;
  /* using Filtration_value = Slicer::Filtration_value; */

  oneapi::tbb::task_arena arena(n_jobs);
  return arena.execute([&] {
    typename Box<value_type>::Point_t basepoint = box.get_lower_corner();
    const std::size_t num_parameters = box.get_dimension();
    std::vector<int> grid_size(num_parameters);
    std::vector<bool> signs(num_parameters);
    int signs_shifts = 0;
    int arg_max_signs_shifts = -1;
    for (std::size_t i = 0; i < num_parameters; i++) {
      auto &a = box.get_lower_corner()[i];
      auto &b = box.get_upper_corner()[i];
      grid_size[i] = static_cast<int>(std::ceil((std::fabs(b - a) / precision))) + 1;
      signs[i] = b > a;
      if (b < a) {
        std::swap(a, b);
        int local_shift;
        if (!direction.size())
          local_shift = grid_size[i];
        else {
          local_shift = direction[i] > 0 ? static_cast<int>(std::ceil(grid_size[i] / direction[i])) : 0;
        }
        if (local_shift > signs_shifts) {
          signs_shifts = std::max(signs_shifts, local_shift);
          arg_max_signs_shifts = i;
        }
      }

      // fix the box
    }
    if (signs_shifts > 0) {
      for (std::size_t i = 0; i < num_parameters; i++)
        grid_size[i] += signs_shifts;  // this may be too much for large num_parameters
      grid_size[arg_max_signs_shifts] = 1;
      if (verbose)
        std::cout << "Had to flatten/shift coordinate " << arg_max_signs_shifts << " by " << signs_shifts << std::endl;
    }
    Module<value_type> out(box);
    box.inflate(2 * precision);  // for infinte summands

    if (verbose) std::cout << "Num parameters : " << num_parameters << std::endl;
    if (verbose) std::cout << "Box : " << box << std::endl;
    if (num_parameters < 1) return out;

    // first line to compute
    // TODO: change here
    // for (auto i = 0u; i < basepoint.size() - 1; i++)
    //   basepoint[i] -= box.get_upper_corner().back();
    // basepoint.back() = 0;
    Line<value_type> current_line(basepoint, direction);
    if (verbose) std::cout << "First line basepoint " << basepoint << std::endl;

    {
      Timer timer("Initializing mma...\n", verbose);
      // fills the first barcode
      slicer.push_to(current_line);
      slicer.initialize_persistence_computation(false);
      auto barcode = slicer.template get_flat_barcode<true>();
      auto num_bars = 0;
      for (const auto &b : barcode) num_bars += b.size();
      out.resize(num_bars, num_parameters);
      out.set_max_dimension(barcode.size() - 1);
      std::size_t i = 0;
      for (unsigned int dim = 0; dim < barcode.size(); ++dim) {
        for ([[maybe_unused]] const auto &bar : barcode[dim]) {
          out.get_summand(i).set_dimension(dim);
          ++i;
        }
      }
      out.add_barcode(current_line, barcode, threshold);

      if (verbose) std::cout << "Instantiated " << num_bars << " summands" << std::endl;
    }
    // TODO : change here
    // std::vector<int> grid_size(num_parameters - 1);
    // auto h = box.get_upper_corner().back() - box.get_lower_corner().back();
    // for (int i = 0; i < num_parameters - 1; i++) {
    //   auto a = box.get_lower_corner()[i];
    //   auto b = box.get_upper_corner()[i];
    //   grid_size[i] =
    //       static_cast<unsigned int>(std::ceil((std::abs(b - a + h) /
    //       precision)));
    // }
    // TODO : change here
    if (verbose) {
      std::cout << "Grid size ";
      for (auto v : grid_size) std::cout << v << " ";
      std::cout << " Signs ";
      if (signs.empty()) {
        std::cout << "[]";
      } else {
        std::cout << "[";
        for (std::size_t i = 0; i < signs.size() - 1; i++) {
          std::cout << signs[i] << ", ";
        }
        std::cout << signs.back() << "]";
      }
      std::cout << std::endl;
      std::cout << "Max error " << precision << std::endl;
    }

    {
      Timer timer("Computing mma...", verbose);
      // actual computation. -1 as line grid is n-1 dim, -1 as we start from 0
      // _rec_mma(out, basepoint, grid_size, num_parameters - 2, slicer,
      // precision,
      //          threshold);
      // TODO : change here

      for (std::size_t i = 1; i < num_parameters; i++) {
        // the loop is on the faces of the lower box
        // should be parallelizable, up to a mutex on out
        if (direction.size() && direction[i] == 0.0) continue;  // skip faces with codim d_i=0
        auto temp_grid_size = grid_size;
        temp_grid_size[i] = 0;
        if (verbose) {
          std::cout << "Face " << i << "/" << num_parameters << " with grid size ";
          for (auto v : temp_grid_size) std::cout << v << " ";
          std::cout << std::endl;
        }
        // if (!direction.size() || direction[0] > 0)
        _rec_mma2<0>(out,
                     typename Line<value_type>::Point_t(basepoint),
                     direction,
                     temp_grid_size,
                     signs,
                     num_parameters - 1,
                     slicer.weak_copy(),
                     precision,
                     threshold);
      }
      // last one, we can destroy basepoint & cie
      if (!direction.size() || direction[0] > 0) {
        grid_size[0] = 0;
        if (verbose) {
          std::cout << "Face " << num_parameters << "/" << num_parameters << " with grid size ";
          for (auto v : grid_size) std::cout << v << " ";
          std::cout << std::endl;
        }
        _rec_mma2<1>(out,
                     std::move(basepoint),
                     direction,
                     grid_size,
                     signs,
                     num_parameters - 1,
                     std::move(slicer),
                     precision,
                     threshold);
      }
    }

    {  // for Timer
      Timer timer("Cleaning output ... ", verbose);
      out.clean();
      if (complete) {
        if (verbose) std::cout << "Completing output ...";
        for (std::size_t i = 0; i < num_parameters; i++) out.fill(precision);
      }
    }  // Timer death
    return out;
  });
};

}  // namespace mma
}  // namespace multiparameter
}  // namespace Gudhi

#endif  // APPR
