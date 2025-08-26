
/*    This file is part of the MMA Library -
 * https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT. See
 * file LICENSE for full license details. Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence
 * class, renaming and cleanup.
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <oneapi/tbb/parallel_for.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "utilities.h"

#include "debug.h"
#include <Persistence_slices_interface.h>
#include <gudhi/One_critical_filtration.h>
#include <gudhi/Multi_critical_filtration.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Line.h>
#include <tbb/parallel_for.h>
#include <tensor/tensor.h>

namespace Gudhi {
namespace multiparameter {
namespace mma {

using Debug::Timer;
using Gudhi::multi_persistence::Box;
using Gudhi::multi_persistence::Line;
template <typename T>
class Module;
template <typename T>
class Summand;

void threshold_filters_list(std::vector<value_type> &filtersList, const Box<value_type> &box);

template <typename value_type>
class Module {
 public:
  using filtration_type = Gudhi::multi_filtration::One_critical_filtration<value_type>;
  using module_type = std::vector<Summand<value_type>>;
  using image_type = std::vector<std::vector<value_type>>;
  using get_2dpixel_value_function_type = std::function<value_type(const typename module_type::const_iterator,
                                                                   const typename module_type::const_iterator,
                                                                   value_type,
                                                                   value_type)>;
  using get_pixel_value_function_type = std::function<value_type(const typename module_type::const_iterator,
                                                                 const typename module_type::const_iterator,
                                                                 std::vector<value_type> &)>;

  Module();
  Module(Box<value_type> &box);

  void resize(unsigned int size);
  Summand<value_type> &at(unsigned int index);
  Summand<value_type> &operator[](size_t index);

  const Summand<value_type> &operator[](const size_t index) const;
  template <class Barcode>
  void add_barcode(const Barcode &barcode);
  void add_barcode(const Line<value_type> &line,
                   const std::vector<std::pair<int, std::pair<value_type, value_type>>> &barcode,
                   const bool threshold);
  void add_barcode(const Line<value_type> &line,
                   const std::vector<std::pair<value_type, value_type>> &barcode,
                   const bool threshold);
  typename module_type::iterator begin();
  typename module_type::iterator end();
  typename module_type::const_iterator begin() const;
  typename module_type::const_iterator end() const;

  void clean();
  void fill(const value_type precision);

  std::vector<image_type> get_vectorization(const value_type delta,
                                            const value_type p,
                                            const bool normalize,
                                            const Gudhi::multi_persistence::Box<value_type> &box,
                                            unsigned int horizontalResolution,
                                            unsigned int verticalResolution);
  std::vector<image_type> get_vectorization(unsigned int horizontalResolution,
                                            unsigned int verticalResolution,
                                            get_2dpixel_value_function_type get_pixel_value) const;
  image_type get_vectorization_in_dimension(const dimension_type dimension,
                                            const value_type delta,
                                            const value_type p,
                                            const bool normalize,
                                            const Gudhi::multi_persistence::Box<value_type> &box,
                                            unsigned int horizontalResolution,
                                            unsigned int verticalResolution);
  image_type get_vectorization_in_dimension(const dimension_type dimension,
                                            unsigned int horizontalResolution,
                                            unsigned int verticalResolution,
                                            get_2dpixel_value_function_type get_pixel_value) const;
  std::vector<value_type> get_landscape_values(const std::vector<value_type> &x, const dimension_type dimension) const;
  image_type get_landscape(const dimension_type dimension,
                           const unsigned int k,
                           const Box<value_type> &box,
                           const std::vector<unsigned int> &resolution) const;
  std::vector<image_type> get_landscapes(const dimension_type dimension,
                                         const std::vector<unsigned int> ks,
                                         const Box<value_type> &box,
                                         const std::vector<unsigned int> &resolution) const;
  void add_summand(Summand<value_type> summand, int degree = -1);
  Box<value_type> get_box() const;
  void set_box(Box<value_type> box);
  unsigned int size() const;
  void infer_box(std::vector<filtration_type> &filters_list);
  dimension_type get_dimension() const;
  module_type get_summands_of_dimension(const int dimension) const;
  std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>>
  get_corners_of_dimension(const int dimension) const;
  MultiDiagram<filtration_type, value_type> get_barcode(const Line<value_type> &l,
                                                        const dimension_type dimension = -1,
                                                        const bool threshold = false) const;
  std::vector<std::vector<std::pair<value_type, value_type>>> get_barcode2(const Line<value_type> &l,
                                                                           const dimension_type dimension) const;
  std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> get_barcodes2(
      const std::vector<Line<value_type>> &lines,
      const dimension_type dimension = -1) const;
  MultiDiagrams<filtration_type, value_type> get_barcodes(const std::vector<Line<value_type>> &lines,
                                                          const dimension_type dimension = -1,
                                                          const bool threshold = false) const;
  MultiDiagrams<filtration_type, value_type> get_barcodes(const std::vector<filtration_type> &basepoints,
                                                          const dimension_type dimension = -1,
                                                          const bool threshold = false) const;
  std::vector<int> euler_curve(const std::vector<filtration_type> &points) const;

  inline Box<value_type> get_bounds() const;
  inline void rescale(const std::vector<value_type> &rescale_factors, int degree);
  inline void translate(const std::vector<value_type> &translation, int degree);

  std::vector<std::vector<value_type>> compute_pixels(const std::vector<std::vector<value_type>> &coordinates,
                                                      const std::vector<int> &degrees,
                                                      const Box<value_type> &box = {},
                                                      const value_type delta = 0.1,
                                                      const value_type p = 1,
                                                      const bool normalize = true,
                                                      const int n_jobs = 0);

  std::vector<value_type> get_interleavings(const Box<value_type> &box);
  using distance_to_idx_type = std::vector<std::vector<int>>;
  distance_to_idx_type compute_distance_idx_to(const std::vector<value_type> &pt, bool full) const;
  std::vector<value_type> compute_distance_to(const std::vector<value_type> &pt, bool negative) const;
  std::vector<std::vector<value_type>> compute_distances_to(const std::vector<std::vector<value_type>> &pt,
                                                            bool negative,
                                                            int n_jobs) const;
  template <typename dtype = value_type, typename indices_type = int32_t>
  void inline compute_distances_to(dtype *data_ptr,
                                   const std::vector<std::vector<value_type>> &pts,
                                   bool negative,
                                   int n_jobs) const;
  using distances_to_idx_type = std::vector<distance_to_idx_type>;
  distances_to_idx_type compute_distances_idx_to(const std::vector<std::vector<value_type>> &pt,
                                                 bool full,
                                                 int n_jobs) const;
  std::vector<value_type> compute_pixels_of_degree(const typename module_type::iterator start,
                                                   const typename module_type::iterator end,
                                                   const value_type delta,
                                                   const value_type p,
                                                   const bool normalize,
                                                   const Box<value_type> &box,
                                                   const std::vector<std::vector<value_type>> &coordinates,
                                                   const int n_jobs = 0);

  using idx_dump_type =
      std::vector<std::vector<std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>>>;
  idx_dump_type to_idx(const std::vector<std::vector<value_type>> &grid) const;
  Module<int64_t> grid_squeeze(const std::vector<std::vector<value_type>> &grid) const;

  std::vector<std::vector<std::vector<int>>> to_flat_idx(const std::vector<std::vector<value_type>> &grid) const;

  std::vector<int> inline get_degree_splits() const;

  inline friend bool operator==(const Module &a, const Module &b) {
    if (a.get_dimension() != b.get_dimension()) return false;
    if (a.box_ != b.box_) return false;
    if (a.size() != b.size()) return false;
    for (auto i : std::views::iota(0u, a.size())) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

 private:
  module_type module_;
  Box<value_type> box_;
  void _compute_2D_image(image_type &image,
                         const typename module_type::iterator start,
                         const typename module_type::iterator end,
                         const value_type delta = 0.1,
                         const value_type p = 1,
                         const bool normalize = true,
                         const Box<value_type> &box = Box<value_type>(),
                         const unsigned int horizontalResolution = 100,
                         const unsigned int verticalResolution = 100);
  void _compute_2D_image(image_type &image,
                         const typename module_type::const_iterator start,
                         const typename module_type::const_iterator end,
                         unsigned int horizontalResolution,
                         unsigned int verticalResolution,
                         get_2dpixel_value_function_type get_pixel_value) const;

  value_type _get_pixel_value(const typename module_type::iterator start,
                              const typename module_type::iterator end,
                              const filtration_type x,
                              const value_type delta,
                              const value_type p,
                              const bool normalize,
                              const value_type moduleWeight) const;

  void _add_bar_with_threshold(const Line<value_type> &line,
                               const std::pair<value_type, value_type> &bar,
                               const bool threshold_in,
                               Summand<value_type> &summand);
};

template <typename value_type>
class Summand {
 public:
  using births_type = Gudhi::multi_filtration::Multi_critical_filtration<value_type, false>;
  using deaths_type = Gudhi::multi_filtration::Multi_critical_filtration<value_type, true>;
  using filtration_type = typename births_type::Generator;  // same for death
  using dimension_type = int;
  Summand();
  Summand(const births_type &birth_corners, const deaths_type &death_corners, dimension_type dimension);
  Summand(const std::vector<filtration_type> &birth_corners,
          const std::vector<filtration_type> &death_corners,
          dimension_type dimension);

  value_type get_interleaving() const;
  value_type get_interleaving(const Box<value_type> &box);
  value_type get_local_weight(const filtration_type &x, const value_type delta) const;

  value_type distance_to_upper(const filtration_type &x, bool negative) const;
  value_type distance_to_lower(const filtration_type &x, bool negative) const;
  value_type distance_to(const filtration_type &x, bool negative) const;
  std::tuple<int, int> distance_idx_to_upper(const filtration_type &x) const;
  std::tuple<int, int> distance_idx_to_lower(const filtration_type &x) const;
  std::vector<int> distance_idx_to(const filtration_type &x, bool full) const;
  std::pair<filtration_type, filtration_type> get_bar(const Line<value_type> &line) const;
  std::pair<value_type, value_type> get_bar2(const Line<value_type> &l) const;
  void add_bar(value_type baseBirth,
               value_type baseDeath,
               const filtration_type &basepoint,
               filtration_type &birth,
               filtration_type &death,
               const bool threshold,
               const Box<value_type> &box);
  void add_bar(const filtration_type &birth, const filtration_type &death);
  void add_bar(const filtration_type &basepoint, value_type birth, value_type death, const Box<value_type> &);

  const std::vector<filtration_type> &get_birth_list() const;
  const std::vector<filtration_type> &get_death_list() const;
  const Gudhi::multi_filtration::Multi_critical_filtration<value_type> &get_upset() const;
  const Gudhi::multi_filtration::Multi_critical_filtration<value_type> &get_downset() const;
  void clean();

  void complete_birth(const value_type precision);
  void complete_death(const value_type precision);

  dimension_type get_dimension() const;
  void set_dimension(dimension_type dimension);

  value_type get_landscape_value(const std::vector<value_type> &x) const;

  friend void swap(Summand &sum1, Summand &sum2) {
    std::swap(sum1.birth_corners_, sum2.birth_corners_);
    std::swap(sum1.death_corners_, sum2.death_corners_);
    std::swap(sum1.distanceTo0_, sum2.distanceTo0_);
    // 	std::swap(sum1.updateDistance_, sum2.updateDistance_);
  };

  friend bool operator==(const Summand &a, const Summand &b) {
    return a.dimension_ == b.dimension_ && a.birth_corners_ == b.birth_corners_ && a.death_corners_ == b.death_corners_;
  }

  bool contains(const filtration_type &x) const;

  inline Box<value_type> get_bounds() const {
    if (birth_corners_.num_generators() == 0) return Box<value_type>();
    auto dimension = birth_corners_.num_parameters();
    filtration_type m(dimension, std::numeric_limits<value_type>::infinity());
    filtration_type M(dimension, -std::numeric_limits<value_type>::infinity());
    for (const auto &corner : birth_corners_) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        m[parameter] = std::min(m[parameter], corner[parameter]);
      }
    }
    for (const auto &corner : death_corners_) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        auto corner_i = corner[parameter];
        if (corner_i != std::numeric_limits<value_type>::infinity())
          M[parameter] = std::max(M[parameter], corner[parameter]);
      }
    }
    return Box(m, M);
  }

  inline void rescale(const std::vector<value_type> &rescale_factors) {
    if (birth_corners_.num_generators() == 0) return;
    auto dimension = birth_corners_.num_parameters();
    for (auto &corner : birth_corners_) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        corner[parameter] *= rescale_factors.at(parameter);
      }
    }
    for (auto &corner : death_corners_) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        corner[parameter] *= rescale_factors.at(parameter);
      }
    }
  }

  inline void translate(const std::vector<value_type> &translation) {
    if (birth_corners_.num_generators() == 0) return;
    auto dimension = birth_corners_.num_parameters();
    for (auto &corner : birth_corners_) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        corner[parameter] += translation.at(parameter);
      }
    }
    for (auto &corner : death_corners_) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        corner[parameter] += translation.at(parameter);
      }
    }
  }

  inline Summand<int64_t> grid_squeeze(const std::vector<std::vector<value_type>> &grid) const;

 private:
  Gudhi::multi_filtration::Multi_critical_filtration<value_type, false>
      birth_corners_;  // TODO : use Multi_critical_filtration
  Gudhi::multi_filtration::Multi_critical_filtration<value_type, true> death_corners_;
  value_type distanceTo0_;
  dimension_type dimension_;

  void _compute_interleaving(const Box<value_type> &box);
  void _add_birth(const filtration_type &birth);
  void _add_death(const filtration_type &death);
  value_type _rectangle_volume(const filtration_type &a, const filtration_type &b) const;
  value_type _get_max_diagonal(const filtration_type &a, const filtration_type &b, const Box<value_type> &box) const;
  value_type d_inf(const filtration_type &a, const filtration_type &b) const;
  void _factorize_min(filtration_type &a, const filtration_type &b);
  void _factorize_max(filtration_type &a, const filtration_type &b);
  static void _clean(std::vector<filtration_type> &list, bool keep_inf = true);

  static inline void _clean(births_type &list, bool keep_inf = true) { list.remove_empty_generators(keep_inf); }

  static inline void _clean(deaths_type &list, bool keep_inf = true) { list.remove_empty_generators(keep_inf); }
};

inline void threshold_filters_list(std::vector<filtration_type> &filtersList, const Box<value_type> &box) {
  return;
  for (unsigned int i = 0; i < filtersList.size(); i++) {
    for (value_type &value : filtersList[i]) {
      value = std::min(std::max(value, box.get_lower_corner()[i]), box.get_upper_corner()[i]);
    }
  }
}

template <class Filtration_value, int axis = 0, bool sign = true>
class LineIterator {
 public:
  using value_type = typename Filtration_value::value_type;
  LineIterator(const Filtration_value &basepoint,
               const Filtration_value &direction,
               value_type precision,
               int num_iterations)
      : precision(precision), remaining_iterations(num_iterations), current_line(std::move(basepoint), direction) {};

  inline LineIterator<Filtration_value, axis, sign> &operator++() {
    //
    auto &basepoint = current_line.base_point();
    if (this->is_finished()) return *this;
    // If we didn't reached the end, go to the next line
    basepoint[axis] += sign ? precision : -precision;
    --remaining_iterations;
    return *this;
  }

  inline const Line<value_type> &operator*() const { return current_line; }

  inline LineIterator<Filtration_value, axis, sign> &next(std::size_t i) {
    auto &basepoint = current_line.base_point();
    if (this->is_finished()) return *this;
    // If we didn't reached the end, go to the next line
    basepoint[i] += sign ? precision : -precision;
    --remaining_iterations;
    return *this;
  }

  inline bool is_finished() const { return remaining_iterations <= 0; }

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
    // if constexpr (axis_ >= 0) {
    //   new_line = *(++line_iterator); // first line is always a persistence
    // } else {
    //   new_line = *line_iterator.next(axis);
    // }
    // copy, no need to add it
    if constexpr (verbose) std::cout << "----------------------------------------------" << std::endl;
    if constexpr (verbose) std::cout << "Line basepoint " << new_line.base_point() << std::endl;
    slicer.push_to(new_line);

    slicer.vineyard_update();
    if constexpr (verbose2) std::cout << slicer << std::endl;
    const auto &diagram = slicer.get_flat_nodim_barcode();
    module.add_barcode(new_line, std::move(diagram), threshold);
  };
};

template <class Filtration_value, class Slicer = SimplicialVineMatrixTruc<>>
void _rec_mma(Module<typename Filtration_value::value_type> &module,
              Filtration_value &basepoint,
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
  Filtration_value basepoint_copy;
  for (int i = 0; i < grid_size[dim_to_iterate]; ++i) {
    // TODO : multithread, but needs matrix to be thread safe + put mutex on
    // module
    pers_copy = current_persistence;
    basepoint_copy = basepoint;
    _rec_mma(module, basepoint_copy, grid_size, dim_to_iterate - 1, pers_copy, precision, threshold);
    basepoint[dim_to_iterate] += precision;
    // current_persistence.push_to(Line(basepoint));
    // current_persistence.vineyard_update();
  }
}

template <int axis, class Filtration_value, class Slicer>
void _rec_mma2(Module<typename Filtration_value::value_type> &module,
               Filtration_value &&basepoint,
               const Filtration_value &direction,
               const std::vector<int> &grid_size,
               const std::vector<bool> &signs,
               int dim_to_iterate,
               Slicer &&current_persistence,
               const value_type precision,
               bool threshold) {
  static_assert(std::is_same_v<typename Filtration_value::value_type, typename Slicer::value_type>);

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
    _rec_mma2<axis, Filtration_value, typename Slicer::ThreadSafe>(module,
                                                                   Filtration_value(basepoint),
                                                                   direction,
                                                                   grid_size,
                                                                   signs,
                                                                   dim_to_iterate - 1,
                                                                   current_persistence.weak_copy(),
                                                                   precision,
                                                                   threshold);
    basepoint[dim_to_iterate] += signs[dim_to_iterate] ? precision : -precision;
    // current_persistence.push_to(Line(basepoint));
    // current_persistence.vineyard_update();
  }
}

template <class Slicer, typename value_type>
Module<value_type> multiparameter_module_approximation(
    Slicer &slicer,
    const Gudhi::multi_filtration::One_critical_filtration<value_type> &direction,
    const value_type precision,
    Box<value_type> &box,
    const bool threshold,
    const bool complete,
    const bool verbose) {
  static_assert(std::is_same_v<typename Slicer::Filtration_value::value_type,
                               value_type>);  // Value type can be exposed to python interface.
  if (verbose) std::cout << "Starting Module Approximation" << std::endl;
  /* using Filtration_value = Slicer::Filtration_value; */

  Gudhi::multi_filtration::One_critical_filtration<value_type> basepoint = box.get_lower_corner();
  const std::size_t num_parameters = box.dimension();
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
      if (!direction.num_parameters())
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
    slicer.compute_persistence();
    auto barcode = slicer.get_flat_barcode();
    auto num_bars = barcode.size();
    out.resize(num_bars);
    /* Filtration_value birthContainer(num_parameters), */
    /* deathContainer(num_parameters); */
    for (std::size_t i = 0; i < num_bars; i++) {
      const auto &[dim, bar] = barcode[i];
      /* const auto &[birth, death] = bar; */
      out[i].set_dimension(dim);
      /* out[i].add_bar(birth, death, basepoint, birthContainer, deathContainer,
       */
      /* threshold, box); */
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
    std::cout << "Grid size " << Gudhi::multi_filtration::One_critical_filtration(grid_size) << " Signs ";
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
      if (direction.num_parameters() && direction[i] == 0.0) continue;  // skip faces with codim d_i=0
      auto temp_grid_size = grid_size;
      temp_grid_size[i] = 0;
      if (verbose)
        std::cout << "Face " << i << "/" << num_parameters << " with grid size "
                  << Gudhi::multi_filtration::One_critical_filtration(temp_grid_size) << std::endl;
      // if (!direction.size() || direction[0] > 0)
      _rec_mma2<0>(out,
                   Gudhi::multi_filtration::One_critical_filtration<value_type>(basepoint),
                   direction,
                   temp_grid_size,
                   signs,
                   num_parameters - 1,
                   slicer.weak_copy(),
                   precision,
                   threshold);
    }
    // last one, we can destroy basepoint & cie
    if (!direction.num_parameters() || direction[0] > 0) {
      grid_size[0] = 0;
      if (verbose)
        std::cout << "Face " << num_parameters << "/" << num_parameters << " with grid size "
                  << Gudhi::multi_filtration::One_critical_filtration(grid_size) << std::endl;
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
};

template <typename value_type>
template <class Barcode>
inline void Module<value_type>::add_barcode(const Barcode &barcode) {
  constexpr const bool verbose = false;
  if (barcode.size() != module_.size()) {
    std::cerr << "Barcode sizes doesn't match. Module is " << std::to_string(module_.size()) << " and barcode is "
              << std::to_string(barcode.size()) << std::endl;
  }
  unsigned int count = 0;
  for (const auto &bar_ : barcode) {
    auto &summand = this->operator[](count++);
    auto &[dim, bar] = bar_;
    auto &[birth_filtration, death_filtration] = bar;
    if constexpr (verbose) std::cout << "Birth " << birth_filtration << " Death " << death_filtration << std::endl;
    summand.add_bar(birth_filtration, death_filtration);
  }
}

template <typename value_type>
inline void Module<value_type>::add_barcode(
    const Line<value_type> &line,
    const std::vector<std::pair<int, std::pair<value_type, value_type>>> &barcode,
    const bool threshold_in) {
  assert(barcode.size() == module_.size() && "Barcode sizes doesn't match.");

  auto count = 0U;
  for (const auto &extBar : barcode) {
    auto &[dim, bar] = extBar;
    _add_bar_with_threshold(line, bar, threshold_in, this->operator[](count++));
  }
}

template <typename value_type>
inline void Module<value_type>::add_barcode(const Line<value_type> &line,
                                            const std::vector<std::pair<value_type, value_type>> &barcode,
                                            const bool threshold_in) {
  assert(barcode.size() == module_.size() && "Barcode sizes doesn't match.");

  auto count = 0U;
  for (const auto &bar : barcode) {
    _add_bar_with_threshold(line, bar, threshold_in, this->operator[](count++));
  }
}

template <typename value_type>
inline void Module<value_type>::_add_bar_with_threshold(const Line<value_type> &line,
                                                        const std::pair<value_type, value_type> &bar,
                                                        const bool threshold_in,
                                                        Summand<value_type> &summand) {
  constexpr const bool verbose = false;
  auto [birth_filtration, death_filtration] = bar;

  if (birth_filtration >= death_filtration) return;

  if constexpr (verbose) {
    std::cout << "--BAR (" << birth_filtration << ", " << death_filtration << ") at basepoint " << line.base_point()
              << " direction " << line.direction() << std::endl;
  }

  auto birth_container = line[birth_filtration];
  if constexpr (verbose) std::cout << " B: " << birth_container << " B*d: " << birth_filtration * line.direction();
  if (birth_container.is_minus_inf()) {
    if (threshold_in) birth_container = box_.get_lower_corner();
  } else {
    bool allInf = true;
    for (std::size_t i = 0U; i < birth_container.num_parameters(); i++) {
      auto t = box_.get_lower_corner()[i];
      if (birth_container[i] < t - 1e-10) birth_container[i] = threshold_in ? t : -filtration_type::T_inf;
      if (birth_container[i] != -filtration_type::T_inf) allInf = false;
    }
    if (allInf) birth_container = filtration_type::minus_inf();
  }

  auto death_container = line[death_filtration];
  if constexpr (verbose) std::cout << " D: " << death_container;
  if (death_container.is_plus_inf()) {
    if (threshold_in) death_container = box_.get_upper_corner();
  } else {
    bool allInf = true;
    for (std::size_t i = 0U; i < death_container.num_parameters(); i++) {
      auto t = box_.get_upper_corner()[i];
      if (death_container[i] > t + 1e-10) death_container[i] = threshold_in ? t : filtration_type::T_inf;
      if (death_container[i] != filtration_type::T_inf) allInf = false;
    }
    if (allInf) death_container = filtration_type::inf();
  }

  if constexpr (verbose) std::cout << " BT: " << birth_container << " DT: " << death_container << std::endl;
  summand.add_bar(birth_container, death_container);
}

template <typename value_type>
inline Module<value_type>::Module() {}

template <typename value_type>
inline Module<value_type>::Module(Box<value_type> &box) : box_(box) {}

template <typename value_type>
inline void Module<value_type>::resize(const unsigned int size) {
  module_.resize(size);
}

template <typename value_type>
inline Summand<value_type> &Module<value_type>::at(const unsigned int index) {
  return module_.at(index);
}

template <typename value_type>
inline Summand<value_type> &Module<value_type>::operator[](const size_t index) {
  return this->module_[index];
}

template <typename value_type>
inline const Summand<value_type> &Module<value_type>::operator[](const size_t index) const {
  return this->module_[index];
}

template <typename value_type>
inline typename Module<value_type>::module_type::iterator Module<value_type>::begin() {
  return module_.begin();
}

template <typename value_type>
inline typename Module<value_type>::module_type::const_iterator Module<value_type>::begin() const {
  return module_.cbegin();
}

template <typename value_type>
inline typename Module<value_type>::module_type::iterator Module<value_type>::end() {
  return module_.end();
}

template <typename value_type>
inline typename Module<value_type>::module_type::const_iterator Module<value_type>::end() const {
  return module_.cend();
}

template <typename value_type>
inline void Module<value_type>::add_summand(Summand<value_type> summand, int degree) {
  if (degree >= 0) summand.set_dimension(degree);
  module_.push_back(summand);
}

/**
 * @brief Remove the empty summands of the output
 *
 * @param output p_output:...
 * @param keep_order p_keep_order:... Defaults to false.
 */

template <typename value_type>
inline void Module<value_type>::clean() {
  module_type tmp;
  for (size_t i = 0u; i < module_.size(); i++) {
    module_[i].clean();
  }
  module_.erase(
      std::remove_if(
          module_.begin(), module_.end(), [](const Summand<value_type> &s) { return s.get_upset().is_plus_inf(); }),
      module_.end());
}

template <typename value_type>
inline void Module<value_type>::fill(const value_type precision) {
  if (module_.empty()) return;

  for (Summand<value_type> &sum : module_) {
    sum.complete_birth(precision);
    sum.complete_death(precision);
  }
}

template <typename value_type>
std::vector<value_type> inline Module<value_type>::get_interleavings(const Box<value_type> &box) {
  std::vector<value_type> out(this->size());
  for (auto i = 0u; i < out.size(); ++i) {
    out[i] = module_[i].get_interleaving(box);
  }
  return out;
}

template <typename value_type>
typename Module<value_type>::distance_to_idx_type inline Module<value_type>::compute_distance_idx_to(
    const std::vector<value_type> &pt,
    bool full) const {
  typename Module<value_type>::distance_to_idx_type out(module_.size(), std::vector<int>(full ? 4 : 2));
  for (auto i = 0u; i < module_.size(); ++i) {
    out[i] = module_[i].distance_idx_to(pt, full);
  }
  return out;
}

template <typename value_type>
std::vector<value_type> inline Module<value_type>::compute_distance_to(const std::vector<value_type> &pt,
                                                                       bool negative) const {
  std::vector<value_type> out(this->size());
  for (auto i = 0u; i < this->size(); ++i) {
    out[i] = module_[i].distance_to(pt, negative);
  }
  return out;
}

template <typename value_type>
std::vector<std::vector<value_type>> inline Module<value_type>::compute_distances_to(
    const std::vector<std::vector<value_type>> &pts,
    bool negative,
    int n_jobs) const {
  std::vector<std::vector<value_type>> out(pts.size(), std::vector<value_type>(this->size()));
  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0u), pts.size(), [&](std::size_t i) {
      tbb::parallel_for(std::size_t(0u), std::size_t(this->size()), [&](std::size_t j) {
        out[i][j] = module_[j].distance_to(pts[i], negative);
      });
    });
  });
  return out;
}

template <typename value_type>
template <typename dtype, typename indices_type>
void inline Module<value_type>::compute_distances_to(dtype *data_ptr,
                                                     const std::vector<std::vector<value_type>> &pts,
                                                     bool negative,
                                                     int n_jobs) const {
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, {static_cast<int>(pts.size()), static_cast<int>(this->size())});
  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0u), pts.size(), [&](std::size_t i) {
      // tbb::parallel_for(std::size_t(0u), std::size_t(this->size()), [&](std::size_t j) {
      dtype *current_ptr = &container[{static_cast<int>(i), 0}];
      for (std::size_t j = 0u; j < this->size(); ++j) {
        *(current_ptr + j) = module_[j].distance_to(pts[i], negative);
      }
    });
    // });
  });
}

template <typename value_type>
typename Module<value_type>::distances_to_idx_type inline Module<value_type>::compute_distances_idx_to(
    const std::vector<std::vector<value_type>> &pts,
    bool full,
    int n_jobs) const {
  Module::distances_to_idx_type out(pts.size(),
                                    Module::distance_to_idx_type(module_.size(), std::vector<int>(full ? 4 : 2)));

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0u), pts.size(), [&](std::size_t i) {
      tbb::parallel_for(std::size_t(0u), std::size_t(this->size()), [&](std::size_t j) {
        out[i][j] = module_[j].distance_idx_to(pts[i], full);
      });
    });
  });
  return out;
}

template <typename value_type>
inline std::vector<std::vector<value_type>> Module<value_type>::compute_pixels(
    const std::vector<std::vector<value_type>> &coordinates,
    const std::vector<int> &degrees,
    const Box<value_type> &box,
    const value_type delta,
    const value_type p,
    const bool normalize,
    const int n_jobs) {
  auto num_degrees = degrees.size();
  auto num_pts = coordinates.size();
  std::vector<std::vector<value_type>> out(num_degrees, std::vector<value_type>(num_pts));

  typename module_type::iterator start;
  typename module_type::iterator end = module_.begin();
  for (auto degree_idx = 0u; degree_idx < num_degrees; degree_idx++) {
    {  // for Timer
      auto d = degrees[degree_idx];
      Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
      start = end;
      while (start != module_.end() && start->get_dimension() != d) start++;
      if (start == module_.end()) break;
      end = start;
      while (end != module_.end() && end->get_dimension() == d) end++;
      out[degree_idx] = compute_pixels_of_degree(start, end, delta, p, normalize, box, coordinates, n_jobs);
    }  // Timer death
  }
  return out;
}

template <typename value_type>
inline typename std::vector<typename Module<value_type>::image_type> Module<value_type>::get_vectorization(
    const value_type delta,
    const value_type p,
    const bool normalize,
    const Box<value_type> &box,
    unsigned int horizontalResolution,
    unsigned int verticalResolution) {
  dimension_type maxDim = module_.back().get_dimension();
  std::vector<Module::image_type> image_vector(maxDim + 1);
  typename module_type::iterator start;
  typename module_type::iterator end = module_.begin();
  for (dimension_type d = 0; d <= maxDim; d++) {
    {  // for Timer
      Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
      start = end;
      while (end != module_.end() && end->get_dimension() == d) end++;
      _compute_2D_image(
          image_vector.at(d), start, end, delta, p, normalize, box, horizontalResolution, verticalResolution);
    }  // Timer death
  }
  return image_vector;
}

template <typename value_type>
inline std::vector<typename Module<value_type>::image_type> Module<value_type>::get_vectorization(
    unsigned int horizontalResolution,
    unsigned int verticalResolution,
    get_2dpixel_value_function_type get_pixel_value) const {
  dimension_type maxDim = module_.back().get_dimension();
  std::vector<Module::image_type> image_vector(maxDim + 1);
  typename module_type::const_iterator start;
  typename module_type::const_iterator end = module_.begin();
  for (dimension_type d = 0; d <= maxDim; d++) {
    {  // for Timer
      Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
      start = end;
      while (end != module_.end() && end->get_dimension() == d) end++;
      _compute_2D_image(image_vector.at(d), start, end, horizontalResolution, verticalResolution, get_pixel_value);
    }  // Timer death
  }
  return image_vector;
}

template <typename value_type>
inline typename Module<value_type>::image_type Module<value_type>::get_vectorization_in_dimension(
    const dimension_type dimension,
    const value_type delta,
    const value_type p,
    const bool normalize,
    const Box<value_type> &box,
    unsigned int horizontalResolution,
    unsigned int verticalResolution) {
  Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

  Module::image_type image;
  typename module_type::iterator start = module_.begin();
  while (start != module_.end() && start->get_dimension() < dimension) start++;
  typename module_type::iterator end = start;
  while (end != module_.end() && end->get_dimension() == dimension) end++;
  _compute_2D_image(image, start, end, delta, p, normalize, box, horizontalResolution, verticalResolution);

  return image;
}

template <typename value_type>
inline typename Module<value_type>::image_type Module<value_type>::get_vectorization_in_dimension(
    const dimension_type dimension,
    unsigned int horizontalResolution,
    unsigned int verticalResolution,
    get_2dpixel_value_function_type get_pixel_value) const {
  Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

  typename Module::image_type image;
  typename module_type::const_iterator start = module_.begin();
  while (start != module_.end() && start->get_dimension() < dimension) start++;
  typename module_type::const_iterator end = start;
  while (end != module_.end() && end->get_dimension() == dimension) end++;
  _compute_2D_image(image, start, end, horizontalResolution, verticalResolution, get_pixel_value);

  return image;
}

template <typename value_type>
std::vector<value_type> Module<value_type>::get_landscape_values(const std::vector<value_type> &x,
                                                                 const dimension_type dimension) const {
  std::vector<value_type> out;
  out.reserve(this->size());
  for (unsigned int i = 0; i < this->size(); i++) {
    const Summand<value_type> &summand = this->module_[i];
    if (summand.get_dimension() == dimension) out.push_back(summand.get_landscape_value(x));
  }
  std::sort(out.begin(), out.end(), [](const value_type x, const value_type y) { return x > y; });
  return out;
}

template <typename value_type>
typename Module<value_type>::image_type Module<value_type>::get_landscape(
    const dimension_type dimension,
    const unsigned int k,
    const Box<value_type> &box,
    const std::vector<unsigned int> &resolution) const {
  // TODO extend in higher dimension (ie, change the image type to a template
  // class)
  Module::image_type image;
  image.resize(resolution[0], std::vector<value_type>(resolution[1]));
  value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / resolution[0];
  value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / resolution[1];
  tbb::parallel_for(0U, resolution[0], [&](unsigned int i) {
    tbb::parallel_for(0U, resolution[1], [&](unsigned int j) {
      auto landscape = this->get_landscape_values(
          {box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j}, dimension);
      image[i][j] = k < landscape.size() ? landscape[k] : 0;
    });
  });
  return image;
}

template <typename value_type>
std::vector<typename Module<value_type>::image_type> Module<value_type>::get_landscapes(
    const dimension_type dimension,
    const std::vector<unsigned int> ks,
    const Box<value_type> &box,
    const std::vector<unsigned int> &resolution) const {
  std::vector<Module::image_type> images(ks.size());
  for (auto &image : images) image.resize(resolution[0], std::vector<value_type>(resolution[1]));
  value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / resolution[0];
  value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / resolution[1];

  tbb::parallel_for(0U, resolution[0], [&](unsigned int i) {
    tbb::parallel_for(0U, resolution[1], [&](unsigned int j) {
      std::vector<value_type> landscapes = this->get_landscape_values(
          {box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j}, dimension);
      for (const auto k : ks) {
        images[k][i][j] = k < landscapes.size() ? landscapes[k] : 0;
      }
    });
  });
  return images;
}

template <typename value_type>
inline Box<value_type> Module<value_type>::get_box() const {
  return this->box_;
}

template <typename value_type>
inline void Module<value_type>::set_box(Box<value_type> box) {
  this->box_ = box;
}

template <typename value_type>
inline unsigned int Module<value_type>::size() const {
  return this->module_.size();
}

template <typename value_type>
inline void Module<value_type>::infer_box(std::vector<filtration_type> &f) {
  this->box_.infer_from_filters(f);
}

template <typename value_type>
inline dimension_type Module<value_type>::get_dimension() const {
  return this->module_.empty() ? -1 : this->module_.back().get_dimension();
}

template <typename value_type>
inline std::vector<Summand<value_type>> Module<value_type>::get_summands_of_dimension(const int dimension) const {
  std::vector<Summand<value_type>> list;
  for (const Summand<value_type> &summand : this->module_) {
    if (summand.get_dimension() == dimension) list.push_back(summand);
  }
  return list;
}

template <typename value_type>
inline std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>>
Module<value_type>::get_corners_of_dimension(const int dimension) const {
  std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>> list;
  for (const Summand<value_type> &summand : this->module_) {
    if (summand.get_dimension() == dimension)
      list.push_back(std::make_pair(
          std::vector<std::vector<value_type>>(summand.get_birth_list().begin(), summand.get_birth_list().end()),
          std::vector<std::vector<value_type>>(summand.get_death_list().begin(), summand.get_death_list().end())));
  }
  return list;
}

template <typename value_type>
std::vector<std::vector<std::pair<value_type, value_type>>> Module<value_type>::get_barcode2(
    const Line<value_type> &l,
    const dimension_type dimension) const {
  constexpr const bool verbose = false;
  std::vector<std::vector<std::pair<value_type, value_type>>> barcode(this->get_dimension() + 1);
  for (auto i = 0; i < this->get_dimension(); ++i) {
    barcode[i].reserve(this->size());
  }
  for (unsigned int i = 0; i < this->size(); i++) {
    const Summand<value_type> &summand = this->module_[i];
    if constexpr (verbose) std::cout << "Summand of dimension " << summand.get_dimension() << std::endl;

    if (dimension != -1 && summand.get_dimension() != dimension) continue;
    /* if (dimension != -1 && summand.get_dimension() > dimension) */
    /* 	break; */
    const auto &pushed_summand = summand.get_bar2(l);

    barcode[summand.get_dimension()].push_back(pushed_summand);
  }
  return barcode;
}

template <typename value_type>
MultiDiagram<typename Module<value_type>::filtration_type, value_type>
Module<value_type>::get_barcode(const Line<value_type> &l, const dimension_type dimension, const bool threshold) const {
  constexpr const bool verbose = false;
  if constexpr (verbose)
    std::cout << "Computing barcode of dimension " << dimension << " and threshold " << threshold << std::endl;
  std::vector<MultiDiagram_point<filtration_type>> barcode(this->size());
  std::pair<value_type, value_type> threshold_bounds;
  if (threshold) threshold_bounds = l.get_bounds(this->box_);
  unsigned int summand_idx = 0;
  for (unsigned int i = 0; i < this->size(); i++) {
    const Summand<value_type> &summand = this->module_[i];
    if constexpr (verbose) std::cout << "Summand of dimension " << summand.get_dimension() << std::endl;

    if (dimension != -1 && summand.get_dimension() != dimension) continue;
    /* if (dimension != -1 && summand.get_dimension() > dimension) */
    /* 	break; */
    auto pushed_summand = summand.get_bar(l);

    filtration_type &pbirth = pushed_summand.first;
    filtration_type &pdeath = pushed_summand.second;
    if constexpr (verbose) std::cout << "BAR : " << pbirth << " " << pdeath << std::endl;
    if (threshold) {
      auto min = l[threshold_bounds.first];
      auto max = l[threshold_bounds.second];
      if (!(pbirth < max) || !(pdeath > min)) {
        /* continue; */  // We still need summands to be aligned. The price to
        // pay is some memory.
        pbirth = std::numeric_limits<filtration_type>::infinity();
        pdeath = pbirth;
      }
      pbirth.push_to_least_common_upper_bound(min);
      pdeath.pull_to_greatest_common_lower_bound(max);
    }
    barcode[summand_idx++] = MultiDiagram_point(summand.get_dimension(), pbirth, pdeath);
  }
  barcode.resize(summand_idx);
  return MultiDiagram<filtration_type, value_type>(barcode);
}

template <typename value_type>
MultiDiagrams<typename Module<value_type>::filtration_type, value_type> Module<value_type>::get_barcodes(
    const std::vector<Line<value_type>> &lines,
    const dimension_type dimension,
    const bool threshold) const {
  unsigned int nlines = lines.size();
  MultiDiagrams<typename Module<value_type>::filtration_type, value_type> out(nlines);
  tbb::parallel_for(0U, nlines, [&](unsigned int i) {
    const Line<value_type> &l = lines[i];
    out[i] = this->get_barcode(l, dimension, threshold);
  });
  return out;
}

template <typename value_type>
std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> Module<value_type>::get_barcodes2(
    const std::vector<Line<value_type>> &lines,
    const dimension_type dimension) const {
  unsigned int nlines = lines.size();
  std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> out(
      this->get_dimension() + 1, std::vector<std::vector<std::pair<value_type, value_type>>>(nlines));
  tbb::parallel_for(0U, nlines, [&](unsigned int i) {
    const Line<value_type> &l = lines[i];
    for (const auto &summand : module_) {
      if (dimension != -1 && summand.get_dimension() != dimension) continue;
      const auto &bar = summand.get_bar2(l);
      out[summand.get_dimension()][i].push_back(bar);
    }
  });
  return out;
}

template <typename value_type>
MultiDiagrams<typename Module<value_type>::filtration_type, value_type> Module<value_type>::get_barcodes(
    const std::vector<filtration_type> &basepoints,
    const dimension_type dimension,
    const bool threshold) const {
  unsigned int nlines = basepoints.size();
  MultiDiagrams<typename Module<value_type>::filtration_type, value_type> out(nlines);
  // for (unsigned int i = 0; i < nlines; i++){
  tbb::parallel_for(0U, nlines, [&](unsigned int i) {
    const Line<value_type> &l = Line<value_type>(basepoints[i]);
    out[i] = this->get_barcode(l, dimension, threshold);
  });
  return out;
}

template <typename value_type>
std::vector<int> Module<value_type>::euler_curve(const std::vector<filtration_type> &points) const {
  unsigned int npts = points.size();
  std::vector<int> out(npts);
  // #pragma omp parallel for
  tbb::parallel_for(0U, static_cast<unsigned int>(out.size()), [&](unsigned int i) {
    auto &euler_char = out[i];
    const filtration_type &point = points[i];
    /* #pragma omp parallel for reduction(+ : euler_char) */
    for (const Summand<value_type> &I : this->module_) {
      if (I.contains(point)) {
        int sign = I.get_dimension() % 2 ? -1 : 1;
        euler_char += sign;
      }
    }
  });
  return out;
}

template <typename value_type>
inline Box<value_type> Module<value_type>::get_bounds() const {
  dimension_type num_parameters = box_.get_lower_corner().num_parameters();
  filtration_type lower_bound(num_parameters, std::numeric_limits<value_type>::infinity());
  filtration_type upper_bound(num_parameters, -std::numeric_limits<value_type>::infinity());
  for (const auto &summand : module_) {
    const auto &summand_bounds = summand.get_bounds();
    const auto &[m, M] = summand_bounds.get_bounding_corners();
    for (auto parameter = 0; parameter < num_parameters; parameter++) {
      lower_bound[parameter] = std::min(m[parameter], lower_bound[parameter]);
      upper_bound[parameter] = std::min(M[parameter], upper_bound[parameter]);
    }
  }
  return Box(lower_bound, upper_bound);
}

template <typename value_type>
inline void Module<value_type>::rescale(const std::vector<value_type> &rescale_factors, int degree) {
  for (auto &summand : module_) {
    if (degree == -1 or summand.get_dimension() == degree) summand.rescale(rescale_factors);
  }
}

template <typename value_type>
inline void Module<value_type>::translate(const std::vector<value_type> &translation, int degree) {
  for (auto &summand : module_) {
    if (degree == -1 or summand.get_dimension() == degree) summand.translate(translation);
  }
}

template <typename value_type>
inline std::vector<value_type> Module<value_type>::compute_pixels_of_degree(
    const typename module_type::iterator start,
    const typename module_type::iterator end,
    const value_type delta,
    const value_type p,
    const bool normalize,
    const Box<value_type> &box,
    const std::vector<std::vector<value_type>> &coordinates,
    const int n_jobs) {
  unsigned int num_pixels = coordinates.size();
  std::vector<value_type> out(num_pixels);
  value_type moduleWeight = 0;
  {  // for Timer
    Debug::Timer timer("Computing module weight ...", verbose);
    for (auto it = start; it != end; it++)  //  precomputes interleaving restricted to box for all summands.
      it->get_interleaving(box);
    if (p == 0) {
      // #pragma omp parallel for reduction(+ : moduleWeight)
      for (auto it = start; it != end; it++) {
        moduleWeight += it->get_interleaving() > 0;
      }
    } else if (p != inf) {
      // #pragma omp parallel for reduction(+ : moduleWeight)
      for (auto it = start; it != end; it++) {
        // /!\ TODO deal with inf summands (for the moment,  depends on the box
        // ...)
        if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
          moduleWeight += std::pow(it->get_interleaving(), p);
      }
    } else {
      // #pragma omp parallel for reduction(std::max : moduleWeight)
      for (auto it = start; it != end; it++) {
        if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
          moduleWeight = std::max(moduleWeight, it->get_interleaving());
      }
    }
  }  // Timer death
  if (verbose) std::cout << "Module " << start->get_dimension() << " has weight : " << moduleWeight << "\n";
  if (!moduleWeight) return out;

  if constexpr (Debug::debug)
    if (moduleWeight < 0) {
      if constexpr (Debug::debug) std::cout << "!! Negative weight !!" << std::endl;
      // 		image.clear();
      return {};
    }

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    tbb::parallel_for(0u, num_pixels, [&](unsigned int i) {
      out[i] = _get_pixel_value(start, end, coordinates[i], delta, p, normalize, moduleWeight);
    });
  });
  return out;
}

template <typename value_type>
inline Module<int64_t> Module<value_type>::grid_squeeze(const std::vector<std::vector<value_type>> &grid) const {
  auto dimension = this->get_dimension();
  Module<int64_t> out(this->size());
  for (auto i = 0u; i < this->size(); ++i) {
    const auto &interval = this->operator[](i);
    out[i] = interval.grid_squeeze(grid);
  }
  return out;
}

template <typename value_type>
inline Summand<int64_t> Summand<value_type>::grid_squeeze(const std::vector<std::vector<value_type>> &grid) const {
  auto dimension = this->get_dimension();
  Summand<int64_t> out(
      compute_coordinates_in_grid(birth_corners_, grid), compute_coordinates_in_grid(death_corners_, grid), dimension_);
  return out;
}

/**
 * dim, summand, (birth/death), num_pts, num_parameters
 */
template <typename value_type>
inline typename Module<value_type>::idx_dump_type Module<value_type>::to_idx(
    const std::vector<std::vector<value_type>> &grid) const {
  unsigned int num_parameters = grid.size();
  auto dimension = this->get_dimension();
  idx_dump_type out(dimension + 1);
  for (auto i = 0u; i < this->size(); ++i) {
    auto &interval = this->operator[](i);
    auto &out_of_dim = out[interval.get_dimension()];
    out_of_dim.reserve(this->size());
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> interval_idx;

    auto &birth_idx = interval_idx.first;
    birth_idx.reserve(interval.get_birth_list().size());
    auto &death_idx = interval_idx.second;
    death_idx.reserve(interval.get_death_list().size());

    for (const auto &pt : interval.get_birth_list()) {
      std::vector<int> pt_idx(pt.size());
      for (auto i = 0u; i < num_parameters; ++i) {
        pt_idx[i] = std::distance(grid[i].begin(), std::lower_bound(grid[i].begin(), grid[i].end(), pt[i]));
      }
      birth_idx.push_back(pt_idx);
    }
    for (const auto &pt : interval.get_death_list()) {
      std::vector<int> pt_idx(pt.size());
      for (auto i = 0u; i < num_parameters; ++i) {
        pt_idx[i] = std::distance(grid[i].begin(), std::lower_bound(grid[i].begin(), grid[i].end(), pt[i]));
      }
      death_idx.push_back(pt_idx);
    }
    out_of_dim.push_back(interval_idx);
  }
  return out;
}

template <typename value_type>
std::vector<int> inline to_grid_coord(const Gudhi::multi_filtration::One_critical_filtration<value_type> &pt,
                                      const std::vector<std::vector<value_type>> &grid) {
  std::size_t num_parameters = grid.size();
  std::vector<int> out(num_parameters);
  if (pt.is_plus_inf() || pt.is_nan()) [[unlikely]] {
    for (size_t i = 0; i < num_parameters; ++i) out[i] = grid[i].size() - 1;
    return out;
  }
  if (pt.is_minus_inf()) [[unlikely]] {
    for (size_t i = 0; i < num_parameters; ++i) out[i] = 0;
    return out;
  }
  // pt has to be of size num_parameters now
  for (size_t i = 0u; i < num_parameters; ++i) {
    if (pt[i] >= grid[i].back()) [[unlikely]]
      out[i] = grid[i].size() - 1;
    else if (pt[i] <= grid[i][0]) [[unlikely]] {
      out[i] = 0;
    } else {
      auto temp = std::distance(grid[i].begin(), std::lower_bound(grid[i].begin(), grid[i].end(), pt[i]));
      if (std::abs(grid[i][temp] - pt[i]) < std::abs(grid[i][temp - 1] - pt[i])) {
        out[i] = temp;
      } else {
        out[i] = temp - 1;
      }
    }
  }
  return out;
}

template <typename value_type>
std::vector<std::vector<std::vector<int>>> inline Module<value_type>::to_flat_idx(
    const std::vector<std::vector<value_type>> &grid) const {
  std::vector<std::vector<std::vector<int>>> out(3);
  auto &idx = out[0];
  auto &births = out[1];
  auto &deaths = out[2];

  idx.resize(2);
  idx[0].resize(this->size());
  idx[1].resize(this->size());

  // some heuristic: usually
  births.reserve(2 * this->size());
  deaths.reserve(2 * this->size());
  for (auto i = 0u; i < this->size(); ++i) {
    auto &interval = this->operator[](i);
    idx[0][i] = interval.get_birth_list().size();
    for (const auto &pt : interval.get_birth_list()) {
      births.push_back(to_grid_coord(pt, grid));
    }
    idx[1][i] = interval.get_death_list().size();
    for (const auto &pt : interval.get_death_list()) {
      deaths.push_back(to_grid_coord(pt, grid));
    }
  }
  return out;
}

template <typename value_type>
std::vector<int> inline Module<value_type>::get_degree_splits() const {
  std::vector<int> splits = {};
  int current_degree = 0;
  for (auto i = 0u; i < this->size(); ++i) {
    const auto &summand = this->operator[](i);
    while (summand.get_dimension() > current_degree) [[unlikely]] {
      current_degree++;
      splits.push_back(i);
    }
  }
  return splits;
}

template <typename value_type>
inline void Module<value_type>::_compute_2D_image(Module::image_type &image,
                                                  const typename module_type::iterator start,
                                                  const typename module_type::iterator end,
                                                  const value_type delta,
                                                  const value_type p,
                                                  const bool normalize,
                                                  const Box<value_type> &box,
                                                  const unsigned int horizontalResolution,
                                                  const unsigned int verticalResolution) {
  image.resize(horizontalResolution, std::vector<value_type>(verticalResolution));
  value_type moduleWeight = 0;
  {  // for Timer
    Debug::Timer timer("Computing module weight ...", verbose);
    for (auto it = start; it != end; it++)  //  precomputes interleaving restricted to box for all summands.
      it->get_interleaving(box);
    if (p == 0) {
      /* #pragma omp parallel for reduction(+ : moduleWeight) */
      for (auto it = start; it != end; it++) {
        moduleWeight += it->get_interleaving() > 0;
      }
    } else if (p != inf) {
      /* #pragma omp parallel for reduction(+ : moduleWeight) */
      for (auto it = start; it != end; it++) {
        // /!\ TODO deal with inf summands (for the moment,  depends on the box
        // ...)
        if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
          moduleWeight += std::pow(it->get_interleaving(), p);
      }
    } else {
      /* #pragma omp parallel for reduction(std::max : moduleWeight) */
      for (auto it = start; it != end; it++) {
        if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
          moduleWeight = std::max(moduleWeight, it->get_interleaving());
      }
    }
  }  // Timer death
  if (verbose) std::cout << "Module " << start->get_dimension() << " has weight : " << moduleWeight << "\n";
  if (!moduleWeight) return;

  if constexpr (Debug::debug)
    if (moduleWeight < 0) {
      if constexpr (Debug::debug) std::cout << "!! Negative weight !!" << std::endl;
      // 		image.clear();
      return;
    }

  value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / horizontalResolution;
  value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / verticalResolution;

  {  // for Timer
    Debug::Timer timer("Computing pixel values ...", verbose);

    tbb::parallel_for(0U, horizontalResolution, [&](unsigned int i) {
      tbb::parallel_for(0U, verticalResolution, [&](unsigned int j) {
        image[i][j] = _get_pixel_value(start,
                                       end,
                                       {box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j},
                                       delta,
                                       p,
                                       normalize,
                                       moduleWeight);
      });
    });
  }  // Timer death
}

template <typename value_type>
inline void Module<value_type>::_compute_2D_image(Module::image_type &image,
                                                  const typename module_type::const_iterator start,
                                                  const typename module_type::const_iterator end,
                                                  unsigned int horizontalResolution,
                                                  unsigned int verticalResolution,
                                                  get_2dpixel_value_function_type get_pixel_value) const {
  image.resize(horizontalResolution, std::vector<value_type>(verticalResolution));
  const Box<value_type> &box = this->box_;
  value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / horizontalResolution;
  value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / verticalResolution;

  {  // for Timer
    Debug::Timer timer("Computing pixel values ...", verbose);

    // #pragma omp parallel for collapse(2)
    // 			for (unsigned int i = 0; i < horizontalResolution; i++)
    // 			{
    // 				for (unsigned int j = 0; j < verticalResolution;
    // j++)
    // 				{
    // 					image[i][j] = get_pixel_value(
    // 						start,
    // 						end,
    // 						box.get_lower_corner()[0] +
    // stepX * i,
    // box.get_lower_corner()[1] + stepY * j);
    // 				}
    // 			}
    tbb::parallel_for(0U, horizontalResolution, [&](unsigned int i) {
      tbb::parallel_for(0U, verticalResolution, [&](unsigned int j) {
        image[i][j] =
            get_pixel_value(start, end, box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j);
      });
    });

  }  // Timer death
}

template <typename value_type>
inline value_type Module<value_type>::_get_pixel_value(const typename module_type::iterator start,
                                                       const typename module_type::iterator end,
                                                       const filtration_type x,
                                                       const value_type delta,
                                                       const value_type p,
                                                       const bool normalize,
                                                       const value_type moduleWeight) const {
  value_type value = 0;
  if (p == 0) {
    /* #pragma omp parallel for reduction(+ : value) */
    for (auto it = start; it != end; it++) {
      value += it->get_local_weight(x, delta);
    }
    if (normalize) value /= moduleWeight;
    return value;
  }
  if (p != inf) {
    /* #pragma omp parallel for reduction(+ : value) */
    for (auto it = start; it != end; it++) {
      value_type summandWeight = it->get_interleaving();
      value_type summandXWeight = it->get_local_weight(x, delta);
      value += std::pow(summandWeight, p) * summandXWeight;
    }
    if (normalize) value /= moduleWeight;
    return value;
  }

  /* #pragma omp parallel for reduction(std::max : value) */
  for (auto it = start; it != end; it++) {
    value = std::max(value, it->get_local_weight(x, delta));
  }
  return value;
}

/////////////////////////////////////////////////

template <typename value_type>
inline Summand<value_type>::Summand()
    : birth_corners_(1, births_type::Generator::T_inf),
      death_corners_(1, -births_type::Generator::T_inf),
      distanceTo0_(-1),
      dimension_(-1) {}

template <typename value_type>
inline Summand<value_type>::Summand(
    const typename std::vector<typename Summand<value_type>::filtration_type> &birth_corners,
    const typename std::vector<typename Summand<value_type>::filtration_type> &death_corners,
    dimension_type dimension)
    : birth_corners_(birth_corners), death_corners_(death_corners), distanceTo0_(-1), dimension_(dimension) {}

template <typename value_type>
inline bool Summand<value_type>::contains(const filtration_type &x) const {
  bool out = false;
  for (const auto &birth : this->birth_corners_) {  // checks if there exists a birth smaller than x
    if (birth <= x) {
      out = true;
      break;
    }
  }
  if (!out) return false;
  out = false;
  for (const auto &death : this->death_corners_) {
    if (x <= death) {
      out = true;
      break;
    }
  }
  return out;
}

template <typename value_type>
inline Summand<value_type>::Summand(const typename Summand<value_type>::births_type &birth_corners,
                                    const typename Summand<value_type>::deaths_type &death_corners,
                                    dimension_type dimension)
    : birth_corners_(birth_corners), death_corners_(death_corners), distanceTo0_(-1), dimension_(dimension) {}

template <typename value_type>
inline value_type Summand<value_type>::get_interleaving(const Box<value_type> &box) {
  _compute_interleaving(box);
  return distanceTo0_;
}

template <typename value_type>
inline value_type Summand<value_type>::get_interleaving() const {
  return distanceTo0_;
}

template <typename value_type>
inline value_type Summand<value_type>::get_local_weight(const filtration_type &x, const value_type delta) const {
  bool rectangle = delta <= 0;

  // TODO: add assert to verify that x.size == birth.size/death.size
  // if they are not infinite.

  filtration_type mini(x.num_parameters());
  filtration_type maxi(x.num_parameters());

  // box on which to compute the local weight
  for (unsigned int i = 0; i < x.size(); i++) {
    mini[i] = delta <= 0 ? x[i] + delta : x[i] - delta;
    maxi[i] = delta <= 0 ? x[i] - delta : x[i] + delta;
  }

  // Pre-allocating
  std::vector<filtration_type> birthList(birth_corners_.num_generators());
  std::vector<filtration_type> deathList(death_corners_.num_generators());
  unsigned int lastEntry = 0;
  for (const filtration_type &birth : birth_corners_) {
    if (birth <= maxi) {
      unsigned int dim = std::max(birth.num_parameters(), mini.num_parameters());
      filtration_type tmpBirth(dim);
      for (unsigned int i = 0; i < dim; i++) {
        auto birthi = birth.num_parameters() > i ? birth[i] : birth[0];
        auto minii = mini.num_parameters() > i ? mini[i] : mini[0];
        tmpBirth[i] = std::max(birthi, minii);
      }

      birthList[lastEntry].swap(tmpBirth);
      lastEntry++;
    }
  }
  birthList.resize(lastEntry);

  // Thresholds birthlist & deathlist to B_inf(x,delta)
  lastEntry = 0;
  for (const filtration_type &death : death_corners_) {
    if (death >= mini) {
      unsigned int dim = std::max(death.num_parameters(), maxi.num_parameters());
      filtration_type tmpDeath(dim);
      for (unsigned int i = 0; i < dim; i++) {
        auto deathi = death.num_parameters() > i ? death[i] : death[0];
        auto maxii = maxi.num_parameters() > i ? maxi[i] : maxi[0];
        tmpDeath[i] = std::min(deathi, maxii);
      }

      deathList[lastEntry].swap(tmpDeath);
      lastEntry++;
    }
  }
  deathList.resize(lastEntry);
  value_type local_weight = 0;
  if (!rectangle) {
    // Local weight is inteleaving to 0 of module restricted to the square
    // #pragma omp parallel for reduction(std::max: local_weight)
    Box<value_type> trivial_box;
    for (const filtration_type &birth : birthList) {
      if (birth.num_parameters() == 0) continue;
      for (const filtration_type &death : deathList) {
        if (death.num_parameters() > 0)
          local_weight = std::max(local_weight,
                                  _get_max_diagonal(birth,
                                                    death,
                                                    trivial_box));  // if box is empty, does not thredhold
                                                                    // (already done before).
      }
    }
    return local_weight / (2 * std::abs(delta));
  } else {
    // local weight is the volume of the largest rectangle in the restricted
    // module #pragma omp parallel for reduction(std::max: local_weight)
    for (const filtration_type &birth : birthList) {
      if (birth.num_parameters() == 0) continue;
      for (const filtration_type &death : deathList) {
        if (death.num_parameters() > 0) local_weight = std::max(local_weight, _rectangle_volume(birth, death));
      }
    }
    return local_weight / std::pow(2 * std::abs(delta), x.num_parameters());
  }
}

template <typename value_type>
inline std::tuple<int, int> Summand<value_type>::distance_idx_to_lower(const filtration_type &x) const {
  value_type distance_to_lower = std::numeric_limits<value_type>::infinity();
  int b_idx = -1;  // argmin_b max_i (b-x)_x
  int param = 0;
  auto count = 0u;
  for (const auto &birth : birth_corners_) {
    value_type temp = -std::numeric_limits<value_type>::infinity();  // max_i(birth - x)_+
    int temp_idx = 0;
    for (auto i = 0u; i < birth.size(); ++i) {
      auto plus = birth[i] - x[i];
      if (plus > temp) {
        temp_idx = i;
        temp = plus;
      }
    }
    if (temp < distance_to_lower) {
      distance_to_lower = temp;
      param = temp_idx;
      b_idx = count;
    }
    ++count;
  }
  return {b_idx, param};
}

template <typename value_type>
inline std::tuple<int, int> Summand<value_type>::distance_idx_to_upper(const filtration_type &x) const {
  value_type distance_to_upper = std::numeric_limits<value_type>::infinity();
  int d_idx = -1;  // argmin_d max_i (x-death)
  int param = 0;
  auto count = 0u;
  for (const auto &death : death_corners_) {
    value_type temp = -std::numeric_limits<value_type>::infinity();  // max_i(death-x)_+
    int temp_idx = 0;
    for (auto i = 0u; i < death.size(); ++i) {
      auto plus = x[i] - death[i];
      if (plus > temp) {
        temp_idx = i;
        temp = plus;
      }
    }
    if (temp < distance_to_upper) {
      distance_to_upper = temp;
      param = temp_idx;
      d_idx = count;
    }
    ++count;
  }
  return {d_idx, param};
}

template <typename value_type>
inline std::vector<int> Summand<value_type>::distance_idx_to(const filtration_type &x, bool full) const {
  const auto &[a, b] = Summand::distance_idx_to_lower(x);
  const auto &[c, d] = Summand::distance_idx_to_upper(x);
  if (full) [[unlikely]]
    return {a, b, c, d};
  else {
    return {a, c};
  }
}

template <typename value_type>
inline value_type Summand<value_type>::distance_to_lower(const filtration_type &x, bool negative) const {
  value_type distance_to_lower = std::numeric_limits<value_type>::infinity();
  for (const auto &birth : birth_corners_) {
    value_type temp = negative ? -std::numeric_limits<value_type>::infinity() : 0;
    for (auto i = 0u; i < birth.size(); ++i) {
      temp = std::max(temp, birth[i] - x[i]);
    }
    distance_to_lower = std::min(distance_to_lower, temp);
  }
  return distance_to_lower;
}

template <typename value_type>
inline value_type Summand<value_type>::distance_to_upper(const filtration_type &x, bool negative) const {
  value_type distance_to_upper = std::numeric_limits<value_type>::infinity();
  for (const auto &death : death_corners_) {
    value_type temp = negative ? -std::numeric_limits<value_type>::infinity() : 0;
    for (auto i = 0u; i < death.size(); ++i) {
      temp = std::max(temp, x[i] - death[i]);
    }
    distance_to_upper = std::min(distance_to_upper, temp);
  }
  return distance_to_upper;
}

template <typename value_type>
inline value_type Summand<value_type>::distance_to(const filtration_type &x, bool negative) const {
  return std::max(Summand::distance_to_lower(x, negative), Summand::distance_to_upper(x, negative));
}

template <typename value_type>
inline std::pair<value_type, value_type> Summand<value_type>::get_bar2(const Line<value_type> &l) const {
  constexpr const bool verbose = false;
  if constexpr (verbose)
    std::cout << "Computing bar of this summand of dimension " << this->get_dimension() << std::endl;
  value_type pushed_birth = std::numeric_limits<value_type>::infinity();
  value_type pushed_death = -pushed_birth;
  for (filtration_type birth : this->get_birth_list()) {
    value_type pb = l.compute_forward_intersection(birth);
    pushed_birth = std::min(pb, pushed_birth);
  }
  //
  for (const filtration_type &death : this->get_death_list()) {
    value_type pd = l.compute_backward_intersection(death);
    pushed_death = std::max(pd, pushed_death);
  }

  if (!(pushed_birth <= pushed_death)) {
    if constexpr (verbose) std::cout << "Birth <!= Death ! Ignoring this value" << std::endl;
    return {inf, inf};
  }
  if constexpr (verbose) {
    std::cout << "Final values" << pushed_birth << " ----- " << pushed_death << std::endl;
  }
  return {pushed_birth, pushed_death};
}

template <typename value_type>
inline std::pair<typename Summand<value_type>::filtration_type, typename Summand<value_type>::filtration_type>
Summand<value_type>::get_bar(const Line<value_type> &l) const {
  constexpr const bool verbose = false;
  if constexpr (verbose)
    std::cout << "Computing bar of this summand of dimension " << this->get_dimension() << std::endl;
  filtration_type pushed_birth = std::numeric_limits<filtration_type>::infinity();
  filtration_type pushed_death = std::numeric_limits<filtration_type>::minus_infinity();
  for (filtration_type birth : this->get_birth_list()) {
    filtration_type pb = l[l.compute_forward_intersection(birth)];
    if constexpr (verbose)
      std::cout << "Updating birth " << pushed_birth << " with " << pb << " pushed at " << birth << " "
                << pushed_birth.is_plus_inf();
    if ((pb <= pushed_birth) || pushed_birth.is_plus_inf()) {
      pushed_birth.swap(pb);
      if constexpr (verbose) std::cout << " swapped !";
    }
    if constexpr (verbose) std::cout << std::endl;
  }
  //
  for (const filtration_type &death : this->get_death_list()) {
    filtration_type pd = l[l.compute_backward_intersection(death)];
    if constexpr (verbose)
      std::cout << "Updating death " << pushed_death << " with " << pd << " pushed at " << death << " "
                << pushed_death.is_minus_inf() << pushed_death[0];
    if ((pd >= pushed_death) || pushed_death.is_minus_inf()) {
      pushed_death.swap(pd);
      if constexpr (verbose) std::cout << " swapped !";
    }
    if constexpr (verbose) std::cout << std::endl;
  }

  if (!(pushed_birth <= pushed_death)) {
    if constexpr (verbose) std::cout << "Birth <!= Death ! Ignoring this value" << std::endl;
    return {std::numeric_limits<filtration_type>::infinity(), std::numeric_limits<filtration_type>::infinity()};
  }
  if constexpr (verbose) {
    std::cout << "Final values" << pushed_birth << " ----- " << pushed_death << std::endl;
  }
  return {pushed_birth, pushed_death};
}

/**
 * @brief Adds the bar @p bar to the indicator module @p summand if @p bar
 * is non-trivial (ie. not reduced to a point or, if @p threshold is true,
 * its thresholded version should not be reduced to a point) .
 *
 * @param bar p_bar: to add to the support of the summand
 * @param summand p_summand: indicator module which is being completed
 * @param basepoint p_basepoint: basepoint of the line of the bar
 * @param birth p_birth: birth container (for memory optimization purposes).
 * Has to be of the size @p basepoint.size()+1.
 * @param death p_death: death container. Same purpose as @p birth but for
 * deathpoint.
 * @param threshold p_threshold: If true, will threshold the bar with @p box.
 * @param box p_box: Only useful if @p threshold is set to true.
 */

template <typename value_type>
inline void Summand<value_type>::add_bar(value_type baseBirth,
                                         value_type baseDeath,
                                         const filtration_type &basepoint,
                                         filtration_type &birth,
                                         filtration_type &death,
                                         const bool threshold,
                                         const Box<value_type> &box) {
  // bar is trivial in that case
  if (baseBirth >= baseDeath) return;
  // #pragma omp simd
  // 		for (unsigned int j = 0; j < birth.size() - 1; j++)
  // 		{
  // 			birth[j] = basepoint[j] + baseBirth;
  // 			death[j] = basepoint[j] + baseDeath;
  // 		}
  // 		birth.back() = baseBirth;
  // 		death.back() = baseDeath;

  /* #pragma omp simd */
  for (unsigned int j = 0; j < birth.size() - 1; j++) {
    value_type temp = basepoint[j] + baseBirth;
    // The box is assumed to contain all of the filtration values, if its
    // outside, its inf.
    birth[j] = temp < box.get_lower_corner()[j] ? negInf : temp;
    temp = basepoint[j] + baseDeath;
    death[j] = temp > box.get_upper_corner()[j] ? inf : temp;
  }
  birth.back() = baseBirth < box.get_lower_corner().back() ? negInf : baseBirth;
  death.back() = baseDeath > box.get_upper_corner().back() ? inf : baseDeath;

  if (threshold) {
    // std::cout << box;
    threshold_down(birth, box, basepoint);
    threshold_up(death, box, basepoint);
  }
  _add_birth(birth);
  _add_death(death);
}

template <typename value_type>
inline void Summand<value_type>::add_bar(const filtration_type &birth, const filtration_type &death) {
  _add_birth(birth);
  _add_death(death);
}

template <typename value_type>
inline void Summand<value_type>::add_bar(const filtration_type &basepoint,
                                         value_type birth,
                                         value_type death,
                                         const Box<value_type> &box) {
  constexpr const bool verbose = false;
  if (birth >= death) return;
  if constexpr (verbose) {
    std::cout << "Bar : " << basepoint + birth << "--" << basepoint + death << std::endl;
  }
  auto inf = std::numeric_limits<value_type>::infinity();
  auto container = basepoint + birth;
  for (auto i = 0u; i < container.size(); i++) {
    if (container[i] < box.get_lower_corner()[i]) container[i] = -inf;
  }
  _add_birth(container);
  container = basepoint + death;
  for (auto i = 0u; i < container.size(); i++) {
    if (container[i] > box.get_upper_corner()[i]) container[i] = inf;
  }
  _add_death(container);
}

template <typename value_type>
inline const std::vector<typename Summand<value_type>::filtration_type> &Summand<value_type>::get_birth_list() const {
  return birth_corners_.get_underlying_container();
}

template <typename value_type>
inline const std::vector<typename Summand<value_type>::filtration_type> &Summand<value_type>::get_death_list() const {
  return death_corners_.get_underlying_container();
}

template <typename value_type>
const Gudhi::multi_filtration::Multi_critical_filtration<value_type> &Summand<value_type>::get_upset() const {
  return birth_corners_;
}

template <typename value_type>
const Gudhi::multi_filtration::Multi_critical_filtration<value_type> &Summand<value_type>::get_downset() const {
  return death_corners_;
};

template <typename value_type>
inline void Summand<value_type>::clean() {
  // birth_corners_.erase(std::remove_if(birth_corners_.begin(),
  //                                     birth_corners_.end(),
  //                                     [](const std::vector<value_type> &bp) {
  //                                       // return std::any_of(
  //                                       //     bp.begin(), bp.end(),
  //                                       //     [](float value) { return !std::isfinite(value); });
  //                                       bp.size() == 0;
  //                                     }),
  //                      birth_corners_.end());
  // birth_corners_.simplify();
  // TODO : clean
}

template <typename value_type>
inline void Summand<value_type>::complete_birth(const value_type precision) {
  if (!birth_corners_.is_finite()) return;

  for (std::size_t i = 0; i < birth_corners_.num_generators(); i++) {
    for (std::size_t j = i + 1; j < birth_corners_.num_generators(); j++) {
      value_type dinf = d_inf(birth_corners_[i], birth_corners_[j]);
      if (dinf < .99 * precision) {  // for machine error ?
        _factorize_min(birth_corners_[i], birth_corners_[j]);
        birth_corners_[j] = std::remove_reference_t<decltype(birth_corners_[j])>::inf();
        i++;
      }
    }
  }
  birth_corners_.simplify();
  // _clean(birth_corners_);
}

template <typename value_type>
inline void Summand<value_type>::complete_death(const value_type precision) {
  if (!death_corners_.is_finite()) return;

  for (std::size_t i = 0; i < death_corners_.num_generators(); i++) {
    for (std::size_t j = i + 1; j < death_corners_.num_generators(); j++) {
      value_type d = d_inf(death_corners_[i], death_corners_[j]);
      if (d < .99 * precision) {
        _factorize_max(death_corners_[i], death_corners_[j]);
        death_corners_[j] = std::remove_reference_t<decltype(death_corners_[j])>::minus_inf();
        i++;
      }
    }
  }

  death_corners_.simplify();
  // _clean(death_corners_);
}

template <typename value_type>
inline dimension_type Summand<value_type>::get_dimension() const {
  return dimension_;
}

template <typename value_type>
inline value_type Summand<value_type>::get_landscape_value(const std::vector<value_type> &x) const {
  value_type out = 0;
  Box<value_type> trivial_box;
  for (const filtration_type &b : this->birth_corners_) {
    for (const filtration_type &d : this->death_corners_) {
      value_type value =
          std::min(this->_get_max_diagonal(b, x, trivial_box), this->_get_max_diagonal(x, d, trivial_box));
      out = std::max(out, value);
    }
  }
  return out;
}

template <typename value_type>
inline void Summand<value_type>::set_dimension(dimension_type dimension) {
  dimension_ = dimension;
}

template <typename value_type>
inline void Summand<value_type>::_compute_interleaving(const Box<value_type> &box) {
  distanceTo0_ = 0;
  /* #pragma omp parallel for reduction(max : distanceTo0_) */
  for (const std::vector<value_type> &birth : birth_corners_) {
    for (const std::vector<value_type> &death : death_corners_) {
      distanceTo0_ = std::max(distanceTo0_, _get_max_diagonal(birth, death, box));
    }
  }
}

/**
 * @brief Adds @p birth to the summand's @p birth_list if it is not induced
 * from the @p birth_list (ie. not comparable or smaller than another birth),
 * and removes unnecessary birthpoints (ie. birthpoints that are induced
 * by @p birth).
 *
 * @param birth_list p_birth_list: birthpoint list of a summand
 * @param birth p_birth: birth to add to the summand
 */

template <typename value_type>
inline void Summand<value_type>::_add_birth(const filtration_type &birth) {
  birth_corners_.add_generator(birth);
  return;

  // // TODO : DEPRECATE THIS OLD CODE
  // if (birth_corners_.empty()) {
  //   birth_corners_.push_back(birth);
  //   return;
  // }

  // for (const auto &current_birth : birth_corners_) {
  //   if (birth >= current_birth) {
  //     return;
  //   }
  // }
  // // this birth value is useful, we can now remove useless other filtrations
  // for (auto &current_birth : birth_corners_) {
  //   if ((!current_birth.empty()) && (birth <= current_birth)) {
  //     current_birth.clear();
  //   }
  // }

  // _clean(birth_corners_);
  // birth_corners_.push_back(birth);
}

/**
 * @brief Adds @p death to the summand's @p death_list if it is not induced
 * from the @p death_list (ie. not comparable or greater than another death),
 * and removes unnecessary deathpoints (ie. deathpoints that are induced
 * by @p death)
 *
 * @param death_list p_death_list: List of deathpoints of a summand
 * @param death p_death: deathpoint to add to this list
 */

template <typename value_type>
inline void Summand<value_type>::_add_death(const filtration_type &death) {
  death_corners_.add_generator(death);
  return;
  // // TODO:  Deprecate this old code
  // if (death_corners_.empty()) {
  //   death_corners_.push_back(death);
  //   return;
  // }

  // for (const auto &current_death : death_corners_) {
  //   if (death <= current_death) {
  //     return;
  //   }
  // }
  // // this death value is useful, we can now remove useless other filtrations
  // for (auto &current_death : death_corners_) {
  //   if (!current_death.empty() && (death >= current_death)) {
  //     current_death.clear();
  //   }
  // }
  // _clean(death_corners_);
  // death_corners_.push_back(death);
}

template <typename value_type>
inline value_type Summand<value_type>::_get_max_diagonal(const filtration_type &birth,
                                                         const filtration_type &death,
                                                         const Box<value_type> &box) const {
  // assumes birth and death to be never NaN
  if constexpr (Debug::debug)
    assert(!birth.is_finite || !death.is_finite || birth.size() == death.size() && "Inputs must be of the same size !");

  value_type s = inf;
  bool threshold_flag = !box.is_trivial();
  if (threshold_flag) {
    unsigned int dim = std::max(birth.size(), box.dimension());
    for (unsigned int i = 0; i < dim; ++i) {
      value_type max_i = box.get_upper_corner().size() > i ? box.get_upper_corner()[i] : inf;
      value_type min_i = box.get_lower_corner().size() > i ? box.get_lower_corner()[i] : negInf;
      value_type t_death = death.is_plus_inf() ? max_i : (death.is_minus_inf() ? -inf : std::min(death[i], max_i));
      value_type t_birth = birth.is_plus_inf() ? inf : (birth.is_minus_inf() ? min_i : std::max(birth[i], min_i));
      s = std::min(s, t_death - t_birth);
    }
  } else {
    unsigned int dim = std::max(birth.size(), death.size());
    for (unsigned int i = 0; i < dim; i++) {
      // if they don't have the same size, then one of them has to (+/-)infinite.
      value_type t_death = death.size() > i ? death[i] : death[0];  // assumes death is never empty
      value_type t_birth = birth.size() > i ? birth[i] : birth[0];  // assumes birth is never empty
      s = std::min(s, t_death - t_birth);
    }
  }

  return s;
}

template <typename value_type>
inline value_type Summand<value_type>::_rectangle_volume(const filtration_type &a, const filtration_type &b) const {
  if constexpr (Debug::debug) assert(a.size() == b.size() && "Inputs must be of the same size !");
  value_type s = b[0] - a[0];
  for (unsigned int i = 1; i < a.size(); i++) {
    s = s * (b[i] - a[i]);
  }
  return s;
}

template <typename value_type>
inline value_type Summand<value_type>::d_inf(const filtration_type &a, const filtration_type &b) const {
  if (a.empty() || b.empty() || a.size() != b.size()) return inf;

  value_type d = std::abs(a[0] - b[0]);
  for (unsigned int i = 1; i < a.size(); i++) d = std::max(d, std::abs(a[i] - b[i]));

  return d;
}

template <typename value_type>
inline void Summand<value_type>::_factorize_min(filtration_type &a, const filtration_type &b) {
  /* if (Debug::debug && (a.empty() || b.empty())) */
  /* { */
  /* 	std::cout << "Empty corners ??\n"; */
  /* 	return; */
  /* } */

  for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::min(a[i], b[i]);
}

template <typename value_type>
inline void Summand<value_type>::_factorize_max(filtration_type &a, const filtration_type &b) {
  /* if (Debug::debug && (a.empty() || b.empty())) */
  /* { */
  /* 	std::cout << "Empty corners ??\n"; */
  /* 	return; */
  /* } */

  for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::max(a[i], b[i]);
}

/**
 * @brief Cleans empty entries of a corner list
 *
 * @param list corner list to clean
 * @param keep_sort If true, will keep the order of the corners,
 * with a computational overhead. Defaults to false.
 */
// WARNING Does permute the output.

template <typename value_type>
inline void Summand<value_type>::_clean(std::vector<filtration_type> &list, bool keep_inf) {
  list.erase(std::remove_if(list.begin(),
                            list.end(),
                            [keep_inf](filtration_type &a) {
                              return a.empty() || ((!keep_inf) && (a.is_plus_inf() || a.is_minus_inf()));
                            }),
             list.end());
}

}  // namespace mma
}  // namespace multiparameter
}  // namespace Gudhi

#endif  // APPR
