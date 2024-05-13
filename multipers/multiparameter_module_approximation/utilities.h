/*    This file is part of the MMA Library -
 * https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT. See
 * file LICENSE for full license details. Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <cassert>
#include <gudhi/Simplex_tree/Simplex_tree_multi.h>
#include <limits>
#include <vector>

namespace Gudhi::multiparameter::mma {

const bool verbose = false;

using index = unsigned int;
using value_type = double;
using filtration_type = Gudhi::multiparameter::multi_filtrations::
    Finitely_critical_multi_filtration<value_type>;
using multifiltration_type = std::vector<filtration_type>;
using python_filtration_type = std::vector<value_type>;
using python_multifiltration_type = std::vector<python_filtration_type>;

using dimension_type = int;
using persistence_pair = std::pair<value_type, value_type>;
using boundary_type = std::vector<index>;
using boundary_matrix = std::vector<boundary_type>;
using permutation_type = std::vector<std::size_t>;
using point_type = Gudhi::multiparameter::multi_filtrations::
    Finitely_critical_multi_filtration<value_type>;
using corner_type = Gudhi::multiparameter::multi_filtrations::
    Finitely_critical_multi_filtration<value_type>;
using corners_type =
    std::pair<std::vector<corner_type>, std::vector<corner_type>>;
// using python_bar =
//     std::pair<std::vector<value_type>,
//               std::vector<value_type>>; // This type is for python
using multipers_barcode = std::vector<std::vector<value_type>>;
const value_type inf = std::numeric_limits<value_type>::infinity();
const value_type negInf = -1 * inf;
using interval_type = std::pair<point_type, point_type>;

// template<typename T>
// bool is_smaller(const std::vector<T>& x, const std::vector<T>& y)
// {
//     for (unsigned int i = 0; i < std::min(x.size(), y.size()); i++)
//         if (x[i] > y[i]) return false;
//     return true;
// }
// template<typename T>
// bool is_greater(const std::vector<T>& x, const std::vector<T>& y)
// {
//     for (unsigned int i = 0; i < std::min(x.size(), y.size()); i++)
//         if (x[i] < y[i]) return false;
//     return true;
// }

struct Bar {
  Bar() : dim(-1), birth(-1), death(-1) {}

  Bar(dimension_type dim, int birth, int death)
      : dim(dim), birth(birth), death(death) {}

  dimension_type dim;
  int birth;
  int death;
};

using barcode_type = std::vector<Bar>;

struct Diagram_point {
  Diagram_point() : dim(-1), birth(-1), death(-1) {}

  Diagram_point(dimension_type dim, value_type birth, value_type death)
      : dim(dim), birth(birth), death(death) {}

  dimension_type dim;
  value_type birth;
  value_type death;
};
using diagram_type = std::vector<Diagram_point>;

template <typename filtration_type = filtration_type>
struct MultiDiagram_point {
public:
  MultiDiagram_point() : dim(-1), birth({}), death({}) {}

  MultiDiagram_point(dimension_type dim, filtration_type birth,
                     filtration_type death)
      : dim(dim), birth(birth), death(death) {}
  dimension_type get_dimension() const { return dim; }
  const filtration_type& get_birth() const { return birth; }
  const filtration_type& get_death() const { return death; }

private:
  dimension_type dim;
  filtration_type birth;
  filtration_type death;
};
template <typename filtration_type, typename value_type>
struct MultiDiagram { // for python interface
  static_assert(std::is_same<typename filtration_type::value_type, value_type>::value,
                "filtration_type and value_type must be the same");
public:
  using iterator =
      typename std::vector<MultiDiagram_point<filtration_type>>::const_iterator;
  MultiDiagram() {}
  MultiDiagram(std::vector<MultiDiagram_point<filtration_type>> &m)
      : multiDiagram(m) {}

  using python_bar =
      std::pair<std::vector<typename filtration_type::value_type>,
                std::vector<typename filtration_type::value_type>>; // This type is for python
  std::vector<python_bar> get_points(
      const dimension_type dimension = -1) const { // dump for python interface
    std::vector<python_bar> out;
    out.reserve(multiDiagram.size());
    for (const MultiDiagram_point<filtration_type> &pt : multiDiagram) {
      if (dimension == -1 || pt.get_dimension() == dimension) {
        if (pt.get_birth().size() > 0 && pt.get_death().size() > 0 &&
            !pt.get_birth().is_inf() && !pt.get_death().is_minus_inf())
          out.push_back({pt.get_birth(), pt.get_death()});
      }
    }
    /* out.shrink_to_fit(); */
    return out;
  }
  std::vector<std::vector<double>> to_multipers(
      const dimension_type dimension = -1) const { // dump for python interface
    std::vector<std::vector<double>> out;
    out.reserve(multiDiagram.size());
    for (const MultiDiagram_point<filtration_type> &pt : multiDiagram) {
      if (dimension == -1 || pt.get_dimension() == dimension) {
        const auto &b = pt.get_birth().template astype<double>();
        const auto &d = pt.get_death().template astype<double>();
        assert(!(b.is_inf() || b.is_minus_inf() || d.is_inf() ||
                 d.is_minus_inf()));
        out.push_back({b[0], d[0], b[1], d[1]});
      }
    }
    out.shrink_to_fit();
    return out;
  }
  iterator begin() const { return this->multiDiagram.begin(); }
  iterator end() const { return this->multiDiagram.end(); }
  unsigned int size() { return this->multiDiagram.size(); }
  void set(std::vector<MultiDiagram_point<filtration_type>> &m) {
    this->multiDiagram.swap(m);
  }
  std::vector<MultiDiagram_point<filtration_type>> &getref() {
    return this->multiDiagram;
  }
  MultiDiagram_point<filtration_type> &operator[](unsigned int i) {
    return this->multiDiagram[i];
  }
  MultiDiagram_point<filtration_type> &at(const unsigned int i) {
    return multiDiagram[i];
  }

private:
  std::vector<MultiDiagram_point<filtration_type>> multiDiagram;
};
template <typename filtration_type, typename value_type > struct MultiDiagrams {
  static_assert(std::is_same<typename filtration_type::value_type, value_type>::value,
                "filtration_type and value_type must be the same");
public:
  using iterator =
      typename std::vector<MultiDiagram<filtration_type, value_type>>::const_iterator;
  using nciterator =
      typename std::vector<MultiDiagram<filtration_type, value_type>>::iterator;
  MultiDiagrams() {}
  MultiDiagrams(unsigned int size) : multiDiagrams(size) {}

  std::vector<std::vector<std::vector<double>>> to_multipers() {
    unsigned int nsummands = this->multiDiagrams.front().size();
    unsigned int nlines = this->multiDiagrams.size();
    // std::vector<std::vector<std::vector<double>>> out(nsummands,
    // std::vector<std::vector<double>>(nlines, std::vector<double>(5)));
    std::vector<std::vector<std::vector<double>>> out(nsummands);
    for (unsigned int i = 0; i < nsummands; i++) {
      out[i].reserve(nlines);
      for (unsigned int j = 0; j < nlines; j++) {
        const MultiDiagram_point<filtration_type> &pt =
            this->multiDiagrams[j][i];
        /* if(pt.get_birth().is_inf() || pt.get_death().is_minus_inf()) */
        /*     out[i].push_back({0, 0, 0, 0,static_cast<value_type>(j)}); */
        /* else */
        double a, b, c, d;
        double inf = std::numeric_limits<double>::infinity();
        if (pt.get_birth().is_inf()) {
          a = 0;
          b = 0;
        } else if (pt.get_birth().is_minus_inf()) {
          a = -inf;
          b = -inf;
        } else {
          a = pt.get_birth()[0];
          b = pt.get_birth()[1];
        }
        if (pt.get_death().is_inf()) {
          c = inf;
          d = inf;
        }
        if (pt.get_death().is_minus_inf()) {
          c = 0;
          d = 0;
        } else {
          c = pt.get_death()[0];
          d = pt.get_death()[1];
        }
        /* out[i].push_back({pt.get_birth()[0], pt.get_death()[0],
         * pt.get_birth()[1], pt.get_death()[1],static_cast<value_type>(j)}); */
        out[i].push_back({a, c, b, d, static_cast<double>(j)});
      }
    }
    return out;
  }
  using __for_python_plot_type =
      std::pair<std::vector<std::pair<double, double>>,
                std::vector<unsigned int>>;
  __for_python_plot_type _for_python_plot(dimension_type dimension = -1,
                                          value_type min_persistence = 0) {
    __for_python_plot_type out;
    auto &bars = out.first;
    auto &summand_idx = out.second;
    bars.reserve(this->multiDiagrams.size() * this->multiDiagrams[0].size() *
                 2);
    summand_idx.reserve(this->multiDiagrams.size() *
                        this->multiDiagrams[0].size());
    for (const MultiDiagram<filtration_type, value_type> &multiDiagram :
         this->multiDiagrams) {
      unsigned int count = 0;
      for (const MultiDiagram_point<filtration_type> &bar : multiDiagram) {
        const auto &birth = bar.get_birth().template astype<double>();
        const auto &death = bar.get_death().template astype<double>();
        if ((dimension == -1 || bar.get_dimension() == dimension) &&
            birth.size() > 1 && death.size() > 1 &&
            (death[0] > birth[0] + min_persistence)) {
          // Checking >1 ensure that filtration is not inf or -inf or nan.
          bars.push_back({birth[0], death[0]});
          bars.push_back({birth[1], death[1]});
          summand_idx.push_back(count);
        }
        count++;
      }
    }
    return out;
  }
  MultiDiagram<filtration_type, value_type> &operator[](const unsigned int i) {
    return multiDiagrams[i];
  }
  MultiDiagram<filtration_type, value_type> &at(const unsigned int i) {
    return multiDiagrams[i];
  }
  iterator begin() const {
    return this->multiDiagrams.begin();
  } // cython bug : iterators like bc in bcs crash)
  iterator end() const { return this->multiDiagrams.end(); }

using python_bar =
    std::pair<std::vector<value_type>,
              std::vector<value_type>>; // This type is for python
  using barcodes = std::vector<std::vector<python_bar>>;
  barcodes get_points() {
    unsigned int nsummands = this->multiDiagrams.front().size();
    unsigned int nlines = this->multiDiagrams.size();
    // std::vector<std::vector<std::vector<double>>> out(nsummands,
    // std::vector<std::vector<double>>(nlines, std::vector<double>(5)));
    barcodes out(nlines, std::vector<python_bar>(nsummands));
    for (unsigned int i = 0; i < nlines; i++) {
      for (unsigned int j = 0; j < nsummands; j++) {
        const MultiDiagram_point<filtration_type> &pt =
            this->multiDiagrams[i][j];
        out[i][j] = {pt.get_birth(), pt.get_death()};
      }
    }
    return out;
  }

  unsigned int size() const { return this->multiDiagrams.size(); }

private:
  std::vector<MultiDiagram<filtration_type, value_type>> multiDiagrams;
  /* inline bool _is_inf(const std::vector<value_type> &truc) const{ */
  /*     for (const auto coord : truc) */
  /*         if (coord != inf)   return false; */
  /*     return true; */
  /* } */
  /* inline bool _is_negInf(const std::vector<value_type> &truc) const{ */
  /*     for (const auto coord : truc) */
  /*         if (coord != negInf)   return false; */
  /*     return true; */
  /* } */
};

void inline threshold_up(point_type &point,
                  const multi_filtrations::Box<value_type> &box,
                  const point_type &basepoint) {
  Gudhi::multiparameter::multi_filtrations::Finitely_critical_multi_filtration
      point_(point);
  // if (is_smaller(point, box.get_upper_corner())) return;
  if (point_ <= box.get_upper_corner())
    return;

  //    if  (verbose && Debug::debug) Debug::disp_vect(point);

  if (basepoint[0] == negInf) [[unlikely]]
    return;

  // ie. point at infinity, assumes [basepoint,0] is smaller than box.second
  if (point.back() == inf) [[unlikely]] {
    //        if (verbose) std::cout << " Infinite point" << std::endl;

    value_type threshold = box.get_upper_corner().back();
    for (unsigned int i = 0; i < point.size(); i++) {
      threshold = std::min(threshold, box.get_upper_corner()[i] - basepoint[i]);
    }
    for (unsigned int i = 0; i < point.size() - 1; i++)
      point[i] = basepoint[i] + threshold;
    point.back() = threshold;

    return;
  }

  // if (!is_greater(point, box.get_bottom_corner())) {
  if (box.get_bottom_corner() <= point_) {
    point[0] = inf; // puts point to infinity
    //        if  (verbose) std::cout << "buggy point" << std::endl;
    return;
  }
  // in this last case, at least 1 coord of point is is_greater than a coord of
  // box.second

  value_type threshold = point[0] - box.get_upper_corner()[0];
  for (std::size_t i = 1; i < point.size(); i++) {
    threshold = std::max(threshold, point[i] - box.get_upper_corner()[i]);
  }

  //    if  (verbose)
  //            std::cout << "Thresholding the point with "<< threshold << " at
  //            ";

  for (std::size_t i = 0; i < point.size(); i++)
    point[i] -= threshold;

  //    if  (verbose && Debug::debug) Debug::disp_vect(point);
}

/**
 * @brief Threshold a point to the positive cone of b=box.first
 * (ie. the set \f$\{x \in \mathbb R^n \mid x \ge b\})
 * along the slope 1 line crossing this point.
 *
 * @param point The point to threshold.
 * @param box box.fist is the point defining where to threshold.
 * @param basepoint Basepoint of the slope 1 line crossing the point.
 * Meant to handle infinite cases (when the point have infinite coordinates,
 * we cannot infer the line).
 */

void inline threshold_down(point_type &point,
                    const multi_filtrations::Box<value_type> &box,
                    const point_type &basepoint) {
  if (basepoint[0] == negInf) [[unlikely]]
    return;

  if (point.back() == inf)
      [[unlikely]] { // ie. point at infinity -> feature never appearing
    return;
  }

  // if (is_greater(point, box.get_bottom_corner())) return;
  if (point >= box.get_bottom_corner())
    return;

  // if (!is_smaller(point, box.get_upper_corner())) {
  if (!(point <= box.get_upper_corner())) {
    point[0] = inf; // puts point to infinity
    return;
  }

  value_type threshold = box.get_bottom_corner()[0] - point[0];
  for (unsigned int i = 1; i < point.size(); i++) {
    threshold = std::max(threshold, box.get_bottom_corner()[i] - point[i]);
  }
  for (unsigned int i = 0; i < point.size(); i++)
    point[i] += threshold;
}

} // namespace Gudhi::multiparameter::mma

//// Different implementations of the matrix columns. Set seems to be the
/// fastest / in our tests. / using Vineyard_matrix_type =
/// RU_matrix<Heap_column>; / using Vineyard_matrix_type =
/// RU_matrix<List_column>; /  using Vineyard_matrix_type =
/// RU_matrix<Vector_column>; / using Vineyard_matrix_type =
/// RU_matrix<Unordered_set_column>;
// #include "ru_matrix.h"
///*#include "heap_column.h"*/
///*#include "list_column.h"*/
//// #include "list_column_2.h"
///*#include "vector_column.h"*/
// #include "set_column.h"
///*#include "unordered_set_column.h"*/
//
// namespace Gudhi::multiparameter::mma {
// using Vineyard_matrix_type = RU_matrix<Set_column>;
//}

#endif // UTILITIES_H
