/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which
 * is released under MIT. See file LICENSE or go to
 * https://gudhi.inria.fr/licensing/ for full license details. Author(s): David
 * Loiseaux
 *
 *    Copyright (C) 2023 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */
/**
 * @file box.h
 * @author David Loiseaux
 * @brief BOX.
 */

#ifndef BOX_H_INCLUDED
#define BOX_H_INCLUDED

#include <gudhi/Debug_utils.h>
#include <ostream>
#include <vector>

#include "Finitely_critical_filtrations.h"

/**
 * @brief Simple box in $\mathbb R^n$ .
 */

namespace Gudhi::multiparameter::multi_filtrations {

template <typename T> class Box {
  using point_type = Finitely_critical_multi_filtration<T>;

public:
  Box();
  Box(T a, T b, T c, T d) : bottomCorner_({a, b}), upperCorner_({c, d}){};
  Box(const point_type &bottomCorner, const point_type &upperCorner);
  Box(const std::pair<point_type, point_type> &box);

  void inflate(T delta);
  const point_type &get_bottom_corner() const;
  const point_type &get_upper_corner() const;
  point_type &get_bottom_corner();
  point_type &get_upper_corner();
  inline void threshold_up(point_type &x) const;
  inline void threshold_down(point_type &x) const;
  bool contains(const point_type &point) const;
  void infer_from_filters(const std::vector<point_type> &Filters_list);
  bool is_trivial() const;
  std::pair<const point_type &, const point_type &> get_pair() const {
    return {bottomCorner_, upperCorner_};
  }
  std::pair<point_type &, point_type &> get_pair() {
    return {bottomCorner_, upperCorner_};
  }
  std::size_t dimension() const { return bottomCorner_.size(); }

private:
  point_type bottomCorner_;
  point_type upperCorner_;
};

template <typename T> inline Box<T>::Box() {}

template <typename T>
inline Box<T>::Box(const point_type &bottomCorner,
                   const point_type &upperCorner)
    : bottomCorner_(bottomCorner), upperCorner_(upperCorner) {
  GUDHI_CHECK(bottomCorner.size() == upperCorner.size() &&
                  bottomCorner <= upperCorner,
              "This box is trivial !");
}

template <typename T>
inline Box<T>::Box(const std::pair<point_type, point_type> &box)
    : bottomCorner_(box.first), upperCorner_(box.second) {}

template <typename T> inline void Box<T>::inflate(T delta) {
  bottomCorner_ -= delta;
  upperCorner_ += delta;
}

/**
 * Define a box containing the filtration values.
 */
template <typename T>
inline void
Box<T>::infer_from_filters(const std::vector<point_type> &Filters_list) {
  int dimension = Filters_list.size();
  int nsplx = Filters_list[0].size();
  std::vector<T> lower(dimension);
  std::vector<T> upper(dimension);
  for (int i = 0; i < dimension; i++) {
    T min = Filters_list[i][0];
    T max = Filters_list[i][0];
    for (int j = 1; j < nsplx; j++) {
      min = std::min(min, Filters_list[i][j]);
      max = std::max(max, Filters_list[i][j]);
    }
    lower[i] = min;
    upper[i] = max;
  }
  bottomCorner_.swap(lower);
  upperCorner_.swap(upper);
}
template <typename T> inline bool Box<T>::is_trivial() const {
  return bottomCorner_.empty() || upperCorner_.empty() ||
         bottomCorner_.size() != upperCorner_.size();
}

template <typename T>
inline const typename Box<T>::point_type &Box<T>::get_bottom_corner() const {
  return bottomCorner_;
}

template <typename T>
inline const typename Box<T>::point_type &Box<T>::get_upper_corner() const {
  return upperCorner_;
}

template <typename T>
inline typename Box<T>::point_type &Box<T>::get_bottom_corner() {
  return bottomCorner_;
}

template <typename T>
inline typename Box<T>::point_type &Box<T>::get_upper_corner() {
  return upperCorner_;
}

template <typename T>
inline bool Box<T>::contains(const point_type &point) const {
  if (point.size() != bottomCorner_.size()) {
    std::cerr << "Box and point are not of the same size." << std::endl;
    return false;
  }

  return bottomCorner_ <= point && point <= upperCorner_;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Box<T> &box) {
  os << "Box -- Bottom corner : ";
  os << box.get_bottom_corner();
  os << ", Top corner : ";
  os << box.get_upper_corner();
  return os;
}

template <typename T> inline void Box<T>::threshold_up(point_type &x) const {
  for (auto i = 0u; i < x.size(); i++) {
    auto t = upperCorner_[i];
    if (x[i] > t)
      x[i] = t;
  }
  return;
}
template <typename T> inline void Box<T>::threshold_down(point_type &x) const {
  for (auto i = 0u; i < x.size(); i++) {
    auto t = bottomCorner_[i];
    if (x[i] < t)
      x[i] = t;
  }
  return;
}

} // namespace Gudhi::multiparameter::multi_filtrations

#endif // BOX_H_INCLUDED
