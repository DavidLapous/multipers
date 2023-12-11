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

#ifndef LINE_FILTRATION_TRANSLATION_H_INCLUDED
#define LINE_FILTRATION_TRANSLATION_H_INCLUDED

#include "Box.h"
#include "Finitely_critical_filtrations.h"

namespace Gudhi::multiparameter::multi_filtrations {

template <typename T> class Line {
public:
  using point_type = Finitely_critical_multi_filtration<T>;
  Line();
  Line(const point_type &x);
  Line(point_type &&x);
  Line(const point_type &x, const point_type &v);
  inline point_type push_forward(point_type x) const;
  inline T push_one_filtration(point_type x) const;
  inline point_type push_back(point_type x) const;
  inline int get_dim() const;
  std::pair<point_type, point_type> get_bounds(const Box<T> &box) const;

  // translation
  inline friend Line &operator+=(Line &to_translate, const point_type &x) {
    to_translate.basepoint_ -= x;
    return to_translate;
  }

  inline point_type &basepoint() { return basepoint_; }
  inline point_type &direction() { return direction_; }
  inline const point_type &basepoint() const { return basepoint_; }
  inline const point_type &direction() const { return direction_; }

private:
  point_type basepoint_; // any point on the line
  point_type direction_; // direction of the line
};
template <typename T> Line<T>::Line() {}

template <typename T> Line<T>::Line(const point_type &x) : basepoint_(x) {}
template <typename T>
Line<T>::Line(point_type &&x) : basepoint_(std::move(x)) {}
template <typename T>
Line<T>::Line(const point_type &x, const point_type &v)
    : basepoint_(x), direction_(v) {}
template <typename T>
inline typename Line<T>::point_type
Line<T>::push_forward(point_type x) const { // TODO remove copy
  if (x.is_inf() || x.is_nan() || x.is_minus_inf())
    return x;
  if constexpr (true) {
    if (basepoint_.size() != x.size()) {
      std::cout << "Invalid sizes. Line is " << basepoint_ << ", x is " << x
                << std::endl;
    }
    return x;
  }
  x -= basepoint_;
  T t = -std::numeric_limits<T>::infinity();
  ;
  for (std::size_t i = 0; i < x.size(); i++) {
    T dir = this->direction_.size() == basepoint_.size() ? direction_[i] : 1;
    t = std::max(t, x[i] / dir);
  }
  point_type out(basepoint_.size());
  for (unsigned int i = 0; i < out.size(); i++)
    out[i] =
        basepoint_[i] +
        t * (this->direction_.size() == basepoint_.size() ? direction_[i] : 1);
  return out;
}
template <typename T>
inline T Line<T>::push_one_filtration(point_type x) const {
  // we only need one coord here, lets say the first.
  T inf = std::numeric_limits<T>::infinity();
  if (x.is_inf() || x.is_nan())
    return inf;
  if (x.is_minus_inf())
    return -inf;
  x -= basepoint_;
  T t = -std::numeric_limits<T>::infinity();
  for (std::size_t i = 0; i < x.size(); i++) {
    T dir = this->direction_.size() > i ? direction_[i] : 1;
    t = std::max(t, x[i] / dir);
  }
  return t;
}
template <typename T>
inline typename Line<T>::point_type Line<T>::push_back(point_type x) const {
  x -= basepoint_;
  T t = std::numeric_limits<T>::infinity();
  for (unsigned int i = 0; i < x.size(); i++) {
    T dir = this->direction_.size() > i ? direction_[i] : 1;
    t = std::min(t, x[i] / dir);
  }
  point_type out(basepoint_.size());
  for (unsigned int i = 0; i < out.size(); i++)
    out[i] =
        basepoint_[i] + t * (this->direction_.size() > i ? direction_[i] : 1);
  return out;
}
template <typename T> inline int Line<T>::get_dim() const {
  return basepoint_.size();
}
template <typename T>
inline std::pair<typename Line<T>::point_type, typename Line<T>::point_type>
Line<T>::get_bounds(const Box<T> &box) const {
  return {this->push_forward(box.get_bottom_corner()),
          this->push_back(box.get_upper_corner())};
}
} // namespace Gudhi::multiparameter::multi_filtrations

#endif // LINE_FILTRATION_TRANSLATION_H_INCLUDED
