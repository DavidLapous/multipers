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

#ifndef FINITELY_CRITICAL_FILTRATIONS_H_
#define FINITELY_CRITICAL_FILTRATIONS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace Gudhi::multiparameter::multi_filtrations {

/** \class Finitely_critical_multi_filtration
 * Finitely_critical_multi_filtration.h
 * gudhi/Simplex_tree/multi_filtrations/Finitely_critical_multi_filtration.h
 * \brief Vector-like filtration value, for multiparameter persistence, with
 * numpy-like methods.
 *
 * \ingroup multiparameter
 *
 * \details Child of `std::vector<T>` that has numpy-like pointwise operators.
 * One critical simplicial filtrations are filtrations such that the lifetime
 * each simplex is a positive cone, e.g.
 *  - \f$ \{ x \in  \mathbb R^2 : x>=(1,2)\} \f$ is valid, while
 *  - \f$ \{ x \in  \mathbb R^2 : x>=(1,2)\} \cap \{x \in  \mathbb R^2 :
 * x>=(2,1)\} \f$ is not- Finitely critical filtrations are filtrations such
 * that the lifetime of each simplex is a union of such cones, e.g.,
 *  - \f$ \{ x \in  \mathbb R^2 : x>=(1,2)\} \cap \{ x \in  \mathbb R^2 :
 * x>=(2,1)\} \f$ is finitely critical, and more particularly 2-critical, while
 *  - \f$ \{ x \in  \mathbb R^2 : x>= \mathrm{epigraph}(y\mapsto e^{-y})\} \f$
 * is not. Non 1-critical filtrations are not supported yet. \tparam T value
 * type of the vector-like.
 */
template <typename T = float>
class Finitely_critical_multi_filtration : public std::vector<T> {
public:
  Finitely_critical_multi_filtration() : std::vector<T>(){};
  Finitely_critical_multi_filtration(int n)
      : std::vector<T>(n, -T_inf){}; // minus infinity by default
  Finitely_critical_multi_filtration(int n, T value)
      : std::vector<T>(n, value){};
  Finitely_critical_multi_filtration(std::initializer_list<T> init)
      : std::vector<T>(init){};
  Finitely_critical_multi_filtration(const std::vector<T> &v)
      : std::vector<T>(v){};
  Finitely_critical_multi_filtration(std::vector<T> &&v)
      : std::vector<T>(std::move(v)){};
  Finitely_critical_multi_filtration(typename std::vector<T>::iterator it_begin,
                                     typename std::vector<T>::iterator it_end)
      : std::vector<T>(it_begin, it_end){};
  Finitely_critical_multi_filtration(
      typename std::vector<T>::const_iterator it_begin,
      typename std::vector<T>::const_iterator it_end)
      : std::vector<T>(it_begin, it_end){};

  operator std::vector<T> &() const { return *this; }
  std::vector<T> get_vector() const {
    return static_cast<std::vector<T>>(*this);
  }

  inline bool is_inf() const {
    if (this->size() != 1)
      return false;
    return *(this->begin()) == T_inf;
  }
  inline bool is_minus_inf() const {
    if (this->size() != 1)
      return false;
    return *(this->begin()) == -T_inf;
  }
  inline bool is_nan() const {
    if (this->size() != 1)
      return false;
    return std::isnan(*(this->begin()));
  }
  inline bool is_finite() const {
    if (this->size() > 1)
      return true;
    if (this->size() == 0)
      return false;
    auto first_value = *(this->begin()); // TODO : Maybe check all entries ?
    if (std::isnan(first_value) || first_value == -T_inf ||
        first_value == T_inf)
      return false;
    return true;
  }

  inline friend bool operator<(const Finitely_critical_multi_filtration &a,
                               const Finitely_critical_multi_filtration &b) {
    if (a.is_inf() || a.is_nan() || b.is_nan() || b.is_minus_inf())
      return false;
    if (b.is_inf() || a.is_minus_inf())
      return true;
    bool isSame = true;
    auto n = a.size();
    assert(a.size() == b.size());
    for (auto i = 0u; i < n; ++i) {
      if (a[i] > b[i])
        return false;
      if (isSame && a[i] != b[i])
        isSame = false;
    }
    if (isSame)
      return false;
    return true;
  }
  inline friend bool operator<=(const Finitely_critical_multi_filtration &a,
                                const Finitely_critical_multi_filtration &b) {
    if (a.is_nan() || b.is_nan())
      return false;
    if (b.is_inf() || a.is_minus_inf())
      return true;
    if (a.is_inf() || b.is_minus_inf())
      return false;
    auto n = a.size();
    assert(a.size() == b.size());
    for (auto i = 0u; i < n; ++i) {
      if (a[i] > b[i])
        return false;
    }
    return true;
  }

  // GREATER THAN OPERATORS
  inline friend bool operator>(const Finitely_critical_multi_filtration &a,
                               const Finitely_critical_multi_filtration &b) {
    return b < a;
  }
  inline friend bool operator>=(const Finitely_critical_multi_filtration &a,
                                const Finitely_critical_multi_filtration &b) {
    return b <= a;
  }

  inline Finitely_critical_multi_filtration &
  operator=(const Finitely_critical_multi_filtration &a) {
    std::vector<T>::operator=(a);
    return *this;
  }

  std::vector<T> &_convert_back() { return *this; }

  inline friend Finitely_critical_multi_filtration &
  operator-=(Finitely_critical_multi_filtration &result,
             const Finitely_critical_multi_filtration &to_substract) {
    std::transform(result.begin(), result.end(), to_substract.begin(),
                   result.begin(), std::minus<T>());
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T> &
  operator+=(Finitely_critical_multi_filtration<T> &result,
             const Finitely_critical_multi_filtration &to_add) {
    std::transform(result.begin(), result.end(), to_add.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  inline friend Finitely_critical_multi_filtration &
  operator-=(Finitely_critical_multi_filtration &result,
             const T &to_substract) {
    // std::transform(result.begin(), result.end(),
    // to_substract.begin(),result.begin(), std::minus<T>());
    for (auto &truc : result) {
      truc -= to_substract;
    }
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T> &
  operator+=(Finitely_critical_multi_filtration<T> &result, const T &to_add) {
    for (auto &truc : result) {
      truc += to_add;
    }
    return result;
  }
  template <class add_type>
  inline friend Finitely_critical_multi_filtration<T>
  operator+(Finitely_critical_multi_filtration<T> result,
            const add_type &to_add) {
    result += to_add;
    return result;
  }
  template <class add_type>
  inline friend Finitely_critical_multi_filtration<T>
  operator-(Finitely_critical_multi_filtration<T> result,
            const add_type &to_add) {
    result -= to_add;
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T> &
  operator*=(Finitely_critical_multi_filtration<T> &result, const T &to_add) {
    for (auto &truc : result) {
      truc *= to_add;
    }
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T> &
  operator/=(Finitely_critical_multi_filtration<T> &result, const T &to_add) {
    for (auto &truc : result) {
      truc /= to_add;
    }
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T>
  operator-(Finitely_critical_multi_filtration<T> result, const T &to_add) {
    for (auto &truc : result) {
      truc -= static_cast<T>(to_add);
    }
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T>
  operator-(const T &to_add,
            const Finitely_critical_multi_filtration<T> &result) {
    return result - to_add;
  }
  inline friend Finitely_critical_multi_filtration<T>
  operator*(Finitely_critical_multi_filtration<T> result, const T &to_add) {
    for (auto &truc : result) {
      truc *= static_cast<T>(to_add);
    }
    return result;
  }
  inline friend Finitely_critical_multi_filtration<T>
  operator*(const T &to_add,
            const Finitely_critical_multi_filtration<T> &result) {
    return result - to_add;
  }

  // template<class array_like>
  inline friend bool
  operator==(const Finitely_critical_multi_filtration<T> &self,
             const Finitely_critical_multi_filtration<T> &to_compare) {
    if (self.size() != to_compare.size())
      return false;
    auto it = to_compare.begin();
    for (auto i = 0u; i < self.size(); i++) {
      if (self.at(i) != *(it++))
        return false;
    }
    return true;
  }

  inline static std::vector<std::vector<T>> to_python(
      const std::vector<Finitely_critical_multi_filtration<T>> &to_convert) {
    return std::vector<std::vector<T>>(to_convert.begin(), to_convert.end());
  }

  inline static std::vector<Finitely_critical_multi_filtration<T>>
  from_python(const std::vector<std::vector<T>> &to_convert) {
    return std::vector<Finitely_critical_multi_filtration<T>>(
        to_convert.begin(), to_convert.end());
  }

  /** \brief This functions take the filtration value `this` and pushes it to
   * the cone \f$ \{ y\in \mathbb R^n : y>=x \} \f$. After calling this method,
   * the value of this is updated to \f$ \mathrm{this} = \min \{ y\in \mathbb
   * R^n : y>=this \}\cap \{ y\in \mathbb R^n : y>=x \}
   * @param[in] x The target filtration value on which to push `this`.
   */
  inline void push_to(const Finitely_critical_multi_filtration<T> &x) {
    if (this->is_inf() || this->is_nan() || x.is_nan() || x.is_minus_inf())
      return;
    if (x.is_inf() || this->is_minus_inf()) {
      *this = x;
      return;
    }
    if (this->size() != x.size()) {
      std::cerr << "Does only work with 1-critical filtrations ! Sizes "
                << this->size() << " and " << x.size() << "are different !"
                << std::endl;
      std::cerr << "This : " << *this << std::endl;
      std::cerr << "arg : " << x << std::endl;
      throw std::logic_error("Bad sizes");
    }
    for (unsigned int i = 0; i < x.size(); i++)
      this->operator[](i) =
          this->operator[](i) > x[i] ? this->operator[](i) : x[i];
  }

  /** \brief This functions take the filtration value `this` and pulls it to the
   * cone \f$ \{ y\in \mathbb R^n : y<=x \} \f$. After calling this method, the
   * value of this is updated to \f$ \mathrm{this} = \max \{ y\in \mathbb R^n :
   * y<=this \}\cap \{ y\in \mathbb R^n : y<=x \}
   * @param[in] x The target filtration value on which to push `this`.
   */
  inline void pull_to(const Finitely_critical_multi_filtration<T> &x) {
    if (x.is_inf() || this->is_nan() || x.is_nan() || this->is_minus_inf())
      return;
    if (this->is_inf() || x.is_minus_inf()) {
      *this = x;
      return;
    }
    if (this->size() != x.size()) {
      std::cerr << "Does only work with 1-critical filtrations ! Sizes "
                << this->size() << " and " << x.size() << "are different !"
                << std::endl;
      std::cerr << "This : " << *this << std::endl;
      std::cerr << "arg : " << x << std::endl;
      throw std::logic_error("Bad sizes");
    }
    for (auto i = 0u; i < x.size(); i++)
      this->operator[](i) =
          this->operator[](i) > x[i] ? x[i] : this->operator[](i);
  }
  // Warning, this function  assumes that the comparisons checks have already
  // been made !
  inline void insert_new(Finitely_critical_multi_filtration to_concatenate) {
    this->insert(this->end(), std::move_iterator(to_concatenate.begin()),
                 std::move_iterator(to_concatenate.end()));
  }

  // scalar product of a filtration value with x.
  inline T linear_projection(const std::vector<T> &x) {
    T projection = 0;
    unsigned int size = std::min(x.size(), this->size());
    for (auto i = 0u; i < size; i++)
      projection += x[i] * this->operator[](i);
    return projection;
  }

  // easy debug
  inline friend std::ostream &
  operator<<(std::ostream &stream,
             const Finitely_critical_multi_filtration<T> &truc) {
    if (truc.is_inf()) {
      stream << "[inf, ..., inf]";
      return stream;
    }
    if (truc.is_minus_inf()) {
      stream << "[-inf, ..., -inf]";
      return stream;
    }
    if (truc.is_nan()) {
      stream << "[NaN]";
      return stream;
    }
    if (truc.empty()) {
      stream << "[]";
      return stream;
    }
    stream << "[";
    for (unsigned int i = 0; i < truc.size() - 1; i++) {
      stream << truc[i] << ", ";
    }
    if (!truc.empty())
      stream << truc.back();
    stream << "]";
    return stream;
  }
  inline T norm() const {
    T out = 0;
    for (auto &stuff : *this)
      out += std::pow(stuff, 2);
    return sqrt(out);
  }
  inline T distance(const Finitely_critical_multi_filtration &other) const {
    T out = 0;
    for (auto i = 0u; i < other.size(); i++) {
      out += std::pow((*this)[i] - other[i], 2);
    }
    return sqrt(out);
  }
  template <typename oned_array>
  inline Finitely_critical_multi_filtration<std::size_t>
  coordinates_in_grid(std::vector<oned_array> grid) {
    Finitely_critical_multi_filtration<std::size_t> coords(this->size());
    assert(grid.size() >= this->size());
    for (auto parameter = 0u; parameter < grid.size(); ++parameter) {
      const auto &filtration = grid[parameter];
      coords[parameter] =
          std::distance(filtration.begin(),
                        std::lower_bound(filtration.begin(), filtration.end(),
                                         this->operator[](parameter)));
    }
    return coords;
  }
  template <typename oned_array>
  inline void coordinates_in_grid_inplace(std::vector<oned_array> grid) {
    Finitely_critical_multi_filtration<std::size_t> coords(this->size());
    assert(grid.size() >= this->size());
    for (auto parameter = 0u; parameter < grid.size(); ++parameter) {
      const auto &filtration = grid[parameter];
      this->operator[](parameter) =
          std::distance(filtration.begin(),
                        std::lower_bound(filtration.begin(), filtration.end(),
                                         this->operator[](parameter)));
    }
  }

public:
  constexpr static T T_inf = std::numeric_limits<T>::has_infinity
                                 ? std::numeric_limits<T>::infinity()
                                 : std::numeric_limits<T>::max();
};

} // namespace Gudhi::multiparameter::multi_filtrations

namespace std {

template <typename T>
class numeric_limits<Gudhi::multiparameter::multi_filtrations::
                         Finitely_critical_multi_filtration<T>> {
public:
  static constexpr bool has_infinity = numeric_limits<T>::has_infinity;

  static Gudhi::multiparameter::multi_filtrations::
      Finitely_critical_multi_filtration<T>
      infinity() throw() {
    return Gudhi::multiparameter::multi_filtrations::
        Finitely_critical_multi_filtration<T>(
            1, std::numeric_limits<T>::infinity());
  };
  static Gudhi::multiparameter::multi_filtrations::
      Finitely_critical_multi_filtration<T>
      minus_infinity() throw() {
    return Gudhi::multiparameter::multi_filtrations::
        Finitely_critical_multi_filtration<T>(
            1, -std::numeric_limits<T>::infinity());
  };
  static Gudhi::multiparameter::multi_filtrations::
      Finitely_critical_multi_filtration<T>
      max() throw() {
    return Gudhi::multiparameter::multi_filtrations::
        Finitely_critical_multi_filtration<T>(1, std::numeric_limits<T>::max());
  };
  static Gudhi::multiparameter::multi_filtrations::
      Finitely_critical_multi_filtration<T>
      quiet_NaN() throw() {
    return Gudhi::multiparameter::multi_filtrations::
        Finitely_critical_multi_filtration<T>(1,
                                              numeric_limits<T>::quiet_NaN());
  };
};

} // namespace std

#endif // FINITELY_CRITICAL_FILTRATIONS_H_
