/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2026/02 Hannah Schreiber: reorganization + small optimizations + documentation
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file Module.h
 * @author David Loiseaux
 * @brief Contains the @ref Gudhi::multi_persistence::Module class.
 */

#ifndef MP_MODULE_H_INCLUDED
#define MP_MODULE_H_INCLUDED

#include <algorithm>  // std::max
#include <cstddef>
#include <ostream>  //std::ostream
#include <utility>
#include <vector>

#ifdef GUDHI_USE_TBB
#include <oneapi/tbb/parallel_for.h>
#endif

#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Summand.h>

namespace Gudhi {
namespace multi_persistence {

/**
 * @class Module Module.h gudhi/Multi_persistence/Module.h
 * @ingroup multi_persistence
 *
 * @brief
 *
 * @tparam T
 */
template <typename T>
class Module {
 public:
  using value_type = T;
  using Dimension = int;
  using Summand_t = Summand<value_type, Dimension>;

 private:
  using Module_t = std::vector<Summand_t>;

 public:
  using iterator = typename Module_t::iterator;             /**< Iterator type. */
  using const_iterator = typename Module_t::const_iterator; /**< Const iterator type. */
  using Index = typename Module_t::size_type;

  static constexpr T T_inf = Summand_t::T_inf;
  static constexpr T T_m_inf = Summand_t::T_m_inf;

  Module() : module_(), box_(), maxDim_(Summand_t::template get_null_value<Dimension>()) {}

  Module(const Box<value_type> &box) : module_(), box_(box), maxDim_(Summand_t::template get_null_value<Dimension>()) {}

  iterator begin() { return module_.begin(); }

  iterator end() { return module_.end(); }

  const_iterator begin() const { return module_.cbegin(); }

  const_iterator end() const { return module_.cend(); }

  const Box<value_type> &get_box() const { return box_; }

  void set_box(const Box<value_type> &box) { box_ = box; }

  // Do not change (max) dimension
  Summand_t &get_summand(Index index) { return module_[index]; }

  const Summand_t &get_summand(Index index) const { return module_[index]; }

  void add_summand(Index i,
                   const Summand_t &summand,
                   Dimension dimension = Summand_t::template get_null_value<Dimension>()) {
    if (module_.size() <= i) resize(i + 1, summand.get_number_of_parameters());
    module_[i] = summand;
    if (dimension != Summand_t::template get_null_value<Dimension>()) module_[i].set_dimension(dimension);
    maxDim_ = std::max(maxDim_, module_[i].get_dimension());
  }

  void add_summand(const Summand_t &summand, Dimension dimension = Summand_t::template get_null_value<Dimension>()) {
    add_summand(module_.size(), summand, dimension);
  }

  std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>>
  get_corners_of_dimension(Dimension dimension) const {
    // TODO: parallelize
    std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>> corners;
    for (const auto &summand : module_) {
      if (summand.get_dimension() == dimension) {
        const auto &birthList = summand.get_upset();
        const auto &deathList = summand.get_downset();
        std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>> corner;
        corner.first.resize(birthList.size());
        corner.second.resize(deathList.size());
        Index i = 0;
        for (const auto &b : birthList) {
          corner.first[i] = std::vector<value_type>(b.begin(), b.end());
          ++i;
        }
        i = 0;
        for (const auto &d : deathList) {
          corner.second[i] = std::vector<value_type>(d.begin(), d.end());
          ++i;
        }
        corners.push_back(std::move(corner));
      }
    }
    return corners;
  }

  [[nodiscard]] Dimension get_max_dimension() const { return maxDim_; }

  void set_max_dimension(Dimension maxDim) { maxDim_ = maxDim; }

  [[nodiscard]] Index size() const { return module_.size(); }

  void resize(Index size, int numberOfParameters) { module_.resize(size, Summand_t(numberOfParameters)); }

  Box<value_type> compute_bounds() const {
    Dimension numParam = box_.get_lower_corner().size();
    typename Box<value_type>::Point_t lower_bound(numParam, T_inf);
    typename Box<value_type>::Point_t upper_bound(numParam, T_m_inf);
    for (const auto &summand : module_) {
      auto summandBounds = summand.compute_bounds();
      const auto &[m, M] = summandBounds.get_bounding_corners();
      for (auto parameter = 0; parameter < numParam; parameter++) {
        lower_bound[parameter] = std::min(m[parameter], lower_bound[parameter]);
        upper_bound[parameter] = std::min(M[parameter], upper_bound[parameter]);
      }
    }
    return Box(lower_bound, upper_bound);
  }

  std::vector<std::vector<std::pair<value_type, value_type>>> get_barcode_from_line(
      const Line<value_type> &l,
      Dimension dimension = Summand_t::template get_null_value<Dimension>()) const {
    std::vector<std::vector<std::pair<value_type, value_type>>> barcode(get_max_dimension() + 1);
    for (Dimension i = 0; i < get_max_dimension(); ++i) {
      barcode[i].reserve(size());
    }
    for (const auto &summand : module_) {
      if (dimension == Summand_t::template get_null_value<Dimension>() || summand.get_dimension() == dimension) {
        barcode[summand.get_dimension()].push_back(summand.get_bar(l));
      }
    }
    return barcode;
  }

  std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> get_barcodes_from_set_of_lines(
      const std::vector<Line<value_type>> &lines,
      Dimension dimension = Summand_t::template get_null_value<Dimension>()) const {
    std::size_t nlines = lines.size();
    std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> barcodes(
        get_max_dimension() + 1, std::vector<std::vector<std::pair<value_type, value_type>>>(nlines));

    auto get_barcode_of_line = [&](std::size_t i) {
      const Line<value_type> &l = lines[i];
      for (const auto &summand : module_) {
        if (dimension == Summand_t::template get_null_value<Dimension>() || summand.get_dimension() == dimension) {
          barcodes[summand.get_dimension()][i].push_back(summand.get_bar(l));
        }
      };
    };

#ifdef GUDHI_USE_TBB
    tbb::parallel_for(std::size_t(0), nlines, get_barcode_of_line);
#else
    for (std::size_t i = 0; i < nlines; ++i) {
      get_barcode_of_line(i);
    }
#endif

    return barcodes;
  }

  void add_barcode(const Line<value_type> &line,
                   const std::vector<std::vector<std::array<value_type, 2>>> &barcode,
                   bool thresholdToBox) {
#ifdef GUDHI_USE_TBB
    std::vector<std::size_t> shifts(barcode.size(), 0U);
    for (std::size_t i = 1U; i < barcode.size(); i++) {
      shifts[i] = shifts[i - 1] + barcode[i - 1].size();
    }
    tbb::parallel_for(size_t(0), barcode.size(), [&](size_t dim) {
      tbb::parallel_for(size_t(0), barcode[dim].size(), [&](size_t j) {
        _add_bar_with_threshold(line, barcode[dim][j], thresholdToBox, get_summand(shifts[dim] + j));
      });
    });
#else
    Index count = 0;
    for (const auto &barDim : barcode) {
      for (const auto &bar : barDim) _add_bar_with_threshold(line, bar, thresholdToBox, get_summand(count++));
    }
#endif
  }

  void clean() {
    module_.erase(
        std::remove_if(module_.begin(), module_.end(), [](const Summand_t &s) { return s.get_upset().is_plus_inf(); }),
        module_.end());
    maxDim_ = Summand_t::template get_null_value<Dimension>();
    for (const auto& sum : module_) maxDim_ = std::max(maxDim_, sum.get_dimension());
  }

  void fill(value_type precision) {
    // TODO: parallelize
    for (Summand_t &sum : module_) {
      sum.complete_birth(precision);
      sum.complete_death(precision);
    }
  }

  void rescale(const std::vector<value_type> &rescaleFactors,
               Dimension dimension = Summand_t::template get_null_value<Dimension>()) {
    for (auto &summand : module_) {
      if (dimension == Summand_t::template get_null_value<Dimension>() || summand.get_dimension() == dimension)
        summand.rescale(rescaleFactors);
    }
  }

  void translate(const std::vector<value_type> &translation,
                 Dimension dimension = Summand_t::template get_null_value<Dimension>()) {
    for (auto &summand : module_) {
      if (dimension == Summand_t::template get_null_value<Dimension>() || summand.get_dimension() == dimension)
        summand.translate(translation);
    }
  }

  void evaluate_in_grid(const std::vector<std::vector<value_type>> &grid) {
#ifdef GUDHI_USE_TBB
    tbb::parallel_for(std::size_t(0), module_.size(), [&](std::size_t i) { module_[i].evaluate_in_grid(grid); });
#else
    for (auto &summand : module_) {
      summand.evaluate_in_grid(grid);
    }
#endif
  }

  friend bool operator==(const Module &a, const Module &b) {
    if (a.get_max_dimension() != b.get_max_dimension()) return false;
    if (a.box_ != b.box_) return false;
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a.get_summand(i) != b.get_summand(i)) return false;
    }
    return true;
  }

  /**
   * @brief Outstream operator.
   */
  friend std::ostream &operator<<(std::ostream &stream, const Module &m) {
    stream << "Module of max dim " << m.maxDim_ << ":\n";
    for (const auto &s : m) {
      stream << s << "\n";
    }
    return stream;
  }

  friend void swap(Module &mod1, Module &mod2) noexcept {
    mod1.module_.swap(mod2.module_);
    swap(mod1.box_, mod2.box_);
    std::swap(mod1.maxDim_, mod2.maxDim_);
  }

 private:
  Module_t module_;
  Box<value_type> box_;
  Dimension maxDim_;

  void _add_bar_with_threshold(const Line<value_type> &line,
                               const std::array<value_type, 2> &bar,
                               bool thresholdToBox,
                               Summand_t &summand) {
    if (bar[0] >= bar[1]) return;

    // TODO: parallelize
    const value_type error = 1e-10;

    auto birthContainer = line[bar[0]];
    bool allInf = true;
    for (std::size_t i = 0; i < birthContainer.size(); i++) {
      value_type t = box_.get_lower_corner()[i];
      if (birthContainer[i] < t - error) birthContainer[i] = thresholdToBox ? t : T_m_inf;
      if (birthContainer[i] != T_m_inf) allInf = false;
    }
    if (allInf) birthContainer.resize(1);

    auto deathContainer = line[bar[1]];
    allInf = true;
    for (std::size_t i = 0; i < deathContainer.size(); i++) {
      value_type t = box_.get_upper_corner()[i];
      if (deathContainer[i] > t + error) deathContainer[i] = thresholdToBox ? t : T_inf;
      if (deathContainer[i] != T_inf) allInf = false;
    }
    if (allInf) deathContainer.resize(1);

    // could be automaticaly converted, but this should avoid one copy?
    typename Summand_t::Births::Generator births(std::move(birthContainer.retrieve_underlying_container()));
    typename Summand_t::Deaths::Generator deaths(std::move(deathContainer.retrieve_underlying_container()));
    summand.add_bar(births, deaths);
    // summand.add_bar(birthContainer, deathContainer);
  }
};

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_MODULE_H_INCLUDED
