/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022-24 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file ru_rep_cycles.h
 * @author Hannah Schreiber
 * @brief Contains the @ref Gudhi::persistence_matrix::RU_representative_cycles class and
 * @ref Gudhi::persistence_matrix::Dummy_ru_representative_cycles structure.
 */

#ifndef PM_RU_REP_CYCLES_H
#define PM_RU_REP_CYCLES_H

#include <cassert>
#include <utility>    //std::move
#include <algorithm>  //std::sort
#include <vector>
// #include <set>
#if BOOST_VERSION >= 108100
#include <boost/unordered/unordered_flat_set.hpp>
#else
#include <unordered_set>
#endif

#include <gudhi/persistence_matrix_options.h>

namespace Gudhi {
namespace persistence_matrix {

/**
 * @ingroup persistence_matrix
 *
 * @brief Empty structure.
 * Inherited instead of @ref RU_representative_cycles, when the computation of the representative cycles
 * were not enabled.
 */
struct Dummy_ru_representative_cycles
{
  friend void swap([[maybe_unused]] Dummy_ru_representative_cycles& d1,
                   [[maybe_unused]] Dummy_ru_representative_cycles& d2) {}
};

// TODO: add coefficients ? Only Z2 token into account for now.
/**
 * @class RU_representative_cycles ru_rep_cycles.h gudhi/Persistence_matrix/ru_rep_cycles.h
 * @ingroup persistence_matrix
 *
 * @brief Class managing the representative cycles for @ref RU_matrix if the option was enabled.
 * 
 * @tparam Master_matrix An instantiation of @ref Matrix from which all types and options are deduced.
 */
template <class Master_matrix>
class RU_representative_cycles 
{
 public:
  using Index = typename Master_matrix::Index;        /**< @ref MatIdx index type. */
  using ID_index = typename Master_matrix::ID_index;  /**< @ref IDIdx index type. */
  using Bar = typename Master_matrix::Bar;            /**< Bar type. */
  using Cycle = typename Master_matrix::Cycle;        /**< Cycle type. */
  using Cycle_border = std::vector<ID_index>;
  using Cycle_borders = std::vector<Cycle_border>;
  struct hashCycle {
    size_t operator()(const Cycle_border& cycle) const {
      std::hash<ID_index> hasher;
      size_t answer = 0;
      for (ID_index i : cycle) {
        answer ^= hasher(i) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
      }
      return answer;
    }
  };
#if BOOST_VERSION >= 108100
  using Cycle_borders_tmp = boost::unordered_flat_set<Cycle_border,hashCycle>;
#else
  using Cycle_borders_tmp = std::unordered_set<Cycle_border,hashCycle>;
#endif
#if BOOST_VERSION >= 108100
  using Cycle_unreduced_borders_tmp = boost::unordered_flat_set<Index>;
#else
  using Cycle_unreduced_borders_tmp = std::unordered_set<Index>;
#endif

  /**
   * @brief Default constructor.
   */
  RU_representative_cycles();
  /**
   * @brief Copy constructor.
   * 
   * @param matrixToCopy Matrix to copy.
   */
  RU_representative_cycles(const RU_representative_cycles& matrixToCopy);
  /**
   * @brief Move constructor.
   * 
   * @param other Matrix to move.
   */
  RU_representative_cycles(RU_representative_cycles&& other) noexcept;

  /**
   * @brief Computes the current representative cycles of the matrix.
   */
  void update_representative_cycles();

  // /**
  //  * @brief Returns the current representative cycles. If the matrix is modified later after the first call,
  //  * @ref update_representative_cycles has to be called to update the returned cycles.
  //  * 
  //  * @return A const reference to a vector of @ref Matrix::Cycle containing all representative cycles.
  //  */
  // const std::vector<Cycle>& get_representative_cycles();
  // /**
  //  * @brief Returns the representative cycle corresponding to the given bar.
  //  * If the matrix is modified later after the first call,
  //  * @ref update_representative_cycles has to be called to update the returned cycles.
  //  * 
  //  * @param bar Bar corresponding to the wanted representative cycle.
  //  * @return A const reference to the representative cycle.
  //  */
  // const Cycle& get_representative_cycle(const Bar& bar);

  /**
   * @brief Returns the current representative cycles. If the matrix is modified later after the first call,
   * @ref update_representative_cycles has to be called to update the returned cycles.
   * 
   * @return A const reference to a vector of @ref Matrix::Cycle containing all representative cycles.
   */
  std::vector<Cycle_borders> get_representative_cycles_as_borders(bool detailed = false);

  /**
   * @brief Assign operator.
   */
  RU_representative_cycles& operator=(RU_representative_cycles other);
  /**
   * @brief Swap operator.
   */
  friend void swap(RU_representative_cycles& base1, RU_representative_cycles& base2) {
    base1.representativeCycles_.swap(base2.representativeCycles_);
    base1.u_transposed_.swap(base2.u_transposed_);
    // base1.birthToCycle_.swap(base2.birthToCycle_);
  }

 private:
  using Master_RU_matrix = typename Master_matrix::Master_RU_matrix;

  std::vector<Index> representativeCycles_; /**< Cycle container. */
  // std::vector<Index> birthToCycle_;         /**< Map from birth index to cycle index. */
  std::vector<Cycle> u_transposed_;

  void _get_initial_borders(Index idx, Cycle_borders_tmp& borders);
  Cycle_border _get_border(Index uIndex);
  Cycle_border _get_dependent_border(Index uIndex);

  constexpr Master_RU_matrix* _matrix() { return static_cast<Master_RU_matrix*>(this); }
  constexpr const Master_RU_matrix* _matrix() const { return static_cast<const Master_RU_matrix*>(this); }
};

template <class Master_matrix>
inline RU_representative_cycles<Master_matrix>::RU_representative_cycles()
{}

template <class Master_matrix>
inline RU_representative_cycles<Master_matrix>::RU_representative_cycles(
    const RU_representative_cycles<Master_matrix>& matrixToCopy)
    : representativeCycles_(matrixToCopy.representativeCycles_),
      // birthToCycle_(matrixToCopy.birthToCycle_),
      u_transposed_(matrixToCopy.u_transposed_)
{}

template <class Master_matrix>
inline RU_representative_cycles<Master_matrix>::RU_representative_cycles(
    RU_representative_cycles<Master_matrix>&& other) noexcept
    : representativeCycles_(std::move(other.representativeCycles_)),
      // birthToCycle_(std::move(other.birthToCycle_)),
      u_transposed_(std::move(other.u_transposed_))
{}

//TODO: u_transposed_ as a vector of Index and another vector giving position + length of each cycle?
template <class Master_matrix>
inline void RU_representative_cycles<Master_matrix>::update_representative_cycles() 
{
  //WARNING: this was only thought for the multipers interface, it is not definitive and does not
  // cover all cases.
  static_assert(Master_matrix::Option_list::has_column_pairings && Master_matrix::Option_list::is_z2,
                "Needs an ID to Pos map.");

  auto rsize = _matrix()->reducedMatrixR_.get_number_of_columns();
  representativeCycles_.clear();
  representativeCycles_.reserve(rsize);

  auto usize = _matrix()->mirrorMatrixU_.get_number_of_columns();
  u_transposed_.clear();
  u_transposed_.resize(usize);

  for (Index i = 0; i < usize; i++) {
    if (_matrix()->reducedMatrixR_.is_zero_column(i)) {
      representativeCycles_.push_back(i);
    }
    if constexpr (Master_matrix::Option_list::column_type == Column_types::HEAP ||
                  Master_matrix::Option_list::column_type == Column_types::VECTOR) {
      // TODO: have a better way to do this. For now, one cannot use the column iterators for that.
      unsigned int j = 0;
      for (const auto& entry : _matrix()->mirrorMatrixU_.get_column(i).get_content()) {
        if (entry != 0){
          u_transposed_[j].push_back(i);
        }
        ++j;
      }
    } else {
      for (const auto& entry : _matrix()->mirrorMatrixU_.get_column(i)) {
        u_transposed_[entry.get_row_index()].push_back(i);
      }
    }
  }
}

// template <class Master_matrix>
// inline const std::vector<typename RU_representative_cycles<Master_matrix>::Cycle>&
// RU_representative_cycles<Master_matrix>::get_representative_cycles() 
// {
//   if (representativeCycles_.empty()) update_representative_cycles();
//   return representativeCycles_;
// }

// template <class Master_matrix>
// inline const typename RU_representative_cycles<Master_matrix>::Cycle&
// RU_representative_cycles<Master_matrix>::get_representative_cycle(const Bar& bar) 
// {
//   if (representativeCycles_.empty()) update_representative_cycles();
//   return representativeCycles_[birthToCycle_[bar.birth]];
// }

//TODO: not store cycle borders in vectors of vectors of vectors
template <class Master_matrix>
inline std::vector<typename RU_representative_cycles<Master_matrix>::Cycle_borders>
RU_representative_cycles<Master_matrix>::get_representative_cycles_as_borders(bool detailed)
{
  static_assert(Master_matrix::Option_list::is_z2, "Only available for Z2 coefficients for now.");

  if (representativeCycles_.empty()) update_representative_cycles();

  std::vector<Cycle_borders> cycles(representativeCycles_.size());
  unsigned int i = 0;
  for (const auto& cycleIndex : representativeCycles_){
    const auto& cycle = u_transposed_[cycleIndex];
    assert(_matrix()->reducedMatrixR_.is_zero_column(cycle.back()));
    if (cycle.size() != 1){ //cycle.size == 1 -> border is empty
      if (detailed) {
        Cycle_borders_tmp cbt;
        for (unsigned int j = 0; j < cycle.size() - 1; ++j){
          assert(!_matrix()->reducedMatrixR_.is_zero_column(cycle[j]));
          _get_initial_borders(cycle[j], cbt);
        }
        cycles[i] = Cycle_borders(cbt.begin(), cbt.end());
        cycles[i].push_back(_get_dependent_border(cycle.back()));
      } else {
        cycles[i].resize(cycle.size());
        for (unsigned int j = 0; j < cycle.size() - 1; ++j){
          cycles[i][j] = _get_border(cycle[j]);
        }
        cycles[i].back() = _get_dependent_border(cycle.back());
      }
    } else {
      cycles[i].push_back({});
    }
    
    ++i;
  }
  return cycles;
}

template <class Master_matrix>
inline RU_representative_cycles<Master_matrix>& RU_representative_cycles<Master_matrix>::operator=(
    RU_representative_cycles<Master_matrix> other) 
{
  representativeCycles_.swap(other.representativeCycles_);
  u_transposed_.swap(other.u_transposed_);
  // birthToCycle_.swap(other.birthToCycle_);
  return *this;
}

//TODO: try avoid storing the vectors in the unordered set, when it is not reduced as it adds run time for hash
//computation and possibly comparaison of vectors. Treat them separately. See if it goes faster, or if it was marginal
template <class Master_matrix>
inline void RU_representative_cycles<Master_matrix>::_get_initial_borders(Index idx,
                                                                          Cycle_borders_tmp& borders) {
  const Cycle& cycle = u_transposed_[idx];
  assert(cycle.back() == idx);

  auto add_to = [](const std::vector<Index>& b, Cycle_borders_tmp& borders){
    auto p = borders.insert(b);
    if (!p.second) borders.erase(p.first);
  };

  for (unsigned int j = 0; j < cycle.size() - 1; ++j){
    _get_initial_borders(cycle[j], borders);
  }

  add_to(_get_dependent_border(idx), borders);
}

template <class Master_matrix>
inline typename RU_representative_cycles<Master_matrix>::Cycle_border
RU_representative_cycles<Master_matrix>::_get_border(Index uIndex) {
  const auto& col = _matrix()->reducedMatrixR_.get_column(uIndex);
  Cycle_border res;

  unsigned int j = 0;
  if constexpr (Master_matrix::Option_list::column_type == Column_types::HEAP ||
                Master_matrix::Option_list::column_type == Column_types::VECTOR) {
    res.reserve(col.size());
    // TODO: have a better way to do this. For now, one cannot use the column iterators for that.
    for (auto i : col.get_content()) {
      if (i != 0) res.push_back(j);
      ++j;
    }
  } else {
    res.resize(col.size());
    for (const auto& entry : col) {
      res[j] = entry.get_row_index();
      ++j;
    }
  }

  return res;
}

template <class Master_matrix>
inline typename RU_representative_cycles<Master_matrix>::Cycle_border
RU_representative_cycles<Master_matrix>::_get_dependent_border(Index uIndex) {
  if (u_transposed_[uIndex].size() == 1){
    return _get_border(uIndex);
  }

  auto add = [](const Master_matrix::Column& col, Cycle_unreduced_borders_tmp& b){
    if constexpr (Master_matrix::Option_list::column_type == Column_types::HEAP ||
                  Master_matrix::Option_list::column_type == Column_types::VECTOR) {
      // TODO: have a better way to do this. For now, one cannot use the column iterators for that.
      unsigned int j = 0;
      for (auto k : col.get_content()) {
        if (k != 0) {
          auto p = b.insert(j);
          if (!p.second) b.erase(p.first);
        }
        ++j;
      }
    } else {
      for (const auto& entry : col) {
        auto p = b.insert(entry.get_row_index());
          if (!p.second) b.erase(p.first);
      }
    }
  };

  Cycle_unreduced_borders_tmp b;
  for (Index i : u_transposed_[uIndex]){
    add(_matrix()->reducedMatrixR_.get_column(i), b);
  }

  Cycle_border res(b.begin(), b.end());
  std::sort(res.begin(), res.end());

  return res;
}

}  // namespace persistence_matrix
}  // namespace Gudhi

#endif  // PM_RU_REP_CYCLES_H
