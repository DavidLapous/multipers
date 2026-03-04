/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       multipers contributors
 *
 *    Copyright (C) 2026 Inria
 */

/**
 * @file phat_bit_tree_column.h
 * @brief Contains the @ref Gudhi::persistence_matrix::Phat_bit_tree_column class.
 */

#ifndef PM_PHAT_BIT_TREE_COLUMN_H
#define PM_PHAT_BIT_TREE_COLUMN_H

#include <functional>

#include <gudhi/Persistence_matrix/columns/column_utilities.h>
#include <gudhi/Persistence_matrix/columns/phat_column_interface.h>

#include <phat/representations/bit_tree_pivot_column.h>

namespace Gudhi {
namespace persistence_matrix {

namespace phat_column_detail {

struct Phat_bit_tree_pivot_policy {
  using Phat_column = phat::bit_tree_column;
  using Phat_index = phat::index;
  using Row_indices = std::vector<Phat_index>;

  static constexpr bool tracks_capacity = true;
  static constexpr bool supports_incremental_xor = true;

  static void assert_rows_valid(const std::vector<Phat_index>& rows)
  {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (rows[i] < 0) {
        throw std::invalid_argument("PHAT bit-tree columns require non-negative row indices.");
      }
      if (i > 0 && rows[i - 1] >= rows[i]) {
        throw std::invalid_argument("PHAT bit-tree columns require strictly increasing row indices.");
      }
    }
  }

  static Phat_index required_capacity(const Row_indices& rows)
  {
    if (rows.empty()) return static_cast<Phat_index>(0);
    if (rows.back() == std::numeric_limits<Phat_index>::max()) {
      throw std::overflow_error("PHAT bit-tree column cannot index beyond numeric limits.");
    }
    return static_cast<Phat_index>(rows.back() + 1);
  }

  static void set_column_from_rows(Phat_column& target,
                                   const std::vector<Phat_index>& rows,
                                   Phat_index capacity_hint)
  {
    assert_rows_valid(rows);

    const Phat_index required = required_capacity(rows);
    if (capacity_hint < required) capacity_hint = required;

    Phat_column rebuilt;
    rebuilt.init(capacity_hint);
    if (!rows.empty()) {
      rebuilt.add_col(rows);
    }
    target = std::move(rebuilt);
  }

  static bool can_incremental_xor(const Phat_column& target, const Row_indices& other, Phat_index capacity)
  {
    static_cast<void>(target);
    if (other.empty()) return true;
    if (capacity <= 0) return false;
    return other.back() < capacity;
  }

  static void xor_rows_in_place(Phat_column& target, const Row_indices& rows)
  {
    for (const Phat_index idx : rows) {
      target.add_index(static_cast<std::size_t>(idx));
    }
  }

  static void sync_rows_from_column(const Phat_column& source, Row_indices& rows)
  {
    rows.clear();
    auto& mutable_source = const_cast<Phat_column&>(source);
    mutable_source.get_col(rows);
    assert_rows_valid(rows);
  }

  static Phat_index get_max_index(const Phat_column& col)
  {
    return col.get_max_index();
  }
};

}  // namespace phat_column_detail

/**
 * @class Phat_bit_tree_column phat_bit_tree_column.h gudhi/Persistence_matrix/columns/phat_bit_tree_column.h
 * @ingroup persistence_matrix
 *
 * @brief PHAT-style bit-tree pivot interface.
 *
 * The storage and arithmetic are based on a PHAT column container.
 * Pivot lookup in boundary mode is delegated to PHAT's @ref phat::bit_tree_column.
 */
template <class Master_matrix>
class Phat_bit_tree_column
    : public Phat_column_interface<Master_matrix, phat_column_detail::Phat_bit_tree_pivot_policy>
{
 private:
  using Base = Phat_column_interface<Master_matrix, phat_column_detail::Phat_bit_tree_pivot_policy>;

 public:
  using ID_index = typename Base::ID_index;
  using Field_element = typename Base::Field_element;

  using Base::Base;
  using Base::operator=;
};

}  // namespace persistence_matrix
}  // namespace Gudhi

/**
 * @ingroup persistence_matrix
 *
 * @brief Hash method for @ref Gudhi::persistence_matrix::Phat_bit_tree_column.
 */
template <class Master_matrix>
struct std::hash<Gudhi::persistence_matrix::Phat_bit_tree_column<Master_matrix> > {
  std::size_t operator()(const Gudhi::persistence_matrix::Phat_bit_tree_column<Master_matrix>& column) const
  {
    return Gudhi::persistence_matrix::hash_column(column);
  }
};

#endif  // PM_PHAT_BIT_TREE_COLUMN_H
