/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       multipers contributors
 *
 *    Copyright (C) 2026 Inria
 */

/**
 * @file phat_vector_column.h
 * @brief Contains the @ref Gudhi::persistence_matrix::Phat_vector_column class.
 */

#ifndef PM_PHAT_VECTOR_COLUMN_H
#define PM_PHAT_VECTOR_COLUMN_H

#include <functional>

#include <gudhi/Persistence_matrix/columns/column_utilities.h>
#include <gudhi/Persistence_matrix/columns/phat_column_interface.h>

#include <phat/helpers/misc.h>

namespace Gudhi {
namespace persistence_matrix {

namespace phat_column_detail {

struct Phat_vector_pivot_policy {
  using Phat_column = phat::column;
  using Phat_index = phat::index;
  using Row_indices = std::vector<Phat_index>;

  static constexpr bool tracks_capacity = false;
  static constexpr bool supports_incremental_xor = false;

  static void assert_rows_valid(const std::vector<Phat_index>&) {}

  static bool can_incremental_xor(const Phat_column&, const Row_indices&, Phat_index) { return false; }

  static void xor_rows_in_place(Phat_column&, const Row_indices&) {}

  static void sync_rows_from_column(const Phat_column& source, Row_indices& rows)
  {
    rows = source;
  }

  static Phat_index required_capacity(const Row_indices& rows)
  {
    if (rows.empty()) return static_cast<Phat_index>(0);
    return static_cast<Phat_index>(rows.back() + 1);
  }

  static void set_column_from_rows(Phat_column& target,
                                   const std::vector<Phat_index>& rows,
                                   [[maybe_unused]] Phat_index capacity_hint)
  {
    target = rows;
  }

  static Phat_index get_max_index(const Phat_column& col)
  {
    return col.empty() ? static_cast<Phat_index>(-1) : col.back();
  }
};

}  // namespace phat_column_detail

/**
 * @class Phat_vector_column phat_vector_column.h gudhi/Persistence_matrix/columns/phat_vector_column.h
 * @ingroup persistence_matrix
 *
 * @brief PHAT-style vector-backed column interface.
 *
 * The storage and arithmetic are based on a PHAT column container.
 */
template <class Master_matrix>
class Phat_vector_column
    : public Phat_column_interface<Master_matrix, phat_column_detail::Phat_vector_pivot_policy>
{
 private:
  using Base = Phat_column_interface<Master_matrix, phat_column_detail::Phat_vector_pivot_policy>;

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
 * @brief Hash method for @ref Gudhi::persistence_matrix::Phat_vector_column.
 */
template <class Master_matrix>
struct std::hash<Gudhi::persistence_matrix::Phat_vector_column<Master_matrix> > {
  std::size_t operator()(const Gudhi::persistence_matrix::Phat_vector_column<Master_matrix>& column) const
  {
    return Gudhi::persistence_matrix::hash_column(column);
  }
};

#endif  // PM_PHAT_VECTOR_COLUMN_H
