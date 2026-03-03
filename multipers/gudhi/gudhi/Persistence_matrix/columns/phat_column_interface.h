/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       multipers contributors
 *
 *    Copyright (C) 2026 Inria
 */

/**
 * @file phat_column_interface.h
 * @brief Common PHAT-backed column implementation for matrix columns.
 */

#ifndef PM_PHAT_COLUMN_INTERFACE_H
#define PM_PHAT_COLUMN_INTERFACE_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/range/iterator_range_core.hpp>

namespace Gudhi {
namespace persistence_matrix {

/**
 * @brief Shared PHAT-backed column implementation.
 *
 * Storage uses a PHAT column container directly (sorted row indices in Z2).
 *
 * @tparam Master_matrix Matrix type.
 * @tparam Pivot_policy Type exposing:
 *   - `using Phat_column`
 *   - `using Phat_index`
 *   - `static Phat_index get_max_index(const Phat_column&)`
 */
template <class Master_matrix, class Pivot_policy>
class Phat_column_interface : public Master_matrix::Row_access_option,
                              public Master_matrix::Column_dimension_option,
                              public Master_matrix::Chain_column_option
{
 public:
  using Master = Master_matrix;
  using Index = typename Master_matrix::Index;
  using ID_index = typename Master_matrix::ID_index;
  using Dimension = typename Master_matrix::Dimension;
  using Field_element = typename Master_matrix::Element;
  using Entry = typename Master_matrix::Matrix_entry;
  using Column_settings = typename Master_matrix::Column_settings;

  using Column_support = typename Pivot_policy::Phat_column;
  using Phat_index = typename Pivot_policy::Phat_index;
  using Row_indices = std::vector<Phat_index>;

  using Entry_view = std::vector<Entry>;
  using iterator = typename Entry_view::const_iterator;
  using const_iterator = typename Entry_view::const_iterator;
  using reverse_iterator = typename Entry_view::const_reverse_iterator;
  using const_reverse_iterator = typename Entry_view::const_reverse_iterator;
  using Content_range = boost::iterator_range<const_iterator>;

 private:
  using RA_opt = typename Master_matrix::Row_access_option;
  using Dim_opt = typename Master_matrix::Column_dimension_option;
  using Chain_opt = typename Master_matrix::Chain_column_option;
  using Field_operators = typename Master_matrix::Field_operators;

  static constexpr ID_index _null_index() { return Master_matrix::template get_null_value<ID_index>(); }

  static void _check_supported_options()
  {
    static_assert(Master_matrix::Option_list::is_z2,
                  "PHAT columns currently support only Z2 coefficients.");
    static_assert(!Master_matrix::Option_list::has_row_access,
                  "PHAT columns currently do not support row access.");
    static_assert(std::is_integral_v<ID_index> && std::is_integral_v<Phat_index>,
                  "PHAT column indices must be integral types.");
  }

  static Phat_index _to_phat_index(ID_index idx)
  {
    if constexpr (std::is_signed_v<ID_index>) {
      if (idx < 0) {
        throw std::invalid_argument("PHAT columns require non-negative row indices.");
      }
    }

    if constexpr (std::numeric_limits<ID_index>::max() > std::numeric_limits<Phat_index>::max()) {
      if (idx > static_cast<ID_index>(std::numeric_limits<Phat_index>::max())) {
        throw std::overflow_error("Row index does not fit into PHAT index type.");
      }
    }

    return static_cast<Phat_index>(idx);
  }

  static ID_index _to_id_index(Phat_index idx) { return static_cast<ID_index>(idx); }

  static ID_index _pivot_from_rows(const Row_indices& rows)
  {
    if (rows.empty()) return _null_index();
    const Phat_index max_idx = rows.back();
    if (max_idx < 0) return _null_index();
    return _to_id_index(max_idx);
  }

  ID_index _pivot_from_storage() const
  {
    const Phat_index max_idx = Pivot_policy::get_max_index(column_);
    if (max_idx < 0) return _null_index();
    return _to_id_index(max_idx);
  }

  static void _normalize(Row_indices& column)
  {
    if (column.empty()) return;

    std::sort(column.begin(), column.end());

    Row_indices reduced;
    reduced.reserve(column.size());
    for (const Phat_index idx : column) {
      if (!reduced.empty() && reduced.back() == idx) {
        reduced.pop_back();
      } else {
        reduced.push_back(idx);
      }
    }

    column.swap(reduced);
  }

  template <class Entry_range>
  static Row_indices _to_sorted_column(const Entry_range& range)
  {
    Row_indices out;
    out.reserve(static_cast<std::size_t>(std::distance(range.begin(), range.end())));
    for (const auto& entry : range) {
      out.push_back(_to_phat_index(Master_matrix::get_row_index(entry)));
    }
    _normalize(out);
    return out;
  }

  void _mark_view_dirty() const { viewDirty_ = true; }

  void _xor_rows_by_symmetric_difference(const Row_indices& other)
  {
    scratch_.clear();
    scratch_.reserve(rows_.size() + other.size());
    std::set_symmetric_difference(
        rows_.begin(), rows_.end(), other.begin(), other.end(), std::back_inserter(scratch_));
    rows_.swap(scratch_);
  }

  void _sync_column_from_rows()
  {
    const Phat_index required_capacity = Pivot_policy::required_capacity(rows_);
    if constexpr (Pivot_policy::tracks_capacity) {
      if (required_capacity > storageCapacity_) storageCapacity_ = required_capacity;
    } else {
      storageCapacity_ = required_capacity;
    }

    Pivot_policy::assert_rows_valid(rows_);
    Pivot_policy::set_column_from_rows(column_, rows_, storageCapacity_);
    rowsDirty_ = false;
  }

  void _sync_rows_from_storage_if_needed() const
  {
    if (!rowsDirty_) return;
    Pivot_policy::sync_rows_from_column(column_, rows_);
    rowsDirty_ = false;
  }

  void _rebuild_entry_view() const
  {
    if (!viewDirty_) return;

    _sync_rows_from_storage_if_needed();

    entryView_.clear();
    entryView_.reserve(rows_.size());
    for (const Phat_index idx : rows_) {
      entryView_.emplace_back(_to_id_index(idx));
    }
    viewDirty_ = false;
  }

  void _set_chain_pivot_from_storage()
  {
    if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
      Chain_opt::operator=(Chain_opt(_pivot_from_storage()));
    }
  }

  bool _xor_with_sorted(const Row_indices& other)
  {
    const ID_index old_pivot = _pivot_from_storage();
    bool use_incremental_update = false;

    if constexpr (Pivot_policy::supports_incremental_xor) {
      use_incremental_update = Pivot_policy::can_incremental_xor(column_, other, storageCapacity_);
      if (use_incremental_update) {
        Pivot_policy::xor_rows_in_place(column_, other);
      }
    }

    if (!use_incremental_update) {
      _xor_rows_by_symmetric_difference(other);
      _sync_column_from_rows();
    } else {
      _xor_rows_by_symmetric_difference(other);
    }

    _mark_view_dirty();
    _set_chain_pivot_from_storage();

    if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
      return old_pivot != _null_index() && _pivot_from_storage() != old_pivot;
    } else {
      return false;
    }
  }

 public:
  Phat_column_interface(Column_settings* colSettings = nullptr)
      : RA_opt(),
        Dim_opt(),
        Chain_opt(),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(false),
        storageCapacity_(static_cast<Phat_index>(0)),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    _check_supported_options();
    _sync_column_from_rows();
  }

  template <class Container = typename Master_matrix::Boundary>
  Phat_column_interface(const Container& nonZeroRowIndices, Column_settings* colSettings)
      : Phat_column_interface(nonZeroRowIndices,
                              nonZeroRowIndices.size() == 0 ? 0 : nonZeroRowIndices.size() - 1,
                              colSettings)
  {
    static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                  "Constructor not available for chain columns, please specify the dimension of the chain.");
  }

  template <class Container = typename Master_matrix::Boundary, class Row_container>
  Phat_column_interface(Index columnIndex,
                        const Container& nonZeroRowIndices,
                        Row_container* rowContainer,
                        Column_settings* colSettings)
      : Phat_column_interface(columnIndex,
                              nonZeroRowIndices,
                              nonZeroRowIndices.size() == 0 ? 0 : nonZeroRowIndices.size() - 1,
                              rowContainer,
                              colSettings)
  {
    static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                  "Constructor not available for chain columns, please specify the dimension of the chain.");
  }

  template <class Container = typename Master_matrix::Boundary,
            class = std::enable_if_t<!std::is_arithmetic_v<Container> > >
  Phat_column_interface(const Container& nonZeroRowIndices, Dimension dimension, Column_settings* colSettings)
      : RA_opt(),
        Dim_opt(dimension),
        Chain_opt(),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(false),
        storageCapacity_(static_cast<Phat_index>(0)),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    _check_supported_options();
    rows_ = _to_sorted_column(nonZeroRowIndices);
    _sync_column_from_rows();
    _set_chain_pivot_from_storage();
  }

  template <class Container = typename Master_matrix::Boundary,
            class Row_container,
            class = std::enable_if_t<!std::is_arithmetic_v<Container> > >
  Phat_column_interface(Index columnIndex,
                        const Container& nonZeroRowIndices,
                        Dimension dimension,
                        Row_container* rowContainer,
                        Column_settings* colSettings)
      : RA_opt(),
        Dim_opt(dimension),
        Chain_opt(),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(false),
        storageCapacity_(static_cast<Phat_index>(0)),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    static_cast<void>(columnIndex);
    static_cast<void>(rowContainer);
    _check_supported_options();
    rows_ = _to_sorted_column(nonZeroRowIndices);
    _sync_column_from_rows();
    _set_chain_pivot_from_storage();
  }

  Phat_column_interface(ID_index idx, Dimension dimension, Column_settings* colSettings)
      : RA_opt(),
        Dim_opt(dimension),
        Chain_opt(idx),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(false),
        storageCapacity_(static_cast<Phat_index>(0)),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    _check_supported_options();
    rows_.push_back(_to_phat_index(idx));
    _sync_column_from_rows();
  }

  Phat_column_interface(ID_index idx, Field_element e, Dimension dimension, Column_settings* colSettings)
      : RA_opt(),
        Dim_opt(dimension),
        Chain_opt(),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(false),
        storageCapacity_(static_cast<Phat_index>(0)),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    _check_supported_options();
    if (Master_matrix::get_coefficient_value(e, operators_) != Field_operators::get_additive_identity()) {
      rows_.push_back(_to_phat_index(idx));
    }
    _sync_column_from_rows();
    _set_chain_pivot_from_storage();
  }

  template <class Row_container>
  Phat_column_interface(Index columnIndex,
                        ID_index idx,
                        Dimension dimension,
                        Row_container* rowContainer,
                        Column_settings* colSettings)
      : RA_opt(),
        Dim_opt(dimension),
        Chain_opt(idx),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(false),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    static_cast<void>(columnIndex);
    static_cast<void>(rowContainer);
    _check_supported_options();
    rows_.push_back(_to_phat_index(idx));
    _sync_column_from_rows();
  }

  template <class Row_container>
  Phat_column_interface(Index columnIndex,
                        ID_index idx,
                        Field_element e,
                        Dimension dimension,
                        Row_container* rowContainer,
                        Column_settings* colSettings)
      : RA_opt(),
        Dim_opt(dimension),
        Chain_opt(),
        column_(),
        rows_(),
        entryView_(),
        viewDirty_(true),
        operators_(Master_matrix::get_operator_ptr(colSettings))
  {
    static_cast<void>(columnIndex);
    static_cast<void>(rowContainer);
    _check_supported_options();
    if (Master_matrix::get_coefficient_value(e, operators_) != Field_operators::get_additive_identity()) {
      rows_.push_back(_to_phat_index(idx));
    }
    _sync_column_from_rows();
    _set_chain_pivot_from_storage();
  }

  Phat_column_interface(const Phat_column_interface& column, Column_settings* colSettings = nullptr)
      : RA_opt(),
        Dim_opt(static_cast<const Dim_opt&>(column)),
        Chain_opt(static_cast<const Chain_opt&>(column)),
        column_(column.column_),
        rows_(column.rows_),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(column.rowsDirty_),
        storageCapacity_(column.storageCapacity_),
        operators_(colSettings == nullptr ? column.operators_ : Master_matrix::get_operator_ptr(colSettings))
  {
    _check_supported_options();
    static_assert(!Master_matrix::Option_list::has_row_access,
                  "Simple copy constructor not available when row access option enabled.");
  }

  template <class Row_container>
  Phat_column_interface(const Phat_column_interface& column,
                        Index columnIndex,
                        Row_container* rowContainer,
                        Column_settings* colSettings = nullptr)
      : RA_opt(),
        Dim_opt(static_cast<const Dim_opt&>(column)),
        Chain_opt(static_cast<const Chain_opt&>(column)),
        column_(column.column_),
        rows_(column.rows_),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(column.rowsDirty_),
        storageCapacity_(column.storageCapacity_),
        operators_(colSettings == nullptr ? column.operators_ : Master_matrix::get_operator_ptr(colSettings))
  {
    static_cast<void>(columnIndex);
    static_cast<void>(rowContainer);
    _check_supported_options();
  }

  Phat_column_interface(Phat_column_interface&& column) noexcept
      : RA_opt(std::move(static_cast<RA_opt&>(column))),
        Dim_opt(std::move(static_cast<Dim_opt&>(column))),
        Chain_opt(std::move(static_cast<Chain_opt&>(column))),
        column_(std::move(column.column_)),
        rows_(std::move(column.rows_)),
        entryView_(),
        viewDirty_(true),
        rowsDirty_(column.rowsDirty_),
        storageCapacity_(column.storageCapacity_),
        operators_(std::exchange(column.operators_, nullptr))
  {
    _check_supported_options();
  }

  ~Phat_column_interface() = default;

  std::vector<Field_element> get_content(int columnLength = -1) const
  {
    _sync_rows_from_storage_if_needed();

    if (columnLength < 0) {
      columnLength = rows_.empty() ? 0 : static_cast<int>(rows_.back()) + 1;
    }

    std::vector<Field_element> container(static_cast<std::size_t>(columnLength),
                                         Field_operators::get_additive_identity());
    for (const Phat_index idx : rows_) {
      const std::size_t i = static_cast<std::size_t>(_to_id_index(idx));
      if (i < container.size()) {
        container[i] = Field_operators::get_multiplicative_identity();
      }
    }

    return container;
  }

  bool is_non_zero(ID_index rowIndex) const
  {
    _sync_rows_from_storage_if_needed();
    return std::binary_search(rows_.begin(), rows_.end(), _to_phat_index(rowIndex));
  }

  [[nodiscard]] bool is_empty() const { return _pivot_from_storage() == _null_index(); }

  [[nodiscard]] std::size_t size() const
  {
    _sync_rows_from_storage_if_needed();
    return rows_.size();
  }

  template <class Row_index_map>
  void reorder(const Row_index_map& valueMap,
               [[maybe_unused]] Index columnIndex = Master_matrix::template get_null_value<Index>())
  {
    _sync_rows_from_storage_if_needed();
    for (Phat_index& idx : rows_) {
      idx = _to_phat_index(valueMap[_to_id_index(idx)]);
    }
    _normalize(rows_);
    _sync_column_from_rows();
    _mark_view_dirty();
    _set_chain_pivot_from_storage();
  }

  void clear()
  {
    rows_.clear();
    _sync_column_from_rows();
    _mark_view_dirty();
    _set_chain_pivot_from_storage();
  }

  void clear(ID_index rowIndex)
  {
    _sync_rows_from_storage_if_needed();
    const Phat_index target = _to_phat_index(rowIndex);
    auto it = std::lower_bound(rows_.begin(), rows_.end(), target);
    if (it != rows_.end() && *it == target) {
      rows_.erase(it);
      _sync_column_from_rows();
      _mark_view_dirty();
      _set_chain_pivot_from_storage();
    }
  }

  ID_index get_pivot() const
  {
    static_assert(Master_matrix::isNonBasic,
                  "Method not available for base columns.");

    if constexpr (!Master_matrix::Option_list::is_of_boundary_type) {
      return Chain_opt::_get_pivot();
    } else {
      return _pivot_from_storage();
    }
  }

  Field_element get_pivot_value() const
  {
    static_assert(Master_matrix::isNonBasic,
                  "Method not available for base columns.");

    if constexpr (Master_matrix::Option_list::is_z2) {
      return 1;
    } else {
      if constexpr (!Master_matrix::Option_list::is_of_boundary_type) {
        if (Chain_opt::_get_pivot() == _null_index()) return Field_element();
      } else {
        if (get_pivot() == _null_index()) return Field_element();
      }
      return Field_element();
    }
  }

  iterator begin() noexcept
  {
    _rebuild_entry_view();
    return entryView_.begin();
  }

  const_iterator begin() const noexcept
  {
    _rebuild_entry_view();
    return entryView_.begin();
  }

  iterator end() noexcept
  {
    _rebuild_entry_view();
    return entryView_.end();
  }

  const_iterator end() const noexcept
  {
    _rebuild_entry_view();
    return entryView_.end();
  }

  reverse_iterator rbegin() noexcept
  {
    _rebuild_entry_view();
    return entryView_.rbegin();
  }

  const_reverse_iterator rbegin() const noexcept
  {
    _rebuild_entry_view();
    return entryView_.rbegin();
  }

  reverse_iterator rend() noexcept
  {
    _rebuild_entry_view();
    return entryView_.rend();
  }

  const_reverse_iterator rend() const noexcept
  {
    _rebuild_entry_view();
    return entryView_.rend();
  }

  Content_range get_non_zero_content_range() const
  {
    _rebuild_entry_view();
    return Content_range(entryView_.begin(), entryView_.end());
  }

  template <class Entry_range>
  Phat_column_interface& operator+=(const Entry_range& column)
  {
    static_assert((!Master_matrix::isNonBasic ||
                   std::is_base_of_v<Phat_column_interface,
                                     std::remove_cv_t<std::remove_reference_t<Entry_range>>>),
                  "For boundary columns, the range has to be a column of same type.");
    static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type),
                  "For chain columns, the given column cannot be constant.");

    Row_indices source = _to_sorted_column(column);
    _xor_with_sorted(source);
    return *this;
  }

  Phat_column_interface& operator+=(Phat_column_interface& column)
  {
    column._sync_rows_from_storage_if_needed();

    if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
      if (_xor_with_sorted(column.rows_)) {
        Chain_opt::_swap_pivots(column);
        Dim_opt::_swap_dimension(column);
      }
    } else {
      _xor_with_sorted(column.rows_);
    }

    return *this;
  }

  Phat_column_interface& operator*=(const Field_element& v)
  {
    const Field_element val = Master_matrix::get_coefficient_value(v, operators_);

    if (val == Field_operators::get_additive_identity()) {
      if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
        throw std::invalid_argument("A chain column should not be multiplied by 0.");
      } else {
        clear();
      }
    }

    return *this;
  }

  template <class Entry_range>
  Phat_column_interface& multiply_target_and_add(const Field_element& val, const Entry_range& column)
  {
    static_assert((!Master_matrix::isNonBasic ||
                   std::is_base_of_v<Phat_column_interface,
                                     std::remove_cv_t<std::remove_reference_t<Entry_range>>>),
                  "For boundary columns, the range has to be a column of same type.");
    static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type),
                  "For chain columns, the given column cannot be constant.");

    const Field_element coeff = Master_matrix::get_coefficient_value(val, operators_);
    if (coeff == Field_operators::get_additive_identity()) return *this;

    Row_indices source = _to_sorted_column(column);
    _xor_with_sorted(source);
    return *this;
  }

  Phat_column_interface& multiply_target_and_add(const Field_element& val, Phat_column_interface& column)
  {
    const Field_element coeff = Master_matrix::get_coefficient_value(val, operators_);
    if (coeff == Field_operators::get_additive_identity()) return *this;

    column._sync_rows_from_storage_if_needed();

    if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
      if (_xor_with_sorted(column.rows_)) {
        Chain_opt::_swap_pivots(column);
        Dim_opt::_swap_dimension(column);
      }
    } else {
      _xor_with_sorted(column.rows_);
    }

    return *this;
  }

  template <class Entry_range>
  Phat_column_interface& multiply_source_and_add(const Entry_range& column, const Field_element& val)
  {
    static_assert((!Master_matrix::isNonBasic ||
                   std::is_base_of_v<Phat_column_interface,
                                     std::remove_cv_t<std::remove_reference_t<Entry_range>>>),
                  "For boundary columns, the range has to be a column of same type.");
    static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type),
                  "For chain columns, the given column cannot be constant.");

    const Field_element coeff = Master_matrix::get_coefficient_value(val, operators_);
    if (coeff == Field_operators::get_additive_identity()) return *this;

    Row_indices source = _to_sorted_column(column);
    _xor_with_sorted(source);
    return *this;
  }

  Phat_column_interface& multiply_source_and_add(Phat_column_interface& column, const Field_element& val)
  {
    const Field_element coeff = Master_matrix::get_coefficient_value(val, operators_);
    if (coeff == Field_operators::get_additive_identity()) return *this;

    column._sync_rows_from_storage_if_needed();

    if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
      if (_xor_with_sorted(column.rows_)) {
        Chain_opt::_swap_pivots(column);
        Dim_opt::_swap_dimension(column);
      }
    } else {
      _xor_with_sorted(column.rows_);
    }

    return *this;
  }

  void push_back(const Entry& entry)
  {
    static_assert(Master_matrix::Option_list::is_of_boundary_type,
                  "`push_back` is not available for Chain matrices.");

    _sync_rows_from_storage_if_needed();

    if (!rows_.empty() && entry.get_row_index() <= _to_id_index(rows_.back())) {
      throw std::invalid_argument("The new row index has to be higher than the current pivot.");
    }

    rows_.push_back(_to_phat_index(entry.get_row_index()));
    _sync_column_from_rows();
    _mark_view_dirty();
  }

  friend bool operator==(const Phat_column_interface& c1, const Phat_column_interface& c2)
  {
    c1._sync_rows_from_storage_if_needed();
    c2._sync_rows_from_storage_if_needed();
    return c1.rows_ == c2.rows_;
  }

  friend bool operator<(const Phat_column_interface& c1, const Phat_column_interface& c2)
  {
    c1._sync_rows_from_storage_if_needed();
    c2._sync_rows_from_storage_if_needed();
    return std::lexicographical_compare(c1.rows_.begin(), c1.rows_.end(), c2.rows_.begin(), c2.rows_.end());
  }

  Phat_column_interface& operator=(const Phat_column_interface& other)
  {
    static_assert(!Master_matrix::Option_list::has_row_access,
                  "= assignment not enabled with row access option.");

    if (this == &other) return *this;

    Dim_opt::operator=(other);
    Chain_opt::operator=(other);
    column_ = other.column_;
    rows_ = other.rows_;
    rowsDirty_ = other.rowsDirty_;
    storageCapacity_ = other.storageCapacity_;
    scratch_.clear();
    operators_ = other.operators_;
    _mark_view_dirty();
    return *this;
  }

  Phat_column_interface& operator=(Phat_column_interface&& other) noexcept
  {
    static_assert(!Master_matrix::Option_list::has_row_access,
                  "= assignment not enabled with row access option.");

    if (this == &other) return *this;

    Dim_opt::operator=(std::move(other));
    Chain_opt::operator=(std::move(other));
    column_ = std::move(other.column_);
    rows_ = std::move(other.rows_);
    rowsDirty_ = other.rowsDirty_;
    storageCapacity_ = other.storageCapacity_;
    scratch_.clear();
    operators_ = std::exchange(other.operators_, nullptr);
    _mark_view_dirty();
    return *this;
  }

  friend void swap(Phat_column_interface& col1, Phat_column_interface& col2) noexcept
  {
    swap(static_cast<RA_opt&>(col1), static_cast<RA_opt&>(col2));
    swap(static_cast<Dim_opt&>(col1), static_cast<Dim_opt&>(col2));
    swap(static_cast<Chain_opt&>(col1), static_cast<Chain_opt&>(col2));
    std::swap(col1.column_, col2.column_);
    col1.rows_.swap(col2.rows_);
    col1.scratch_.swap(col2.scratch_);
    std::swap(col1.rowsDirty_, col2.rowsDirty_);
    std::swap(col1.storageCapacity_, col2.storageCapacity_);
    std::swap(col1.operators_, col2.operators_);
    col1._mark_view_dirty();
    col2._mark_view_dirty();
  }

 protected:
  const Column_support& _column_support() const { return column_; }

 private:
  Column_support column_;
  mutable Row_indices rows_;
  Row_indices scratch_;
  mutable Entry_view entryView_;
  mutable bool viewDirty_;
  mutable bool rowsDirty_;
  Phat_index storageCapacity_;
  Field_operators const* operators_;
};

}  // namespace persistence_matrix
}  // namespace Gudhi

#endif  // PM_PHAT_COLUMN_INTERFACE_H
