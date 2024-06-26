/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022-23 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef PM_INTRUSIVE_SET_COLUMN_H
#define PM_INTRUSIVE_SET_COLUMN_H

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <utility>	//std::swap, std::move & std::exchange

#include <boost/intrusive/set.hpp>

#include <gudhi/Debug_utils.h>
#include <gudhi/Persistence_matrix/columns/cell_constructors.h>

namespace Gudhi {
namespace persistence_matrix {

template<class Master_matrix, class Cell_constructor = New_cell_constructor<typename Master_matrix::Cell_type> >
class Intrusive_set_column : public Master_matrix::Row_access_option, 
							  public Master_matrix::Column_dimension_option,
							  public Master_matrix::Chain_column_option
{
public:
	using Master = Master_matrix;
	using Field_operators = typename Master_matrix::Field_operators;
	using Field_element_type = typename Master_matrix::element_type;
	using index = typename Master_matrix::index;
	using id_index = typename Master_matrix::id_index;
	using dimension_type = typename Master_matrix::dimension_type;

	using Cell = typename Master_matrix::Cell_type;
	using Column_type = boost::intrusive::set <
							Cell, 
							boost::intrusive::constant_time_size<false>, 
							boost::intrusive::base_hook< typename Master_matrix::base_hook_matrix_set_column > 
						>;
	using iterator = typename Column_type::iterator;
	using const_iterator = typename Column_type::const_iterator;
	using reverse_iterator = typename Column_type::reverse_iterator;
	using const_reverse_iterator = typename Column_type::const_reverse_iterator;

	Intrusive_set_column(Field_operators* operators = nullptr, Cell_constructor* cellConstructor = nullptr);
	template<class Container_type = typename Master_matrix::boundary_type>
	Intrusive_set_column(const Container_type& nonZeroRowIndices, Field_operators* operators, Cell_constructor* cellConstructor);	//has to be a boundary for boundary, has no sense for chain if dimension is needed
	template<class Container_type = typename Master_matrix::boundary_type, class Row_container_type>
	Intrusive_set_column(index columnIndex, const Container_type& nonZeroRowIndices, Row_container_type *rowContainer, Field_operators* operators, Cell_constructor* cellConstructor);	//has to be a boundary for boundary, has no sense for chain if dimension is needed
	template<class Container_type = typename Master_matrix::boundary_type>
	Intrusive_set_column(const Container_type& nonZeroChainRowIndices, dimension_type dimension, Field_operators* operators, Cell_constructor* cellConstructor);	//dimension gets ignored for base
	template<class Container_type = typename Master_matrix::boundary_type, class Row_container_type>
	Intrusive_set_column(index columnIndex, const Container_type& nonZeroChainRowIndices, dimension_type dimension, Row_container_type *rowContainer, Field_operators* operators, Cell_constructor* cellConstructor);	//dimension gets ignored for base
	Intrusive_set_column(const Intrusive_set_column& column, Field_operators* operators = nullptr, Cell_constructor* cellConstructor = nullptr);
	template<class Row_container_type>
	Intrusive_set_column(const Intrusive_set_column& column, index columnIndex, Row_container_type *rowContainer, Field_operators* operators = nullptr, Cell_constructor* cellConstructor = nullptr);
	Intrusive_set_column(Intrusive_set_column&& column) noexcept;
	~Intrusive_set_column();

	std::vector<Field_element_type> get_content(int columnLength = -1) const;
	bool is_non_zero(id_index rowIndex) const;
	bool is_empty() const;
	std::size_t size() const;

	//****************
	//only for base and boundary
	template<class Map_type>
	void reorder(const Map_type& valueMap);	//used for lazy row swaps
	void clear();
	void clear(id_index rowIndex);
	//****************

	//****************
	//only for chain and boundary
	id_index get_pivot() const;
	Field_element_type get_pivot_value() const;
	//****************

	iterator begin() noexcept;
	const_iterator begin() const noexcept;
	iterator end() noexcept;
	const_iterator end() const noexcept;
	reverse_iterator rbegin() noexcept;
	const_reverse_iterator rbegin() const noexcept;
	reverse_iterator rend() noexcept;
	const_reverse_iterator rend() const noexcept;

	template<class Cell_range>
	Intrusive_set_column& operator+=(const Cell_range& column);	//for base & boundary except vector
	Intrusive_set_column& operator+=(Intrusive_set_column &column);	//for chain and vector

	Intrusive_set_column& operator*=(unsigned int v);

	//this = v * this + column
	template<class Cell_range>
	Intrusive_set_column& multiply_and_add(const Field_element_type& val, const Cell_range& column);	//for base & boundary except vector
	Intrusive_set_column& multiply_and_add(const Field_element_type& val, Intrusive_set_column& column);	//for chain and vector
	//this = this + column * v
	template<class Cell_range>
	Intrusive_set_column& multiply_and_add(const Cell_range& column, const Field_element_type& val);	//for base & boundary except vector
	Intrusive_set_column& multiply_and_add(Intrusive_set_column& column, const Field_element_type& val);	//for chain and vector

	friend bool operator==(const Intrusive_set_column& c1, const Intrusive_set_column& c2){
		if (&c1 == &c2) return true;

		if constexpr (Master_matrix::Option_list::is_z2){
			return c1.column_ == c2.column_;
		} else {
			auto it1 = c1.column_.begin();
			auto it2 = c2.column_.begin();
			if (c1.column_.size() != c2.column_.size()) return false;
			while (it1 != c1.column_.end() && it2 != c2.column_.end()) {
				if (it1->get_row_index() != it2->get_row_index() || it1->get_element() != it2->get_element())
					return false;
				++it1; ++it2;
			}
			return true;
		}
	}
	friend bool operator<(const Intrusive_set_column& c1, const Intrusive_set_column& c2){
		if (&c1 == &c2) return false;

		if constexpr (Master_matrix::Option_list::is_z2){
			return c1.column_ < c2.column_;
		} else {
			auto it1 = c1.column_.begin();
			auto it2 = c2.column_.begin();
			while (it1 != c1.column_.end() && it2 != c2.column_.end()) {
				if (it1->get_row_index() != it2->get_row_index())
					return it1->get_row_index() < it2->get_row_index();
				if (it1->get_element() != it2->get_element())
					return it1->get_element() < it2->get_element();
				++it1; ++it2;
			}
			return it2 != c2.column_.end();
		}
	}

	// void set_operators(Field_operators* operators){ operators_ = operators; }

	//Disabled with row access.
	Intrusive_set_column& operator=(const Intrusive_set_column& other);

	friend void swap(Intrusive_set_column& col1, Intrusive_set_column& col2){
		swap(static_cast<typename Master_matrix::Row_access_option&>(col1),
			 static_cast<typename Master_matrix::Row_access_option&>(col2));
		swap(static_cast<typename Master_matrix::Column_dimension_option&>(col1),
			 static_cast<typename Master_matrix::Column_dimension_option&>(col2));
		swap(static_cast<typename Master_matrix::Chain_column_option&>(col1),
			 static_cast<typename Master_matrix::Chain_column_option&>(col2));
		col1.column_.swap(col2.column_);
		std::swap(col1.operators_, col2.operators_);
		std::swap(col1.cellPool_, col2.cellPool_);
	}

private:
	using ra_opt = typename Master_matrix::Row_access_option;
	using dim_opt = typename Master_matrix::Column_dimension_option;
	using chain_opt = typename Master_matrix::Chain_column_option;

	//Cloner object function for boost intrusive container
	struct new_cloner
	{
		new_cloner(Cell_constructor* cellPool) : cellPool_(cellPool)
		{};

		Cell *operator()(const Cell& clone_this)
		{
			return cellPool_->construct(clone_this);
		}

		Cell_constructor* cellPool_;
	};

	//The disposer object function for boost intrusive container
	struct delete_disposer
	{
		delete_disposer(){};
		delete_disposer(Intrusive_set_column* col) : col_(col)
		{};

		void operator()(Cell *delete_this)
		{
			if constexpr (Master_matrix::Option_list::has_row_access)
				col_->unlink(delete_this);
			col_->cellPool_->destroy(delete_this);
		}

		Intrusive_set_column* col_;
	};

	Column_type column_;
	Field_operators* operators_;
	Cell_constructor* cellPool_;

	void _delete_cell(iterator& it);
	void _insert_cell(const Field_element_type& value, id_index rowIndex, const iterator& position);
	void _insert_cell(id_index rowIndex, const iterator& position);
	template<class Cell_range>
	bool _add(const Cell_range& column);
	template<class Cell_range>
	bool _multiply_and_add(const Field_element_type& val, const Cell_range& column);
	template<class Cell_range>
	bool _multiply_and_add(const Cell_range& column, const Field_element_type& val);

	void _verifyCellConstructor(){
		if (cellPool_ == nullptr){
			if constexpr (std::is_same_v<Cell_constructor, New_cell_constructor<typename Master_matrix::Cell_type> >){
				cellPool_ = &Master_matrix::defaultCellConstructor;
			} else {
				throw std::invalid_argument("Cell constructor pointer cannot be null.");
			}
		}
	}
};

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(Field_operators* operators, Cell_constructor* cellConstructor) : ra_opt(), dim_opt(), chain_opt(), operators_(operators),
	  cellPool_(cellConstructor)
{
	if (operators_ == nullptr && cellPool_ == nullptr) return;	//to allow default constructor which gives a dummy column
	_verifyCellConstructor();
}

template<class Master_matrix, class Cell_constructor>
template<class Container_type>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(const Container_type &nonZeroRowIndices, Field_operators* operators, Cell_constructor* cellConstructor)
	: ra_opt(), 
	  dim_opt(nonZeroRowIndices.size() == 0 ? 0 : nonZeroRowIndices.size() - 1), 
	  chain_opt(),
	  operators_(operators),
	  cellPool_(cellConstructor)
{
	static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type, 
						"Constructor not available for chain columns, please specify the dimension of the chain.");

	_verifyCellConstructor();

	if constexpr (Master_matrix::Option_list::is_z2){
		for (id_index id : nonZeroRowIndices){
			_insert_cell(id, column_.end());
		}
	} else {
		for (const auto& p : nonZeroRowIndices){
			_insert_cell(operators_->get_value(p.second), p.first, column_.end());
		}
	}
}

template<class Master_matrix, class Cell_constructor>
template<class Container_type, class Row_container_type>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(
	index columnIndex, const Container_type &nonZeroRowIndices, Row_container_type *rowContainer, Field_operators* operators, Cell_constructor* cellConstructor) 
	: ra_opt(columnIndex, rowContainer), 
	  dim_opt(nonZeroRowIndices.size() == 0 ? 0 : nonZeroRowIndices.size() - 1),
	  chain_opt([&]{
			if constexpr (Master_matrix::Option_list::is_z2){
				return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : *std::prev(nonZeroRowIndices.end());
			} else {
				return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : std::prev(nonZeroRowIndices.end())->first;
			}
		}()),
	  operators_(operators),
	  cellPool_(cellConstructor)
{
	static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type, 
						"Constructor not available for chain columns, please specify the dimension of the chain.");

	_verifyCellConstructor();

	if constexpr (Master_matrix::Option_list::is_z2){
		for (id_index id : nonZeroRowIndices){
			_insert_cell(id, column_.end());
		}
	} else {
		for (const auto& p : nonZeroRowIndices){
			_insert_cell(operators_->get_value(p.second), p.first, column_.end());
		}
	}
}

template<class Master_matrix, class Cell_constructor>
template<class Container_type>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(
	const Container_type &nonZeroRowIndices, dimension_type dimension, Field_operators* operators, Cell_constructor* cellConstructor) 
	: ra_opt(), 
	  dim_opt(dimension),
	  chain_opt([&]{
			if constexpr (Master_matrix::Option_list::is_z2){
				return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : *std::prev(nonZeroRowIndices.end());
			} else {
				return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : std::prev(nonZeroRowIndices.end())->first;
			}
		}()),
	  operators_(operators),
	  cellPool_(cellConstructor)
{
	_verifyCellConstructor();

	if constexpr (Master_matrix::Option_list::is_z2){
		for (id_index id : nonZeroRowIndices){
			_insert_cell(id, column_.end());
		}
	} else {
		for (const auto& p : nonZeroRowIndices){
			_insert_cell(operators_->get_value(p.second), p.first, column_.end());
		}
	}
}

template<class Master_matrix, class Cell_constructor>
template<class Container_type, class Row_container_type>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(
	index columnIndex, const Container_type &nonZeroRowIndices, dimension_type dimension, Row_container_type *rowContainer, Field_operators* operators, Cell_constructor* cellConstructor) 
	: ra_opt(columnIndex, rowContainer), 
	  dim_opt(dimension),
	  chain_opt([&]{
			if constexpr (Master_matrix::Option_list::is_z2){
				return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : *std::prev(nonZeroRowIndices.end());
			} else {
				return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : std::prev(nonZeroRowIndices.end())->first;
			}
		}()),
	  operators_(operators),
	  cellPool_(cellConstructor)
{
	_verifyCellConstructor();

	if constexpr (Master_matrix::Option_list::is_z2){
		for (id_index id : nonZeroRowIndices){
			_insert_cell(id, column_.end());
		}
	} else {
		for (const auto& p : nonZeroRowIndices){
			_insert_cell(operators_->get_value(p.second), p.first, column_.end());
		}
	}
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(const Intrusive_set_column &column, Field_operators* operators, Cell_constructor* cellConstructor) 
	: ra_opt(), 
	  dim_opt(static_cast<const dim_opt&>(column)), 
	  chain_opt(static_cast<const chain_opt&>(column)),
	  operators_(operators == nullptr ? column.operators_ : operators),
	  cellPool_(cellConstructor == nullptr ? column.cellPool_ : cellConstructor)
{
	static_assert(!Master_matrix::Option_list::has_row_access,
			"Simple copy constructor not available when row access option enabled. Please specify the new column index and the row container.");

	column_.clone_from(column.column_, new_cloner(cellPool_), delete_disposer(this));
}

template<class Master_matrix, class Cell_constructor>
template<class Row_container_type>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(
	const Intrusive_set_column &column, index columnIndex, Row_container_type *rowContainer, Field_operators* operators, Cell_constructor* cellConstructor) 
	: ra_opt(columnIndex, rowContainer), 
	  dim_opt(static_cast<const dim_opt&>(column)), 
	  chain_opt(static_cast<const chain_opt&>(column)),
	  operators_(operators == nullptr ? column.operators_ : operators),
	  cellPool_(cellConstructor == nullptr ? column.cellPool_ : cellConstructor)
{
	for (const Cell& cell : column.column_){
		if constexpr (Master_matrix::Option_list::is_z2){
			_insert_cell(cell.get_row_index(), column_.end());
		} else {
			_insert_cell(cell.get_element(), cell.get_row_index(), column_.end());
		}
	}
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::Intrusive_set_column(Intrusive_set_column &&column) noexcept 
	: ra_opt(std::move(static_cast<ra_opt&>(column))), 
	  dim_opt(std::move(static_cast<dim_opt&>(column))), 
	  chain_opt(std::move(static_cast<chain_opt&>(column))), 
	  column_(std::move(column.column_)),
	  operators_(std::exchange(column.operators_, nullptr)),
	  cellPool_(std::exchange(column.cellPool_, nullptr))
{}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor>::~Intrusive_set_column()
{
	column_.clear_and_dispose(delete_disposer(this));
}

template<class Master_matrix, class Cell_constructor>
inline std::vector<typename Intrusive_set_column<Master_matrix,Cell_constructor>::Field_element_type> 
Intrusive_set_column<Master_matrix,Cell_constructor>::get_content(int columnLength) const
{
	if (columnLength < 0 && column_.size() > 0) columnLength = column_.rbegin()->get_row_index() + 1;
	else if (columnLength < 0) return std::vector<Field_element_type>();

	std::vector<Field_element_type> container(columnLength);
	for (auto it = column_.begin(); it != column_.end() && it->get_row_index() < static_cast<id_index>(columnLength); ++it){
		if constexpr (Master_matrix::Option_list::is_z2){
			container[it->get_row_index()] = 1;
		} else {
			container[it->get_row_index()] = it->get_element();
		}
	}
	return container;
}

template<class Master_matrix, class Cell_constructor>
inline bool Intrusive_set_column<Master_matrix,Cell_constructor>::is_non_zero(id_index rowIndex) const
{
	return column_.find(Cell(rowIndex)) != column_.end();
}

template<class Master_matrix, class Cell_constructor>
inline bool Intrusive_set_column<Master_matrix,Cell_constructor>::is_empty() const
{
	return column_.empty();
}

template<class Master_matrix, class Cell_constructor>
inline std::size_t Intrusive_set_column<Master_matrix,Cell_constructor>::size() const{
	return column_.size();
}

template<class Master_matrix, class Cell_constructor>
template<class Map_type>
inline void Intrusive_set_column<Master_matrix,Cell_constructor>::reorder(const Map_type &valueMap)
{
	static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type, 
						"Method not available for chain columns.");

	Column_type newSet;

	if constexpr (Master_matrix::Option_list::has_row_access) {
		for (auto it = column_.begin(); it != column_.end(); ) {
			Cell *new_cell = cellPool_->construct(ra_opt::columnIndex_, valueMap.at(it->get_row_index()));
			if constexpr (!Master_matrix::Option_list::is_z2){
				new_cell->set_element(it->get_element());
			}
			newSet.insert(newSet.end(), *new_cell);
			_delete_cell(it);	//increases it
			if constexpr (Master_matrix::Option_list::has_intrusive_rows)	//intrusive list
				ra_opt::insert_cell(new_cell->get_row_index(), new_cell);
		}

		//when row is a set, all cells have to be deleted first, to avoid colliding when inserting
		if constexpr (!Master_matrix::Option_list::has_intrusive_rows){		//set
			for (Cell& cell : newSet) {
				ra_opt::insert_cell(cell.get_row_index(), &cell);
			}
		}
	} else {
		for (auto it = column_.begin(); it != column_.end(); ) {
			Cell *new_cell = cellPool_->construct(valueMap.at(it->get_row_index()));
			if constexpr (!Master_matrix::Option_list::is_z2){
				new_cell->set_element(it->get_element());
			}
			newSet.insert(newSet.end(), *new_cell);
			_delete_cell(it);	//increases it
		}
	}

	column_.swap(newSet);
}

template<class Master_matrix, class Cell_constructor>
inline void Intrusive_set_column<Master_matrix,Cell_constructor>::clear()
{
	static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type, 
						"Method not available for chain columns as a base element should not be empty.");

	column_.clear_and_dispose(delete_disposer(this));
}

template<class Master_matrix, class Cell_constructor>
inline void Intrusive_set_column<Master_matrix,Cell_constructor>::clear(id_index rowIndex)
{
	static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type, 
						"Method not available for chain columns.");

	auto it = column_.find(Cell(rowIndex));
	if (it != column_.end()){
		_delete_cell(it);
	}
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::id_index Intrusive_set_column<Master_matrix,Cell_constructor>::get_pivot() const
{
	static_assert(Master_matrix::isNonBasic, "Method not available for base columns.");	//could technically be, but is the notion usefull then?

	if constexpr (Master_matrix::Option_list::is_of_boundary_type){
		if (column_.empty()) return -1;
		return column_.rbegin()->get_row_index();
	} else {
		return chain_opt::get_pivot();
	}
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::Field_element_type Intrusive_set_column<Master_matrix,Cell_constructor>::get_pivot_value() const
{
	static_assert(Master_matrix::isNonBasic, "Method not available for base columns.");	//could technically be, but is the notion usefull then?

	if constexpr (Master_matrix::Option_list::is_z2){
		return 1;
	} else {
		if constexpr (Master_matrix::Option_list::is_of_boundary_type){
			if (column_.empty()) return 0;
			return column_.rbegin()->get_element();
		} else {
			if (chain_opt::get_pivot() == -1) return 0;
			auto it = column_.find(Cell(chain_opt::get_pivot()));
			GUDHI_CHECK(it != column_.end(), "Pivot not found only if the column was misused.");
			return it->get_element();
		}
	}
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::begin() noexcept
{
	return column_.begin();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::const_iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::begin() const noexcept
{
	return column_.begin();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::end() noexcept
{
	return column_.end();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::const_iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::end() const noexcept
{
	return column_.end();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::reverse_iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::rbegin() noexcept
{
	return column_.rbegin();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::const_reverse_iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::rbegin() const noexcept
{
	return column_.rbegin();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::reverse_iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::rend() noexcept
{
	return column_.rend();
}

template<class Master_matrix, class Cell_constructor>
inline typename Intrusive_set_column<Master_matrix,Cell_constructor>::const_reverse_iterator
Intrusive_set_column<Master_matrix,Cell_constructor>::rend() const noexcept
{
	return column_.rend();
}

template<class Master_matrix, class Cell_constructor>
template<class Cell_range>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::operator+=(const Cell_range &column)
{
	static_assert((!Master_matrix::isNonBasic || std::is_same_v<Cell_range, Intrusive_set_column>), 
					"For boundary columns, the range has to be a column of same type to help ensure the validity of the base element.");	//could be removed, if we give the responsability to the user.
	static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type), 
					"For chain columns, the given column cannot be constant.");

	_add(column);

	return *this;
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::operator+=(Intrusive_set_column &column)
{
	if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
		//assumes that the addition never zeros out this column. 
		if (_add(column)) {
			chain_opt::swap_pivots(column);
			dim_opt::swap_dimension(column);
		}
	} else {
		_add(column);
	}

	return *this;
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::operator*=(unsigned int v)
{
	if constexpr (Master_matrix::Option_list::is_z2){
		if (v % 2 == 0){
			if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
				throw std::invalid_argument("A chain column should not be multiplied by 0.");
			} else {
				clear();
			}
		}
	} else {
		Field_element_type val = operators_->get_value(v);

		if (val == Field_operators::get_additive_identity()) {
			if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
				throw std::invalid_argument("A chain column should not be multiplied by 0.");
			} else {
				clear();
			}
			return *this;
		}

		if (val == Field_operators::get_multiplicative_identity()) return *this;

		for (Cell& cell : column_){
			cell.get_element() = operators_->multiply(cell.get_element(), val);
			if constexpr (Master_matrix::Option_list::has_row_access)
				ra_opt::update_cell(cell);
		}
	}

	return *this;
}

template<class Master_matrix, class Cell_constructor>
template<class Cell_range>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::multiply_and_add(const Field_element_type& val, const Cell_range& column)
{
	static_assert((!Master_matrix::isNonBasic || std::is_same_v<Cell_range, Intrusive_set_column>), 
					"For boundary columns, the range has to be a column of same type to help ensure the validity of the base element.");	//could be removed, if we give the responsability to the user.
	static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type), 
					"For chain columns, the given column cannot be constant.");

	if constexpr (Master_matrix::Option_list::is_z2){
		if (val){
			_add(column);
		} else {
			clear();
			_add(column);
		}
	} else {
		_multiply_and_add(val, column);
	}

	return *this;
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::multiply_and_add(const Field_element_type& val, Intrusive_set_column& column)
{
	if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
		//assumes that the addition never zeros out this column. 
		if constexpr (Master_matrix::Option_list::is_z2){
			if (val){
				if (_add(column)){
					chain_opt::swap_pivots(column);
					dim_opt::swap_dimension(column);
				}
			} else {
				throw std::invalid_argument("A chain column should not be multiplied by 0.");
			}
		} else {
			if (_multiply_and_add(val, column)){
				chain_opt::swap_pivots(column);
				dim_opt::swap_dimension(column);
			}
		}
	} else {
		if constexpr (Master_matrix::Option_list::is_z2){
			if (val){
				_add(column);
			} else {
				clear();
				_add(column);
			}
		} else {
			_multiply_and_add(val, column);
		}
	}

	return *this;
}

template<class Master_matrix, class Cell_constructor>
template<class Cell_range>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::multiply_and_add(const Cell_range& column, const Field_element_type& val)
{
	static_assert((!Master_matrix::isNonBasic || std::is_same_v<Cell_range, Intrusive_set_column>), 
					"For boundary columns, the range has to be a column of same type to help ensure the validity of the base element.");	//could be removed, if we give the responsability to the user.
	static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type), 
					"For chain columns, the given column cannot be constant.");

	if constexpr (Master_matrix::Option_list::is_z2){
		if (val){
			_add(column);
		}
	} else {
		_multiply_and_add(column, val);
	}

	return *this;
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::multiply_and_add(Intrusive_set_column& column, const Field_element_type& val)
{
	if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
		//assumes that the addition never zeros out this column. 
		if constexpr (Master_matrix::Option_list::is_z2){
			if (val){
				if (_add(column)){
					chain_opt::swap_pivots(column);
					dim_opt::swap_dimension(column);
				}
			}
		} else {
			if (_multiply_and_add(column, val)){
				chain_opt::swap_pivots(column);
				dim_opt::swap_dimension(column);
			}
		}
	} else {
		if constexpr (Master_matrix::Option_list::is_z2){
			if (val){
				_add(column);
			}
		} else {
			_multiply_and_add(column, val);
		}
	}

	return *this;
}

template<class Master_matrix, class Cell_constructor>
inline Intrusive_set_column<Master_matrix,Cell_constructor> &
Intrusive_set_column<Master_matrix,Cell_constructor>::operator=(const Intrusive_set_column& other)
{
	static_assert(!Master_matrix::Option_list::has_row_access, "= assignement not enabled with row access option.");

	dim_opt::operator=(other);
	chain_opt::operator=(other);

	//order is important
	column_.clear_and_dispose(delete_disposer(this));
	operators_ = other.operators_;
	cellPool_ = other.cellPool_;
	column_.clone_from(other.column_, new_cloner(cellPool_), delete_disposer(this));

	return *this;
}

template<class Master_matrix, class Cell_constructor>
inline void Intrusive_set_column<Master_matrix,Cell_constructor>::_delete_cell(iterator &it)
{
	it = column_.erase_and_dispose(it, delete_disposer(this));
}

template<class Master_matrix, class Cell_constructor>
inline void Intrusive_set_column<Master_matrix,Cell_constructor>::_insert_cell(
		const Field_element_type &value, id_index rowIndex, const iterator &position)
{
	if constexpr (Master_matrix::Option_list::has_row_access){
		Cell *new_cell = cellPool_->construct(ra_opt::columnIndex_, rowIndex);
		new_cell->set_element(value);
		column_.insert(position, *new_cell);
		ra_opt::insert_cell(rowIndex, new_cell);
	} else {
		Cell *new_cell = cellPool_->construct(rowIndex);
		new_cell->set_element(value);
		column_.insert(position, *new_cell);
	}
}

template<class Master_matrix, class Cell_constructor>
inline void Intrusive_set_column<Master_matrix,Cell_constructor>::_insert_cell(
		id_index rowIndex, const iterator &position)
{
	if constexpr (Master_matrix::Option_list::has_row_access){
		Cell *new_cell = cellPool_->construct(ra_opt::columnIndex_, rowIndex);
		column_.insert(position, *new_cell);
		ra_opt::insert_cell(rowIndex, new_cell);
	} else {
		Cell *new_cell = cellPool_->construct(rowIndex);
		column_.insert(position, *new_cell);
	}
}

template<class Master_matrix, class Cell_constructor>
template<class Cell_range>
inline bool Intrusive_set_column<Master_matrix,Cell_constructor>::_add(const Cell_range &column)
{
	bool pivotIsZeroed = false;

	for (const Cell &cell : column) {
		auto it1 = column_.find(cell);
		if (it1 != column_.end()) {
			if constexpr (Master_matrix::Option_list::is_z2){
				if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
					if (it1->get_row_index() == chain_opt::get_pivot()) pivotIsZeroed = true;
				}
				_delete_cell(it1);
			} else {
				it1->get_element() = operators_->add(it1->get_element(), cell.get_element());
				if (it1->get_element() == Field_operators::get_additive_identity()){
					if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
						if (it1->get_row_index() == chain_opt::get_pivot()) pivotIsZeroed = true;
					}
					_delete_cell(it1);
				} else {
					if constexpr (Master_matrix::Option_list::has_row_access)
						ra_opt::update_cell(*it1);
				}
			}
		} else {
			if constexpr (Master_matrix::Option_list::is_z2){
				_insert_cell(cell.get_row_index(), column_.end());
			} else {
				_insert_cell(cell.get_element(), cell.get_row_index(), column_.end());
			}
		}
	}

	return pivotIsZeroed;
}

template<class Master_matrix, class Cell_constructor>
template<class Cell_range>
inline bool Intrusive_set_column<Master_matrix,Cell_constructor>::_multiply_and_add(const Field_element_type& val, const Cell_range& column)
{
	bool pivotIsZeroed = false;

	if (val == 0u) {
		if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
			throw std::invalid_argument("A chain column should not be multiplied by 0.");
			//this would not only mess up the base, but also the pivots stored.
		} else {
			clear();
		}
	}

	//cannot use usual addition method, because all values of column_ have to be multiplied

	auto itTarget = column_.begin();
	auto itSource = column.begin();
	while (itTarget != column_.end() && itSource != column.end())
	{
		if (itTarget->get_row_index() < itSource->get_row_index()) {
			itTarget->get_element() = operators_->multiply(itTarget->get_element(), val);
			if constexpr (Master_matrix::Option_list::has_row_access)
				ra_opt::update_cell(*itTarget);
			++itTarget;
		} else if (itTarget->get_row_index() > itSource->get_row_index()) {
			_insert_cell(itSource->get_element(), itSource->get_row_index(), itTarget);
			++itSource;
		} else {
			itTarget->get_element() = operators_->multiply_and_add(itTarget->get_element(), val, itSource->get_element());
			if (itTarget->get_element() == Field_operators::get_additive_identity()){
				if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
					if (itTarget->get_row_index() == chain_opt::get_pivot()) pivotIsZeroed = true;
				}
				_delete_cell(itTarget);
			} else {
				if constexpr (Master_matrix::Option_list::has_row_access)
					ra_opt::update_cell(*itTarget);
				++itTarget;
			}
			++itSource;
		}
	}

	while (itTarget != column_.end()){
		itTarget->get_element() = operators_->multiply(itTarget->get_element(), val);
		if constexpr (Master_matrix::Option_list::has_row_access)
			ra_opt::update_cell(*itTarget);
		itTarget++;
	}

	while (itSource != column.end()) {
		_insert_cell(itSource->get_element(), itSource->get_row_index(), column_.end());
		++itSource;
	}

	return pivotIsZeroed;
}

template<class Master_matrix, class Cell_constructor>
template<class Cell_range>
inline bool Intrusive_set_column<Master_matrix,Cell_constructor>::_multiply_and_add(const Cell_range& column, const Field_element_type& val)
{
	if (val == 0u) {
		return false;
	}

	bool pivotIsZeroed = false;

	for (const Cell &cell : column) {
		auto it1 = column_.find(cell);
		if (it1 != column_.end()) {
			it1->get_element() = operators_->multiply_and_add(cell.get_element(), val, it1->get_element());
			if (it1->get_element() == Field_operators::get_additive_identity()){
				if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type){
					if (it1->get_row_index() == chain_opt::get_pivot()) pivotIsZeroed = true;
				}
				_delete_cell(it1);
			} else {
				if constexpr (Master_matrix::Option_list::has_row_access)
					ra_opt::update_cell(*it1);
			}
		} else {
			_insert_cell(operators_->multiply(cell.get_element(), val), cell.get_row_index(), column_.end());
		}
	}

	return pivotIsZeroed;
}

} //namespace persistence_matrix
} //namespace Gudhi

template<class Master_matrix, class Cell_constructor>
struct std::hash<Gudhi::persistence_matrix::Intrusive_set_column<Master_matrix,Cell_constructor> >
{
	size_t operator()(const Gudhi::persistence_matrix::Intrusive_set_column<Master_matrix,Cell_constructor>& column) const
	{
		std::size_t seed = 0;
		for (const auto& cell : column){
			seed ^= std::hash<unsigned int>()(cell.get_row_index() * static_cast<unsigned int>(cell.get_element())) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

#endif // PM_INTRUSIVE_SET_COLUMN_H
