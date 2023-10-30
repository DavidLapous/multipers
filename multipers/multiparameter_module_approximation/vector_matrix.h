/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef VECTOR_MATRIX_H
#define VECTOR_MATRIX_H

#include <utility>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <algorithm>

#include "utilities.h"  //type definitions

namespace Gudhi::multiparameter::mma {

template<class Column_type>
class Vector_matrix
{
public:
    Vector_matrix();
    Vector_matrix(std::vector<boundary_type>& orderedBoundaries);
    Vector_matrix(unsigned int numberOfColumns);
    Vector_matrix(Vector_matrix<Column_type>& matrixToCopy);
    Vector_matrix(Vector_matrix&& other) noexcept;

    void insert_boundary(index columnIndex, boundary_type& boundary);
    void get_boundary(index columnIndex, boundary_type& container);
    void insert_column(index columnIndex, Column_type column);
    Column_type& get_column(index columnIndex);

    void reduce(barcode_type& barcode);

    index get_last_index() const;
    dimension_type get_max_dim() const;
    unsigned int get_number_of_columns() const;

    void swap_columns(index columnIndex1, index columnIndex2);
    void swap_rows(index rowIndex1, index rowIndex2);
    void swap_at_indices(index index1, index index2);

    void zero_cell(index columnIndex, index rowIndex);
    void zero_column(index columnIndex);

    void add_to(index sourceColumnIndex, index targetColumnIndex);
    bool is_zero_cell(index columnIndex, index rowIndex) const;
    bool is_zero_column(index columnIndex);

    dimension_type get_column_dimension(index columnIndex) const;

    Vector_matrix<Column_type>& operator=(Vector_matrix<Column_type> other);
    template<class Friend_column_type>
    friend void swap(Vector_matrix<Friend_column_type>& matrix1,
                     Vector_matrix<Friend_column_type>& matrix2);

private:
    std::vector<Column_type> matrix_;
    dimension_type maxDim_;
    std::vector<unsigned int> indexToRow_;
    std::vector<index> rowToIndex_;
    bool rowSwapped_;

    void _clear_column(index columnIndex);
    void _orderRows();
};

template<class Column_type>
inline Vector_matrix<Column_type>::Vector_matrix() : maxDim_(0), rowSwapped_(false)
{}

template<class Column_type>
inline Vector_matrix<Column_type>::Vector_matrix(
        std::vector<boundary_type>& orderedBoundaries) : maxDim_(0), rowSwapped_(false)
{
    matrix_.resize(orderedBoundaries.size());
    indexToRow_.resize(orderedBoundaries.size());
    rowToIndex_.resize(orderedBoundaries.size());
    for (unsigned int i = 0; i < orderedBoundaries.size(); i++){
        boundary_type& b = orderedBoundaries.at(i);
        matrix_.at(i) = Column_type(b);
        if (maxDim_ < static_cast<int>(b.size()) - 1) maxDim_ = b.size() - 1;
        indexToRow_.at(i) = i;
        rowToIndex_.at(i) = i;
    }
    if (maxDim_ == -1) maxDim_ = 0;
}

template<class Column_type>
inline Vector_matrix<Column_type>::Vector_matrix(unsigned int numberOfColumns)
    : maxDim_(0), rowSwapped_(false)
{
    matrix_.resize(numberOfColumns);
    indexToRow_.resize(numberOfColumns);
    rowToIndex_.resize(numberOfColumns);
    for (unsigned int i = 0; i < numberOfColumns; i++){
        indexToRow_.at(i) = i;
        rowToIndex_.at(i) = i;
    }
}

template<class Column_type>
inline Vector_matrix<Column_type>::Vector_matrix(
        Vector_matrix<Column_type>& matrixToCopy) : maxDim_(0), rowSwapped_(false)
{
    matrix_.resize(matrixToCopy.get_number_of_columns());
    indexToRow_.resize(matrix_.size());
    rowToIndex_.resize(matrix_.size());
    for (unsigned int i = 0; i < matrix_.size(); i++){
        matrix_.at(i) = Column_type(matrixToCopy.get_column(i));
        indexToRow_.at(i) = i;
        rowToIndex_.at(i) = i;
    }
    maxDim_ = matrixToCopy.get_max_dim();
}

template<class Column_type>
inline Vector_matrix<Column_type>::Vector_matrix(Vector_matrix<Column_type> &&other) noexcept
    : matrix_(std::move(other.matrix_)),
      maxDim_(std::exchange(other.maxDim_, 0)),
      indexToRow_(std::move(other.indexToRow_)),
      rowToIndex_(std::move(other.rowToIndex_)),
      rowSwapped_(std::exchange(other.rowSwapped_, 0))
{}

template<class Column_type>
inline void Vector_matrix<Column_type>::insert_boundary(
        index columnIndex, boundary_type& boundary)
{
    if (rowSwapped_) _orderRows();

    if (matrix_.size() <= columnIndex) {
        for (int i = matrix_.size(); i <= columnIndex; i++){
            indexToRow_.push_back(i);
            rowToIndex_.push_back(i);
        }
        matrix_.resize(columnIndex + 1);
    }
    matrix_.at(columnIndex) = Column_type(boundary);
    if (maxDim_ < boundary.size() - 1) maxDim_ = boundary.size() - 1;
}

template<class Column_type>
inline void Vector_matrix<Column_type>::get_boundary(
        index columnIndex, boundary_type& container)
{
    matrix_.at(columnIndex).get_content(container);
	for (unsigned int& v : container) v = rowToIndex_.at(v);
	std::sort(container.begin(), container.end());
}

template<class Column_type>
inline void Vector_matrix<Column_type>::insert_column(
        index columnIndex, Column_type column)
{
    if (rowSwapped_) _orderRows();

    dimension_type dim = column.get_dimension();
    if (matrix_.size() <= columnIndex) {
        for (unsigned int i = matrix_.size(); i <= columnIndex; i++){
            indexToRow_.push_back(i);
            rowToIndex_.push_back(i);
        }
        matrix_.resize(columnIndex + 1);
    }
    std::swap(matrix_.at(columnIndex), column);
    if (maxDim_ < dim) maxDim_ = dim;
}

template<class Column_type>
inline Column_type& Vector_matrix<Column_type>::get_column(index columnIndex)
{
    if (rowSwapped_) _orderRows();
    return matrix_.at(columnIndex);
}

template<class Column_type>
inline void Vector_matrix<Column_type>::reduce(barcode_type& barcode)
{
    if (rowSwapped_) _orderRows();
    std::unordered_map<index, index> pivotsToColumn;

    for (int d = maxDim_; d > 0; d--){
        for (unsigned int i = 0; i < matrix_.size(); i++){
            if (!(matrix_.at(i).is_empty()) && matrix_.at(i).get_dimension() == d)
            {
                Column_type &curr = matrix_.at(i);
                int pivot = curr.get_pivot();

                while (pivot != -1 && pivotsToColumn.find(pivot) != pivotsToColumn.end()){
                    curr.add(matrix_.at(pivotsToColumn.at(pivot)));
                    pivot = curr.get_pivot();
                }

                if (pivot != -1){
                    pivotsToColumn.emplace(pivot, i);
                    _clear_column(pivot);
                    barcode.push_back(Bar(d - 1, pivot, i));
                } else {
                    _clear_column(i);
                }
            }
        }
    }
}

template<class Column_type>
inline index Vector_matrix<Column_type>::get_last_index() const
{
    return matrix_.size() - 1;
}

template<class Column_type>
inline dimension_type Vector_matrix<Column_type>::get_max_dim() const
{
    return maxDim_;
}

template<class Column_type>
inline unsigned int Vector_matrix<Column_type>::get_number_of_columns() const
{
    return matrix_.size();
}

template<class Column_type>
inline void Vector_matrix<Column_type>::swap_columns(index columnIndex1, index columnIndex2)
{
    std::swap(matrix_.at(columnIndex1), matrix_.at(columnIndex2));
}

template<class Column_type>
inline void Vector_matrix<Column_type>::swap_rows(index rowIndex1, index rowIndex2)
{
    rowSwapped_ = true;
    std::swap(rowToIndex_.at(indexToRow_.at(rowIndex1)), rowToIndex_.at(indexToRow_.at(rowIndex2)));
    std::swap(indexToRow_.at(rowIndex1), indexToRow_.at(rowIndex2));
}

template<class Column_type>
inline void Vector_matrix<Column_type>::swap_at_indices(index index1, index index2)
{
    swap_columns(index1, index2);
    swap_rows(index1, index2);
}

template<class Column_type>
inline void Vector_matrix<Column_type>::zero_cell(index columnIndex, index rowIndex)
{
    matrix_.at(columnIndex).clear(indexToRow_.at(rowIndex));
}

template<class Column_type>
inline void Vector_matrix<Column_type>::zero_column(index columnIndex)
{
    _clear_column(columnIndex);
}

template<class Column_type>
inline void Vector_matrix<Column_type>::add_to(index sourceColumnIndex, index targetColumnIndex)
{
    matrix_.at(targetColumnIndex).add(matrix_.at(sourceColumnIndex));
}

template<class Column_type>
inline bool Vector_matrix<Column_type>::is_zero_cell(index columnIndex, index rowIndex) const
{
    return !(matrix_.at(columnIndex).contains(indexToRow_.at(rowIndex)));
}

template<class Column_type>
inline bool Vector_matrix<Column_type>::is_zero_column(index columnIndex)
{
    return matrix_.at(columnIndex).is_empty();
}

template<class Column_type>
inline dimension_type Vector_matrix<Column_type>::get_column_dimension(index columnIndex) const
{
    return matrix_.at(columnIndex).get_dimension();
}

template<class Column_type>
inline Vector_matrix<Column_type> &Vector_matrix<Column_type>::operator=(
        Vector_matrix<Column_type> other)
{
    std::swap(matrix_, other.matrix_);
    std::swap(maxDim_, other.maxDim_);
    std::swap(indexToRow_, other.indexToRow_);
    std::swap(rowToIndex_, other.rowToIndex_);
    std::swap(rowSwapped_, other.rowSwapped_);
    return *this;
}

template<class Column_type>
inline void Vector_matrix<Column_type>::_clear_column(index columnIndex)
{
    matrix_.at(columnIndex).clear();
}

template<class Column_type>
inline void Vector_matrix<Column_type>::_orderRows()
{
    for (unsigned int i = 0; i < matrix_.size(); i++){
        matrix_.at(i).reorder(rowToIndex_);
    }
    for (unsigned int i = 0; i < matrix_.size(); i++){
        indexToRow_.at(i) = i;
        rowToIndex_.at(i) = i;
    }
    rowSwapped_ = false;
}

template<class Column_type>
inline void swap(Vector_matrix<Column_type>& matrix1, Vector_matrix<Column_type>& matrix2)
{
    std::swap(matrix1.matrix_, matrix2.matrix_);
    std::swap(matrix1.maxDim_, matrix2.maxDim_);
    std::swap(matrix1.indexToRow_, matrix2.indexToRow_);
    std::swap(matrix1.rowToIndex_, matrix2.rowToIndex_);
    std::swap(matrix1.rowSwapped_, matrix2.rowSwapped_);
}

}   //namespace Vineyard

#endif // VECTOR_MATRIX_H
