/*    This file is part of the MMA Library -
 * https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT. See
 * file LICENSE for full license details. Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef RU_MATRIX_H
#define RU_MATRIX_H

#include <vector>

#include "utilities.h" //type definitions
#include "vector_matrix.h"

namespace Gudhi::multiparameter::mma {

template <class Column_type> class RU_matrix {
public:
  RU_matrix();
  RU_matrix(boundary_matrix &orderedBoundaries);
  RU_matrix(int numberOfColumns);
  RU_matrix(RU_matrix &matrixToCopy);
  RU_matrix(RU_matrix &&other) noexcept;

  void insert_boundary(index columnIndex, boundary_type &boundary);
  dimension_type get_dimension(index index);
  unsigned int get_number_of_simplices();

  void initialize();
  void vine_swap(index index);
  const barcode_type &get_current_barcode();

  void print_matrices(); // for debug

  RU_matrix<Column_type> &operator=(RU_matrix<Column_type> other);
  template <class Friend_column_type>
  friend void swap(RU_matrix<Column_type> &matrix1,
                   RU_matrix<Friend_column_type> &matrix2);

private:
  Vector_matrix<Column_type> reducedMatrixR_;
  Vector_matrix<Column_type> mirrorMatrixU_;
  barcode_type barcode_;
  std::vector<int> indexToBar_;

  void _initialize_U();
  void _swap_at_index(index index);
  void _add_to(index sourceIndex, index targetIndex);
  void _positive_vine_swap(index index);
  void _negative_vine_swap(index index);
  void _positive_negative_vine_swap(index index);
  void _negative_positive_vine_swap(index index);
};

template <class Column_type> inline RU_matrix<Column_type>::RU_matrix() {}

template <class Column_type>
inline RU_matrix<Column_type>::RU_matrix(boundary_matrix &orderedBoundaries)
    : reducedMatrixR_(orderedBoundaries),
      mirrorMatrixU_(orderedBoundaries.size()) {
  _initialize_U();
}

template <class Column_type>
inline RU_matrix<Column_type>::RU_matrix(int numberOfColumns)
    : reducedMatrixR_(numberOfColumns), mirrorMatrixU_(numberOfColumns) {
  _initialize_U();
}

template <class Column_type>
inline RU_matrix<Column_type>::RU_matrix(RU_matrix &matrixToCopy)
    : reducedMatrixR_(matrixToCopy.reducedMatrixR_),
      mirrorMatrixU_(matrixToCopy.mirrorMatrixU_),
      barcode_(matrixToCopy.barcode_), indexToBar_(matrixToCopy.indexToBar_) {}

template <class Column_type>
inline RU_matrix<Column_type>::RU_matrix(
    RU_matrix<Column_type> &&other) noexcept
    : reducedMatrixR_(std::move(other.reducedMatrixR_)),
      mirrorMatrixU_(std::move(other.mirrorMatrixU_)),
      barcode_(std::move(other.barcode_)),
      indexToBar_(std::move(other.indexToBar_)) {}

template <class Column_type>
inline void RU_matrix<Column_type>::insert_boundary(index columnIndex,
                                                    boundary_type &boundary) {
  reducedMatrixR_.insert_boundary(columnIndex, boundary);
  boundary_type id(1, columnIndex);
  mirrorMatrixU_.insert_column(columnIndex, Column_type(id));
}

template <class Column_type>
inline dimension_type RU_matrix<Column_type>::get_dimension(index index) {
  return reducedMatrixR_.get_column_dimension(index);
}

template <class Column_type>
inline unsigned int RU_matrix<Column_type>::get_number_of_simplices() {
  return reducedMatrixR_.get_number_of_columns();
}

template <class Column_type> inline void RU_matrix<Column_type>::initialize() {
  std::unordered_map<index, index> pivotsToColumn;
  indexToBar_.resize(reducedMatrixR_.get_number_of_columns(), -1);

  for (unsigned int i = 0; i < reducedMatrixR_.get_number_of_columns(); i++) {
    if (!(reducedMatrixR_.is_zero_column(i))) {
      Column_type &curr = reducedMatrixR_.get_column(i);
      int pivot = curr.get_pivot();

      while (pivot != -1 &&
             pivotsToColumn.find(pivot) != pivotsToColumn.end()) {
        curr.add(reducedMatrixR_.get_column(pivotsToColumn.at(pivot)));
        mirrorMatrixU_.get_column(pivotsToColumn.at(pivot))
            .add(mirrorMatrixU_.get_column(i));
        pivot = curr.get_pivot();
      }

      if (pivot != -1) {
        pivotsToColumn.emplace(pivot, i);
        barcode_.at(indexToBar_.at(pivot)).death = i;
        indexToBar_.at(i) = indexToBar_.at(pivot);
      } else {
        barcode_.push_back(Bar(get_dimension(i), i, -1));
        indexToBar_.at(i) = barcode_.size() - 1;
      }
    } else {
      barcode_.push_back(Bar(get_dimension(i), i, -1));
      indexToBar_.at(i) = barcode_.size() - 1;
    }
  }
}

template <class Column_type>
inline void RU_matrix<Column_type>::vine_swap(index index) {
  if (index >= reducedMatrixR_.get_number_of_columns() - 1)
    return;

  bool iIsPositive =
      (barcode_.at(indexToBar_.at(index)).birth == static_cast<int>(index));
  bool iiIsPositive = (barcode_.at(indexToBar_.at(index + 1)).birth ==
                       static_cast<int>(index) + 1);

  if (iIsPositive && iiIsPositive)
    _positive_vine_swap(index);
  else if (!iIsPositive && !iiIsPositive)
    _negative_vine_swap(index);
  else if (iIsPositive && !iiIsPositive)
    _positive_negative_vine_swap(index);
  else
    _negative_positive_vine_swap(index);
}

template <class Column_type>
inline const barcode_type &RU_matrix<Column_type>::get_current_barcode() {
  return barcode_;
}

template <class Column_type>
inline void RU_matrix<Column_type>::print_matrices() {
  boundary_type b;

  std::cout << "R:\n";
  for (unsigned int i = 0; i < reducedMatrixR_.get_number_of_columns(); i++) {
    reducedMatrixR_.get_boundary(i, b);
    if (b.empty()) {
      std::cout << "-\n";
    } else {
      for (unsigned int i : b)
        std::cout << i << " ";
      std::cout << "\n";
      b.clear();
    }
  }
  std::cout << "\n";

  std::cout << "U:\n";
  for (unsigned int i = 0; i < mirrorMatrixU_.get_number_of_columns(); i++) {
    mirrorMatrixU_.get_boundary(i, b);
    if (b.empty()) {
      std::cout << "-\n";
    } else {
      for (unsigned int i : b)
        std::cout << i << " ";
      std::cout << "\n";
      b.clear();
    }
  }
  std::cout << "\n";
}

template <class Column_type>
inline RU_matrix<Column_type> &
RU_matrix<Column_type>::operator=(RU_matrix<Column_type> other) {
  std::swap(reducedMatrixR_, other.reducedMatrixR_);
  std::swap(mirrorMatrixU_, other.mirrorMatrixU_);
  std::swap(barcode_, other.barcode_);
  std::swap(indexToBar_, other.indexToBar_);
  return *this;
}

template <class Column_type>
inline void RU_matrix<Column_type>::_initialize_U() {
  boundary_type id(1);
  for (unsigned int i = 0; i < reducedMatrixR_.get_number_of_columns(); i++) {
    id.at(0) = i;
    mirrorMatrixU_.insert_column(i, Column_type(id));
  }
}

template <class Column_type>
inline void RU_matrix<Column_type>::_swap_at_index(index index) {
  reducedMatrixR_.swap_at_indices(index, index + 1);
  mirrorMatrixU_.swap_at_indices(index, index + 1);
}

template <class Column_type>
inline void RU_matrix<Column_type>::_add_to(index sourceIndex,
                                            index targetIndex) {
  reducedMatrixR_.add_to(sourceIndex, targetIndex);
  mirrorMatrixU_.add_to(targetIndex, sourceIndex);
}

template <class Column_type>
inline void RU_matrix<Column_type>::_positive_vine_swap(index index) {
  int iDeath = barcode_.at(indexToBar_.at(index)).death;
  int iiDeath = barcode_.at(indexToBar_.at(index + 1)).death;

  if (get_dimension(index) == get_dimension(index + 1)) {
    if (!mirrorMatrixU_.is_zero_cell(index, index + 1))
      mirrorMatrixU_.zero_cell(index, index + 1);

    if (iDeath != -1 && iiDeath != -1 &&
        !(reducedMatrixR_.is_zero_cell(iiDeath, index))) {
      if (iDeath < iiDeath) {
        _swap_at_index(index);
        _add_to(iDeath, iiDeath);

        barcode_.at(indexToBar_.at(index)).birth = index + 1;
        barcode_.at(indexToBar_.at(index + 1)).birth = index;
        std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));

        return;
      }

      if (iiDeath < iDeath) {
        _swap_at_index(index);
        _add_to(iiDeath, iDeath);

        return;
      }
    }

    _swap_at_index(index);

    if (iDeath != -1 || iiDeath == -1 ||
        reducedMatrixR_.is_zero_cell(iiDeath, index + 1)) {
      barcode_.at(indexToBar_.at(index)).birth = index + 1;
      barcode_.at(indexToBar_.at(index + 1)).birth = index;
      std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));
    }

    return;
  }

  _swap_at_index(index);

  barcode_.at(indexToBar_.at(index)).birth = index + 1;
  barcode_.at(indexToBar_.at(index + 1)).birth = index;
  std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));
}

template <class Column_type>
inline void RU_matrix<Column_type>::_negative_vine_swap(index index) {
  if (get_dimension(index) == get_dimension(index + 1) &&
      !mirrorMatrixU_.is_zero_cell(index, index + 1)) {
    _add_to(index, index + 1);
    _swap_at_index(index);

    if (barcode_.at(indexToBar_.at(index)).birth <
        barcode_.at(indexToBar_.at(index + 1)).birth) {
      barcode_.at(indexToBar_.at(index)).death = index + 1;
      barcode_.at(indexToBar_.at(index + 1)).death = index;
      std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));

      return;
    }

    _add_to(index, index + 1);

    return;
  }

  _swap_at_index(index);

  barcode_.at(indexToBar_.at(index)).death = index + 1;
  barcode_.at(indexToBar_.at(index + 1)).death = index;
  std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));
}

template <class Column_type>
inline void RU_matrix<Column_type>::_positive_negative_vine_swap(index index) {
  if (get_dimension(index) == get_dimension(index + 1) &&
      !mirrorMatrixU_.is_zero_cell(index, index + 1))
    mirrorMatrixU_.zero_cell(index, index + 1);

  _swap_at_index(index);

  barcode_.at(indexToBar_.at(index)).birth = index + 1;
  barcode_.at(indexToBar_.at(index + 1)).death = index;
  std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));
}

template <class Column_type>
inline void RU_matrix<Column_type>::_negative_positive_vine_swap(index index) {
  if (get_dimension(index) == get_dimension(index + 1) &&
      !mirrorMatrixU_.is_zero_cell(index, index + 1)) {
    _add_to(index, index + 1);
    _swap_at_index(index);
    _add_to(index, index + 1);

    return;
  }

  _swap_at_index(index);

  barcode_.at(indexToBar_.at(index)).death = index + 1;
  barcode_.at(indexToBar_.at(index + 1)).birth = index;
  std::swap(indexToBar_.at(index), indexToBar_.at(index + 1));
}

template <class Column_type>
inline void swap(RU_matrix<Column_type> &matrix1,
                 RU_matrix<Column_type> &matrix2) {
  std::swap(matrix1.reducedMatrixR_, matrix2.reducedMatrixR_);
  std::swap(matrix1.mirrorMatrixU_, matrix2.mirrorMatrixU_);
  matrix1.barcode_.swap(matrix2.barcode_);
  matrix1.indexToBar_.swap(matrix2.indexToBar_);
}

} // namespace Gudhi::multiparameter::mma

#endif // RU_MATRIX_H
