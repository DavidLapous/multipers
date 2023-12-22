/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which
 * is released under MIT. See file LICENSE or go to
 * https://gudhi.inria.fr/licensing/ for full license details. Author(s): Hannah
 * Schreiber
 *
 *    Copyright (C) 2022 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */
/**
 * @file options.h
 * @author Hannah Schreiber
 * @brief Interface of the matrix for MMA
 */

#ifndef MMA_INTERFACE_MATRIX_H
#define MMA_INTERFACE_MATRIX_H

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include <gudhi/Fields/Z2_field.h>
#include <gudhi/matrix.h>
#include <gudhi/persistence_matrix_options.h>

namespace Gudhi::multiparameter::interface {

template <Gudhi::persistence_matrix::Column_types column_type =
              Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct Multi_persistence_options
    : Gudhi::persistence_matrix::Default_options<
          true, Gudhi::persistence_matrix::Z2_field_element, column_type,
          false> {
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = true;
};

template <Gudhi::persistence_matrix::Column_types column_type =
              Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct Multi_persistence_Clement_options
    : Gudhi::persistence_matrix::Default_options<
          true, Gudhi::persistence_matrix::Z2_field_element, column_type,
          false> {
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = true;
  static const bool is_of_boundary_type = false;
  static const bool is_indexed_by_position =
      true; // useless if has_vine_update = false, as the two indexing
            // strategies only differ when swaps occur.
};

template <Gudhi::persistence_matrix::Column_types column_type =
              Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct No_vine_multi_persistence_options
    : Gudhi::persistence_matrix::Default_options<
          true, Gudhi::persistence_matrix::Z2_field_element, column_type,
          false> {
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = false;
};

template <class Matrix_options, class Boundary_matrix_type>
class Persistence_backend_matrix {
private:
  using matrix_type = Gudhi::persistence_matrix::Matrix<Matrix_options>;

public:
  using bar = typename matrix_type::Bar;

  class Barcode_iterator
      : public boost::iterator_facade<Barcode_iterator, const bar &,
                                      boost::forward_traversal_tag> {
  public:
    Barcode_iterator(const typename matrix_type::barcode_type *barcode,
                     const std::vector<std::size_t> *inv)
        : barcode_(barcode->size() == 0 ? nullptr : barcode),
          perm_(barcode->size() == 0 ? nullptr : inv), currPos_(0) {
      auto &b = barcode_->operator[](currPos_);
      currBar_.dim = b.dim;
      currBar_.birth = perm_->operator[](b.birth);
      currBar_.death = b.death == -1 ? -1 : perm_->operator[](b.death);
    }

    Barcode_iterator() : barcode_(nullptr), perm_(nullptr), currPos_(0) {}

  private:
    // mandatory for the boost::iterator_facade inheritance.
    friend class boost::iterator_core_access;

    const typename matrix_type::barcode_type *barcode_;
    const std::vector<std::size_t> *perm_;
    std::size_t currPos_;
    bar currBar_;

    bool equal(Barcode_iterator const &other) const {
      return barcode_ == other.barcode_ && perm_ == other.perm_ &&
             currPos_ == other.currPos_;
    }

    const bar &dereference() const { return currBar_; }

    void increment() {
      ++currPos_;
      if (currPos_ == barcode_->size()) {
        barcode_ = nullptr;
        perm_ = nullptr;
        currPos_ = 0;
      } else {
        auto &b = barcode_->operator[](currPos_);
        currBar_.dim = b.dim;
        currBar_.birth = perm_->operator[](b.birth);
        currBar_.death = b.death == -1 ? -1 : perm_->operator[](b.death);
      }
    }
  };
  class Barcode {
  public:
    using iterator = Barcode_iterator;
    Barcode(matrix_type &matrix, const std::vector<std::size_t> *perm)
        : barcode_(&matrix.get_current_barcode()), perm_(perm) {}

    iterator begin() const { return Barcode_iterator(barcode_, perm_); }

    iterator end() const { return Barcode_iterator(); }

    std::size_t size() const { return barcode_->size(); }

    inline friend std::ostream &operator<<(std::ostream &stream,
                                           Barcode &structure) {
      stream << "Barcode: " << structure.size() << "\n";
      for (const auto bar : structure) {
        stream << "[" << bar.dim << "] ";
        stream << bar.birth << ", " << bar.death;
        stream << "\n";
      }
      stream << "\n";
      return stream;
    }

  private:
    const typename matrix_type::barcode_type *barcode_;
    const std::vector<std::size_t> *perm_;
  };

  Persistence_backend_matrix() : permutation_(nullptr){};
  Persistence_backend_matrix(Boundary_matrix_type &boundaries,
                             std::vector<std::size_t> &permutation)
      : matrix_(boundaries.size()), permutation_(&permutation) {
    const bool verbose = false;
    if constexpr (verbose)
      std::cout << "Constructing matrix..." << std::endl;
    std::vector<std::size_t> permutationInv(permutation.size());
    std::vector<std::size_t> boundary_container;
    std::size_t c = 0;
    for (std::size_t i : permutation) {
      permutationInv[i] = c++;
      boundary_container.resize(boundaries[i].size());
      if constexpr (verbose)
        std::cout << i << "/" << permutation.size() << " dimension "
                  << boundaries.dimension(i) << "..." << std::endl
                  << std::flush;
      for (std::size_t j = 0; j < boundaries[i].size(); ++j) {
        boundary_container[j] = permutationInv[boundaries[i][j]];
      }
      matrix_.insert_boundary(boundary_container, boundaries.dimension(i));
    }
  }
  Persistence_backend_matrix(const Persistence_backend_matrix &toCopy)
      : matrix_(toCopy.matrix_), permutation_(toCopy.permutation_) {}
  Persistence_backend_matrix(Persistence_backend_matrix &&other) noexcept
      : matrix_(std::move(other.matrix_)),
        permutation_(std::exchange(other.permutation_, nullptr)) {}

  Persistence_backend_matrix &operator=(Persistence_backend_matrix other) {
    swap(matrix_, other.matrix_);
    std::swap(permutation_, other.permutation_);
    return *this;
  }

  friend void swap(Persistence_backend_matrix &be1,
                   Persistence_backend_matrix &be2) {
    swap(be1.matrix_, be2.matrix_);
    std::swap(be1.permutation_, be2.permutation_);
    be1.permutationInv_.swap(be2.permutationInv_);
  }

  int get_dimension(int i) { return matrix_.get_column_dimension(i); }

  void vine_swap(int i) { matrix_.vine_swap(i); }

  Barcode get_barcode() { return Barcode(matrix_, permutation_); }

  inline friend std::ostream &
  operator<<(std::ostream &stream, Persistence_backend_matrix &structure) {
    stream << "[\n";
    for (auto i = 0u; i < structure.matrix_.get_number_of_columns(); i++) {
      stream << "[";
      for (const auto &stuff : structure.matrix_.get_column(i))
        stream << stuff << ", ";
      stream << "]\n";
    }

    stream << "]\n";
    return stream;
  }

private:
  matrix_type matrix_;
  std::vector<std::size_t> *permutation_;
};

} // namespace Gudhi::multiparameter::interface
#endif // MMA_INTERFACE_MATRIX_H
