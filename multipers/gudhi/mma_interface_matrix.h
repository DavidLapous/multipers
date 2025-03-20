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
#include <cstdint>
#include <limits>
#include <ostream>
#include <utility>
#include <vector>

#include <gudhi/Matrix.h>
#include <gudhi/persistence_matrix_options.h>

namespace Gudhi {
namespace multiparameter {
namespace truc_interface {

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct Multi_persistence_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = true;
  static const bool can_retrieve_representative_cycles = true;
};

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct Multi_persistence_Clement_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = true;
  static const bool is_of_boundary_type = false;
  static const Gudhi::persistence_matrix::Column_indexation_types column_indexation_type =
      Gudhi::persistence_matrix::Column_indexation_types::POSITION;
  static const bool can_retrieve_representative_cycles = true;
};

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET>
struct No_vine_multi_persistence_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_matrix_maximal_dimension_access = false;
  static const bool has_column_pairings = true;
  static const bool has_vine_update = false;
};

template <Gudhi::persistence_matrix::Column_types column_type = Gudhi::persistence_matrix::Column_types::INTRUSIVE_SET, bool row_access = true>
struct fix_presentation_options : Gudhi::persistence_matrix::Default_options<column_type, true> {
  using Index = std::uint32_t;
  static const bool has_row_access = row_access;
  static const bool has_map_column_container = false;
  static const bool has_removable_columns = false;  // WARN : idx will change if map is not true
};

template <class Matrix_options, class Boundary_matrix_type>
class Persistence_backend_matrix {
 public:
  using matrix_type = Gudhi::persistence_matrix::Matrix<Matrix_options>;
  using options = Matrix_options;
  using cycle_type = typename matrix_type::Cycle;
  static const bool is_vine = Matrix_options::has_vine_update;

  using bar = typename matrix_type::Bar;
  //   using index = typename matrix_type::index;
  //   using id_index = typename matrix_type::id_index;
  using pos_index = typename matrix_type::Pos_index;
  using Index = typename matrix_type::Index;
  using dimension_type = typename matrix_type::Dimension;

  class Barcode_iterator : public boost::iterator_facade<Barcode_iterator, const bar &, boost::forward_traversal_tag> {
   public:
    Barcode_iterator(const typename matrix_type::Barcode *barcode, const std::vector<Index> *inv)
        : barcode_(barcode->size() == 0 ? nullptr : barcode), perm_(barcode->size() == 0 ? nullptr : inv), currPos_(0) {
      if (barcode_ != nullptr && perm_ != nullptr) {
        auto &b = barcode_->operator[](currPos_);
        currBar_.dim = b.dim;
        currBar_.birth = perm_->operator[](b.birth);
        currBar_.death = b.death == static_cast<pos_index>(-1) ? -1 : perm_->operator[](b.death);
      }
    }

    Barcode_iterator() : barcode_(nullptr), perm_(nullptr), currPos_(0) {}

   private:
    // mandatory for the boost::iterator_facade inheritance.
    friend class boost::iterator_core_access;

    const typename matrix_type::Barcode *barcode_;
    const std::vector<Index> *perm_;
    std::size_t currPos_;
    bar currBar_;

    bool equal(Barcode_iterator const &other) const {
      return barcode_ == other.barcode_ && perm_ == other.perm_ && currPos_ == other.currPos_;
    }

    const bar &dereference() const { return currBar_; }

    void increment() {
      constexpr const bool debug = false;
      ++currPos_;
      if (currPos_ == barcode_->size()) {
        barcode_ = nullptr;
        perm_ = nullptr;
        currPos_ = 0;
      } else {
        auto &b = barcode_->operator[](currPos_);
        currBar_.dim = b.dim;
        currBar_.birth = perm_->operator[](b.birth);
        if (debug && currBar_.birth > std::numeric_limits<decltype(currBar_.birth)>::max() / 2) {
          std::cout << currBar_ << std::endl;
          std::cout << "while " << b.birth;
          std::cout << "  " << perm_->size();
        }
        currBar_.death = b.death == static_cast<pos_index>(-1) ? -1 : perm_->operator[](b.death);
      }
    }
  };

  class Barcode {
   public:
    using iterator = Barcode_iterator;

    Barcode(matrix_type &matrix, const std::vector<Index> *perm) : barcode_(&matrix.get_current_barcode()) {
      const bool debug = false;
      if constexpr (Matrix_options::has_vine_update) {
        perm_ = perm;
      } else {
        perm_.reserve(perm->size());
        for (const auto &stuff : *perm)
          if (stuff < perm->size())  // number of generators
            perm_.push_back(stuff);
      }
      if constexpr (debug) {
        std::cout << "Built matrix of size " << matrix.get_number_of_columns() << " / " << perm->size() << std::endl;
      }
    }

    iterator begin() const {
      if constexpr (Matrix_options::has_vine_update) {
        return Barcode_iterator(barcode_, perm_);
      } else {
        return Barcode_iterator(barcode_, &this->perm_);
      }
    }

    iterator end() const { return Barcode_iterator(); }

    /* using bar = typename matrix_type::Bar; */
    /* const bar& operator[](std::size_t i){ */
    /*   return barcode_->at(); */
    /* } */
    std::size_t size() const { return barcode_->size(); }

    inline friend std::ostream &operator<<(std::ostream &stream, Barcode &structure) {
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
    const typename matrix_type::Barcode *barcode_;
    typename std::conditional<Matrix_options::has_vine_update, const std::vector<Index> *, std::vector<Index>>::type
        perm_;
  };

  Persistence_backend_matrix() : permutation_(nullptr) {};

  Persistence_backend_matrix(const Boundary_matrix_type &boundaries, std::vector<Index> &permutation)
      : matrix_(boundaries.size()), permutation_(&permutation) {
    static_assert(Matrix_options::is_of_boundary_type || Matrix_options::has_vine_update, "Clement implies vine.");
    constexpr const bool verbose = false;
    if constexpr (verbose) std::cout << "Constructing matrix..." << std::endl;
    std::vector<Index> permutationInv(permutation_->size());
    std::vector<Index> boundary_container;
    std::size_t c = 0;
    for (auto i : *permutation_) {
      if (i >= boundaries.size()) {
        c++;
        continue;
      }
      permutationInv[i] = c++;
      boundary_container.resize(boundaries[i].size());
      if constexpr (verbose)
        std::cout << i << "/" << permutation_->size() << " c= " << c << " dimension " << boundaries.dimension(i)
                  << "..." << std::endl
                  << std::flush;
      for (std::size_t j = 0; j < boundaries[i].size(); ++j) {
        boundary_container[j] = permutationInv[boundaries[i][j]];
      }
      std::sort(boundary_container.begin(), boundary_container.end());
      matrix_.insert_boundary(c - 1, boundary_container, boundaries.dimension(i));
    }
  }

  Persistence_backend_matrix(const Persistence_backend_matrix &toCopy)
      : matrix_(toCopy.matrix_), permutation_(toCopy.permutation_) {}

  Persistence_backend_matrix(Persistence_backend_matrix &&other) noexcept
      : matrix_(std::move(other.matrix_)), permutation_(std::exchange(other.permutation_, nullptr)) {}

  Persistence_backend_matrix &operator=(Persistence_backend_matrix other) {
    swap(matrix_, other.matrix_);
    std::swap(permutation_, other.permutation_);
    return *this;
  }

  friend void swap(Persistence_backend_matrix &be1, Persistence_backend_matrix &be2) {
    swap(be1.matrix_, be2.matrix_);
    std::swap(be1.permutation_, be2.permutation_);
  }

  inline dimension_type get_dimension(pos_index i) { return matrix_.get_column_dimension(i); }

  inline void vine_swap(pos_index i) { matrix_.vine_swap(i); }

  inline Barcode get_barcode() { return Barcode(matrix_, permutation_); }

  inline std::size_t size() const { return this->matrix_.get_number_of_columns(); }

  inline friend std::ostream &operator<<(std::ostream &stream, Persistence_backend_matrix &structure) {
    stream << "[\n";
    for (auto i = 0u; i < structure.matrix_.get_number_of_columns(); i++) {
      stream << "[";
      for (const auto &stuff : structure.matrix_.get_column(i)) stream << stuff << ", ";
      stream << "]\n";
    }

    stream << "]\n";
    return stream;
  }

  inline std::vector<std::vector<std::vector<unsigned int>>> get_representative_cycles(bool update,
                                                                                       bool detailed = false) {
    // Only used when vineyard, so shrunk permutation i.e.
    // without the -1, is permutation as we keep inf values (they can become
    // finite) cf barcode perm which is copied to remove the -1
    std::vector<unsigned int> permutation2;
    permutation2.reserve(permutation_->size());
    for (auto i : *permutation_) {
      if (i >= matrix_.get_number_of_columns()) {
        continue;
      }
      permutation2.push_back(i);
    }
    constexpr const bool verbose = false;
    if (update) [[likely]]
      matrix_.update_representative_cycles();
    std::vector<std::vector<std::vector<unsigned int>>> current_cycles =
        matrix_.get_representative_cycles_as_borders(detailed);
    for (auto &cycle : current_cycles) {
      if constexpr (verbose) std::cout << "Cycle (matrix_ order): ";
      for (auto &border : cycle) {
        for (auto &b : border) {
          b = permutation2[b];
        }
      }
    }
    return current_cycles;
  }

  inline void _update_permutation_ptr(std::vector<Index> &perm) { permutation_ = &perm; }

 private:
  matrix_type matrix_;
  std::vector<Index> *permutation_;
};

}  // namespace truc_interface
}  // namespace multiparameter
}  // namespace Gudhi
#endif  // MMA_INTERFACE_MATRIX_H
