/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which
 * is released under MIT. See file LICENSE or go to
 * https://gudhi.inria.fr/licensing/ for full license details. Author(s): Hannah
 * Schreiber
 *
 *    Copyright (C) 2024 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */
/**
 * @file options.h
 * @author Hannah Schreiber
 * @brief Interface of the matrix for MMA
 */

#ifndef MMA_INTERFACE_COH_H
#define MMA_INTERFACE_COH_H

#include <cstddef>
#include <limits>
#include <ostream>
#include <utility>
#include <vector>

#include <gudhi/persistence_interval.h>
#include <gudhi/Persistent_cohomology.h>

namespace Gudhi {
namespace multiparameter {
namespace truc_interface {

template <class Structure>
class Boundary_matrix_as_filtered_complex_for_coh {
 public:
  using Simplex_key = std::uint32_t;
  using Simplex_handle = Simplex_key;
  using Filtration_value = Simplex_key;
  using Dimension = int;

  class Filtration_range_iterator
      : public boost::iterator_facade<Filtration_range_iterator, const Simplex_handle &, boost::forward_traversal_tag> {
   public:
    Filtration_range_iterator(const std::vector<Simplex_handle> *new_to_old_perm = nullptr)
        : new_to_old_perm_(new_to_old_perm), currPos_(0) {
      if (new_to_old_perm_ != nullptr) {
        find_next_valid_index();
      }
    }

   private:
    // mandatory for the boost::iterator_facade inheritance.
    friend class boost::iterator_core_access;

    const std::vector<Simplex_handle> *new_to_old_perm_;
    std::size_t currPos_;

    bool equal(Filtration_range_iterator const &other) const {
      return new_to_old_perm_ == other.new_to_old_perm_ && currPos_ == other.currPos_;
    }

    const Simplex_handle &dereference() const { return (*new_to_old_perm_)[currPos_]; }

    void increment() {
      ++currPos_;
      find_next_valid_index();
    }

    void set_to_end() {
      new_to_old_perm_ = nullptr;
      currPos_ = 0;
    }

    void find_next_valid_index() {
      while (currPos_ < new_to_old_perm_->size() && dereference() >= new_to_old_perm_->size()) ++currPos_;
      if (currPos_ == new_to_old_perm_->size()) set_to_end();
    }
  };

  class Filtration_simplex_range {
   public:
    using iterator = Filtration_range_iterator;

    Filtration_simplex_range(std::vector<Simplex_handle> const *new_to_old_perm)
        : new_to_old_perm_(new_to_old_perm), size_(-1) {}

    iterator begin() const { return iterator(new_to_old_perm_); }

    iterator end() const { return iterator(); }

    std::size_t size() {
      if (size_ == static_cast<std::size_t>(-1)) {
        size_ = 0;
        for (auto i : *new_to_old_perm_) {
          if (i < new_to_old_perm_->size()) ++size_;
        }
      }
      return size_;
    }

    inline friend std::ostream &operator<<(std::ostream &stream, const Filtration_simplex_range &range) {
      stream << "Filtration_simplex_range: ";
      for (const auto idx : range) {
        stream << idx << " ";
      }
      stream << "\n";
      return stream;
    }

   private:
    std::vector<Simplex_handle> const *new_to_old_perm_;
    std::size_t size_;
  };

  Boundary_matrix_as_filtered_complex_for_coh() : boundaries_(nullptr), new_to_old_perm_(nullptr) {}

  Boundary_matrix_as_filtered_complex_for_coh(const Structure &boundaries,
                                              const std::vector<Simplex_handle> &permutation)
      : boundaries_(&boundaries),
        new_to_old_perm_(&permutation),
        keys_(permutation.size(), -1),
        keysToSH_(permutation.size(), -1) {
    assert(permutation.size() == boundaries.size());
  }

  friend void swap(Boundary_matrix_as_filtered_complex_for_coh &be1, Boundary_matrix_as_filtered_complex_for_coh &be2) {
    std::swap(be1.boundaries_, be2.boundaries_);
    std::swap(be1.new_to_old_perm_, be2.new_to_old_perm_);
    be1.keys_.swap(be2.keys_);
    be1.keysToSH_.swap(be2.keysToSH_);
  }

  std::size_t num_simplices() const { return boundaries_->size(); }

  Filtration_value filtration(Simplex_handle sh) const {
    return sh == null_simplex() ? std::numeric_limits<Filtration_value>::max() : keys_[sh];
  }

  Dimension dimension() const { return boundaries_->max_dimension(); }

  Dimension dimension(Simplex_handle sh) const { return sh == null_simplex() ? -1 : boundaries_->dimension(sh); }

  void assign_key(Simplex_handle sh, Simplex_key key) {
    if (sh != null_simplex()) keys_[sh] = key;
    if (key != null_key()) keysToSH_[key] = sh;
  }

  Simplex_key key(Simplex_handle sh) const { return sh == null_simplex() ? null_key() : keys_[sh]; }

  static constexpr Simplex_key null_key() { return static_cast<Simplex_key>(-1); }

  Simplex_handle simplex(Simplex_key key) const { return key == null_key() ? null_simplex() : keysToSH_[key]; }

  static constexpr Simplex_handle null_simplex() { return static_cast<Simplex_handle>(-1); }

  std::pair<Simplex_handle, Simplex_handle> endpoints(Simplex_handle sh) const {
    if (sh == null_simplex()) return {null_simplex(), null_simplex()};
    assert(dimension(sh) == 1);
    const auto &col = (*boundaries_)[sh];
    assert(col.size() == 2);
    return {col[0], col[1]};
  }

  Filtration_simplex_range filtration_simplex_range() const { return Filtration_simplex_range(new_to_old_perm_); }

  const std::vector<Simplex_handle> &boundary_simplex_range(Simplex_handle sh) const { return (*boundaries_)[sh]; }

  void set_permutation(const std::vector<Simplex_handle> &permutation) { new_to_old_perm_ = &permutation; }

  friend std::ostream &operator<<(std::ostream &stream, const Boundary_matrix_as_filtered_complex_for_coh &structure) {
    std::vector<Simplex_key> inv(structure.new_to_old_perm_->size());
    for (unsigned int i = 0; i < structure.new_to_old_perm_->size(); ++i) {
      inv[(*structure.new_to_old_perm_)[i]] = i;
    }

    stream << "[\n";
    for (auto i : structure.filtration_simplex_range()) {
      stream << "[";
      for (const auto &stuff : structure.boundary_simplex_range(i)) stream << inv[stuff] << ", ";
      stream << "]\n";
    }

    stream << "]\n";
    return stream;
  }

 private:
  Structure const *boundaries_;
  std::vector<Simplex_handle> const *new_to_old_perm_;
  std::vector<Simplex_key> keys_;
  std::vector<Simplex_handle> keysToSH_;
};

template <class Boundary_matrix_type>
class Persistence_backend_cohomology {
 public:
  using MatrixComplex = Boundary_matrix_as_filtered_complex_for_coh<Boundary_matrix_type>;
  using pos_index = typename MatrixComplex::Simplex_key;
  using dimension_type = typename MatrixComplex::Dimension;
  using bar = Gudhi::persistence_matrix::Persistence_interval<dimension_type, typename MatrixComplex::Simplex_handle>;
  using Barcode = std::vector<bar>;
  using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
  using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<MatrixComplex, Field_Zp>;
  static const bool is_vine = false;
  using Index = typename MatrixComplex::Simplex_handle;
  using cycle_type = void;

  Persistence_backend_cohomology() {};

  Persistence_backend_cohomology(const Boundary_matrix_type &boundaries, std::vector<Index> &permutation)
      : matrix_(boundaries, permutation) {}

  friend void swap(Persistence_backend_cohomology &be1, Persistence_backend_cohomology &be2) {
    swap(be1.matrix_, be2.matrix_);
  }

  dimension_type get_dimension(pos_index i) { return matrix_.dimension(matrix_.simplex(i)); }

  Barcode get_barcode() {
    Persistent_cohomology pcoh(matrix_, true);
    pcoh.init_coefficients(2);
    pcoh.compute_persistent_cohomology(0, false);

    const auto &pairs = pcoh.get_persistent_pairs();
    Barcode barcode(pairs.size());

    unsigned int i = 0;
    for (const auto &p : pairs) {
      auto &b = barcode[i];
      b.dim = matrix_.dimension(get<0>(p));
      b.birth = get<0>(p);
      b.death = get<1>(p);
      ++i;
    }

    return barcode;
  }

  std::size_t size() { return matrix_.num_simplices(); }

  friend std::ostream &operator<<(std::ostream &stream, Persistence_backend_cohomology &structure) {
    stream << structure.matrix_ << "\n";
    return stream;
  }

  void _update_permutation_ptr(std::vector<Index> &perm) { matrix_.set_permutation(perm); }

 private:
  MatrixComplex matrix_;
};

}  // namespace truc_interface
}  // namespace multiparameter
}  // namespace Gudhi

#endif  // MMA_INTERFACE_COH_H
