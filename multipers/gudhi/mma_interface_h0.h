
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

#ifndef MMA_INTERFACE_H0_H
#define MMA_INTERFACE_H0_H

#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>

#include <naive_merge_tree.h>

namespace Gudhi {
namespace multiparameter {
namespace truc_interface {

template <class Boundary_matrix_type>
class Persistence_backend_h0 {
 public:
  using pos_index = int;
  using dimension_type = int;
  using cycle_type = std::vector<unsigned int>;
  static const bool is_vine = true;

  std::vector<cycle_type> get_representative_cycles([[maybe_unused]] bool update = true) const { throw "Unimplemented"; }

  class Barcode_iterator : public boost::iterator_facade<Barcode_iterator, const Bar &, boost::forward_traversal_tag> {
   public:
    Barcode_iterator(const std::vector<Bar> *barcode, const std::vector<std::size_t> *inv)
        : barcode_(barcode->size() == 0 ? nullptr : barcode), perm_(barcode->size() == 0 ? nullptr : inv), currPos_(0) {
      if (barcode->size() != 0) {
        auto &b = barcode_->operator[](currPos_);
        currBar_.dim = b.dim;
        currBar_.birth = perm_->operator[](b.birth);
        currBar_.death = b.death == -1 ? -1 : perm_->operator[](b.death);
      }
    }

    Barcode_iterator() : barcode_(nullptr), perm_(nullptr), currPos_(0) {}

   private:
    // mandatory for the boost::iterator_facade inheritance.
    friend class boost::iterator_core_access;

    const std::vector<Bar> *barcode_;
    const std::vector<std::size_t> *perm_;
    std::size_t currPos_;
    Bar currBar_;

    bool equal(Barcode_iterator const &other) const {
      return barcode_ == other.barcode_ && perm_ == other.perm_ && currPos_ == other.currPos_;
    }

    const Bar &dereference() const { return currBar_; }

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

    Barcode(Naive_merge_forest &mf, const std::vector<std::size_t> *perm) : barcode_(&mf.get_barcode()), perm_(perm) {}

    iterator begin() const { return Barcode_iterator(barcode_, perm_); }

    iterator end() const { return Barcode_iterator(); }

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
    const std::vector<Bar> *barcode_;
    const std::vector<std::size_t> *perm_;
  };

  Persistence_backend_h0() {}

  Persistence_backend_h0(const Boundary_matrix_type &boundaries, std::vector<std::size_t> &permutation)
      : pers_(boundaries.size(), boundaries.num_vertices()),
        boundaries_(&boundaries),
        permutation_(&permutation),
        permutationInv_(permutation_->size()) {
    unsigned int c = 0;
    for (std::size_t i : *permutation_) {
      if (i == static_cast<std::size_t>(-1)) {
        c++;
        continue;
      }
      permutationInv_[i] = c;
      if (boundaries.dimension(i) == 0) {
        pers_.add_vertex(c++);
      } else if (boundaries.dimension(i) == 1) {
        pers_.add_edge(c++, permutationInv_[boundaries[i][0]], permutationInv_[boundaries[i][1]]);
      } else {
        std::cout << "Simplex of dimension > 1 got ignored.\n";
      }
    }
    pers_.initialize();
  }

  Persistence_backend_h0(const Persistence_backend_h0 &toCopy)
      : pers_(toCopy.pers_),
        boundaries_(toCopy.boundaries_),
        permutation_(toCopy.permutation_),
        permutationInv_(toCopy.permutationInv_) {}

  Persistence_backend_h0(Persistence_backend_h0 &&other) noexcept
      : pers_(std::move(other.pers_)),
        boundaries_(std::exchange(other.boundaries_, nullptr)),
        permutation_(std::exchange(other.permutation_, nullptr)),
        permutationInv_(std::move(other.permutationInv_)) {}

  Persistence_backend_h0 &operator=(Persistence_backend_h0 other) {
    swap(pers_, other.pers_);
    std::swap(boundaries_, other.boundaries_);
    std::swap(permutation_, other.permutation_);
    permutationInv_.swap(other.permutationInv_);
    return *this;
  }

  friend void swap(Persistence_backend_h0 &be1, Persistence_backend_h0 &be2) {
    swap(be1.pers_, be2.pers_);
    std::swap(be1.boundaries_, be2.boundaries_);
    std::swap(be1.permutation_, be2.permutation_);
    be1.permutationInv_.swap(be2.permutationInv_);
  }

  dimension_type get_dimension(pos_index i) { return pers_.get_dimension(i); }

  void vine_swap(pos_index i) {
    if (pers_.get_dimension(i) == 0) {
      if (pers_.get_dimension(i + 1) == 1) {
        const auto &boundary = boundaries_->operator[](permutation_->operator[](i + 1));
        pers_.vertex_edge_swap(i, permutationInv_[boundary[0]], permutationInv_[boundary[1]]);
        std::swap(permutationInv_[permutation_->operator[](i)], permutationInv_[permutation_->operator[](i + 1)]);
        return;
      }
      pers_.vertex_swap(i);
      std::swap(permutationInv_[permutation_->operator[](i)], permutationInv_[permutation_->operator[](i + 1)]);
      return;
    }

    if (pers_.get_dimension(i + 1) == 1) {
      const auto &boundary1 = boundaries_->operator[](permutation_->operator[](i));
      const auto &boundary2 = boundaries_->operator[](permutation_->operator[](i + 1));
      pers_.edge_edge_swap(i,
                           permutationInv_[boundary1[0]],
                           permutationInv_[boundary1[1]],
                           permutationInv_[boundary2[0]],
                           permutationInv_[boundary2[1]]);
      std::swap(permutationInv_[permutation_->operator[](i)], permutationInv_[permutation_->operator[](i + 1)]);
      return;
    }

    const auto &boundary = boundaries_->operator[](permutation_->operator[](i));
    pers_.edge_vertex_swap(i, permutationInv_[boundary[0]], permutationInv_[boundary[1]]);
    std::swap(permutationInv_[permutation_->operator[](i)], permutationInv_[permutation_->operator[](i + 1)]);
  }

  Barcode get_barcode() { return Barcode(pers_, permutation_); }

  std::size_t size() {
    throw;  // TODO:
  }

  inline friend std::ostream &operator<<(std::ostream &stream, Persistence_backend_h0 &structure) {
    stream << structure.pers_;
    stream << std::endl;
    return stream;
  }

  inline void _update_permutation_ptr(std::vector<std::size_t> &perm) { permutation_ = &perm; }

 private:
  Naive_merge_forest pers_;
  const Boundary_matrix_type *boundaries_;
  std::vector<std::size_t> *permutation_;
  std::vector<std::size_t> permutationInv_;
};
}  // namespace truc_interface
}  // namespace multiparameter
}  // namespace Gudhi
#endif  // MMA_INTERFACE_H0_H
