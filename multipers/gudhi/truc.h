#pragma once
#include "gudhi/Simplex_tree/Simplex_tree_multi.h"
#include "multiparameter_module_approximation/format_python-cpp.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <oneapi/tbb/task_group.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Gudhi::multiparameter::interface {

using Filtration_value =
    multi_filtrations::Finitely_critical_multi_filtration<float>;

class PresentationStructure {
public:
  PresentationStructure() {}
  /* SimplicialStructure &operator=(const SimplicialStructure &) = default; */

  PresentationStructure(
      const std::vector<std::vector<unsigned int>> &generators,
      const std::vector<int> &generator_dimensions)
      : generators(generators), generator_dimensions(generator_dimensions),
        num_vertices_(0) {
    for (const auto &stuff : generator_dimensions) {
      if (stuff == 0)
        num_vertices_++;
    }
  };

  PresentationStructure(const PresentationStructure &other)
      : generators(other.generators),
        generator_dimensions(other.generator_dimensions),
        num_vertices_(other.num_vertices_),
        max_dimension_(other.max_dimension_) {}
  /* PresentationStructure &operator=(const PresentationStructure &other) { */
  /*   generators = other.generators; */
  /*   generator_dimensions = other.generator_dimensions; */
  /*   num_vertices_ = other.num_vertices_; */
  /*   max_dimension_ = other.max_dimension_; */
  /**/
  /*   return *this; */
  /* } */

  std::vector<unsigned int> operator[](std::size_t i) {
    return generators[i];
  } // needs to be iterable (begin, end, size)
  inline int dimension(std::size_t i) const { return generator_dimensions[i]; };
  inline friend std::ostream &operator<<(std::ostream &stream,
                                         PresentationStructure &structure) {
    stream << "Boundary:\n";
    stream << "{";
    for (const auto &stuff : structure.generators) {
      stream << "{";
      for (auto truc : stuff)
        stream << truc << ", ";

      if (!stuff.empty())
        stream << "\b"
               << "\b ";

      stream << "},\n";
    }
    stream << "}\n";
    stream << "Degrees: (max " << structure.max_dimension() << ")\n";
    stream << "{";
    for (const auto &stuff : structure.generator_dimensions)
      stream << stuff << ", ";
    if (structure.size() > 0) {
      stream << "\b"
             << "\b";
    }
    stream << "}\n";
    return stream;
  }
  inline void to_stream(std::ostream &stream,
                        const std::vector<std::size_t> &order) {
    for (const auto &i : order) {
      const auto &stuff = this->operator[](i);
      stream << i << " : [";
      for (const auto &truc : stuff)
        stream << truc << ", ";
      stream << "]\n";
    }
    /* return stream; */
  }
  inline std::size_t size() const { return generators.size(); };
  unsigned int num_vertices() const { return num_vertices_; };
  unsigned int max_dimension() {
    if (max_dimension_ < 0) {
      max_dimension_ = *std::max_element(generator_dimensions.begin(),
                                         generator_dimensions.end());
    }
    return max_dimension_;
  }

private:
  std::vector<std::vector<unsigned int>> generators;
  std::vector<int> generator_dimensions;
  unsigned int num_vertices_;
  int max_dimension_ = -1;
};

class SimplicialStructure {
public:
  template <typename SimplexTree> void from_simplextree(SimplexTree &st) {
    auto [boundary, filtration] = Gudhi::multiparameter::mma::st2bf(st);
    this->boundaries = boundary;
    this->num_vertices_ = st.num_vertices();
    this->max_dimension_ = st.dimension();
  }
  SimplicialStructure() {}
  /* SimplicialStructure &operator=(const SimplicialStructure &) = default; */

  SimplicialStructure(const std::vector<std::vector<unsigned int>> &boundaries,
                      unsigned int num_vertices, unsigned int max_dimension)
      : boundaries(boundaries), num_vertices_(num_vertices),
        max_dimension_(max_dimension){

        };

  std::vector<unsigned int> operator[](std::size_t i) {
    return boundaries[i];
  } // needs to be iterable (begin, end, size)
  int dimension(std::size_t i) const {
    return boundaries[i].size() == 0 ? 0 : boundaries[i].size() - 1;
  };
  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const SimplicialStructure &structure) {
    stream << "{";
    for (const auto &stuff : structure.boundaries) {
      stream << "{";
      for (auto truc : stuff)
        stream << truc << ", ";

      if (!stuff.empty())
        stream << "\b"
               << "\b ";

      stream << "},\n";
    }
    stream << "}\n";
    return stream;
  }
  inline void to_stream(std::ostream &stream,
                        const std::vector<std::size_t> &order) {
    for (const auto &i : order) {
      const auto &stuff = this->operator[](i);
      stream << i << " : [";
      for (const auto &truc : stuff)
        stream << truc << ", ";
      stream << "]\n";
    }
    /* return stream; */
  }
  inline std::size_t size() const { return boundaries.size(); };
  inline unsigned int num_vertices() { return num_vertices_; }
  inline unsigned int max_dimension() { return max_dimension_; }

private:
  std::vector<std::vector<unsigned int>> boundaries;
  unsigned int num_vertices_;
  unsigned int max_dimension_;
};

template <class PersBackend, class Structure,
          class MultiFiltration = Gudhi::multiparameter::multi_filtrations::
              Finitely_critical_multi_filtration<float>>
class Truc {
public:
  using Filtration_value = MultiFiltration;
  // CONSTRUCTORS.
  //  - Need everything of the same size, generator order is a PERMUTATION
  //
  Truc(const Structure &structure,
       const std::vector<MultiFiltration> &generator_filtration_values)
      : generator_filtration_values(generator_filtration_values),
        structure(structure), generator_order(structure.size()),
        filtration_container(structure.size()) {
    std::iota(generator_order.begin(), generator_order.end(), 0); // range
  };
  template <class SimplexTree> Truc(SimplexTree *simplextree) {
    auto [boundary, filtration] = mma::st2bf(*simplextree);
    structure = SimplicialStructure(boundary, (*simplextree).num_vertices(),
                                    (*simplextree).dimension());
    generator_filtration_values = filtration;
    generator_order = std::vector<std::size_t>(structure.size());
    std::iota(generator_order.begin(), generator_order.end(), 0); // range
    filtration_container.resize(structure.size());
  }

  Truc(const std::vector<std::vector<unsigned int>> &generator_maps,
       const std::vector<int> &generator_dimensions,
       const std::vector<MultiFiltration> &generator_filtrations)
      : generator_filtration_values(generator_filtrations),
        generator_order(generator_filtrations.size(), 0),
        structure(PresentationStructure(generator_maps, generator_dimensions)),
        filtration_container(generator_filtrations.size()) {
    std::iota(generator_order.begin(), generator_order.end(), 0); // range
  }

  Truc(){};

  Truc &operator=(const Truc &other) {
    generator_filtration_values = other.generator_filtration_values;
    generator_order = other.generator_order;
    structure = other.structure;
    persistence = other.persistence;
    filtration_container = other.filtration_container;
    return *this;
  };

  inline std::size_t num_generators() const { return structure.size(); }

  template <class SubFiltration, bool original_order = true>
  inline void push_to(const SubFiltration &f) {

    /* std::vector<NewFilrationType> out(this->num_generators()); */
    auto &out = this->filtration_container;

    out.resize(
        this->num_generators()); // for some reasons it is necessary FIXME
    for (auto i = 0u; i < out.size(); i++) {
      if constexpr (original_order) {
        out[i] = f.push_one_filtration(generator_filtration_values[i]);
      } else {
        out[i] = f.push_one_filtration(
            generator_filtration_values[generator_order[i]]);
      }
    }
  }
  template <class array1d> void set_one_filtration(const array1d &truc) {
    assert(truc.size() == this->num_generators());
    this->filtration_container = truc;
  }
  inline const std::vector<typename MultiFiltration::value_type> &
  get_one_filtration() const {
    return this->filtration_container;
  }

  void compute_persistence() {
    auto &f = this->filtration_container;
    if (f.size() != this->num_generators()) {
      std::cerr << "Filtration is of the wrong size. Not computing."
                << std::endl;
      return;
    }
    /* generator_order = std::vector<std::size_t>( */
    /*     this->num_generators()); // not necessary, as its size should be
     * defined */
    /*                              // at construction time */
    std::iota(generator_order.begin(), generator_order.end(),
              0); // we have to reset here, even though we're already doing this
                  // at construction time
    // We sort by dimension as this will imply less vine swaps
    std::sort(generator_order.begin(), generator_order.end(),
              [&](std::size_t i, std::size_t j) {
                if (structure.dimension(i) > structure.dimension(j))
                  return false;
                if (structure.dimension(i) < structure.dimension(j))
                  return true;
                return f[i] < f[j]
                    /*  || */
                    /* (f[i] == f[j] && */
                    /*  structure.dimension(i) < structure.dimension(j)) */
                    ;
              });
    persistence = PersBackend(structure, generator_order);
    /* persistence.initialize_persistence(); */
  };

  void vineyard_update() {
    auto &one_filtration = this->filtration_container;
    const bool verbose = false;
    /* static_assert(PersBackend::has_vine_update); */
    // the first false is to get the generator order
    // insertion sort
    auto n = this->num_generators();
    if constexpr (verbose) {
      std::cout << "Vine updates : ";
    }
    for (std::size_t i = 0; i < n; i++) {
      auto j = i;
      while (j > 0 &&
             persistence.get_dimension(j) == persistence.get_dimension(j - 1) &&
             one_filtration[generator_order[j]] <
                 one_filtration[generator_order[j - 1]]) {
        if constexpr (verbose) {
          std::cout << j - 1 << ", ";
        }
        persistence.vine_swap(j - 1);
        std::swap(generator_order[j - 1], generator_order[j]);
        j--;
      }
    }
    if constexpr (verbose) {
      std::cout << std::endl;
    }
  }
  using split_barcode =
      std::vector<std::vector<std::pair<typename MultiFiltration::value_type,
                                        typename MultiFiltration::value_type>>>;
  inline split_barcode get_barcode() {
    auto barcode_indices = this->persistence.get_barcode();
    split_barcode out(this->structure.max_dimension() + 1);
    const bool verbose = false;
    /* auto count = 0u; */
    for (const auto &bar : barcode_indices) {
      /* std::cout << "BAR : " << bar.birth << " " << bar.death << "\n"; */
      auto inf =
          std::numeric_limits<typename MultiFiltration::value_type>::infinity();

      auto birth_filtration = this->filtration_container[bar.birth];
      auto death_filtration = inf;
      if (bar.death != -1)
        death_filtration = this->filtration_container[bar.death];

      if constexpr (verbose) {
        std::cout << "BAR: " << bar.birth << "(" << birth_filtration << ")"
                  << " --" << bar.death << "(" << death_filtration << ")"
                  << " dim " << bar.dim << std::endl;
      }

      if (birth_filtration < death_filtration)
        out[bar.dim].push_back({birth_filtration, death_filtration});
      else {
        out[bar.dim].push_back({inf, inf});
      }
    }
    return out;
  }

  using flat_barcode =
      std::vector<std::pair<int, std::pair<value_type, value_type>>>;
  flat_barcode get_flat_barcode() {
    const bool verbose = false;
    const auto &barcode_indices = this->persistence.get_barcode();
    auto num_bars = barcode_indices.size();
    flat_barcode out(num_bars);
    if (num_bars <= 0)
      return out;
    auto idx = 0u;
    auto inf =
        std::numeric_limits<typename MultiFiltration::value_type>::infinity();
    for (const auto &bar : barcode_indices) {
      typename MultiFiltration::value_type birth_filtration = inf;
      auto death_filtration = -birth_filtration;
      if (bar.death == -1)
        death_filtration = inf;
      else
        death_filtration = this->filtration_container[bar.death];
      birth_filtration = this->filtration_container[bar.birth];
      if constexpr (verbose) {
        std::cout << "PAIRING : " << bar.birth << " / " << bar.death << " dim "
                  << bar.dim << std::endl;
      }
      if constexpr (verbose) {
        std::cout << "PAIRING filtration : " << birth_filtration << " "
                  << death_filtration << " dim " << bar.dim << std::endl;
      }

      if (birth_filtration < death_filtration)
        out[idx] = {bar.dim, {birth_filtration, death_filtration}};
      else {
        out[idx] = {bar.dim, {inf, inf}};
      }
      idx++;
    }
    return out;
  }

  inline friend std::ostream &operator<<(std::ostream &stream, Truc &truc) {
    stream << "-------------------- Truc \n";
    stream << "--- Structure \n";
    stream << truc.structure;
    /* stream << "-- Dimensions (max " << truc.structure.max_dimension() <<
     * ")\n"; */
    /* stream << "{"; */
    /* for (auto i = 0u; i < truc.num_generators(); i++) */
    /*   stream << truc.structure.dimension(i) << ", "; */
    /* stream << "\b" */
    /*        << "\b"; */
    /* stream << "}" << std::endl; */
    stream << "--- Order \n";
    stream << "{";
    for (const auto &idx : truc.generator_order)
      stream << idx << ", ";
    stream << "}" << std::endl;

    stream << "--- Current slice filtration\n";
    stream << "{";
    for (const auto &stuff : truc.filtration_container)
      stream << stuff << ", ";
    stream << "\b"
           << "\b";
    stream << "}" << std::endl;

    stream << "--- Filtrations \n";
    for (const auto &i : truc.generator_order) {
      stream << i << " : ";
      const auto &stuff = truc.generator_filtration_values[i];
      stream << stuff << "\n";
    }
    stream << "--- PersBackend \n";
    stream << truc.persistence;

    return stream;
  }
  inline std::string to_str() {
    std::stringstream stream;
    stream << *this;
    return stream.str();
  }
  inline std::pair<MultiFiltration, MultiFiltration> get_bounding_box() const {
    auto out = std::pair(generator_filtration_values[0],
                         generator_filtration_values[0]);
    auto &a = out.first;
    auto &b = out.second;
    for (const auto &filtration_value : generator_filtration_values) {
      a.pull_to(filtration_value);
      b.push_to(filtration_value);
    }
    return out;
  }
  const std::vector<MultiFiltration> &get_filtration_values() const {
    return generator_filtration_values;
  }
  const std::vector<int> get_dimensions() const {
    std::size_t n = this->num_generators();
    std::vector<int> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = structure.dimension(i);
    }
    return out;
  }
  const std::vector<std::vector<unsigned int>> get_boundaries() {
    std::size_t n = this->num_generators();
    std::vector<std::vector<unsigned int>> out(n);
    for (auto i = 0u; i < n; ++i) {
      out[i] = this->structure[i];
    }
    return out;
  }

private:
  std::vector<MultiFiltration>
      generator_filtration_values;          // defined at construction time
  std::vector<std::size_t> generator_order; // size fixed at construction time,
  Structure structure;                      // defined at construction time
  std::vector<typename MultiFiltration::value_type>
      filtration_container; // filtration of the current slice
  PersBackend persistence;  // generated by the structure, and generator_order.
};

} // namespace Gudhi::multiparameter::interface
