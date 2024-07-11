#pragma once
#include "multiparameter_module_approximation/format_python-cpp.h"
#include <algorithm>
#include <boost/mpl/aux_/na_fwd.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
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

// using Filtration_value =
//     multi_filtrations::Finitely_critical_multi_filtration<float>;

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
    max_dimension_ = generator_dimensions.size() > 0
                         ? *std::max_element(generator_dimensions.begin(),
                                             generator_dimensions.end())
                         : 0;
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
        stream << "\b" << "\b ";

      stream << "},\n";
    }
    stream << "}\n";
    stream << "Degrees: (max " << structure.max_dimension() << ")\n";
    stream << "{";
    for (const auto &stuff : structure.generator_dimensions)
      stream << stuff << ", ";
    if (structure.size() > 0) {
      stream << "\b" << "\b";
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
  unsigned int max_dimension() const { return max_dimension_; }
  int prune_above_dimension(int dim) {
    int idx = std::lower_bound(generator_dimensions.begin(),
                               generator_dimensions.end(), dim + 1) -
              generator_dimensions.begin();
    generators.resize(idx);
    generator_dimensions.resize(idx);
    max_dimension_ =
        generator_dimensions.size() ? generator_dimensions.back() : -1;
    return idx;
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
    auto [filtration, boundary] =
        Gudhi::multiparameter::mma::simplextree_to_ordered_bf(st);
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
        stream << "\b" << "\b ";

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
  inline unsigned int max_dimension() const { return max_dimension_; }

  int prune_above_dimension(int dim) { throw; }

private:
  std::vector<std::vector<unsigned int>> boundaries;
  unsigned int num_vertices_;
  unsigned int max_dimension_;
};

template <class PersBackend, class Structure, class MultiFiltration>
class Truc {
public:
  using Filtration_value = MultiFiltration;
  using value_type = typename MultiFiltration::value_type;
  using split_barcode =
      std::vector<std::vector<std::pair<typename MultiFiltration::value_type,
                                        typename MultiFiltration::value_type>>>;
  template <typename value_type = value_type>
  using flat_barcode =
      std::vector<std::pair<int, std::pair<value_type, value_type>>>;

  template <typename value_type = value_type>
  using flat_nodim_barcode = std::vector<std::pair<value_type, value_type>>;
  // CONSTRUCTORS.
  //  - Need everything of the same size, generator order is a PERMUTATION
  //
  Truc(const Structure &structure,
       const std::vector<MultiFiltration> &generator_filtration_values)
      : generator_filtration_values(generator_filtration_values),
        generator_order(structure.size()), structure(structure),
        filtration_container(structure.size()) {
    std::iota(generator_order.begin(), generator_order.end(), 0); // range
  };
  template <class SimplexTree> Truc(SimplexTree *simplextree) {
    auto [filtration, boundary] = mma::simplextree_to_ordered_bf(*simplextree);
    structure = SimplicialStructure(boundary, (*simplextree).num_vertices(),
                                    (*simplextree).dimension());
    generator_filtration_values.resize(filtration.size());
    for (auto i = 0u; i < filtration.size(); i++)
      generator_filtration_values[i] =
          filtration[i]; // there is a copy here. TODO : deal with it.
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
    persistence._update_permutation_ptr(generator_order);
    return *this;
  };

  inline std::size_t num_generators() const { return structure.size(); }
  inline std::size_t num_parameters() const {
    return num_generators() == 0
               ? 0
               : this->generator_filtration_values[0].num_parameters();
  }

  template <class SubFiltration, bool original_order = true>
  inline void push_to_out(
      const SubFiltration &f,
      std::vector<typename MultiFiltration::value_type> &filtration_container,
      const std::vector<std::size_t> &generator_order) const {

    /* std::vector<NewFilrationType> out(this->num_generators()); */

    // filtration_container.resize(
    //     this->num_generators()); // for some reasons it is necessary FIXME
    for (std::size_t i = 0u; i < this->num_generators(); i++) {
      if constexpr (original_order) {
        filtration_container[i] =
            f.push_forward2(generator_filtration_values[i]);
      } else {
        filtration_container[i] =
            f.push_forward2(generator_filtration_values[generator_order[i]]);
      }
    }
  }
  template <class SubFiltration, bool original_order = true>
  inline void push_to(const SubFiltration &f) {
    this->push_to_out<SubFiltration, original_order>(
        f, this->filtration_container, this->generator_order);
  }
  template <class array1d> inline void set_one_filtration(const array1d &truc) {
    assert(truc.size() == this->num_generators());
    this->filtration_container = truc;
  }
  inline const std::vector<typename MultiFiltration::value_type> &
  get_one_filtration() const {
    return this->filtration_container;
  }

  inline PersBackend compute_persistence_out(
      const std::vector<typename MultiFiltration::value_type> &one_filtration,
      std::vector<std::size_t>
          &out_gen_order) { // needed ftm as PersBackend only points there

    if (one_filtration.size() != this->num_generators()) {
      throw;
    }
    out_gen_order.resize(this->num_generators());
    std::iota(out_gen_order.begin(), out_gen_order.end(),
              0); // we have to reset here, even though we're already doing this
    std::sort(out_gen_order.begin(), out_gen_order.end(),
              [&](std::size_t i, std::size_t j) {
                if (structure.dimension(i) > structure.dimension(j))
                  return false;
                if (structure.dimension(i) < structure.dimension(j))
                  return true;
                return one_filtration[i] < one_filtration[j];
              });
    if constexpr (!PersBackend::is_vine) {
      for (std::size_t &i : out_gen_order)
        if (one_filtration[i] == MultiFiltration::T_inf) {
          // i = static_cast<std::size_t>(-1); // max
          i = std::numeric_limits<std::size_t>::max();
        }
    }
    if constexpr (false) {
      std::cout << "[";
      for (auto i : out_gen_order) {

        std::cout << i << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "[";
      for (auto i : one_filtration) {

        std::cout << i << ",";
      }
      std::cout << "]" << std::endl;
    }
    return PersBackend(
        structure,
        out_gen_order); // FIXME : PersBackend is not const on struct
  }

  inline void compute_persistence() {
    this->persistence = this->compute_persistence_out(
        this->filtration_container, this->generator_order);
  };

  // TODO : static ?
  inline void vineyard_update(
      PersBackend &persistence,
      const std::vector<typename MultiFiltration::value_type> &one_filtration,
      std::vector<std::size_t> &generator_order) const {
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
  inline void vineyard_update() {
    vineyard_update(this->persistence, this->filtration_container,
                    this->generator_order);
  }
  inline split_barcode
  get_barcode(PersBackend &persistence,
              const std::vector<typename MultiFiltration::value_type>
                  &filtration_container) {
    auto barcode_indices = persistence.get_barcode();
    split_barcode out(this->structure.max_dimension() +
                      1); // TODO : This doesn't allow for negative dimensions
    const bool verbose = false;
    const bool debug = false;
    auto inf = MultiFiltration::T_inf;
    for (const auto &bar : barcode_indices) {
      if constexpr (verbose)
        std::cout << "BAR : " << bar.birth << " " << bar.death << "\n";
      if constexpr (debug) {
        if (bar.birth >= filtration_container.size() || bar.birth < 0) {
          std::cout << "Trying to add an incompatible birth... ";
          std::cout << bar.birth << std::endl;
          std::cout << "Death is " << bar.death << std::endl;
          std::cout << "Max size is " << filtration_container.size()
                    << std::endl;
          continue;
        }
        if (bar.dim > static_cast<int>(this->structure.max_dimension())) {
          std::cout << "Incompatible dimension detected... " << bar.dim
                    << std::endl;
          std::cout << "While max dim is " << this->structure.max_dimension()
                    << std::endl;
          continue;
        }
      }

      auto birth_filtration = filtration_container[bar.birth];
      auto death_filtration = inf;
      if (bar.death != static_cast<typename PersBackend::pos_index>(-1))
        death_filtration = filtration_container[bar.death];

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
  inline split_barcode get_barcode() {
    return get_barcode(this->persistence, this->filtration_container);
  }

  template <typename value_type = value_type>
  static inline flat_nodim_barcode<value_type> get_flat_nodim_barcode(
      PersBackend &persistence,
      std::vector<typename MultiFiltration::value_type> &filtration_container) {
    const bool verbose = false;
    const auto &barcode_indices = persistence.get_barcode();
    auto num_bars = barcode_indices.size();
    flat_nodim_barcode<value_type> out(num_bars);
    if (num_bars <= 0)
      return out;
    auto idx = 0u;
    value_type inf = std::numeric_limits<value_type>::has_infinity
                         ? std::numeric_limits<value_type>::infinity()
                         : std::numeric_limits<value_type>::max();
    for (const auto &bar : barcode_indices) {
      value_type birth_filtration = inf;
      value_type death_filtration = -birth_filtration;
      if (bar.death == static_cast<typename PersBackend::pos_index>(-1))
        death_filtration = inf;
      else
        death_filtration =
            static_cast<value_type>(filtration_container[bar.death]);
      birth_filtration =
          static_cast<value_type>(filtration_container[bar.birth]);
      if constexpr (verbose) {
        std::cout << "PAIRING : " << bar.birth << " / " << bar.death << " dim "
                  << bar.dim << std::endl;
      }
      if constexpr (verbose) {
        std::cout << "PAIRING filtration : " << birth_filtration << " "
                  << death_filtration << " dim " << bar.dim << std::endl;
      }

      if (birth_filtration < death_filtration)
        out[idx] = {birth_filtration, death_filtration};
      else {
        out[idx] = {inf, inf};
      }
      idx++;
    }
    return out;
  }
  template <typename value_type = value_type>
  static inline flat_barcode<value_type> get_flat_barcode(
      PersBackend &persistence,
      std::vector<typename MultiFiltration::value_type> &filtration_container) {
    const bool verbose = false;
    const auto &barcode_indices = persistence.get_barcode();
    auto num_bars = barcode_indices.size();
    flat_barcode<value_type> out(num_bars);
    if (num_bars <= 0)
      return out;
    auto idx = 0u;
    value_type inf = std::numeric_limits<value_type>::has_infinity
                         ? std::numeric_limits<value_type>::infinity()
                         : std::numeric_limits<value_type>::max();
    for (const auto &bar : barcode_indices) {
      value_type birth_filtration = inf;
      value_type death_filtration = -birth_filtration;
      if (bar.death == static_cast<typename PersBackend::pos_index>(-1))
        death_filtration = inf;
      else
        death_filtration =
            static_cast<value_type>(filtration_container[bar.death]);
      birth_filtration =
          static_cast<value_type>(filtration_container[bar.birth]);
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
  template <typename value_type = value_type>
  inline flat_barcode<value_type> get_flat_barcode() {
    return get_flat_barcode(this->persistence, this->filtration_container);
  }
  template <typename value_type = value_type>
  inline flat_nodim_barcode<value_type> get_flat_nodim_barcode() {
    return get_flat_nodim_barcode(this->persistence,
                                  this->filtration_container);
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
    stream << "\b" << "\b";
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
  inline std::pair<typename MultiFiltration::OneCritical,
                   typename MultiFiltration::OneCritical>
  get_bounding_box() const {
    using OC = typename MultiFiltration::OneCritical;
    // assert(!generator_filtration_values.empty());
    OC a = OC::inf();
    OC b = -1 * a;
    for (const auto &filtration_value : generator_filtration_values) {
      if constexpr (MultiFiltration::is_multi_critical) {
        a.pull_to(filtration_value.factorize_below());
        b.push_to(filtration_value.factorize_above());
      } else {
        a.pull_to(filtration_value);
        b.push_to(filtration_value);
      }
    }
    return {a, b};
  }
  std::vector<typename MultiFiltration::OneCritical>
  get_filtration_values() const {
    if constexpr (MultiFiltration::is_multi_critical) {
      std::vector<typename MultiFiltration::OneCritical> out;
      out.reserve(generator_filtration_values
                      .size()); // at least this, will dooble later
      for (std::size_t i = 0; i < generator_filtration_values.size(); ++i) {
        for (const auto &f : generator_filtration_values[i]) {
          out.push_back(f);
        }
      }
      return out;
    } else {
      return generator_filtration_values; // copy not necessary for Onecritical
    } // (could return const&)
  }
  std::vector<MultiFiltration> &get_filtrations() {
    return generator_filtration_values;
  }
  const std::vector<MultiFiltration> &get_filtrations() const {
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

  inline void prune_above_dimension(int max_dim) {
    int idx = structure.prune_above_dimension(max_dim);
    generator_filtration_values.resize(idx);
    generator_order.resize(idx);
    filtration_container.resize(idx); 
  }

  const std::vector<std::vector<unsigned int>> get_boundaries() {
    std::size_t n = this->num_generators();
    std::vector<std::vector<unsigned int>> out(n);
    for (auto i = 0u; i < n; ++i) {
      out[i] = this->structure[i];
    }
    return out;
  }

  Truc<PersBackend, Structure, typename MultiFiltration::template Base<int32_t>>
  coarsen_on_grid(
      const std::vector<std::vector<typename MultiFiltration::value_type>>
          grid) {
    std::vector<typename MultiFiltration::template Base<int32_t>> coords(
        this->num_generators());
    for (std::size_t gen = 0u; gen < coords.size(); ++gen) {
      coords[gen] = generator_filtration_values[gen].coordinates_in_grid(grid);
    }
    return Truc<PersBackend, Structure,
                typename MultiFiltration::template Base<int32_t>>(structure,
                                                                  coords);
  }
  inline void coarsen_on_grid_inplace(
      const std::vector<std::vector<typename MultiFiltration::value_type>>
          &grid,
      bool coordinate = true) {
    for (auto gen = 0u; gen < this->num_generators(); ++gen) {
      generator_filtration_values[gen].coordinates_in_grid_inplace(grid,
                                                                   coordinate);
    }
  }

  // dim, num_cycle_of_dim, num_faces_in_cycle, vertices_in_face
  std::vector<std::vector<std::vector<std::vector<unsigned int>>>>
  get_representative_cycles(bool update = true,
                            const std::set<int> &dims = {}) {
    // iterable iterable simplex key
    auto cycles_key = persistence.get_representative_cycles(update);
    auto num_cycles = cycles_key.size();
    std::vector<std::vector<std::vector<std::vector<unsigned int>>>> out(
        structure.max_dimension() + 1);
    for (auto &cycles_of_dim : out)
      cycles_of_dim.reserve(num_cycles);
    for (auto &cycle : cycles_key) {
      if (structure.dimension(cycle[0]) == 0 &&
          (dims.size() == 0 || dims.contains(0))) {
        out[0].push_back({{static_cast<unsigned int>(cycle[0])}});
        continue;
      }
      int cycle_dim =
          structure.dimension(cycle[0]); // all faces have the same dim
      // if (!(dims.size() == 0 || dims.contains(cycle_dim)))
      //   continue;
      std::vector<std::vector<unsigned int>> cycle_faces(cycle.size());
      for (std::size_t i = 0; i < cycle_faces.size(); ++i) {
        const auto &B = structure[cycle[i]];
        cycle_faces[i].resize(B.size());
        for (std::size_t j = 0; j < B.size(); ++j) {
          cycle_faces[i][j] = static_cast<unsigned int>(B[j]);
        }
      }
      out[cycle_dim].push_back(cycle_faces);
    }
    return out;
  }
  const std::vector<std::size_t> get_current_order() const {
    return generator_order;
  }
  const PersBackend &get_persistence() const { return persistence; }
  PersBackend &get_persistence() { return persistence; }

  // TrucThread get_safe_thread() { return TrucThread(*this); }

  class TrucThread {

  public:
    using Filtration_value = MultiFiltration;
    using value_type = typename MultiFiltration::value_type;
    using ThreadSafe = TrucThread;
    inline TrucThread(const Truc &truc)
        : truc_ptr(&truc), generator_order(truc.get_current_order()),
          filtration_container(truc.get_one_filtration()),
          persistence(truc.get_persistence()) {
      persistence._update_permutation_ptr(generator_order);
    };
    inline TrucThread(const TrucThread &truc)
        : truc_ptr(truc.truc_ptr), generator_order(truc.get_current_order()),
          filtration_container(truc.get_one_filtration()),
          persistence(truc.get_persistence()) {
      persistence._update_permutation_ptr(generator_order);
    };
    inline TrucThread weak_copy() const { return TrucThread(*truc_ptr); }

    inline std::pair<MultiFiltration, MultiFiltration>
    get_bounding_box() const {
      return truc_ptr->get_bounding_box();
    }
    inline const std::vector<MultiFiltration> &get_filtration_values() const {
      return truc_ptr->get_filtration_values();
    }
    inline const std::vector<int> &get_dimensions() const {
      return truc_ptr->get_dimensions();
    }
    inline const std::vector<std::vector<unsigned int>> &
    get_boundaries() const {
      return truc_ptr->get_boundaries();
    }
    inline void coarsen_on_grid_inplace(
        const std::vector<std::vector<typename MultiFiltration::value_type>>
            &grid,
        bool coordinate = true) {
      truc_ptr->coarsen_on_grid_inplace(grid, coordinate);
    }
    template <typename Subfiltration>
    inline void push_to(const Subfiltration &f) {
      truc_ptr->push_to_out(f, this->filtration_container,
                            this->generator_order);
    }

    inline std::vector<typename PersBackend::cycle_type>
    get_representative_cycles(bool update = true) {
      return truc_ptr->get_representative_cycles(update);
    }
    inline void compute_persistence() {
      persistence = this->truc_ptr->compute_persistence_out(
          this->persistence, this->generator_order);
    }
    inline void vineyard_update() {
      truc_ptr->vineyard_update(this->persistence, this->filtration_container,
                                this->generator_order);
    }
    template <typename value_type = value_type>
    inline flat_barcode<value_type> get_flat_barcode() {
      return truc_ptr->get_flat_barcode(this->persistence,
                                        this->filtration_container);
    }
    template <typename value_type = value_type>
    inline flat_nodim_barcode<value_type> get_flat_nodim_barcode() {
      return truc_ptr->get_flat_nodim_barcode(this->persistence,
                                              this->filtration_container);
    }
    inline split_barcode get_barcode() {
      return truc_ptr->get_barcode(this->persistence,
                                   this->filtration_container);
    }

  private:
    const Truc *truc_ptr;
    std::vector<std::size_t>
        generator_order; // size fixed at construction time,
    std::vector<typename MultiFiltration::value_type>
        filtration_container; // filtration of the current slice
    PersBackend persistence; // generated by the structure, and generator_order.

  }; // class TrucThread

public:
  using ThreadSafe = TrucThread; // for outside

  TrucThread weak_copy() const { return TrucThread(*this); }

private:
  std::vector<MultiFiltration>
      generator_filtration_values; // defined at construction time. Const
  std::vector<std::size_t> generator_order; // size fixed at construction time,
  Structure structure; // defined at construction time. Const
  std::vector<typename MultiFiltration::value_type>
      filtration_container; // filtration of the current slice
  PersBackend persistence;  // generated by the structure, and generator_order.

}; // class Truc

} // namespace Gudhi::multiparameter::interface
