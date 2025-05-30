#pragma once
#include "gudhi/Matrix.h"
#include "gudhi/mma_interface_matrix.h"
#include "gudhi/Multi_persistence/Line.h"
#include "multiparameter_module_approximation/format_python-cpp.h"
#include <gudhi/One_critical_filtration.h>
#include <gudhi/Multi_critical_filtration.h>
#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/mutex.h>
#include <ostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>  //std::invoke_result
#include "scc_io.h"

namespace Gudhi {
namespace multiparameter {
namespace truc_interface {
using index_type = std::uint32_t;

template <typename T, typename = void>
struct has_columns : std::false_type {};

template <typename T>
struct has_columns<T, std::void_t<typename T::options>> : std::true_type {};

class PresentationStructure {
 public:
  PresentationStructure() {}

  /* SimplicialStructure &operator=(const SimplicialStructure &) = default; */

  PresentationStructure(const std::vector<std::vector<index_type>> &generators,
                        const std::vector<int> &generator_dimensions)
      : generators(generators), generator_dimensions(generator_dimensions), num_vertices_(0) {
    for (const auto &stuff : generator_dimensions) {
      if (stuff == 0) num_vertices_++;
    }
    max_dimension_ = generator_dimensions.size() > 0
                         ? *std::max_element(generator_dimensions.begin(), generator_dimensions.end())
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

  const std::vector<unsigned int> &operator[](std::size_t i) const {
    return generators[i];
  }  // needs to be iterable (begin, end, size)

  std::vector<unsigned int> &operator[](std::size_t i) {
    return generators[i];
  }  // needs to be iterable (begin, end, size)

  inline int dimension(std::size_t i) const { return generator_dimensions[i]; };

  inline friend std::ostream &operator<<(std::ostream &stream, const PresentationStructure &structure) {
    stream << "Boundary:\n";
    stream << "{\n";
    for (auto i : std::views::iota(0u, structure.size())) {
      const auto &stuff = structure.generators[i];
      stream << i << ": {";
      for (auto truc : stuff) stream << truc << ", ";

      if (!stuff.empty()) stream << "\b" << "\b ";

      stream << "},\n";
    }
    stream << "}\n";
    stream << "Degrees: (max " << structure.max_dimension() << ")\n";
    stream << "{";
    for (const auto &stuff : structure.generator_dimensions) stream << stuff << ", ";
    if (structure.size() > 0) {
      stream << "\b" << "\b";
    }
    stream << "}\n";
    return stream;
  }

  inline void to_stream(std::ostream &stream, const std::vector<index_type> &order) {
    for (const auto &i : order) {
      const auto &stuff = this->operator[](i);
      stream << i << " : [";
      for (const auto &truc : stuff) stream << truc << ", ";
      stream << "]\n";
    }
    /* return stream; */
  }

  inline std::size_t size() const { return generators.size(); };

  unsigned int num_vertices() const { return num_vertices_; };

  unsigned int max_dimension() const { return max_dimension_; }

  int prune_above_dimension(int dim) {
    int idx = std::lower_bound(generator_dimensions.begin(), generator_dimensions.end(), dim + 1) -
              generator_dimensions.begin();
    generators.resize(idx);
    generator_dimensions.resize(idx);
    max_dimension_ = generator_dimensions.size() ? generator_dimensions.back() : -1;
    return idx;
  }

  PresentationStructure permute(const std::vector<index_type> &order) const {
    if (order.size() > generators.size()) {
      throw std::invalid_argument("Permutation order must have the same size as the number of generators.");
    }
    index_type flag = -1;
    std::vector<index_type> inverse_order(generators.size(), flag);
    for (std::size_t i = 0; i < order.size(); i++) {
      inverse_order[order[i]] = i;
    }
    std::vector<std::vector<index_type>> new_generators(order.size());
    std::vector<int> new_generator_dimensions(order.size());

    for (auto i : std::views::iota(0u, order.size())) {
      new_generators[i].reserve(generators[order[i]].size());
      for (std::size_t j = 0; j < generators[order[i]].size(); j++) {
        index_type stuff = inverse_order[generators[order[i]][j]];
        if (stuff != flag) new_generators[i].push_back(stuff);
      }
      std::sort(new_generators[i].begin(), new_generators[i].end());
      new_generator_dimensions[i] = generator_dimensions[order[i]];
    }
    return PresentationStructure(new_generators, new_generator_dimensions);
  }

  void update_matrix(std::vector<std::vector<index_type>> &new_gens) { std::swap(generators, new_gens); }

 private:
  std::vector<std::vector<index_type>> generators;
  std::vector<int> generator_dimensions;
  unsigned int num_vertices_;
  int max_dimension_ = -1;
};

class SimplicialStructure {
 public:
  template <typename SimplexTree>
  void from_simplextree(SimplexTree &st) {
    auto [filtration, boundary] = Gudhi::multiparameter::mma::simplextree_to_ordered_bf(st);
    this->boundaries = boundary;
    this->num_vertices_ = st.num_vertices();
    this->max_dimension_ = st.dimension();
  }

  SimplicialStructure() {}

  /* SimplicialStructure &operator=(const SimplicialStructure &) = default; */

  SimplicialStructure(const std::vector<std::vector<index_type>> &boundaries,
                      unsigned int num_vertices,
                      unsigned int max_dimension)
      : boundaries(boundaries), num_vertices_(num_vertices), max_dimension_(max_dimension) {

        };

  const std::vector<unsigned int> &operator[](std::size_t i) const {
    return boundaries[i];
  }  // needs to be iterable (begin, end, size)

  std::vector<unsigned int> &operator[](std::size_t i) {
    return boundaries[i];
  }  // needs to be iterable (begin, end, size)

  int dimension(std::size_t i) const { return boundaries[i].size() == 0 ? 0 : boundaries[i].size() - 1; };

  inline friend std::ostream &operator<<(std::ostream &stream, const SimplicialStructure &structure) {
    stream << "{";
    for (const auto &stuff : structure.boundaries) {
      stream << "{";
      for (auto truc : stuff) stream << truc << ", ";

      if (!stuff.empty()) stream << "\b" << "\b ";

      stream << "},\n";
    }
    stream << "}\n";
    return stream;
  }

  inline void to_stream(std::ostream &stream, const std::vector<index_type> &order) {
    for (const auto &i : order) {
      const auto &stuff = this->operator[](i);
      stream << i << " : [";
      for (const auto &truc : stuff) stream << truc << ", ";
      stream << "]\n";
    }
    /* return stream; */
  }

  inline std::size_t size() const { return boundaries.size(); };

  inline unsigned int num_vertices() const { return num_vertices_; }

  inline unsigned int max_dimension() const { return max_dimension_; }

  int prune_above_dimension([[maybe_unused]] int dim) { throw "Not implemented"; }

 private:
  std::vector<std::vector<index_type>> boundaries;
  unsigned int num_vertices_;
  unsigned int max_dimension_;
};

template <class PersBackend, class Structure, class MultiFiltration>
class Truc {
 public:
  using Filtration_value = MultiFiltration;
  using MultiFiltrations = std::vector<MultiFiltration>;
  using value_type = typename MultiFiltration::value_type;
  using split_barcode =
      std::vector<std::vector<std::pair<typename MultiFiltration::value_type, typename MultiFiltration::value_type>>>;
  using split_barcode_idx = std::vector<std::vector<std::pair<int, int>>>;
  template <typename value_type = value_type>
  using flat_barcode = std::vector<std::pair<int, std::pair<value_type, value_type>>>;

  template <typename value_type = value_type>
  using flat_nodim_barcode = std::vector<std::pair<value_type, value_type>>;

  // CONSTRUCTORS.
  //  - Need everything of the same size, generator order is a PERMUTATION
  //
  Truc(const Structure &structure, const std::vector<MultiFiltration> &generator_filtration_values)
      : generator_filtration_values(generator_filtration_values),
        generator_order(structure.size()),
        structure(structure),
        filtration_container(structure.size()) {
    std::iota(generator_order.begin(), generator_order.end(), 0);  // range
  };

  template <class SimplexTree>
  Truc(SimplexTree *simplextree) {
    auto [filtration, boundary] = mma::simplextree_to_ordered_bf(*simplextree);
    structure = SimplicialStructure(boundary, (*simplextree).num_vertices(), (*simplextree).dimension());
    generator_filtration_values.resize(filtration.size());
    for (auto i = 0u; i < filtration.size(); i++)
      generator_filtration_values[i] = filtration[i];  // there is a copy here. TODO : deal with it.
    generator_order = std::vector<index_type>(structure.size());
    std::iota(generator_order.begin(), generator_order.end(), 0);  // range
    filtration_container.resize(structure.size());
  }

  Truc(const std::vector<std::vector<index_type>> &generator_maps,
       const std::vector<int> &generator_dimensions,
       const std::vector<MultiFiltration> &generator_filtrations)
      : generator_filtration_values(generator_filtrations),
        generator_order(generator_filtrations.size(), 0),
        structure(PresentationStructure(generator_maps, generator_dimensions)),
        filtration_container(generator_filtrations.size()) {
    std::iota(generator_order.begin(), generator_order.end(), 0);  // range
  }

  Truc(const Truc &other)
      : generator_filtration_values(other.generator_filtration_values),
        generator_order(other.generator_order),
        structure(other.structure),
        filtration_container(other.filtration_container),
        persistence(other.persistence) {
    persistence._update_permutation_ptr(generator_order);
  }

  Truc &operator=(Truc other) {
    if (this != &other) {
      generator_filtration_values = other.generator_filtration_values;
      generator_order = other.generator_order;
      structure = other.structure;
      filtration_container = other.filtration_container;
      persistence = other.persistence;
      persistence._update_permutation_ptr(generator_order);
    }
    return *this;
  }

  Truc() {};

  inline bool dimension_order(const index_type &i, const index_type &j) const {
    return structure.dimension(i) < structure.dimension(j);
  };

  inline bool colexical_order(const index_type &i, const index_type &j) const {
    if (structure.dimension(i) > structure.dimension(j)) return false;
    if (structure.dimension(i) < structure.dimension(j)) return true;
    if constexpr (MultiFiltration::is_multicritical())  // TODO : this may not be the best
      throw "Not implemented in the multicritical case";

    for (int idx = generator_filtration_values[i].num_parameters() - 1; idx >= 0; --idx) {
      if (generator_filtration_values[i][idx] < generator_filtration_values[j][idx])
        return true;
      else if (generator_filtration_values[i][idx] > generator_filtration_values[j][idx])
        return false;
    }
    return false;
  };

  // TODO : inside of MultiFiltration
  inline static bool lexical_order(const MultiFiltration &a, const MultiFiltration &b) {
    if constexpr (MultiFiltration::is_multicritical())  // TODO : this may not be the best
      throw "Not implemented in the multicritical case";
    if (a.is_plus_inf() || a.is_nan() || b.is_minus_inf()) return false;
    if (b.is_plus_inf() || b.is_nan() || a.is_minus_inf()) return true;
    for (auto idx = 0u; idx < a.num_parameters(); ++idx) {
      if (a[idx] < b[idx])
        return true;
      else if (a[idx] > b[idx])
        return false;
    }
    return false;
  };

  inline bool lexical_order(const index_type &i, const index_type &j) const {
    if (structure.dimension(i) > structure.dimension(j)) return false;
    if (structure.dimension(i) < structure.dimension(j)) return true;
    if constexpr (MultiFiltration::is_multicritical())  // TODO : this may not be the best
      throw "Not implemented in the multicritical case";

    for (int idx = 0; idx < generator_filtration_values[i].num_parameters(); ++idx) {
      if (generator_filtration_values[i][idx] < generator_filtration_values[j][idx])
        return true;
      else if (generator_filtration_values[i][idx] > generator_filtration_values[j][idx])
        return false;
    }
    return false;
  };

  inline Truc permute(const std::vector<index_type> &permutation) const {
    auto num_new_gen = permutation.size();
    if (permutation.size() > this->num_generators()) {
      throw std::invalid_argument("Invalid permutation size. Got " + std::to_string(num_new_gen) + " expected " +
                                  std::to_string(this->num_generators()) + ".");
    }
    std::vector<MultiFiltration> new_filtration(num_new_gen);
    for (auto i : std::views::iota(0u, num_new_gen)) {  // assumes permutation has correct indices.
      new_filtration[i] = generator_filtration_values[permutation[i]];
    }
    return Truc(structure.permute(permutation), new_filtration);
  }

  template <class Fun>
  inline std::pair<Truc, std::vector<index_type>> rearange_sort(const Fun &&fun) const {
    std::vector<index_type> permutation(generator_order.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    tbb::parallel_sort(permutation.begin(), permutation.end(), [&](std::size_t i, std::size_t j) { return fun(i, j); });
    return {permute(permutation), permutation};
  }

  std::pair<Truc, std::vector<index_type>> colexical_rearange() const {
    return rearange_sort([this](std::size_t i, std::size_t j) { return this->colexical_order(i, j); });
  }

  template <bool generator_only = false>
  std::conditional_t<generator_only, std::pair<std::vector<std::vector<index_type>>, MultiFiltrations>, Truc>
  projective_cover_kernel(int dim) {
    if constexpr (MultiFiltration::is_multicritical() || !std::is_same_v<Structure, PresentationStructure> ||
                  !has_columns<PersBackend>::value)  // TODO : this may not be the best
    {
      throw std::invalid_argument("Not implemented for this Truc");
    } else {
      // TODO : this only works for 2 parameter modules. Optimize w.r.t. this.
      const bool verbose = false;
      // filtration values are assumed to be dim + colexicographically sorted
      // vector seem to be good here
      using SmallMatrix = Gudhi::persistence_matrix::Matrix<
          Gudhi::multiparameter::truc_interface::fix_presentation_options<PersBackend::options::column_type, false>>;

      int nd = 0;
      int ndpp = 0;
      for (auto i : std::views::iota(0u, structure.size())) {
        if (structure.dimension(i) == dim) {
          nd++;
        } else if (structure.dimension(i) == dim + 1) {
          ndpp++;
        } else {
          throw std::invalid_argument("This truc contains bad dimensions. Got " +
                                      std::to_string(structure.dimension(i)) + " expected " + std::to_string(dim) +
                                      " or " + std::to_string(dim + 1) + " in position " + std::to_string(i) + "  .");
        }
      }
      if (ndpp == 0)
        throw std::invalid_argument("Given dimension+1 has no simplices. Got " + std::to_string(nd) + " " +
                                    std::to_string(ndpp) + ".");
      // lexico iterator
      auto lex_cmp = [&](const MultiFiltration &a, const MultiFiltration &b) { return lexical_order(a, b); };

      struct SmallQueue {
        SmallQueue() {};

        struct MFWrapper {
          MFWrapper(const MultiFiltration &g) : g(g) {};

          MFWrapper(const MultiFiltration &g, int col) : g(g) { some_cols.insert(col); }

          MFWrapper(const MultiFiltration &g, std::initializer_list<int> cols)
              : g(g), some_cols(cols.begin(), cols.end()) {}

          inline void insert(int col) const { some_cols.insert(col); }

          inline bool operator<(const MFWrapper &other) const { return lexical_order(g, other.g); }

         public:
          MultiFiltration g;
          mutable std::set<int> some_cols;
        };

        inline void insert(const MultiFiltration &g, int col) {
          auto it = queue.find(g);
          if (it != queue.end()) {
            it->insert(col);
          } else {
            queue.emplace(g, col);
          }
        };

        inline void insert(const MultiFiltration &g, const std::initializer_list<int> &cols) {
          auto it = queue.find(g);
          if (it != queue.end()) {
            for (int c : cols) it->insert(c);
          } else {
            queue.emplace(g, cols);
          }
        };

        inline bool empty() const { return queue.empty(); }

        inline MultiFiltration pop() {
          if (queue.empty()) [[unlikely]]
            throw std::runtime_error("Queue is empty");

          auto out = std::move(*queue.begin());
          queue.erase(queue.begin());
          std::swap(last_cols, out.some_cols);
          return out.g;
        }

        const auto &get_current_cols() const { return last_cols; }

       private:
        std::set<MFWrapper> queue;
        std::set<int> last_cols;
      };

      SmallQueue lexico_it;
      SmallMatrix M(nd + ndpp);
      for (int i = 0; i < nd + ndpp; i++) {
        const auto &b = structure[i];
        M.insert_boundary(b);
      }
      SmallMatrix N(nd + ndpp);  // slave
      for (auto i : std::views::iota(0u, static_cast<unsigned int>(nd + ndpp))) {
        N.insert_boundary({i});
      };

      auto get_fil = [&](int i) -> MultiFiltration & { return generator_filtration_values[i]; };
      auto get_pivot = [&](int j) -> int {
        const auto &col = M.get_column(j);
        return col.size() > 0 ? (*col.rbegin()).get_row_index() : -1;
      };

      if constexpr (verbose) {
        std::cout << "Initial matrix (" << nd << " + " << ndpp << "):" << std::endl;
        for (int i = 0; i < nd + ndpp; i++) {
          std::cout << "Column " << i << " : {";
          for (const auto &j : M.get_column(i)) std::cout << j << " ";
          std::cout << "} | " << get_fil(i) << std::endl;
        }
      }

      // TODO : pivot caches are small : maybe use a flat container instead ?
      std::vector<std::set<int>> pivot_cache(nd + ndpp);  // this[pivot] = cols of given pivot (<=nd)
      std::vector<bool> reduced_columns(nd + ndpp);       // small cache
      MultiFiltration grid_value;

      std::vector<std::vector<index_type>> out_structure;
      out_structure.reserve(2 * ndpp);
      std::vector<MultiFiltration> out_filtration;
      out_filtration.reserve(2 * ndpp);
      std::vector<int> out_dimension;
      out_dimension.reserve(2 * ndpp);
      if constexpr (!generator_only) {
        for (auto i : std::views::iota(nd, nd + ndpp)) {
          out_structure.push_back({});
          out_filtration.push_back(this->get_filtration_values()[i]);
          out_dimension.push_back(this->structure.dimension(i));
        }
      }
      // pivot cache
      if constexpr (verbose) {
        std::cout << "Initial pivot cache:\n";
      }
      for (int j : std::views::iota(nd, nd + ndpp)) {
        int col_pivot = get_pivot(j);
        if (col_pivot < 0) {
          reduced_columns[j] = true;
          continue;
        };
        auto &current_pivot_cache = pivot_cache[col_pivot];
        current_pivot_cache.emplace_hint(current_pivot_cache.cend(), j);  // j is increasing
      }
      if constexpr (verbose) {
        int i = 0;
        for (const auto &cols : pivot_cache) {
          std::cout << " - " << i++ << " : ";
          for (const auto &col : cols) {
            std::cout << col << " ";
          }
          std::cout << std::endl;
        }
      }

      // if constexpr (!use_grid) {
      if constexpr (verbose) std::cout << "Initial grid queue:\n";
      for (int j : std::views::iota(nd, nd + ndpp)) {
        int col_pivot = get_pivot(j);
        if (col_pivot < 0) continue;
        lexico_it.insert(get_fil(j), j);
        auto it = pivot_cache[col_pivot].find(j);
        if (it == pivot_cache[col_pivot].end()) [[unlikely]]
          throw std::runtime_error("Column " + std::to_string(j) + " not in pivot cache");
        it++;
        // for (int k : pivot_cache[col_pivot]) {
        for (auto _k = it; _k != pivot_cache[col_pivot].end(); ++_k) {
          int k = *_k;
          if (k <= j) [[unlikely]]
            throw std::runtime_error("Column " + std::to_string(k) + " is not a future column");
          auto prev = get_fil(k);
          prev.push_to_least_common_upper_bound(get_fil(j));
          if constexpr (verbose) std::cout << " -  (" << j << ", " << k << ") are interacting at " << prev << "\n";
          lexico_it.insert(std::move(prev), k);
        }
      }
      // TODO : check poset cache ?
      if constexpr (verbose) std::cout << std::flush;
      // }
      auto reduce_column = [&](int j) -> bool {
        int pivot = get_pivot(j);
        if constexpr (verbose) std::cout << "Reducing column " << j << " with pivot " << pivot << "\n";
        if (pivot < 0) {
          if (!reduced_columns[j]) {
            std::vector<index_type> _b(N.get_column(j).begin(), N.get_column(j).end());
            for (auto &stuff : _b) stuff -= nd;
            out_structure.push_back(std::move(_b));
            out_filtration.emplace_back(grid_value.begin(), grid_value.end());
            if constexpr (!generator_only) out_dimension.emplace_back(this->structure.dimension(j) + 1);
            reduced_columns[j] = true;
          }
          return false;
        }
        if constexpr (verbose) std::cout << "Previous registered pivot : " << *pivot_cache[pivot].begin() << std::endl;
        // WARN : we lazy update variables linked with col j...
        if (pivot_cache[pivot].size() == 0) {
          return false;
        }
        for (int k : pivot_cache[pivot]) {
          if (k >= j) {  // cannot reduce more here. this is a (local) pivot.
            return false;
          }
          if (get_fil(k) <= grid_value) {
            M.add_to(k, j);
            N.add_to(k, j);
            // std::cout << "Adding " << k << " to " << j << " at grid time " << grid_value << std::endl;
            pivot_cache[pivot].erase(j);
            // WARN : we update the pivot cache after the update loop
            if (get_pivot(j) >= pivot) {
              throw std::runtime_error("Addition failed ? current " + std::to_string(get_pivot(j)) + " previous " +
                                       std::to_string(pivot));
            }
            return true;  // pivot has changed
          }
        }
        return false;  // for loop exhausted (j may not be there because of lazy)
      };
      auto chores_after_new_pivot = [&](int j) {
        int col_pivot = get_pivot(j);
        if (col_pivot < 0) {
          if (!reduced_columns[j]) throw std::runtime_error("Empty column should have been detected before");
          return;
        };
        auto [it, was_there] = pivot_cache[col_pivot].insert(j);
        it++;
        // if constexpr (!use_grid) {
        for (auto _k = it; _k != pivot_cache[col_pivot].end(); ++_k) {
          int k = *_k;
          if (k <= j) [[unlikely]]
            throw std::runtime_error("(chores)  col " + std::to_string(k) + " is not a future column");
          if (get_fil(k) >= get_fil(j)) continue;
          auto prev = get_fil(k);
          prev.push_to_least_common_upper_bound(get_fil(j));
          if (lex_cmp(grid_value, prev)) {
            if constexpr (verbose)
              std::cout << "(chores) Updating grid queue, (" << j << ", " << k << ") are interacting at " << prev
                        << std::endl;
            lexico_it.insert(prev, k);
          }
        }
        // }
      };
      if constexpr (verbose) {
        std::cout << "Initially reduced columns: [";
        for (int i = 0; i < nd + ndpp; i++) {
          std::cout << reduced_columns[i] << ", ";
        }
        std::cout << "]" << std::endl;
      }
      while (!lexico_it.empty()) {
        // if constexpr (use_grid) {
        //   grid_value = lexico_it.next();
        // } else {
        grid_value = std::move(lexico_it.pop());
        // }
        if constexpr (verbose) {
          std::cout << "Grid value: " << grid_value << std::endl;
          std::cout << "    Reduced cols: ";
          for (int i = 0; i < nd + ndpp; i++) {
            std::cout << reduced_columns[i] << ", ";
          }
          std::cout << "]" << std::endl;
        }

        for (int i : lexico_it.get_current_cols()) {
          if constexpr (false) {
            if ((reduced_columns[i] || !(get_fil(i) <= grid_value))) continue;
            if ((get_fil(i) > grid_value)) break;
          }
          while (reduce_column(i));
          chores_after_new_pivot(i);
        }
      }
      // std::cout<< grid_.str() << std::endl;
      if constexpr (generator_only)
        return {out_structure, out_dimension};
      else {
        return Truc(out_structure, out_dimension, out_filtration);
      }
    }
  }

  template <bool ignore_inf>
  std::vector<std::pair<int, std::vector<index_type>>> get_current_boundary_matrix() {
    std::vector<index_type> permutation(generator_order.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    if constexpr (ignore_inf) {
      permutation.erase(std::remove_if(permutation.begin(),
                                       permutation.end(),
                                       [&](std::size_t val) {
                                         return filtration_container[val] == MultiFiltration::Generator::T_inf;
                                       }),
                        permutation.end());
      tbb::parallel_sort(permutation.begin(), permutation.end());
    }
    tbb::parallel_sort(permutation.begin(), permutation.end(), [&](std::size_t i, std::size_t j) {
      if (structure.dimension(i) > structure.dimension(j)) return false;
      if (structure.dimension(i) < structure.dimension(j)) return true;
      return filtration_container[i] < filtration_container[j];
    });

    std::vector<std::pair<int, std::vector<index_type>>> matrix(permutation.size());

    std::vector<index_type> permutationInv(generator_order.size());
    std::size_t newPos = 0;
    for (std::size_t oldPos : permutation) {
      permutationInv[oldPos] = newPos;
      auto &boundary = matrix[newPos].second;
      boundary.resize(structure[oldPos].size());
      for (std::size_t j = 0; j < structure[oldPos].size(); ++j) {
        boundary[j] = permutationInv[structure[oldPos][j]];
      }
      std::sort(boundary.begin(), boundary.end());
      matrix[newPos].first = structure.dimension(oldPos);
      ++newPos;
    }

    return matrix;
  }

  inline std::size_t num_generators() const { return structure.size(); }

  inline std::size_t num_parameters() const {
    return num_generators() == 0 ? 0 : this->generator_filtration_values[0].num_parameters();
  }

  inline const Structure &get_structure() const { return structure; }

  template <class SubFiltration, bool original_order = true>
  inline void push_to_out(const SubFiltration &f,
                          std::vector<typename MultiFiltration::value_type> &filtration_container,
                          const std::vector<index_type> &generator_order) const {
    /* std::vector<NewFilrationType> out(this->num_generators()); */

    // filtration_container.resize(
    //     this->num_generators()); // for some reasons it is necessary FIXME
    for (std::size_t i = 0u; i < this->num_generators(); i++) {
      if constexpr (original_order) {
        filtration_container[i] = f.compute_forward_intersection(generator_filtration_values[i]);
      } else {
        filtration_container[i] = f.compute_forward_intersection(generator_filtration_values[generator_order[i]]);
      }
    }
  }

  template <class SubFiltration, bool original_order = true>
  inline void push_to(const SubFiltration &f) {
    this->push_to_out<SubFiltration, original_order>(f, this->filtration_container, this->generator_order);
  }

  template <class array1d>
  inline void set_one_filtration(const array1d &truc) {
    if (truc.size() != this->num_generators())
      throw std::invalid_argument("(setting one filtration) Bad size. Got " + std::to_string(truc.size()) +
                                  " expected " + std::to_string(this->num_generators()));
    this->filtration_container = truc;
  }

  inline const std::vector<typename MultiFiltration::value_type> &get_one_filtration() const {
    return this->filtration_container;
  }

  inline PersBackend compute_persistence_out(
      const std::vector<typename MultiFiltration::value_type> &one_filtration,
      std::vector<index_type> &out_gen_order,
      const bool ignore_inf) const {  // needed ftm as PersBackend only points there
    constexpr const bool verbose = false;
    if (one_filtration.size() != this->num_generators()) {
      throw std::runtime_error("The one parameter filtration doesn't have a proper size.");
    }
    out_gen_order.resize(this->num_generators());
    std::iota(out_gen_order.begin(),
              out_gen_order.end(),
              0);  // we have to reset here, even though we're already doing this
    std::sort(out_gen_order.begin(), out_gen_order.end(), [&](index_type i, index_type j) {
      if (structure.dimension(i) > structure.dimension(j)) return false;
      if (structure.dimension(i) < structure.dimension(j)) return true;
      return one_filtration[i] < one_filtration[j];
    });
    if (!PersBackend::is_vine && ignore_inf) {
      if constexpr (verbose) {
        std::cout << "Removing infinite simplices" << std::endl;
      }
      for (auto &i : out_gen_order)
        if (one_filtration[i] == MultiFiltration::Generator::T_inf) {
          // TODO : later
          // int d = structure.dimension(i);
          // d = d == 0 ? 1 : 0;
          // if (degrees.size()>d || degrees[d] || degrees[d-1])
          //   continue;
          i = std::numeric_limits<typename std::remove_reference_t<decltype(out_gen_order)>::value_type>::max();
        }
    }
    if constexpr (false) {
      std::cout << structure << std::endl;
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
    return PersBackend(structure, out_gen_order);
  }

  inline bool has_persistence() const { return this->persistence.size(); };

  inline void compute_persistence(const bool ignore_inf = true) {
    this->persistence = this->compute_persistence_out(
        // this->filtration_container, this->generator_order, degrees); // TODO
        // : later
        this->filtration_container,
        this->generator_order,
        ignore_inf);
  };

  // TODO : static ?
  inline void vineyard_update(PersBackend &persistence,
                              const std::vector<typename MultiFiltration::value_type> &one_filtration,
                              std::vector<index_type> &generator_order) const {
    constexpr const bool verbose = false;
    /* static_assert(PersBackend::has_vine_update); */
    // the first false is to get the generator order
    // insertion sort
    auto n = this->num_generators();
    if constexpr (verbose) {
      std::cout << "Vine updates : ";
    }
    for (std::size_t i = 0; i < n; i++) {
      auto j = i;
      while (j > 0 && persistence.get_dimension(j) == persistence.get_dimension(j - 1) &&
             one_filtration[generator_order[j]] < one_filtration[generator_order[j - 1]]) {
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
    vineyard_update(this->persistence, this->filtration_container, this->generator_order);
  }

  inline split_barcode_idx get_barcode_idx(
      PersBackend &persistence) const {
    auto barcode_indices = persistence.get_barcode();
    split_barcode_idx out(this->structure.max_dimension() + 1);  // TODO : This doesn't allow for negative dimensions
    for (const auto &bar : barcode_indices) {
      int death = bar.death == static_cast<typename PersBackend::pos_index>(-1) ? -1 : bar.death;
      out[bar.dim].push_back({bar.birth, death});
    }
    return out;
  }

  // puts the degree-ordered bc starting out_ptr, and returns the "next" pointer.
  // corresond to an array of shape (num_bar, 2);
  template <bool return_shape = false>
  inline std::conditional_t<return_shape, std::pair<std::vector<int>, int*>, int*> get_barcode_idx(
      PersBackend &persistence,
      int *start_ptr) const {
    const auto &bc = persistence.barcode();
    if (bc.size() == 0) return start_ptr;
    std::vector<int> shape(this->structure.max_dimension());
    for (const auto &b : bc) shape[b.dim]++;
    // dim in barcode may be unsorted...
    std::vector<int *> ptr_shifts(shape.size());
    int shape_cumsum = 0;
    for (auto i : std::views::iota(0u, bc.size())) {
      if (i != 0u) shape_cumsum += shape[i - 1];
      // 2 for (birth, death)
      ptr_shifts[i] = 2 * shape_cumsum + start_ptr;
    }
    for (const auto &b : bc) {
      int *current_loc = ptr_shifts[b.dim];
      *(current_loc++) = b.birth;
      *(current_loc++) = b.death == static_cast<typename PersBackend::pos_index>(-1) ? -1 : b.death;
    }

    if constexpr (return_shape)
      return {shape, ptr_shifts.back()};
    else
      return ptr_shifts.back();
  }

    

  inline split_barcode get_barcode(
      PersBackend &persistence,
      const std::vector<typename MultiFiltration::value_type> &filtration_container) const {
    auto barcode_indices = persistence.get_barcode();
    split_barcode out(this->structure.max_dimension() + 1);  // TODO : This doesn't allow for negative dimensions
    constexpr const bool verbose = false;
    constexpr const bool debug = false;
    const auto inf = MultiFiltration::Generator::T_inf;
    for (const auto &bar : barcode_indices) {
      if constexpr (verbose) std::cout << "BAR : " << bar.birth << " " << bar.death << "\n";
      if constexpr (debug) {
        if (bar.birth >= filtration_container.size() || bar.birth < 0) {
          std::cout << "Trying to add an incompatible birth... ";
          std::cout << bar.birth << std::endl;
          std::cout << "Death is " << bar.death << std::endl;
          std::cout << "Max size is " << filtration_container.size() << std::endl;
          continue;
        }
        if (bar.dim > static_cast<int>(this->structure.max_dimension())) {
          std::cout << "Incompatible dimension detected... " << bar.dim << std::endl;
          std::cout << "While max dim is " << this->structure.max_dimension() << std::endl;
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
      if (birth_filtration <= death_filtration)
        out[bar.dim].push_back({birth_filtration, death_filtration});
      else {
        out[bar.dim].push_back({inf, inf});
      }
    }
    return out;
  }

  inline split_barcode get_barcode() { return get_barcode(this->persistence, this->filtration_container); }

  inline split_barcode_idx get_barcode_idx() { return get_barcode_idx(this->persistence); }

  template <typename value_type = value_type>
  static inline flat_nodim_barcode<value_type> get_flat_nodim_barcode(
      PersBackend &persistence,
      std::vector<typename MultiFiltration::value_type> &filtration_container) {
    constexpr const bool verbose = false;
    const auto &barcode_indices = persistence.get_barcode();
    auto num_bars = barcode_indices.size();
    flat_nodim_barcode<value_type> out(num_bars);
    if (num_bars <= 0) return out;
    auto idx = 0u;
    const value_type inf = MultiFiltration::Generator::T_inf;
    for (const auto &bar : barcode_indices) {
      value_type birth_filtration = inf;
      value_type death_filtration = -birth_filtration;
      if (bar.death == static_cast<typename PersBackend::pos_index>(-1))
        death_filtration = inf;
      else
        death_filtration = static_cast<value_type>(filtration_container[bar.death]);
      birth_filtration = static_cast<value_type>(filtration_container[bar.birth]);
      if constexpr (verbose) {
        std::cout << "PAIRING : " << bar.birth << " / " << bar.death << " dim " << bar.dim << std::endl;
      }
      if constexpr (verbose) {
        std::cout << "PAIRING filtration : " << birth_filtration << " " << death_filtration << " dim " << bar.dim
                  << std::endl;
      }

      if (birth_filtration <= death_filtration)
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
    constexpr const bool verbose = false;
    const auto &barcode_indices = persistence.get_barcode();
    auto num_bars = barcode_indices.size();
    flat_barcode<value_type> out(num_bars);
    if (num_bars <= 0) return out;
    auto idx = 0u;
    const value_type inf = MultiFiltration::Generator::T_inf;
    for (const auto &bar : barcode_indices) {
      value_type birth_filtration = inf;
      value_type death_filtration = -birth_filtration;
      if (bar.death == static_cast<typename PersBackend::pos_index>(-1))
        death_filtration = inf;
      else
        death_filtration = static_cast<value_type>(filtration_container[bar.death]);
      birth_filtration = static_cast<value_type>(filtration_container[bar.birth]);
      if constexpr (verbose) {
        std::cout << "PAIRING : " << bar.birth << " / " << bar.death << " dim " << bar.dim << std::endl;
      }
      if constexpr (verbose) {
        std::cout << "PAIRING filtration : " << birth_filtration << " " << death_filtration << " dim " << bar.dim
                  << std::endl;
      }

      if (birth_filtration <= death_filtration)
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
    return get_flat_nodim_barcode(this->persistence, this->filtration_container);
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
    for (const auto &idx : truc.generator_order) stream << idx << ", ";
    stream << "}" << std::endl;

    stream << "--- Current slice filtration\n";
    stream << "{";
    for (const auto &stuff : truc.filtration_container) stream << stuff << ", ";
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

  inline std::pair<typename MultiFiltration::Generator, typename MultiFiltration::Generator> get_bounding_box() const {
    using OC = typename MultiFiltration::Generator;
    // assert(!generator_filtration_values.empty());
    OC a = OC::inf();
    OC b = -1 * a;
    for (const auto &filtration_value : generator_filtration_values) {
      if constexpr (MultiFiltration::is_multi_critical) {
        a.pull_to_greatest_common_lower_bound(factorize_below(filtration_value));
        b.push_to_least_common_upper_bound(factorize_above(filtration_value));
      } else {
        a.pull_to_greatest_common_lower_bound(filtration_value);
        b.push_to_least_common_upper_bound(filtration_value);
      }
    }
    return {a, b};
  }

  inline std::vector<typename MultiFiltration::Generator> get_filtration_values() const {
    if constexpr (MultiFiltration::is_multi_critical) {
      std::vector<typename MultiFiltration::Generator> out;
      out.reserve(generator_filtration_values.size());  // at least this, will dooble later
      for (std::size_t i = 0; i < generator_filtration_values.size(); ++i) {
        for (const auto &f : generator_filtration_values[i]) {
          out.push_back(f);
        }
      }
      return out;
    } else {
      return generator_filtration_values;  // copy not necessary for Generator
    }  // (could return const&)
  }

  inline std::vector<MultiFiltration> &get_filtrations() { return generator_filtration_values; }

  inline const std::vector<MultiFiltration> &get_filtrations() const { return generator_filtration_values; }

  inline const std::vector<int> get_dimensions() const {
    std::size_t n = this->num_generators();
    std::vector<int> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = structure.dimension(i);
    }
    return out;
  }

  inline int get_dimension(int i) const { return structure.dimension(i); }

  inline void prune_above_dimension(int max_dim) {
    int idx = structure.prune_above_dimension(max_dim);
    generator_filtration_values.resize(idx);
    generator_order.resize(idx);
    filtration_container.resize(idx);
  }

  inline const std::vector<std::vector<index_type>> get_boundaries() {
    std::size_t n = this->num_generators();
    std::vector<std::vector<index_type>> out(n);
    for (auto i = 0u; i < n; ++i) {
      out[i] = this->structure[i];
    }
    return out;
  }

  auto coarsen_on_grid(const std::vector<std::vector<typename MultiFiltration::value_type>>& grid) {
    using return_type = decltype(std::declval<MultiFiltration>().template as_type<std::int32_t>());
    std::vector<return_type> coords(this->num_generators());
    // for (std::size_t gen = 0u; gen < coords.size(); ++gen) { // TODO : parallelize
    //   coords[gen] = compute_coordinates_in_grid<int32_t>(generator_filtration_values[gen], grid);
    // }
    tbb::parallel_for(static_cast<std::size_t>(0u), coords.size(), [&](std::size_t gen){
      coords[gen] = compute_coordinates_in_grid<int32_t>(generator_filtration_values[gen], grid);
    });
    return Truc<PersBackend, Structure, return_type>(structure, coords);
  }

  inline void coarsen_on_grid_inplace(const std::vector<std::vector<typename MultiFiltration::value_type>> &grid,
                                      bool coordinate = true) {
    for (auto gen = 0u; gen < this->num_generators(); ++gen) {
      generator_filtration_values[gen].project_onto_grid(grid, coordinate);
    }
  }

  // dim, num_cycle_of_dim, num_faces_in_cycle, vertices_in_face
  inline std::vector<std::vector<std::vector<std::vector<index_type>>>> get_representative_cycles(
      bool update = true,
      bool detailed = false) {
    // iterable iterable simplex key
    auto cycles_key = persistence.get_representative_cycles(update, detailed);
    auto num_cycles = cycles_key.size();
    std::vector<std::vector<std::vector<std::vector<index_type>>>> out(structure.max_dimension() + 1);
    for (auto &cycles_of_dim : out) cycles_of_dim.reserve(num_cycles);
    for (const auto &cycle : cycles_key) {
      int cycle_dim = 0;        // for more generality, should be minimal dimension instead
      if (!cycle[0].empty()) {  // if empty, cycle has no border -> assumes dimension 0 even if it could be min dim
        cycle_dim = structure.dimension(cycle[0][0]) + 1;  // all faces have the same dim
      }
      out[cycle_dim].push_back(cycle);
    }
    return out;
  }

  const std::vector<index_type> &get_current_order() const { return generator_order; }

  const PersBackend &get_persistence() const { return persistence; }

  PersBackend &get_persistence() { return persistence; }

  // TrucThread get_safe_thread() { return TrucThread(*this); }

  class TrucThread {
   public:
    using Filtration_value = MultiFiltration;
    using value_type = typename MultiFiltration::value_type;
    using ThreadSafe = TrucThread;

    inline TrucThread(const Truc &truc)
        : truc_ptr(&truc),
          generator_order(truc.get_current_order()),
          filtration_container(truc.get_one_filtration()),
          persistence(truc.get_persistence()) {
      persistence._update_permutation_ptr(generator_order);
    };

    inline TrucThread(const TrucThread &truc)
        : truc_ptr(truc.truc_ptr),
          generator_order(truc.get_current_order()),
          filtration_container(truc.get_one_filtration()),
          persistence(truc.get_persistence()) {
      persistence._update_permutation_ptr(generator_order);
    };

    inline TrucThread weak_copy() const { return TrucThread(*truc_ptr); }

    inline bool has_persistence() const { return this->persistence.size(); };

    inline const PersBackend &get_persistence() const { return persistence; }

    inline PersBackend &get_persistence() { return persistence; }

    inline std::pair<MultiFiltration, MultiFiltration> get_bounding_box() const { return truc_ptr->get_bounding_box(); }

    inline const std::vector<index_type> &get_current_order() const { return generator_order; }

    inline const std::vector<MultiFiltration> &get_filtrations() const { return truc_ptr->get_filtrations(); }

    inline const std::vector<int> &get_dimensions() const { return truc_ptr->get_dimensions(); }

    inline const std::vector<std::vector<index_type>> &get_boundaries() const { return truc_ptr->get_boundaries(); }

    inline void coarsen_on_grid_inplace(const std::vector<std::vector<typename MultiFiltration::value_type>> &grid,
                                        bool coordinate = true) {
      truc_ptr->coarsen_on_grid_inplace(grid, coordinate);
    }

    template <typename Subfiltration>
    inline void push_to(const Subfiltration &f) {
      truc_ptr->push_to_out(f, this->filtration_container, this->generator_order);
    }

    inline std::vector<typename PersBackend::cycle_type> get_representative_cycles(bool update = true) {
      return truc_ptr->get_representative_cycles(update);
    }

    inline void compute_persistence(const bool ignore_inf = true) {
      this->persistence =
          this->truc_ptr->compute_persistence_out(this->filtration_container, this->generator_order, ignore_inf);
    };

    inline void vineyard_update() {
      truc_ptr->vineyard_update(this->persistence, this->filtration_container, this->generator_order);
    }

    template <typename value_type = value_type>
    inline flat_barcode<value_type> get_flat_barcode() {
      return truc_ptr->get_flat_barcode(this->persistence, this->filtration_container);
    }

    template <typename value_type = value_type>
    inline flat_nodim_barcode<value_type> get_flat_nodim_barcode() {
      return truc_ptr->get_flat_nodim_barcode(this->persistence, this->filtration_container);
    }

    inline split_barcode get_barcode() { return truc_ptr->get_barcode(this->persistence, this->filtration_container); }

    inline split_barcode_idx get_barcode_idx() {
      return truc_ptr->get_barcode_idx(this->persistence);
    }

    inline std::size_t num_generators() const { return this->truc_ptr->structure.size(); }

    inline std::size_t num_parameters() const {
      return num_generators() == 0 ? 0 : this->get_filtrations()[0].num_parameters();
    }

    inline const std::vector<typename MultiFiltration::value_type> &get_one_filtration() const {
      return this->filtration_container;
    }

    inline std::vector<typename MultiFiltration::value_type> &get_one_filtration() {
      return this->filtration_container;
    }

    template <class array1d>
    inline void set_one_filtration(const array1d &truc) {
      if (truc.size() != this->num_generators())
        throw std::invalid_argument("(setting one filtration) Bad size. Got " + std::to_string(truc.size()) +
                                    " expected " + std::to_string(this->num_generators()));
      this->filtration_container = truc;
    }

   private:
    const Truc *truc_ptr;
    std::vector<index_type> generator_order;                                 // size fixed at construction time,
    std::vector<typename MultiFiltration::value_type> filtration_container;  // filtration of the current slice
    PersBackend persistence;  // generated by the structure, and generator_order.

  };  // class TrucThread

  /*
   * returns barcodes of the f(multipers)
   *
   */
  template <typename Fun, typename Fun_arg, bool idx = false, bool custom = false>
  inline std::conditional_t<idx, std::vector<split_barcode_idx>, std::vector<split_barcode>>
  barcodes(Fun &&f, const std::vector<Fun_arg> &args, const bool ignore_inf = true) {
    if (args.size() == 0) {
      return {};
    }
    std::conditional_t<idx, std::vector<split_barcode_idx>, std::vector<split_barcode>> out(args.size());

    if constexpr (PersBackend::is_vine) {
      if constexpr (custom)
        this->set_one_filtration(f(args[0]));
      else
        this->push_to(f(args[0]));
      this->compute_persistence();
      if constexpr (idx)
        out[0] = this->get_barcode_idx();
      else
        out[0] = this->get_barcode();
      for (auto i = 1u; i < args.size(); ++i) {
        if constexpr (custom)
          this->set_one_filtration(f(args[i]));
        else
          this->push_to(f(args[i]));
        this->vineyard_update();
        if constexpr (idx)
          out[i] = this->get_barcode_idx();
        else
          out[i] = this->get_barcode();
      }

    } else {
      ThreadSafe local_template = this->weak_copy();
      tbb::enumerable_thread_specific<ThreadSafe> thread_locals(local_template);
      tbb::parallel_for(static_cast<std::size_t>(0), args.size(), [&](const std::size_t &i) {
        ThreadSafe &s = thread_locals.local();
        if constexpr (custom)
          s.set_one_filtration(f(args[i]));
        else
          s.push_to(f(args[i]));
        s.compute_persistence(ignore_inf);
        if constexpr (idx) {
          out[i] = s.get_barcode_idx();
        } else
          out[i] = s.get_barcode();
      });
    }
    return out;
  }

  // FOR Python interface, but I'm not fan. Todo: do the lambda function in
  // cython?
  inline std::vector<split_barcode> persistence_on_lines(const std::vector<std::vector<value_type>> &basepoints,
                                                         bool ignore_inf) {
    return barcodes(
        [](const std::vector<value_type> &basepoint) { return Gudhi::multi_persistence::Line<value_type>(basepoint); },
        basepoints,
        ignore_inf);
  }

  inline std::vector<split_barcode_idx> custom_persistences(const value_type *filtrations, int size, bool ignore_inf) {
    std::vector<const value_type *> args(size);
    for (auto i = 0; i < size; ++i) args[i] = filtrations + this->num_generators() * i;

    auto fun = [&](const value_type *one_filtration_ptr) {
      std::vector<value_type> fil(this->num_generators());
      for (auto i : std::views::iota(0u, this->num_generators())) {
        fil[i] = *(one_filtration_ptr + i);
      }
      return std::move(fil);
    };
    return barcodes<decltype(fun), const value_type *, true, true>(std::move(fun), args, ignore_inf);
  }

  inline std::vector<split_barcode> persistence_on_lines(
      const std::vector<std::pair<std::vector<value_type>, std::vector<value_type>>> &bp_dirs,
      bool ignore_inf) {
    return barcodes(
        [](const std::pair<std::vector<value_type>, std::vector<value_type>> &bpdir) {
          return Gudhi::multi_persistence::Line<value_type>(bpdir.first, bpdir.second);
        },
        bp_dirs,
        ignore_inf);
  }

  void build_from_scc_file(const std::string &inFilePath,
                           bool isRivetCompatible = false,
                           bool isReversed = false,
                           int shiftDimensions = 0) {
    *this = read_scc_file<Truc>(inFilePath, isRivetCompatible, isReversed, shiftDimensions);
  }

  void write_to_scc_file(const std::string &outFilePath,
                         int numberOfParameters = -1,
                         int degree = -1,
                         bool rivetCompatible = false,
                         bool IgnoreLastGenerators = false,
                         bool stripComments = false,
                         bool reverse = false) {
    write_scc_file<Truc>(
        outFilePath, *this, numberOfParameters, degree, rivetCompatible, IgnoreLastGenerators, stripComments, reverse);
  }

 public:
  using ThreadSafe = TrucThread;  // for outside

  TrucThread weak_copy() const { return TrucThread(*this); }

  // TODO: declare method here instead of scc_io.h
  // it is just temporary, until Truc is cleaned up
  // friend void write_scc_file<Truc>(const std::string &outFilePath,
  //                                  const Truc &slicer,
  //                                  int numberOfParameters,
  //                                  int degree,
  //                                  bool rivetCompatible,
  //                                  bool IgnoreLastGenerators,
  //                                  bool stripComments,
  //                                  bool reverse);

 private:
  MultiFiltrations generator_filtration_values;                            // defined at construction time. Const
  std::vector<index_type> generator_order;                                 // size fixed at construction time
  Structure structure;                                                     // defined at construction time. Const
  std::vector<typename MultiFiltration::value_type> filtration_container;  // filtration of the current slice
  PersBackend persistence;  // generated by the structure, and generator_order.

};  // class Truc

}  // namespace truc_interface
}  // namespace multiparameter
}  // namespace Gudhi
