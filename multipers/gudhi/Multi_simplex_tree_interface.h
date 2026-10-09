/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux, Hannah Schreiber
 *
 *    Copyright (C) 2026 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file Multi_simplex_tree_interface.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the @ref Gudhi::multi_persistence::Multi_simplex_tree_interface class for python bindings.
 */

#ifndef MP_PY_MULTI_SIMPLEX_TREE_H_INCLUDED
#define MP_PY_MULTI_SIMPLEX_TREE_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include <gudhi/Simplex_tree.h>
#include <gudhi/Slicer.h>
#include <gudhi/simple_mdspan.h>
#include <python_interfaces/numpy_utils.h>
#include <python_interfaces/Simplex_tree_interface.h>
#include <gudhi/multi_simplex_tree_helpers.h>
#include <gudhi/Multi_persistence/Line.h>
#include <gudhi/Multi_persistence/utils.h>
#include <gudhi/Multi_parameter_filtration_value.h>

#include "interface_helpers.h"
#include "interface_helper_structs.h"

namespace Gudhi {
namespace multi_persistence {

using Simplex_tree_std = Simplex_tree<Simplex_tree_options_for_python>;
template <class MultiFiltrationValue>
using Simplex_tree_multi = Simplex_tree<Simplex_tree_options_multidimensional_filtration<MultiFiltrationValue>>;

/**
 * @private
 */
template <class MultiFiltrationValue>
class Multi_simplex_tree_interface : public Simplex_tree_multi<MultiFiltrationValue> {
 public:
  using Options = Simplex_tree_options_multidimensional_filtration<MultiFiltrationValue>;
  using Base = Simplex_tree<Options>;
  using Filtration_value = MultiFiltrationValue;
  using value_type = typename Filtration_value::value_type;
  using Vertex_handle = typename Base::Vertex_handle;
  using Simplex_handle = typename Base::Simplex_handle;
  using Simplex = std::vector<Vertex_handle>;
  using Complex_simplex_iterator = typename Base::Complex_simplex_iterator;
  using Skeleton_simplex_iterator = typename Base::Skeleton_simplex_iterator;
  using Boundary_simplex_iterator = typename Base::Boundary_simplex_iterator;
  template <typename U>
  using Tensor1D = nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>;
  template <typename U>
  using Tensor2D = nanobind::ndarray<const U, nanobind::ndim<2>>;
  template <typename U>
  using Tensor3D = nanobind::ndarray<const U, nanobind::ndim<3>>;

  Multi_simplex_tree_interface() : Base(), filtrationGrid_(nanobind::none()) {};

  Multi_simplex_tree_interface(int numParam) : Base(), filtrationGrid_(nanobind::none()) {
    Base::set_num_parameters(numParam <= 0 ? 2 : numParam);
  };

  Multi_simplex_tree_interface(const Base& st) : Base(st), filtrationGrid_(nanobind::none()) {};
  Multi_simplex_tree_interface(Base&& st) : Base(std::move(st)), filtrationGrid_(nanobind::none()) {};

  Multi_simplex_tree_interface& operator=(const Base& st) {
    Base::operator=(st);
    // do we want to reset the filtration grid here?
    return *this;
  }

  Multi_simplex_tree_interface& operator=(Base&& st) {
    Base::operator=(std::move(st));
    // do we want to reset the filtration grid here?
    return *this;
  }

  template <typename OtherMultiFiltrationValue>
  void copy_from(const Multi_simplex_tree_interface<OtherMultiFiltrationValue>& other, int numParam = -1) {
    {
      nanobind::gil_scoped_release release;
      Base::clear();
      Base::copy_from(other, [numParam](const auto& fil) -> Filtration_value {
        if constexpr (std::is_same_v<Filtration_value, OtherMultiFiltrationValue>) {
          if (numParam >= 0 && static_cast<std::size_t>(numParam) != fil.num_parameters()) {
            return fil.copy(numParam, fil.num_generators());
          } else {
            return fil;
          }
        } else {
          if (numParam >= 0 && static_cast<std::size_t>(numParam) != fil.num_parameters()) {
            return fil.copy(numParam, fil.num_generators()).template as_type<Filtration_value>();
          } else {
            return fil.template as_type<Filtration_value>();
          }
        }
      });
      if (numParam >= 0) Base::set_num_parameters(numParam);
    }
    filtrationGrid_ = other.get_filtration_grid();
  }

  template <class OtherMultiFiltrationValue, class PersistenceAlgorithm>
  void copy_from(const Slicer<OtherMultiFiltrationValue, PersistenceAlgorithm>& other,
                 int maxDim = -1,
                 int numParam = -1) {
    filtrationGrid_ = nanobind::none();
    Base st;
    {
      nanobind::gil_scoped_release release;
      Base::clear();
      st = build_simplex_tree_from_complex<Options>(other.get_filtered_complex(), maxDim, numParam);
      st.set_num_parameters(numParam >= 0 ? numParam : other.get_number_of_parameters());
    }
    *this = std::move(st);
  }

  Multi_simplex_tree_interface& from_std(nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> state,
                                         int dimension,
                                         int num_parameters,
                                         nanobind::object default_values) {
    if (state.size() != 0) {
      Filtration_value fil = detail::_cast_to_filtration_value<Filtration_value>(default_values, num_parameters);
      {
        nanobind::gil_scoped_release release;
        char const* buffer_start = state.data();
        Gudhi::Simplex_tree_interface st;
        st.deserialize(buffer_start, state.size());
        *this = Gudhi::multi_persistence::make_multi_dimensional<Options>(st, fil, dimension);
      }
    }
    return *this;
  }

  [[nodiscard]] nanobind::object get_filtration_grid() const { return filtrationGrid_; }

  void set_filtration_grid(nanobind::object grid) {
    if (grid.is_none()) {
      filtrationGrid_ = nanobind::none();
      return;
    }

    detail::_verify_grid_validity(grid);
    filtrationGrid_ = grid;
  }

  bool find_simplex(nanobind::object simplex) const {
    return (_get_handle_from_vertices(simplex) != Base::null_simplex());
  }

  bool insert_single_simplex(Tensor1D<Vertex_handle> vertices, nanobind::object filtrationValue) {
    std::pair<Simplex_handle, bool> result;

    if (filtrationValue.is_none()) {
      nanobind::gil_scoped_release release;
      result = _insert_single_simplex(Numpy_span(vertices));
    } else {
      Filtration_value fil =
          detail::_cast_to_filtration_value<Filtration_value>(filtrationValue, Base::num_parameters());
      nanobind::gil_scoped_release release;
      result = _insert_single_simplex(Numpy_span(vertices), fil);
    }

    if (result.first != Base::null_simplex()) Base::clear_filtration();
    return result.second;
  }

  Multi_simplex_tree_interface& insert_batch(Tensor1D<Vertex_handle> vertices,
                                             Tensor2D<Vertex_handle> vertex_array,
                                             nanobind::object filtrationValues) {
    auto v_view = vertex_array.view();
    const std::size_t dim = v_view.shape(0) - 1;
    const std::size_t numSimplices = v_view.shape(1);

    std::vector<Filtration_value> fils;
    if (!filtrationValues.is_none()) {
      nanobind::ndarray<> array;
      if (nanobind::try_cast(filtrationValues, array, false) && array.size() == 0) {
        detail::_require_cpu_array(array);
      } else {
        fils = detail::_cast_to_filtration_value_array<Filtration_value>(filtrationValues, Base::num_parameters());
      }
    }
    Base::clear_filtration();

    if (fils.empty()) {
      {
        nanobind::gil_scoped_release release;
        Base::insert_batch_vertices(Numpy_span(vertices), Filtration_value::minus_inf(Base::num_parameters()));
        if (dim > 0) {
          for (std::size_t i = 0; i < numSimplices; ++i) {
            _insert_single_simplex(make_element_range(&v_view(0, i), v_view, false));
          }
        }
      }
      return *this;
    }

    if (fils.size() < numSimplices)
      throw std::invalid_argument("Filtration value array does not have a value for every simplex.");

    {
      nanobind::gil_scoped_release release;
      if constexpr (!Filtration_value::ensures_1_criticality()) {
        // small optimisation, but don't work in both cases because of weird insertion strategy
        Base::insert_batch_vertices(Numpy_span(vertices), Filtration_value::inf(Base::num_parameters()));
      }
      for (std::size_t i = 0; i < numSimplices; ++i) {
        _insert_single_simplex(make_element_range(&v_view(0, i), v_view, false), fils[i]);
      }
    }
    return *this;
  }

  Multi_simplex_tree_interface& remove_maximal_simplex(nanobind::object simplex) {
    auto sh = _get_handle_from_vertices(simplex);
    {
      nanobind::gil_scoped_release release;
      Base::remove_maximal_simplex(sh);
      Base::clear_filtration();
    }
    return *this;
  }

  Multi_simplex_tree_interface& expand(int max_dim) {
    {
      nanobind::gil_scoped_release release;
      Base::expansion(max_dim);
      Base::make_filtration_non_decreasing();
    }
    return *this;
  }

  nanobind::object get_simplex_filtration_value(Tensor1D<Vertex_handle> simplex,
                                                bool viewIfPossible = true,
                                                bool raw = false) {
    Simplex_handle sh = Base::find(Numpy_span(simplex));
    if (sh == Base::null_simplex())
      throw std::invalid_argument("Cannot return the filtration value of a simplex that is not in the complex");
    auto& f = Base::get_filtration_value(sh);

    if (raw)
      return detail::_get_raw_filtration_data(
          viewIfPossible ? nanobind::find(this) : nanobind::handle(), f, !viewIfPossible);

    // view not possible for Degree_rips_bifiltration
    if constexpr (!detail::_is_degree_rips<MultiFiltrationValue>()) {
      if (viewIfPossible) return detail::_get_raw_filtration_data(nanobind::find(this), f, false);
    }
    return nanobind::cast(detail::_get_filtration_array(f));
  }

  Multi_simplex_tree_interface& assign_simplex_filtration(Tensor1D<Vertex_handle> vertices,
                                                          nanobind::object filtrationValue) {
    Filtration_value fil = Filtration_value::minus_inf(Base::num_parameters());
    if (!filtrationValue.is_none()) {
      fil = detail::_cast_to_filtration_value<Filtration_value>(filtrationValue, Base::num_parameters());
    }

    {
      nanobind::gil_scoped_release release;
      Simplex_handle sh = Base::find(Numpy_span(vertices));
      if (sh == Base::null_simplex())
        throw std::invalid_argument("Cannot assign a filtration to a simplex that is not in the complex");

      Base::assign_filtration(sh, fil);
      Base::clear_filtration();
    }

    return *this;
  }

  auto get_simplices_of_dimension(int dimension) const {
    if (dimension < 0) throw std::invalid_argument("Dimension cannot be negative.");

    std::size_t numSimplices = 0;
    std::vector<Vertex_handle> simplices;
    {
      nanobind::gil_scoped_release release;

      for ([[maybe_unused]] auto sh : Base::dimension_simplex_range(dimension)) ++numSimplices;
      simplices.resize(numSimplices * (dimension + 1));

      std::size_t i = 0;
      for (auto sh : Base::dimension_simplex_range(dimension)) {
        for (auto vertex : Base::simplex_vertex_range(sh)) {
          simplices[i] = vertex;
          ++i;
        }
      }
    }

    return _wrap_as_numpy_array(std::move(simplices), numSimplices, static_cast<std::size_t>(dimension) + 1);
  }

  template <typename T = value_type>
  nanobind::ndarray<nanobind::numpy, T> get_edge_list() const {
    // TODO: generalize for more parameters? As edges is already a std::vector, it should not be too difficult.
    if (Base::num_parameters() != 2) throw std::logic_error("Method only implemented for 2-parameter filtrations.");

    std::size_t numEdges = 0;
    std::vector<T> edges;

    {
      nanobind::gil_scoped_release release;

      for ([[maybe_unused]] auto sh : Base::dimension_simplex_range(1)) ++numEdges;
      edges.resize(numEdges * 4);

      std::size_t i = 0;
      for (auto sh : Base::dimension_simplex_range(1)) {
        for (auto vertex : Base::simplex_vertex_range(sh)) {
          edges[i] = static_cast<T>(vertex);
          ++i;
        }
        const auto& f = Base::get_filtration_value(sh);
        if (f.num_parameters() != 2)
          throw std::runtime_error(
              "Inconsistency between number of parameters of the simplex tree (=2) and the number of parameters of its "
              "filtration values (!=2).");
        edges[i] = static_cast<T>(f(0, 0));
        edges[i + 1] = static_cast<T>(f(0, 1));
        i += 2;
      }
    }

    return _wrap_as_numpy_array(std::move(edges), numEdges, 4);
  }

  auto get_simplex_python_iterator() {
    return _make_iterator("simplex_iterator", Complex_simplex_iterator(this), Complex_simplex_iterator());
  }

  auto get_skeleton_python_iterator(int dimension) {
    return _make_iterator("skeleton_iterator", Skeleton_simplex_iterator(this, dimension), Skeleton_simplex_iterator());
  }

  auto get_boundary_python_iterator(nanobind::object simplex) {
    auto bd_sh = _get_handle_from_vertices(simplex);
    if (bd_sh == Base::null_simplex()) throw std::runtime_error("simplex not found - cannot find boundaries");
    return _make_iterator("boundary_iterator", Boundary_simplex_iterator(this, bd_sh), Boundary_simplex_iterator(this));
  }

  // TODO: homogenize format with Slicer
  nanobind::tuple get_filtration_values(Tensor1D<int> degrees) const {
    if (degrees.size() == 0) return nanobind::tuple();
    // assumes degrees has no duplicates and is sorted
    auto view = degrees.view();
    std::vector<std::vector<value_type>> values;
    std::size_t numParam = Base::num_parameters();
    {
      nanobind::gil_scoped_release release;

      std::vector<int> degreeIndex(std::max(std::min(Base::dimension(), view(view.shape(0) - 1)), -1) + 1, -1);
      if (degreeIndex.empty()) {
        std::size_t numSimplices = 0;
        for (const auto& sh : Base::complex_simplex_range()) {
          const auto& f = Base::get_filtration_value(sh);
          if (numParam != f.num_parameters())
            throw std::runtime_error("Inconsistent number of parameters in stored filtration values");
          numSimplices += f.num_generators();
        }
        values.emplace_back(numParam * numSimplices);
        Gudhi::Simple_mdspan view(values[0].data(), numParam, numSimplices);
        std::size_t i = 0;
        for (const auto& sh : Base::complex_simplex_range()) {
          const auto& f = Base::get_filtration_value(sh);
          for (std::size_t g = 0; g < f.num_generators(); ++g) {
            for (std::size_t p = 0; p < numParam; ++p) view(p, i) = f(g, p);
            ++i;
          }
        }
      } else {
        std::size_t searchStart = 0;
        std::vector<std::size_t> numSimplices(degreeIndex.size());
        for (const auto& sh : Base::complex_simplex_range()) {
          const auto& f = Base::get_filtration_value(sh);
          if (numParam != f.num_parameters())
            throw std::runtime_error("Inconsistent number of parameters in stored filtration values");
          const std::size_t dim = Base::dimension(sh);
          if (dim < degreeIndex.size()) numSimplices[dim] += f.num_generators();
        }
        while (view(searchStart) < 0) ++searchStart;  // if all are negative, we are not in this case
        values.resize(degrees.size() - searchStart);
        for (std::size_t i = searchStart; i < degrees.size(); ++i) {
          const auto d = static_cast<std::size_t>(view(i));
          if (d < degreeIndex.size()) {
            degreeIndex[d] = i - searchStart;
            values[i - searchStart].resize(numSimplices[d] * numParam);
          }
        }
        std::vector<std::size_t> currState(values.size(), 0);
        for (const auto& sh : Base::complex_simplex_range()) {
          const auto& f = Base::get_filtration_value(sh);
          const auto dim = Base::dimension(sh);
          if (std::find(view.data() + searchStart, view.data() + view.shape(0), dim) != view.data() + view.shape(0)) {
            Gudhi::Simple_mdspan view(values[degreeIndex[dim]].data(), numParam, numSimplices[dim]);
            auto& i = currState[degreeIndex[dim]];
            for (std::size_t g = 0; g < f.num_generators(); ++g) {
              for (std::size_t p = 0; p < numParam; ++p) view(p, i) = f(g, p);
              ++i;
            }
          }
        }
      }
    }

    return Gudhi::python::_build_tuple(values.size(), [&](std::size_t d) {
      return _wrap_as_numpy_array(std::move(values[d]), numParam, values[d].size() / numParam);
    });
  }

  nanobind::tuple get_point_indices(Tensor2D<value_type> pts, Tensor1D<std::int32_t> dims) const {
    auto map = _build_idx_map(dims);

    auto viewPts = pts.view();
    std::size_t numParam = map.size();
    std::vector<std::int32_t> indices(pts.size() * numParam, -1);
    std::vector<std::array<std::int32_t, 2>> unmappedValues;

    {
      nanobind::gil_scoped_release release;

      Gudhi::Simple_mdspan indexView(indices.data(), viewPts.shape(0), numParam);
      for (std::size_t i = 0; i < viewPts.shape(0); ++i) {
        for (std::size_t p = 0; p < numParam; ++p) {
          const auto& paramMap = map[p];
          auto it = paramMap.find(viewPts(i, p));
          if (it == paramMap.end()) {
            unmappedValues.push_back({static_cast<std::int32_t>(i), static_cast<std::int32_t>(p)});
          } else {
            indexView(i, p) = it->second;
          }
        }
      }
    }

    return nanobind::make_tuple(_wrap_as_numpy_array(std::move(indices), viewPts.shape(0), numParam),
                                _wrap_as_numpy_array(std::move(unmappedValues)));
  }

  Multi_simplex_tree_interface& fill_lowerstar(nanobind::object filtration, int axis) {
    if (filtration.is_none()) throw std::invalid_argument("Filtration values cannot be None.");
    // assuming Base::num_parameters() was properly set
    if (axis < 0) axis += Base::num_parameters();
    if (axis < 0 || axis >= Base::num_parameters()) throw std::invalid_argument("Axis is not a valid parameter index.");

    auto cast_as_vector = [&]() -> void {
      std::vector<value_type> val;
      if (!nanobind::try_cast<std::vector<value_type>>(filtration, val))
        throw std::invalid_argument("Filtration values must be either iterable[U] or ndarray[U, ndim=1].");
      // if vertex indexing is continuous, this catches all filtration size problems
      // but if not, the condition is not sufficient, but at least not wrong
      if (val.size() < Base::num_vertices())
        throw std::invalid_argument("Vertex filtration values does not have a value for every vertex.");
      {
        nanobind::gil_scoped_release release;
        Gudhi::multi_persistence::fill_axis_with_lowerstar(*this, val, static_cast<std::size_t>(axis));
      }
    };
    auto cast_first_as_tensor_then_as_vector = [&]<typename U>() -> void {
      if (Tensor1D<U> val; nanobind::try_cast<Tensor1D<U>>(filtration, val, false)) {
        if (val.shape(0) < Base::num_vertices())
          throw std::invalid_argument("Vertex filtration values does not have a value for every vertex.");
        {
          nanobind::gil_scoped_release release;
          Gudhi::multi_persistence::fill_axis_with_lowerstar(*this, Numpy_span(val), static_cast<std::size_t>(axis));
        }
        return;
      }
      cast_as_vector();
    };
    detail::_dispatch_dtype(
        filtration,
        cast_first_as_tensor_then_as_vector,
        [this]() -> void {
          if (Base::num_vertices() != 0)
            throw std::invalid_argument("Vertex filtration values is empty, but not the simplex tree.");
        },
        cast_as_vector);
    return *this;
  }

  Multi_simplex_tree_interface& fill_distance_matrix(Tensor2D<value_type> distanceMatrix,
                                                     int axis,
                                                     value_type nodeValue) {
    // assuming Base::num_parameters() was properly set
    if (axis < 0) axis += Base::num_parameters();
    if (axis < 0 || axis >= Base::num_parameters()) throw std::invalid_argument("Axis is not a valid parameter index.");

    if (distanceMatrix.ndim() != 2 || distanceMatrix.shape(0) < Base::num_vertices() ||
        distanceMatrix.shape(1) < Base::num_vertices())
      throw std::invalid_argument(
          "Distance matrix has to be a squared 2-dimensional matrix with entries for at least all vertices in the "
          "simplex tree.");
    {
      nanobind::gil_scoped_release release;
      Gudhi::multi_persistence::fill_axis_with_distance_matrix(
          *this, Numpy_2d_span(distanceMatrix), nodeValue, static_cast<std::size_t>(axis));
    }
    return *this;
  }

  template <typename U>
  Multi_simplex_tree_interface& coarsen_on_grid(const std::vector<std::vector<U>>& grid, bool coordinates = true) {
    if (static_cast<int>(grid.size()) < Base::num_parameters()) {
      throw std::invalid_argument("Grid and simplex tree do not agree on number of parameters.");
    }
    {
      nanobind::gil_scoped_release release;
      for (auto sh : Base::complex_simplex_range()) {
        Base::get_filtration_value(sh).project_onto_grid(grid, coordinates);
      }
    }
    return *this;
  }

  template <typename U>
  Multi_simplex_tree_interface& coarsen_on_grid(const std::vector<Tensor1D<U>>& grid, bool coordinates = true) {
    std::vector<Numpy_span<U>> views(grid.begin(), grid.end());
    if (static_cast<int>(grid.size()) < Base::num_parameters()) {
      throw std::invalid_argument("Grid and simplex tree do not agree on number of parameters.");
    }
    {
      nanobind::gil_scoped_release release;
      for (auto sh : Base::complex_simplex_range()) {
        Base::get_filtration_value(sh).project_onto_grid(views, coordinates);
      }
    }
    return *this;
  }

  Multi_simplex_tree_interface& clean_filtration_grid() {
    if (filtrationGrid_.is_none()) throw std::runtime_error("No grid to clean.");
    auto usedCoordinates = detail::Compacted_squeezed_filtration_grid::collect_used_squeezed_coordinates(*this);
    detail::Compacted_squeezed_filtration_grid compact(filtrationGrid_, usedCoordinates);
    filtrationGrid_ = compact.filtrationGrid;
    return coarsen_on_grid(compact.coordinates, true);
  }

  Multi_simplex_tree_interface& simplify_all_filtration_values() {
    {
      nanobind::gil_scoped_release release;
      for (auto sh : Base::complex_simplex_range()) {
        Base::get_filtration_value(sh).simplify();
      }
    }
    return *this;
  }

  template <typename U>
  Multi_simplex_tree_interface& normalize_filtration_values(const std::optional<Tensor2D<U>>& box) {
    if constexpr (MultiFiltrationValue::Storage_policy::has_an_implicit_axis) {
      throw nanobind::type_error("Degree-Rips slicers cannot be affinely normalized.");
    } else if constexpr (!std::is_floating_point_v<value_type>) {
      throw nanobind::type_error("Normalize filtration requires a floating-point dtype for slicers.");
    } else {
      {
        nanobind::gil_scoped_release release;
        auto for_each = [this](auto&& to_apply) {
          Base::for_each_simplex([this, &to_apply](Simplex_handle sh, [[maybe_unused]] int dim) {
            auto& f = Base::get_filtration_value(sh);
            to_apply(f);
          });
        };
        if (box.has_value()) {
          if (box->shape(0) != 2 || box->shape(1) != static_cast<std::size_t>(Base::num_parameters()))
            throw std::invalid_argument("Box must have shape (2, num_parameters).");
          auto boxView = Numpy_2d_span(*box);
          auto lowerView = boxView[0];
          auto upperView = boxView[1];
          normalize_filtration_values_in_complex(
              *this, for_each, {lowerView.begin(), lowerView.end(), upperView.begin(), upperView.end()});
        } else {
          normalize_filtration_values_in_complex(*this, for_each);
        }
      }
      return *this;
    }
  }

  Multi_simplex_tree_interface build_unsqueezed_from(const std::vector<std::vector<value_type>>& grid) const {
    Base out;
    {
      nanobind::gil_scoped_release release;
      out = Base(*this, [&](const Filtration_value& fil) -> Filtration_value {
        return evaluate_coordinates_in_grid<value_type>(fil, grid);
      });
    }
    return {std::move(out)};
  }

  template <typename U = value_type>
  Multi_simplex_tree_interface build_bifiltration_from_edges(Tensor2D<U> edges, int expansionDimension) const {
    auto edgeView = edges.view();
    if (edgeView.shape(1) != 4) {
      throw std::invalid_argument("Expected edge array with shape (n_edges, 4). Got (" +
                                  std::to_string(edgeView.shape(0)) + ", " + std::to_string(edgeView.shape(1)) + ").");
    }

    Multi_simplex_tree_interface out;
    out.set_num_parameters(2);
    out.filtrationGrid_ = get_filtration_grid();
    {
      nanobind::gil_scoped_release release;
      for (auto sh : Base::skeleton_simplex_range(0)) {
        auto& fil = Base::get_filtration_value(sh);
        // or better just throw if fil.num_parameters() != 2 ?
        auto fil2param = fil.num_parameters() == 2 ? fil : fil.copy(2, fil.num_generators());
        out._insert_single_simplex(Base::simplex_vertex_range(sh), fil2param);
      }

      std::array<Vertex_handle, 2> edge;
      Filtration_value fil(2);
      for (std::size_t i = 0; i < edgeView.shape(0); ++i) {
        edge[0] = static_cast<Vertex_handle>(edgeView(i, 0));
        edge[1] = static_cast<Vertex_handle>(edgeView(i, 1));
        fil(0, 0) = static_cast<value_type>(edgeView(i, 2));
        fil(0, 1) = static_cast<value_type>(edgeView(i, 3));
        out._insert_single_simplex(edge, fil);
      }

      if (expansionDimension > 0) {
        out.expansion(expansionDimension);
      }
    }
    return out;
  }

  template <typename U = value_type>
  auto project_on_line_to_std(Tensor1D<U> basepoint, Tensor1D<U> direction, int dimension) const {
    std::vector<char> buffer;
    {
      nanobind::gil_scoped_release release;
      Numpy_span baseView(basepoint);
      Numpy_span dirView(direction);
      Line<U> line = Line<U>(baseView.begin(), baseView.end(), dirView.begin(), dirView.end());
      auto st = Gudhi::multi_persistence::make_one_dimensional<Gudhi::Simplex_tree_options_for_python>(
          *this, line, dimension);
      // serialize to be able to transfer it to a python simplex tree imported from gudhi and not multipers
      buffer.resize(st.get_serialization_size());
      st.serialize(buffer.data(), buffer.size());
    }
    return _wrap_as_numpy_array(std::move(buffer), buffer.size());
  }

  nanobind::tuple serialize() const {
    std::size_t buffer_size;
    std::unique_ptr<char[]> buffer;
    {
      nanobind::gil_scoped_release release;
      buffer_size = Base::get_serialization_size();
      buffer.reset(new char[buffer_size]);  // no leak in case serialize throws
      // also adds version
      Base::serialize(buffer.get(), buffer_size);
    }
    return nanobind::make_tuple(filtrationGrid_, _wrap_as_numpy_array(std::move(buffer), buffer_size));
  }

  void deserialize(nanobind::tuple state) {
    if (nanobind::len(state) != 2)
      throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");

    nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> data;
    if (!nanobind::try_cast<nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy>>(state[1], data, false))
      throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
    {
      nanobind::gil_scoped_release release;
      // also checks version
      Base::deserialize(data.data(), data.size());
    }
    set_filtration_grid(state[0]);
  }

 private:
  nanobind::object filtrationGrid_;

  template <class Iterator>
  class Simplex_filtration_iterator : public boost::iterator_facade<Simplex_filtration_iterator<Iterator>,
                                                                    nanobind::tuple,
                                                                    boost::forward_traversal_tag,
                                                                    nanobind::tuple> {
   public:
    Simplex_filtration_iterator(const Iterator& start, Multi_simplex_tree_interface const* tree = nullptr)
        : curr_(start), tree_(tree) {}

   private:
    friend class boost::iterator_core_access;

    bool equal(Simplex_filtration_iterator const& other) const { return curr_ == other.curr_; }

    nanobind::tuple dereference() const {
      // here just in case, but should never happen as never directly used in python
      if (tree_ == nullptr) throw std::runtime_error("Iterator is at the end of the range.");
      return tree_->_get_simplex_and_filtration(*curr_);
    }

    void increment() { ++curr_; }

    Iterator curr_;
    Multi_simplex_tree_interface const* tree_;
  };

  template <class VertexRange>
  std::pair<Simplex_handle, bool> _insert_single_simplex(const VertexRange& vertices) {
    return Base::insert_simplex_and_subfaces(
        Base::Filtration_maintenance::INCREASE_NEW, vertices, Filtration_value::minus_inf(Base::num_parameters()));
  }

  template <class VertexRange>
  std::pair<Simplex_handle, bool> _insert_single_simplex(const VertexRange& vertices,
                                                         Filtration_value& filtrationValue) {
    // I still don't understand why 1-critical and k-critical simplices are not inserted with the same strategy.
    // That just feels inconsistent. If they are not used in the same situation, you could allow to pass
    // the strategy instead to make sense, no? In particular when the user could have completely different
    // use cases than the examples here.
    if constexpr (Filtration_value::ensures_1_criticality()) {
      return Base::insert_simplex_and_subfaces(Base::Filtration_maintenance::INCREASE_NEW, vertices, filtrationValue);
    } else {
      // TODO: insert_simplex_and_subfaces calls unify_lifetimes which calls add_generator which assumes filtration
      // is simplified. As simplify is not exactly cheap, we could also only call it when we not know if it is
      // simplified higher in the call chain. That is, add the simplify for external inserts and not use it for
      // internal inserts when we know for sure that it is already simplified.
      filtrationValue.simplify();
      return Base::insert_simplex_and_subfaces(Base::Filtration_maintenance::LOWER_EXISTING, vertices, filtrationValue);
    }
  }

  Simplex_handle _get_handle_from_vertices(nanobind::object simplex) const {
    auto cast_as_iterable = [&]() -> Simplex_handle {
      return Base::find(detail::as_cpp_range<Vertex_handle>(simplex));
    };
    auto cast_first_as_tensor_then_as_iterable = [&]<typename U>() -> Simplex_handle {
      if (Tensor1D<U> val; nanobind::try_cast<Tensor1D<U>>(simplex, val, false)) {
        nanobind::gil_scoped_release release;
        return Base::find(Numpy_span(val));
      }
      return cast_as_iterable();
    };
    return detail::_dispatch_dtype(
        simplex,
        cast_first_as_tensor_then_as_iterable,
        []() -> Simplex_handle { return Base::null_simplex(); },
        []() -> Simplex_handle { throw std::invalid_argument("Simplex has to be an iterable of int or float."); });
  }

  nanobind::tuple _get_simplex_and_filtration(Simplex_handle sh) const {
    Simplex simplex;
    for (auto vertex : Base::simplex_vertex_range(sh)) {
      simplex.push_back(vertex);
    }
    std::reverse(simplex.begin(), simplex.end());
    const auto& fil = Base::get_filtration_value(sh);
    return nanobind::make_tuple(_wrap_as_numpy_array(std::move(simplex), simplex.size()),
                                detail::_get_filtration_array(fil));
  }

  template <class Iterator>
  auto _make_iterator(const char* name, Iterator start, Iterator end) const {
    return nanobind::make_iterator(nanobind::type<Multi_simplex_tree_interface>(),
                                   name,
                                   Simplex_filtration_iterator<Iterator>(start, this),
                                   Simplex_filtration_iterator<Iterator>(end));
  }

  std::vector<std::map<value_type, std::int32_t>> _build_idx_map(Tensor1D<std::int32_t> dimensionsByParam) const {
    auto viewDims = dimensionsByParam.view();
    std::size_t numParam = Base::num_parameters();
    if (viewDims.shape(0) < numParam) throw std::invalid_argument("Not enough dimensions for all parameters.");

    std::int32_t maxDim = *std::ranges::max_element(viewDims.data(), viewDims.data() + viewDims.shape(0));
    std::int32_t minDim = *std::ranges::min_element(viewDims.data(), viewDims.data() + viewDims.shape(0));
    // if there is at least one -1, we have to test for every parameter
    maxDim = minDim >= 0 ? maxDim : Base::dimension();

    std::vector<std::map<value_type, std::int32_t>> map(numParam);
    std::int32_t idx = 0;
    // has to be a fixed order for the idx to make sense outside of this method
    for (auto sh : Base::complex_simplex_range()) {
      const auto& fil = Base::filtration(sh);
      if (fil.num_generators() > 1) throw std::invalid_argument("Multicritical not supported yet");
      if (numParam != fil.num_parameters())
        throw std::runtime_error("Inconsistent number of parameters in stored filtration values");
      const std::int32_t dim = Base::dimension(sh);
      if (dim <= maxDim) {
        for (std::size_t p = 0; p < numParam; ++p) {
          if (viewDims(p) == -1 || viewDims(p) == dim) {
            // stores only the first encountered filtration value element with that value
            map[p].try_emplace(fil(0, p), idx);
          }
        }
      }
      ++idx;
    }

    return map;
  }
};

template <class MultiSimplexTreeInterface>
inline MultiSimplexTreeInterface deserialize_multi_simplex_tree_from_python(nanobind::tuple state) {
  MultiSimplexTreeInterface st;
  st.deserialize(state);
  return st;
}

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_MULTI_SIMPLEX_TREE_H_INCLUDED
