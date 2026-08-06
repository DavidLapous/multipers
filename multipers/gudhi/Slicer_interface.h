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
 * @file Slicer_interface.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the @ref Gudhi::multi_persistence::Slicer_interface class for python bindings.
 */

#ifndef MP_PY_SLICER_H_INCLUDED
#define MP_PY_SLICER_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <oneapi/tbb/parallel_for.h>

#include <gudhi/simple_mdspan.h>
#include <gudhi/slicer_helpers.h>
#include <gudhi/Slicer.h>
#include <gudhi/multi_persistence_landscapes.h>
#include <gudhi/Degree_rips_bifiltration.h>
#include <gudhi/Multi_persistence/Line.h>
#include <gudhi/Multi_persistence/utils.h>
#include <python_interfaces/numpy_utils.h>

#include "Simplex_tree_multi_interface.h"
#include "python_interfaces/construction_utils.h"
#include "slicer_interface_helpers.h"
#include "interface_helper_structs.h"
#include "ext_interface/nanobind_wrapper_types.hpp"

namespace Gudhi {
namespace multi_persistence {

/**
 * @private
 */
template <class MultiFiltrationValue, class PersistenceAlgorithm>
class Slicer_interface {
 public:
  using value_type = typename MultiFiltrationValue::value_type;
  using Slicer_t = Slicer<MultiFiltrationValue, PersistenceAlgorithm>;
  using Complex = typename Slicer_t::Complex;
  using Dimension = typename Slicer_t::Dimension;
  using Index = typename Slicer_t::Index;
  template <typename U>
  using Tensor1D = nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>;
  template <typename U>
  using Tensor2D = nanobind::ndarray<const U, nanobind::ndim<2>>;

  // as this needs to be updated only when the serialization strategy changes, 255 updates should be enough?
  static constexpr std::uint8_t SERIALIZATION_VERSION = 0;

  static constexpr value_type T_inf = MultiFiltrationValue::T_inf;     /**< Infinity. */
  static constexpr value_type T_m_inf = MultiFiltrationValue::T_m_inf; /**< Minus infinity. */

  Slicer_interface()
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {};

  template <class OtherMultiFiltrationValue, class OtherPersistenceAlgorithm>
  Slicer_interface(const Slicer_interface<OtherMultiFiltrationValue, OtherPersistenceAlgorithm> &other)
      : slicer_(other.get_slicer()),
        filtrationGrid_(other.get_filtration_grid()),
        generatorBasis_(other.get_generator_basis()),
        presDegree_(other.get_pres_degree()),
        isMinPres_(other.is_min_pres()),
        isMinRes_(other.is_min_res()) {}

  template <class OtherMultiFiltrationValue, class OtherPersistenceAlgorithm>
  Slicer_interface(const Slicer_interface<OtherMultiFiltrationValue, OtherPersistenceAlgorithm> &other,
                   Slicer_t &&slicer)
      : slicer_(std::move(slicer)),
        filtrationGrid_(other.get_filtration_grid()),
        generatorBasis_(other.get_generator_basis()),
        presDegree_(other.get_pres_degree()),
        isMinPres_(other.is_min_pres()),
        isMinRes_(other.is_min_res()) {}

  // use Simplex_tree_multi_interface<OtherMultiFiltrationValue> instead once the weird wrapper thing is removed
  template <class OtherMultiFiltrationValue>
  Slicer_interface(multipers::nanobind_helpers::PySimplexTree<
                   Gudhi::multiparameter::python_interface::Simplex_tree_multi_interface<OtherMultiFiltrationValue>,
                   typename OtherMultiFiltrationValue::value_type> &simplexTree)
      : slicer_(), filtrationGrid_(simplexTree.filtration_grid), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    {
      nanobind::gil_scoped_release release;
      slicer_ = Gudhi::multi_persistence::build_slicer_from_simplex_tree<Slicer_t>(simplexTree.tree);
    }
  }

  Slicer_interface(const std::string &path, int shiftDimension, bool isRivetCompatible = false, bool isReversed = false)
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    {
      nanobind::gil_scoped_release release;
      slicer_ = Gudhi::multi_persistence::build_slicer_from_scc_file<Slicer_t>(
          path, isRivetCompatible, isReversed, shiftDimension);
    }
  }

  Slicer_interface(const std::vector<std::vector<Index>> &generator_maps,
                   const std::vector<Index> &generator_dimensions,
                   const std::vector<std::vector<value_type>> &filtration_values)
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    static_assert(MultiFiltrationValue::ensures_1_criticality(),
                  "Slicer constructor only available for 1-critical filtration values. Use sequence[sequence[U]] for "
                  "filtration value type.");
    _build_slicer(generator_maps, generator_dimensions, filtration_values);
  }

  Slicer_interface(const std::vector<std::vector<Index>> &generator_maps,
                   const std::vector<Index> &generator_dimensions,
                   const std::vector<std::vector<std::vector<value_type>>> &filtration_values)
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    static_assert(!MultiFiltrationValue::ensures_1_criticality(),
                  "Slicer constructor only available for k-critical filtration values. Use "
                  "sequence[sequence[sequence[U]]] for filtration value type.");
    _build_slicer(generator_maps, generator_dimensions, filtration_values);
  }

  template <typename I>
  Slicer_interface(const std::vector<std::vector<Index>> &generator_maps,
                   Tensor1D<I> generator_dimensions,
                   nanobind::object filtration_values)
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    if constexpr (MultiFiltrationValue::ensures_1_criticality()) {
      auto cast_as_vector = [&]() -> void {
        std::vector<std::vector<value_type>> val;
        if (!nanobind::try_cast<std::vector<std::vector<value_type>>>(filtration_values, val))
          throw std::invalid_argument("Filtration values must be either iterable[iterable[U]] or ndarray[U, ndim=2].");
        _build_slicer(generator_maps, Numpy_span(generator_dimensions), val);
      };
      auto cast_first_as_tensor_then_as_vector = [&]<typename U>() -> void {
        if (Tensor2D<U> val; nanobind::try_cast<Tensor2D<U>>(filtration_values, val, false)) {
          _build_slicer(generator_maps, Numpy_span(generator_dimensions), Numpy_2d_span(val));
          return;
        }
        cast_as_vector();
      };
      detail::_dispatch_dtype(filtration_values, cast_first_as_tensor_then_as_vector, []() -> void {}, cast_as_vector);
    } else {
      auto cast_as_vector = [&]() -> void {
        std::vector<std::vector<std::vector<value_type>>> val;
        if (!nanobind::try_cast<std::vector<std::vector<std::vector<value_type>>>>(filtration_values, val))
          throw std::invalid_argument(
              "Filtration values must be either iterable[iterable[iterable[U]]] or iterable[ndarray[U, ndim=2]].");
        _build_slicer(generator_maps, Numpy_span(generator_dimensions), val);
      };
      auto cast_first_as_tensor_then_as_vector = [&]<typename U>() -> void {
        if (std::vector<Tensor2D<U>> val; nanobind::try_cast<std::vector<Tensor2D<U>>>(filtration_values, val, false)) {
          // Tensors have to stay alive to use Numpy_2d_span, so val is necessary
          std::vector<Numpy_2d_span<U>> fils(val.begin(), val.end());
          _build_slicer(generator_maps, Numpy_span(generator_dimensions), fils);
          return;
        }
        cast_as_vector();
      };
      detail::_dispatch_dtype(filtration_values, cast_first_as_tensor_then_as_vector, []() -> void {}, cast_as_vector);
    }
  }

  template <typename I1, typename I2, typename F>
  Slicer_interface(Tensor1D<I1> boundary_indptr,
                   Tensor1D<I2> boundary_flat,
                   Tensor1D<I2> generator_dimensions,
                   Tensor2D<F> grades_flat)
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    auto boundaryDelimitersView = boundary_indptr.view();
    auto boundariesView = boundary_flat.view();
    auto dimensionsView = generator_dimensions.view();
    auto filValuesView = grades_flat.view();

    if (boundaryDelimitersView.shape(0) == 0) {
      if (boundariesView.shape(0) != 0 || dimensionsView.shape(0) != 0 || filValuesView.shape(0) != 0)
        throw std::invalid_argument("Invalid packed input, shapes do not coincide.");
      return;
    }
    std::size_t numGen = boundaryDelimitersView.shape(0) - 1;
    if (boundaryDelimitersView(numGen) > boundariesView.shape(0))
      throw std::invalid_argument("Boundary index ptr and flat boundaries are not coherent.");
    if (dimensionsView.shape(0) != numGen || filValuesView.shape(0) != numGen)
      throw std::invalid_argument("Invalid packed input, shapes do not coincide.");

    // do we really want to test here if the values of boundary_indptr are positive and increasing integers,
    // and the values of boundary_flat and generator_dimensions positive integers ?
    // those testes are not that cheap anymore and do not guarantee no crashes as there still will be some
    // if the boundaries are not valid boundaries or the filtration values do not yield a valid filtration order etc.
    // At some point, the user has to take responsibilities...

    detail::Flat_2D_array_span boundaries(boundary_indptr, boundary_flat);
    Numpy_span dimensions(generator_dimensions);
    Numpy_2d_span filValues(grades_flat);

    _build_slicer(boundaries, dimensions, filValues);
  }

  // std::vector<unsigned int> imposed by Gudhi::cubical_complex::Bitmap_cubical_complex
  Slicer_interface(Tensor2D<value_type> image, const std::vector<unsigned int> &shape)
      : slicer_(), filtrationGrid_(nanobind::none()), presDegree_(-1), isMinPres_(false), isMinRes_(false) {
    Numpy_2d_span imageView(image);
    if (imageView.size() == 0 || shape.size() == 0 || shape[0] == 0) return;
    {
      nanobind::gil_scoped_release release;
      std::vector<MultiFiltrationValue> vertices;
      vertices.reserve(imageView.size());
      for (std::size_t i = 0; i < imageView.size(); ++i) {
        auto rowView = imageView[i];
        vertices.emplace_back(rowView.begin(), rowView.end());
      }
      slicer_ = Gudhi::multi_persistence::build_slicer_from_bitmap<Slicer_t>(vertices, shape);
    }
  }

  template <class OtherMultiFiltrationValue, class OtherPersistenceAlgorithm>
  Slicer_interface &copy(const Slicer_interface<OtherMultiFiltrationValue, OtherPersistenceAlgorithm> &other) {
    *this = Slicer_interface(other);
    return *this;
  }

  // use Simplex_tree_multi_interface<OtherMultiFiltrationValue> instead once the weird wrapper thing is removed
  template <class OtherMultiFiltrationValue>
  Slicer_interface &copy(
      multipers::nanobind_helpers::PySimplexTree<
          Gudhi::multiparameter::python_interface::Simplex_tree_multi_interface<OtherMultiFiltrationValue>,
          typename OtherMultiFiltrationValue::value_type> &other) {
    *this = Slicer_interface(other);
    return *this;
  }

  Slicer_t &get_slicer() { return slicer_; }

  const Slicer_t &get_slicer() const { return slicer_; }

  [[nodiscard]] nanobind::object get_filtration_grid() const { return filtrationGrid_; }

  void set_filtration_grid(nanobind::object grid) {
    if (grid.is_none()) {
      filtrationGrid_ = nanobind::none();
      return;
    }

    // throws if it does not pass the check
    // returns false if valid but empty
    if (_verify_grid_validity(grid)) {
      filtrationGrid_ = grid;
      return;
    }

    filtrationGrid_ = nanobind::none();
  }

  [[nodiscard]] std::optional<detail::Generator_basis_data> get_generator_basis() const { return generatorBasis_; }

  void set_generator_basis(const std::optional<detail::Generator_basis_data> &basis) {
    if (basis.has_value()) {
      generatorBasis_ = basis;
      return;
    }
    generatorBasis_.reset();
  }

  void set_generator_basis(nanobind::dict basis) {
    if (basis.empty() || basis.is_none()) {
      generatorBasis_.reset();
      return;
    }
    generatorBasis_ = detail::Generator_basis_data(basis);
  }

  template <class Complex, class GeneratorMatrix>
  void set_generator_basis(const Complex &complex, int degree, GeneratorMatrix &generatorMatrix) {
    generatorBasis_ = detail::Generator_basis_data(complex, degree, generatorMatrix);
  }

  [[nodiscard]] int get_min_pres_degree() const { return isMinPres_ ? presDegree_ : -1; }

  void set_min_pres_degree(int degree, bool isMinRes = false) {
    isMinPres_ = degree >= 0;
    if (isMinPres_) presDegree_ = degree;
    isMinRes_ = isMinPres_ && isMinRes;
  }

  [[nodiscard]] int get_pres_degree() const { return presDegree_; }

  [[nodiscard]] bool is_pres() const { return presDegree_ >= 0; }

  void set_is_pres(int degree, bool isMinPres = false) {
    presDegree_ = degree;
    isMinPres_ = degree >= 0 ? isMinPres : false;
    isMinRes_ = isMinPres_;
  }

  [[nodiscard]] bool is_min_pres() const { return isMinPres_; }

  [[nodiscard]] bool is_min_res() const { return isMinRes_; }

  void set_is_min_res(bool isMinRes) {
    if (isMinRes && !isMinPres_)
      throw std::invalid_argument("Cannot mark a slicer as `is_minres` without a valid `minpres_degree`.");
    isMinRes_ = isMinRes;
  }

  [[nodiscard]] int size() const { return slicer_.get_number_of_cycle_generators(); }

  [[nodiscard]] int get_number_of_cycle_generators() const { return slicer_.get_number_of_cycle_generators(); }

  [[nodiscard]] int get_number_of_parameters() const { return slicer_.get_number_of_parameters(); }

  [[nodiscard]] nanobind::object get_max_dimension() const {
    auto dim = slicer_.get_max_dimension();
    if (dim == Complex::nullDimension) return nanobind::float_(-std::numeric_limits<double>::infinity());
    return nanobind::int_(dim);
  }

  [[nodiscard]] auto get_dimensions() const {
    const auto &dims = slicer_.get_dimensions();
    // no transfer of ownership, dies together with the slicer
    return _wrap_view_as_numpy_array(nanobind::find(this), dims.data(), dims.size());
  }

  [[nodiscard]] nanobind::tuple get_boundaries() const {
    const auto &boundaries = slicer_.get_boundaries();
    return Gudhi::python::_build_tuple(boundaries.size(), [&](std::size_t b) {
      // no transfer of ownership, dies together with the slicer
      return _wrap_view_as_numpy_array(nanobind::find(this), boundaries[b].data(), boundaries[b].size());
    });
  }

  [[nodiscard]] nanobind::tuple get_flat_boundaries() const {
    std::vector<std::uint64_t> startIndices;
    std::vector<std::uint32_t> boundaries;

    {
      nanobind::gil_scoped_release release;
      const auto &b = slicer_.get_boundaries();
      startIndices.resize(b.size() + 1, 0);
      for (std::size_t i = 0; i < b.size(); ++i) {
        startIndices[i + 1] = startIndices[i] + b[i].size();
      }
      boundaries.reserve(startIndices.back());
      for (const auto &bi : b) {
        boundaries.insert(boundaries.end(), bi.begin(), bi.end());
      }
    }

    return nanobind::make_tuple(_wrap_as_numpy_array(std::move(startIndices), startIndices.size()),
                                _wrap_as_numpy_array(std::move(boundaries), boundaries.size()));
  }

  [[nodiscard]] nanobind::object get_filtration_value(int index, bool viewIfPossible = true, bool raw = false) {
    int size = slicer_.get_number_of_cycle_generators();
    if (index < 0) index += size;
    if (index < 0 || index >= size) throw std::out_of_range("Generator index out of range.");

    auto &f = slicer_.get_filtration_value(index);

    if (raw) return detail::_get_raw_filtration_data(f, !viewIfPossible);

    // view not possible for Degree_rips_bifiltration
    if constexpr (!detail::_is_degree_rips<MultiFiltrationValue>()) {
      if (viewIfPossible) return detail::_get_raw_filtration_data(f, false);
    }
    return nanobind::cast(_get_filtration_array(f));
  }

  [[nodiscard]] nanobind::object get_all_filtration_values(bool compact,
                                                           bool viewIfPossible = true,
                                                           bool raw = false) {
    auto &filts = slicer_.get_filtration_values();

    // view not possible for compact
    if (compact) {
      // raw makes only a difference for Degree_rips
      if constexpr (detail::_is_degree_rips<MultiFiltrationValue>()) {
        if (raw) return detail::_get_compact_filtration_data(filts);
      }
      return _get_compact_filtration_array(filts, slicer_.get_number_of_parameters());
    }

    if (raw) {
      return Gudhi::python::_build_tuple(
          filts.size(), [&](std::size_t i) { return detail::_get_raw_filtration_data(filts[i], !viewIfPossible); });
    }

    // view not possible for Degree_rips_bifiltration
    if constexpr (!detail::_is_degree_rips<MultiFiltrationValue>()) {
      if (viewIfPossible) {
        return Gudhi::python::_build_tuple(
            filts.size(), [&](std::size_t i) { return detail::_get_raw_filtration_data(filts[i], false); });
      }
    }
    return _get_filtration_array(filts, slicer_.get_number_of_parameters());
  }

  [[nodiscard]] auto get_current_slice() const {
    const auto &slice = slicer_.get_slice();
    // no transfer of ownership, dies together with the slicer
    return _wrap_view_as_numpy_array(nanobind::find(this), slice.data(), slice.size());
  }

  template <typename U>
  Slicer_interface &push_to_line(Tensor1D<U> basepoint, std::optional<Tensor1D<U>> direction) {
    {
      nanobind::gil_scoped_release release;
      Numpy_span baseView(basepoint);
      Line<U> line;
      if (direction.has_value()) {
        Numpy_span dirView(*direction);
        line = Line<U>(baseView.begin(), baseView.end(), dirView.begin(), dirView.end());
      } else {
        line = Line<U>(baseView.begin(), baseView.end());
      }
      slicer_.push_to(line);
    }
    return *this;
  }

  Slicer_interface &make_filtration_non_decreasing() {
    {
      nanobind::gil_scoped_release release;
      // validity of grid was already tested when set
      slicer_.make_filtration_non_decreasing();
    }
    return *this;
  }

  Slicer_interface &simplify_all_filtration_values() {
    {
      nanobind::gil_scoped_release release;
      for (auto &f : slicer_.get_filtration_values()) f.simplify();
    }
    return *this;
  }

  Slicer_interface &prune_above_dimension(int max_dimension) {
    {
      nanobind::gil_scoped_release release;
      slicer_.prune_above_dimension(max_dimension);
    }
    return *this;
  }

  template <typename U>
  Slicer_interface &coarsen_on_grid(const std::vector<std::vector<U>> &grid, bool coordinates) {
    {
      nanobind::gil_scoped_release release;
      slicer_.coarsen_on_grid(grid, coordinates);
    }
    return *this;
  }

  template <typename U>
  Slicer_interface &coarsen_on_grid(const std::vector<Tensor1D<U>> &grid, bool coordinates) {
    std::vector<Numpy_span<U>> views(grid.begin(), grid.end());
    {
      nanobind::gil_scoped_release release;
      slicer_.coarsen_on_grid(views, coordinates);
    }
    return *this;
  }

  template <typename U>
  Slicer_interface &normalize_filtration_values(const std::optional<Tensor2D<U>> &box) {
    if constexpr (std::is_same_v<MultiFiltrationValue,
                                 Gudhi::multi_filtration::Degree_rips_bifiltration<
                                     value_type,
                                     MultiFiltrationValue::has_negative_cones(),
                                     MultiFiltrationValue::ensures_1_criticality()>>) {
      throw nanobind::type_error("Degree-Rips slicers cannot be affinely normalized.");
    } else if constexpr (!std::is_floating_point_v<value_type>) {
      throw nanobind::type_error("Normalize filtration requires a floating-point dtype for slicers.");
    } else {
      {
        nanobind::gil_scoped_release release;
        if (box.has_value()) {
          if (box->shape(0) != 2 || box->shape(1) != slicer_.get_number_of_parameters())
            throw std::invalid_argument("Box must have shape (2, num_parameters).");
          auto boxView = Numpy_2d_span(*box);
          auto lowerView = boxView[0];
          auto upperView = boxView[1];
          slicer_.normalize_filtration_values({lowerView.begin(), lowerView.end(), upperView.begin(), upperView.end()});
        } else {
          slicer_.normalize_filtration_values();
        }
      }
      return *this;
    }
  }

  Slicer_interface &clean_filtration_grid() {
    if (filtrationGrid_.is_none()) throw std::runtime_error("No grid to clean.");
    auto usedCoordinates = detail::Compacted_squeezed_filtration_grid::collect_used_squeezed_coordinates(slicer_);
    detail::Compacted_squeezed_filtration_grid compact(filtrationGrid_, usedCoordinates);
    filtrationGrid_ = compact.filtrationGrid;
    return coarsen_on_grid(compact.coordinates, true);
  }

  Slicer_interface &initialize_persistence_computation(bool ignoreInf) {
    {
      nanobind::gil_scoped_release release;
      slicer_.initialize_persistence_computation(ignoreInf);
    }
    return *this;
  }

  Slicer_interface &update_persistence_computation(bool ignoreInf) {
    {
      nanobind::gil_scoped_release release;
      slicer_.update_persistence_computation(ignoreInf);
    }
    return *this;
  }

  nanobind::tuple compute_persistence_on_slices(Tensor2D<value_type> slices, bool ignoreInf) {
    std::vector<typename Slicer_t::template Multi_dimensional_flat_barcode<value_type>> barcodes;
    {
      nanobind::gil_scoped_release release;
      barcodes = persistence_on_slices<value_type>(slicer_, Numpy_2d_span(slices), ignoreInf);
    }

    return Gudhi::python::_build_tuple(barcodes.size(), [&](std::size_t i) {
      return Gudhi::python::_build_tuple(barcodes[i].size(), [&](std::size_t j) {
        return _wrap_as_numpy_array(std::move(barcodes[i][j]), barcodes[i][j].size(), 2);
      });
    });
  }

  [[nodiscard]] nanobind::tuple get_barcode() {
    typename Slicer_t::template Multi_dimensional_flat_barcode<value_type> barcode;
    {
      nanobind::gil_scoped_release release;
      barcode = slicer_.template get_flat_barcode<true>();
    }

    return Gudhi::python::_build_tuple(barcode.size(), [&](std::size_t i) {
      return _wrap_as_numpy_array(std::move(barcode[i]), barcode[i].size(), 2);
    });
  }

  [[nodiscard]] nanobind::tuple get_barcode_as_indices() {
    typename Slicer_t::template Multi_dimensional_flat_barcode<int> barcode;
    {
      nanobind::gil_scoped_release release;
      barcode = slicer_.template get_flat_barcode<true, int, true>();
    }

    return Gudhi::python::_build_tuple(barcode.size(), [&](std::size_t i) {
      return _wrap_as_numpy_array(std::move(barcode[i]), barcode[i].size(), 2);
    });
  }

  template <typename U>
  auto compute_landscapes_on_grid(Tensor1D<U> xGrid,
                                  Tensor1D<U> yGrid,
                                  Tensor1D<U> direction,
                                  std::size_t xStride,
                                  std::size_t yStride,
                                  double dt,
                                  int degree,
                                  Tensor1D<std::int32_t> ks,
                                  int n_jobs,
                                  bool ignoreInf = true) {
    auto xView = xGrid.view();
    auto yView = yGrid.view();
    auto dirView = direction.view();
    auto kView = ks.view();

    const std::size_t nx = xView.shape(0);
    const std::size_t ny = yView.shape(0);

    if (get_number_of_parameters() != 2)
      throw nanobind::value_error("Landscapes can only be computed for bi-filtrations.");
    if (nx == 0 || ny == 0) throw nanobind::value_error("Landscape grid axes must be non-empty.");
    if (direction.shape(0) != 2) throw nanobind::value_error("Landscape direction must be two-dimensional.");
    if (xStride == 0 || yStride == 0) throw nanobind::value_error("Landscape grid strides must be strictly positive.");
    if (!std::isfinite(dt) || dt <= 0.0)
      throw nanobind::value_error("Landscape grid step must be finite and strictly positive.");
    if (nx > std::numeric_limits<std::size_t>::max() / ny)
      throw nanobind::value_error("Landscape output grid is too large.");
    if (degree < 0) throw nanobind::value_error("Degree has to be positive.");

    for (std::size_t i = 0; i < nx; ++i) {
      if (!std::isfinite(xView(i))) throw nanobind::value_error("Landscape x-grid must be finite.");
    }
    for (std::size_t i = 0; i < ny; ++i) {
      if (!std::isfinite(yView(i))) throw nanobind::value_error("Landscape y-grid must be finite.");
    }
    for (std::size_t i = 0; i < dirView.shape(0); ++i) {
      if (!std::isfinite(dirView(i))) throw nanobind::value_error("Landscape direction must be finite.");
      if (dirView(i) <= 0.0) throw nanobind::value_error("Landscape direction must be strictly positive.");
    }
    for (std::size_t i = 0; i < kView.shape(0); ++i) {
      if (kView(i) < 0) throw nanobind::value_error("Landscape ks must be strictly positive.");
    }

    std::vector<double> out;

    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_slicer_landscapes_on_grid(slicer_,
                                                                        Numpy_span(xGrid),
                                                                        Numpy_span(yGrid),
                                                                        Numpy_span(direction),
                                                                        xStride,
                                                                        yStride,
                                                                        dt,
                                                                        degree,
                                                                        Numpy_span(ks),
                                                                        ignoreInf,
                                                                        n_jobs);
    }

    return _wrap_as_numpy_array(std::move(out), kView.shape(0), nx, ny);
  }

  [[nodiscard]] nanobind::object get_representative_cycles(
      bool update,
      const std::optional<Dimension> &dimension,
      nanobind::object barcodeIndices,
      const std::optional<Tensor1D<Index>> &pointsToIntersect) {
    auto get_cycle_list = [](auto &cycles) {
      nanobind::list outCycles;
      for (auto &c : cycles) {
        if (!c.empty()) {
          outCycles.append(Gudhi::python::_build_tuple(
              c.size(), [&](std::size_t b) { return _wrap_as_numpy_array(std::move(c[b]), c[b].size()); }));
        }
      }
      return outCycles;
    };

    if (dimension.has_value()) {
      std::optional<Tensor1D<std::int64_t>> indices;
      if (!barcodeIndices.is_none()) {
        Tensor1D<std::int64_t> tmp;
        if (!nanobind::try_cast<Tensor1D<std::int64_t>>(barcodeIndices, tmp, false))
          throw std::invalid_argument(
              "When dimension is specified, barcode_indices has to be either None or a 1D numpy array.");
        indices = std::move(tmp);
      }
      auto cycles = _get_cycle_boundaries(update, *dimension, indices, pointsToIntersect);
      return get_cycle_list(cycles);
    }

    std::optional<Tensor2D<std::int64_t>> indices;
    if (!barcodeIndices.is_none()) {
      Tensor2D<std::int64_t> tmp;
      if (!nanobind::try_cast<Tensor2D<std::int64_t>>(barcodeIndices, tmp, false))
        throw std::invalid_argument(
            "When dimension is not specified, barcode_indices has to be either None or a 2D numpy array.");
      indices = std::move(tmp);
    }
    auto cycles = _get_cycle_boundaries(update, indices, pointsToIntersect);
    return Gudhi::python::_build_tuple(cycles.size(), [&](std::size_t dim) { return get_cycle_list(cycles[dim]); });
  }

  [[nodiscard]] nanobind::object get_most_persistent_cycles(int dim, int n, bool update, bool idx) {
    if (dim < 0 || n < 0) throw std::invalid_argument("Dimension and number of cycles have to be positive.");

    std::vector<std::vector<Index>> cycleIdx;
    std::vector<std::vector<std::vector<Index>>> out;
    {
      nanobind::gil_scoped_release release;
      cycleIdx = slicer_.get_n_most_persistent_cycles(dim, n, update);

      if (!idx && !cycleIdx.empty()) {
        out.resize(cycleIdx.size());
        tbb::parallel_for(std::size_t(0), cycleIdx.size(), [&](std::size_t idx) {
          _get_cycle_boundary(out[idx], cycleIdx[idx], dim);
        });
      }
    }

    if (cycleIdx.empty()) return nanobind::make_tuple();

    if (n == 1) {
      if (idx) return nanobind::cast(_wrap_as_numpy_array(std::move(cycleIdx[0]), cycleIdx[0].size()));
      return Gudhi::python::_build_tuple(
          out[0].size(), [&](std::size_t i) { return _wrap_as_numpy_array(std::move(out[0][i]), out[0][i].size()); });
    }

    if (idx) {
      return Gudhi::python::_build_tuple(cycleIdx.size(), [&](std::size_t i) {
        return _wrap_as_numpy_array(std::move(cycleIdx[i]), cycleIdx[i].size());
      });
    }
    return Gudhi::python::_build_tuple(out.size(), [&](std::size_t i) {
      return Gudhi::python::_build_tuple(
          out[i].size(), [&](std::size_t b) { return _wrap_as_numpy_array(std::move(out[i][b]), out[i][b].size()); });
    });
  }

  Slicer_interface &write_to_scc_file(const std::string &outFilePath,
                                      int degree,
                                      bool rivetCompatible,
                                      bool ignoreLastGenerators,
                                      bool stripComments,
                                      bool reverse) {
    {
      nanobind::gil_scoped_release release;
      write_slicer_to_scc_file(
          outFilePath, slicer_, degree, rivetCompatible, ignoreLastGenerators, stripComments, reverse);
    }
    return *this;
  }

  Slicer_interface &sort_slicer_co_lexically() {
    std::pair<Slicer_t, std::vector<Index>> outSlicer;
    {
      nanobind::gil_scoped_release release;
      outSlicer = build_permuted_slicer(slicer_);
    }
    // TODO: complex has an internal sort, could be worth interfacing to avoid copy?
    slicer_ = outSlicer.first;
    return *this;
  }

  [[nodiscard]] nanobind::object build_colexical_permuted_slicer(bool returnPermutation) const {
    std::pair<Slicer_t, std::vector<Index>> outSlicer;
    {
      nanobind::gil_scoped_release release;
      outSlicer = build_permuted_slicer(slicer_);
    }

    Slicer_interface out(*this, std::move(outSlicer.first));

    if (returnPermutation)
      return nanobind::make_tuple(out, _wrap_as_numpy_array(std::move(outSlicer.second), outSlicer.second.size()));

    return nanobind::cast(out);
  }

  [[nodiscard]] Slicer_interface build_slicer_as_permutation(const std::vector<Index> &permutation) const {
    Slicer_t outSlicer;
    {
      nanobind::gil_scoped_release release;
      outSlicer = build_permuted_slicer(slicer_, permutation);
    }
    return {*this, std::move(outSlicer)};
  }

  template <typename U>
  auto build_coarsen_on_grid(const std::vector<std::vector<U>> &grid) const {
    using S = decltype(build_slicer_coarsen_on_grid(slicer_, grid));

    S outSlicer;
    {
      nanobind::gil_scoped_release release;
      outSlicer = build_slicer_coarsen_on_grid(slicer_, grid);
    }

    return Slicer_interface<typename S::Filtration_value, typename S::Persistence>(*this, std::move(outSlicer));
  }

  [[nodiscard]] Slicer_interface build_from_projective_cover_kernel(std::optional<int> dimension) const {
    if (generatorBasis_.has_value()) {
      throw nanobind::value_error(
          "compute_kernel_projective_cover does not transport `_generator_basis`;"
          " discard the basis explicitly before this transformation.");
    }

    Slicer_t outSlicer;

    if (slicer_.get_number_of_cycle_generators() == 0) return {*this, std::move(outSlicer)};

    int dim = dimension.has_value() ? *dimension : static_cast<int>(slicer_.get_max_dimension());
    {
      nanobind::gil_scoped_release release;
      outSlicer = build_slicer_from_projective_cover_kernel(slicer_, dim);
    }
    return {*this, std::move(outSlicer)};
  }

  [[nodiscard]] std::string to_string() const {
    std::stringstream stream;
    stream << slicer_;
    return stream.str();
  }

  template <class OtherMultiFiltrationValue, class OtherPersistenceAlgorithm>
  bool operator==(const Slicer_interface<OtherMultiFiltrationValue, OtherPersistenceAlgorithm> &other) const {
    bool res;
    {
      nanobind::gil_scoped_release release;
      // the boundaries in the two slicers have to be ordered the same for them to be equal
      // can potentially be generalized by ordering them the same before comparing
      // but that adds even more annoying complexity
      res = (slicer_.get_dimensions() == other.get_slicer().get_dimensions() &&
             slicer_.get_boundaries() == other.get_slicer().get_boundaries());
    }
    return res && _has_same_filtration_values(other);
  }

  friend char *serialize_value_to_char_buffer(const Slicer_interface &value, char *start) {
    char *curr = start;
    curr = serialize_value_to_char_buffer(value.slicer_, curr);
    bool hasBasis = value.generatorBasis_.has_value();
    curr = serialize_value_to_char_buffer(hasBasis, curr);
    if (hasBasis) curr = serialize_value_to_char_buffer(*value.generatorBasis_, curr);
    curr = serialize_value_to_char_buffer(value.presDegree_, curr);
    curr = serialize_value_to_char_buffer(value.isMinPres_, curr);
    curr = serialize_value_to_char_buffer(value.isMinRes_, curr);
    return curr;
  }

  friend const char *deserialize_value_from_char_buffer(Slicer_interface &value, const char *start) {
    const char *curr = start;
    curr = deserialize_value_from_char_buffer(value.slicer_, curr);
    bool hasBasis;
    curr = deserialize_value_from_char_buffer(hasBasis, curr);
    if (hasBasis) {
      value.generatorBasis_.emplace();
      curr = deserialize_value_from_char_buffer(*value.generatorBasis_, curr);
    }
    curr = deserialize_value_from_char_buffer(value.presDegree_, curr);
    curr = deserialize_value_from_char_buffer(value.isMinPres_, curr);
    curr = deserialize_value_from_char_buffer(value.isMinRes_, curr);
    return curr;
  }

  friend std::size_t get_serialization_size_of(const Slicer_interface &value) {
    std::size_t size = get_serialization_size_of(value.slicer_);
    bool hasBasis = value.generatorBasis_.has_value();
    size += get_serialization_size_of(hasBasis);
    if (hasBasis) size += get_serialization_size_of(*value.generatorBasis_);
    size += get_serialization_size_of(value.presDegree_);
    size += get_serialization_size_of(value.isMinPres_);
    size += get_serialization_size_of(value.isMinRes_);
    return size;
  }

 private:
  Slicer_t slicer_;
  nanobind::object filtrationGrid_;
  std::optional<detail::Generator_basis_data> generatorBasis_;
  int presDegree_;
  bool isMinPres_;
  bool isMinRes_;

  static auto _get_filtration_array(const MultiFiltrationValue &f) {
    std::vector<value_type> values(f.num_generators() * f.num_parameters());
    Gudhi::Simple_mdspan view(values.data(), f.num_generators(), f.num_parameters());
    {
      nanobind::gil_scoped_release release;
      for (std::size_t g = 0; g < f.num_generators(); ++g) {
        for (std::size_t p = 0; p < f.num_parameters(); ++p) {
          view(g, p) = f(g, p);
        }
      }
    }
    if constexpr (MultiFiltrationValue::ensures_1_criticality()) {
      return _wrap_as_numpy_array(std::move(values), f.num_parameters());
    } else {
      return _wrap_as_numpy_array(std::move(values), f.num_generators(), f.num_parameters());
    }
  }

  static nanobind::tuple _get_compact_filtration_array(const typename Complex::Filtration_value_container &filts,
                                                       int numParam) {
    std::vector<value_type> values;
    std::vector<std::int64_t> startIndices(filts.size() + 1, 0);

    {
      nanobind::gil_scoped_release release;
      for (std::size_t i = 0; i < filts.size(); ++i) {
        startIndices[i + 1] = startIndices[i] + filts[i].num_generators();
      }
      values.resize(startIndices.back() * numParam);
      for (std::size_t i = 0; i < filts.size(); ++i) {
        const auto &f = filts[i];
        if (numParam != f.num_parameters())
          throw std::runtime_error("Inconsistent number of parameters in stored filtration values");
        Gudhi::Simple_mdspan view(&values[startIndices[i] * numParam], f.num_generators(), numParam);
        for (std::size_t g = 0; g < f.num_generators(); ++g) {
          for (std::size_t p = 0; p < numParam; ++p) {
            view(g, p) = f(g, p);
          }
        }
      }
    }

    return nanobind::make_tuple(_wrap_as_numpy_array(std::move(startIndices), startIndices.size()),
                                _wrap_as_numpy_array(std::move(values), startIndices.back(), numParam));
  }

  static nanobind::object _get_filtration_array(const typename Complex::Filtration_value_container &filts,
                                                int numParam) {
    if constexpr (MultiFiltrationValue::ensures_1_criticality()) {
      std::vector<value_type> values(filts.size() * numParam);
      {
        nanobind::gil_scoped_release release;
        Gudhi::Simple_mdspan view(values.data(), filts.size(), numParam);
        for (std::size_t i = 0; i < filts.size(); ++i) {
          const auto &f = filts[i];
          if (numParam != f.num_parameters())
            throw std::runtime_error("Inconsistent number of parameters in stored filtration values");
          for (int p = 0; p < numParam; ++p) {
            view(i, p) = f(0, p);
          }
        }
      }

      return nanobind::cast(_wrap_as_numpy_array(std::move(values), filts.size(), numParam));
    } else {
      std::vector<std::vector<value_type>> values(filts.size());
      {
        nanobind::gil_scoped_release release;
        for (std::size_t i = 0; i < filts.size(); ++i) {
          const auto &f = filts[i];
          if (numParam != f.num_parameters())
            throw std::runtime_error("Inconsistent number of parameters in stored filtration values");
          values[i].resize(f.num_generators() * numParam);
          Gudhi::Simple_mdspan view(values[i].data(), f.num_generators(), numParam);
          for (std::size_t g = 0; g < f.num_generators(); ++g) {
            for (int p = 0; p < numParam; ++p) {
              view(g, p) = f(g, p);
            }
          }
        }
      }

      // Storing the items in values and releasing the gil only once is probably faster than
      // storing the items directly in the tuple and releasing the gil at each construction ?
      return Gudhi::python::_build_tuple(filts.size(), [&](std::size_t i) {
        return _wrap_as_numpy_array(std::move(values[i]), filts[i].num_generators(), numParam);
      });
    }
  }

  template <typename U>
  static bool _check_has_sorted_rows(Tensor2D<U> grid) {
    auto view = grid.view();
    std::size_t rows = view.shape(0), cols = view.shape(1);

    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 1; j < cols; ++j)
        if (view(i, j - 1) > view(i, j))
          throw nanobind::type_error("Expected grid rows to be sorted by increasing values.");

    return rows != 0 && cols != 0;  // returns false if the grid is valid but empty
  }

  template <typename U>
  static bool _check_has_sorted_rows(nanobind::iterable grid) {
    bool hasNonEmptyRows = false;
    for (nanobind::handle row : grid) {
      if (!nanobind::isinstance<nanobind::iterable>(row))
        throw nanobind::type_error("Expected each row to be iterable.");

      bool hasPrev = false;
      U prev = 0;

      for (nanobind::handle elem : nanobind::cast<nanobind::iterable>(row)) {
        U val;
        if (!nanobind::try_cast<U>(elem, val)) throw nanobind::type_error("Expected arithmetic elements in the grid.");

        if (hasPrev && val < prev)
          throw nanobind::type_error("Expected rows of the grid to be ordered by increasing value.");

        prev = val;
        hasPrev = true;
      }
      hasNonEmptyRows |= hasPrev;
    }

    return hasNonEmptyRows;  // returns false if the grid is valid but empty
  }

  template <class B, class D, class F>
  void _build_slicer(const B &boundaries, const D &dimensions, const F &filValues) {
    {
      nanobind::gil_scoped_release release;
      Complex cpx(boundaries, dimensions, filValues);
      slicer_ = Slicer_t(std::move(cpx));
    }
  }

  template <class OtherMultiFiltrationValue, class OtherPersistenceAlgorithm>
  bool _has_same_filtration_values(
      const Slicer_interface<OtherMultiFiltrationValue, OtherPersistenceAlgorithm> &other) const {
    using F1_double = decltype(std::declval<MultiFiltrationValue>().template as_type<double>());
    using F2_double = decltype(std::declval<OtherMultiFiltrationValue>().template as_type<double>());

    auto are_equal = [](const auto &a, const auto &b) {
      // we already know they have the same size
      return std::equal(a.begin(), a.end(), b.begin(), [](auto f1, auto f2) {
        return Gudhi::multi_filtration::are_equal_filtration_values(f1, f2);
      });
    };

    const auto &filtsA = slicer_.get_filtration_values();
    const auto &filtsB = other.get_slicer().get_filtration_values();

    if (filtsA.size() != filtsB.size()) return false;

    nanobind::object otherGrid = other.get_filtration_grid();

    if (filtrationGrid_.is_none() && otherGrid.is_none()) {
      nanobind::gil_scoped_release release;
      return are_equal(filtsA, filtsB);
    }

    // if filtrationGrid_ is not None for one of them, the filtration values to compare are in the grid
    // as two different grids can still yield the same filtration values, it is not sufficient to compare the grids
    // also, once translated, the filtration values could be not minimal, i.e. the values have to
    // be explicitly constructed

    std::vector<F1_double> transA;
    std::vector<F2_double> transB;

    if (filtrationGrid_.is_none()) {
      std::vector<std::vector<double>> grid;
      if (!nanobind::try_cast<std::vector<std::vector<double>>>(otherGrid, grid))
        throw std::runtime_error("Stored filtration grid in other did not have a valid format.");
      nanobind::gil_scoped_release release;
      transB.reserve(filtsB.size());
      for (const auto &f : filtsB) {
        transB.push_back(evaluate_coordinates_in_grid(f, grid));
      }
      return are_equal(filtsA, transB);
    }

    if (otherGrid.is_none()) {
      std::vector<std::vector<double>> grid;
      if (!nanobind::try_cast<std::vector<std::vector<double>>>(filtrationGrid_, grid))
        throw std::runtime_error("Stored filtration grid did not have a valid format.");
      nanobind::gil_scoped_release release;
      transA.reserve(filtsA.size());
      for (const auto &f : filtsA) {
        transA.push_back(evaluate_coordinates_in_grid(f, grid));
      }
      return are_equal(transA, filtsB);
    }

    std::vector<std::vector<double>> gridA;
    std::vector<std::vector<double>> gridB;
    if (!nanobind::try_cast<std::vector<std::vector<double>>>(filtrationGrid_, gridA))
      throw std::runtime_error("Stored filtration grid did not have a valid format.");
    if (!nanobind::try_cast<std::vector<std::vector<double>>>(otherGrid, gridB))
      throw std::runtime_error("Stored filtration grid in other did not have a valid format.");
    nanobind::gil_scoped_release release;
    for (std::size_t i = 0; i < filtsA.size(); ++i) {
      transA.push_back(evaluate_coordinates_in_grid(filtsA[i], gridA));
      transB.push_back(evaluate_coordinates_in_grid(filtsB[i], gridB));
    }
    return are_equal(transA, transB);
  }

  [[nodiscard]] bool _verify_grid_validity(nanobind::object grid) const {
    // special case of ndarray is more efficient then general nanobind::iterable
    if (nanobind::ndarray<> arr; nanobind::try_cast<nanobind::ndarray<>>(grid, arr, false)) {
      if (arr.ndim() != 2) throw nanobind::type_error("Expected a 2D grid.");
      return detail::_dispatch_dtype(
          grid,
          [&]<typename U>() { return _check_has_sorted_rows<U>(Tensor2D<U>(arr)); },
          []() { return true; },
          []() -> bool { throw nanobind::type_error("Unsupported element type."); });
    }

    if (!nanobind::isinstance<nanobind::iterable>(grid))
      throw nanobind::type_error("Expected a grid as a 2D array or an iterable of iterables.");

    return detail::_dispatch_dtype(
        grid,
        [&]<typename U>() { return _check_has_sorted_rows<U>(nanobind::cast<nanobind::iterable>(grid)); },
        []() { return true; },
        []() -> bool { throw nanobind::type_error("Unsupported element type."); });
  }

  void _get_cycle_boundary(std::vector<std::vector<Index>> &outCycle, const std::vector<Index> &cycle, int dim) const {
    if (cycle.size() == 0) throw std::runtime_error("A cycle should not be empty");
    if (generatorBasis_.has_value() && dim == generatorBasis_->degree) {
      outCycle = generatorBasis_->expand_cycle(cycle);
    } else if (slicer_.get_boundary(cycle[0]).empty()) {
      outCycle = {std::vector<Index>{}};
    } else {
      outCycle.resize(cycle.size());
      for (std::size_t i = 0; i < cycle.size(); ++i) {
        outCycle[i] = slicer_.get_boundary(cycle[i]);
      }
    }
  }

  std::vector<std::vector<std::vector<std::vector<Index>>>> _get_cycle_boundaries(
      bool update,
      const std::optional<Tensor2D<std::int64_t>> &barcodeIndices,
      const std::optional<Tensor1D<Index>> &pointsToIntersect) {
    std::unordered_set<Index> points;
    if (pointsToIntersect.has_value()) {
      if (generatorBasis_.has_value())
        PyErr_WarnEx(PyExc_UserWarning,
                     " When there is a generator basis, points to intersect are ignored for dimensions different of 1 "
                     "for now: to be implemented.",
                     1);
      Numpy_span view(*pointsToIntersect);
      points.reserve(view.size());
      points.insert(view.begin(), view.end());
    }

    std::vector<std::vector<std::vector<std::vector<Index>>>> out;

    {
      nanobind::gil_scoped_release release;
      auto cycleIdx = slicer_.get_representative_cycles(update);
      out.resize(cycleIdx.size());
      std::vector<std::array<std::int64_t, 3>> blocks;

      if (barcodeIndices.has_value()) {
        Numpy_2d_span view(*barcodeIndices);
        std::vector<std::int64_t> sizeByDim(cycleIdx.size(), 0);
        for (std::int64_t barDim : view) ++sizeByDim[barDim];
        for (std::size_t dim = 0; dim < cycleIdx.size(); ++dim) {
          out[dim].resize(sizeByDim[dim]);
          sizeByDim[dim] = 0;
        }
        for (Index i = 0; i < view.size(); ++i) {
          auto barIdx = view[i];
          if (barIdx.size() != 2) throw std::invalid_argument("`barcode_indices` has to be of shape (*, 2).");
          blocks.push_back({barIdx[0], barIdx[1], sizeByDim[barIdx[0]]});
          ++sizeByDim[barIdx[0]];
        }
      } else {
        for (std::int64_t dim = 0; dim < static_cast<std::int64_t>(cycleIdx.size()); ++dim) {
          out[dim].resize(cycleIdx[dim].size());
          for (std::int64_t c = 0; c < static_cast<std::int64_t>(cycleIdx[dim].size()); ++c) {
            blocks.push_back({dim, c, c});
          }
        }
      }
      detail::Representative_cycle_intersection inter(slicer_.get_boundaries(), slicer_.get_dimensions(), points);
      if (pointsToIntersect.has_value() && !generatorBasis_.has_value()) {
        // pre-initialize cache in sequential loop to avoid problems in parallelization
        inter.initialize_cache(blocks.size(), [&](std::size_t i) -> const auto & {
          auto [dim, cIdx, cOut] = blocks[i];
          return cycleIdx[dim][cIdx];
        });
      }
      tbb::parallel_for(std::size_t(0), blocks.size(), [&](std::size_t blockIdx) {
        auto [dim, cIdx, cOut] = blocks[blockIdx];
        const auto &cycle = cycleIdx[dim][cIdx];
        if (!pointsToIntersect.has_value() || generatorBasis_.has_value() || inter.intersects(cycle)) {
          auto &outCycle = out[dim][cOut];
          _get_cycle_boundary(outCycle, cycle, dim);
          // TODO: intersects version for generatorBasis_ and remove this if
          if (dim == 1 && generatorBasis_.has_value() && pointsToIntersect.has_value() &&
              !inter.dim_1_boundaries_intersects(outCycle)) {
            outCycle.clear();  // to mark as to be ignored
          }
        }
      });
    }

    return out;
  }

  std::vector<std::vector<std::vector<Index>>> _get_cycle_boundaries(
      bool update,
      Dimension dimension,
      const std::optional<Tensor1D<std::int64_t>> &barcodeIndices,
      const std::optional<Tensor1D<Index>> &pointsToIntersect) {
    std::unordered_set<Index> points;
    if (pointsToIntersect.has_value()) {
      if (dimension != 1 && generatorBasis_.has_value()) {
        PyErr_WarnEx(PyExc_UserWarning,
                     "When there is a generator basis, points to intersect are ignored for dimensions different of 1 "
                     "for now: to be implemented.",
                     1);
      } else {
        Numpy_span view(*pointsToIntersect);
        points.reserve(view.size());
        points.insert(view.begin(), view.end());
      }
    }

    std::vector<std::vector<std::vector<Index>>> out;

    {
      nanobind::gil_scoped_release release;
      auto cycleIdx = slicer_.get_representative_cycles_in_dim(dimension, update);

      detail::Representative_cycle_intersection inter(slicer_.get_boundaries(), slicer_.get_dimensions(), points);
      auto compute_boundaries = [&](const auto &range) {
        if (pointsToIntersect.has_value() && !generatorBasis_.has_value()) {
          // pre-initialize cache in sequential loop to avoid problems in parallelization
          inter.initialize_cache(range.size(), [&](std::size_t i) -> const auto & { return cycleIdx[range[i]]; });
        }
        tbb::parallel_for(std::size_t(0), range.size(), [&](std::size_t idx) {
          const auto &cycle = cycleIdx[range[idx]];
          if (!pointsToIntersect.has_value() || generatorBasis_.has_value() || inter.intersects(cycle)) {
            auto &outCycle = out[idx];
            _get_cycle_boundary(outCycle, cycle, dimension);
            // TODO: intersects version for generatorBasis_ and remove this if
            if (dimension == 1 && generatorBasis_.has_value() && pointsToIntersect.has_value() &&
                !inter.dim_1_boundaries_intersects(outCycle)) {
              outCycle.clear();  // to mark as to be ignored
            }
          }
        });
      };

      if (barcodeIndices.has_value()) {
        Numpy_span view(*barcodeIndices);
        out.resize(view.size());
        compute_boundaries(view);
      } else {
        out.resize(cycleIdx.size());
        std::vector<std::int64_t> id(cycleIdx.size());
        std::iota(id.begin(), id.end(), 0);
        compute_boundaries(id);
      }
    }

    return out;
  }
};

template <class SlicerInterface>
inline SlicerInterface deserialize_slicer_from_python(nanobind::tuple state) {
  if (nanobind::len(state) != 3)
    throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
  std::uint8_t version;
  if (!nanobind::try_cast<std::uint8_t>(state[0], version, false))
    throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
  if (version < SlicerInterface::SERIALIZATION_VERSION)
    throw std::invalid_argument(
        "Given state to deserialize is not compatible with current multipers version: try an older release");
  if (version > SlicerInterface::SERIALIZATION_VERSION)
    throw std::invalid_argument(
        "Given state to deserialize is not compatible with current multipers version: try an newer release");

  nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> data;
  if (!nanobind::try_cast<nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy>>(state[2], data, false))
    throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
  SlicerInterface slicer;
  {
    nanobind::gil_scoped_release release;
    deserialize_value_from_char_buffer(slicer, data.data());
  }
  slicer.set_filtration_grid(state[1]);
  return slicer;
}

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_SLICER_H_INCLUDED
