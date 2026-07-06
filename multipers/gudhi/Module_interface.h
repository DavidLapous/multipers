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
 * @file Module_interface.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the @ref Gudhi::multi_persistence::Module_interface class for python bindings.
 */

#ifndef MP_PY_MODULE_H_INCLUDED
#define MP_PY_MODULE_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/range/any_range.hpp>
#include <boost/range/adaptor/type_erased.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <gudhi/Debug_utils.h>
#include <gudhi/simple_mdspan.h>
#include <gudhi/Multi_persistence/Module.h>
#include <gudhi/Multi_persistence/module_helpers.h>
#include <gudhi/Multi_persistence/Line.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/utils.h>
#include <python_interfaces/numpy_utils.h>

namespace Gudhi {
namespace multi_persistence {

/**
 * @private
 */
template <typename T>
class Module_interface {
 public:
  using value_type = T;
  using Dimension = typename Module<value_type>::Dimension;
  using Summand_t = typename Module<value_type>::Summand_t;
  using iterator = typename Module<value_type>::iterator;
  using const_iterator = typename Module<value_type>::const_iterator;
  using Summand_of_dimension_range =
      boost::any_range<Summand_t, boost::forward_traversal_tag, const Summand_t &, std::ptrdiff_t>;
  using Tensor1D = nanobind::ndarray<const value_type, nanobind::ndim<1>, nanobind::any_contig>;
  using Tensor2D = nanobind::ndarray<const value_type, nanobind::ndim<2> >;
  template <typename IntegerType>
  using IntTensor1D = nanobind::ndarray<const IntegerType, nanobind::ndim<1>, nanobind::any_contig>;
  using Box_t = std::vector<T>;

  static constexpr T T_inf = Summand_t::T_inf;     /**< Infinity. */
  static constexpr T T_m_inf = Summand_t::T_m_inf; /**< Minus infinity. */

  Module_interface() = default;

  Module_interface(const Box<value_type> &box) : module_(), box_(get_flat_box(box)) {}

  Module_interface(const Box_t &box) : module_(), box_(box) {}

  Module_interface(Tensor2D box) : module_(), box_(get_flat_box_from_tensor(box)) {}

  Module_interface(Module<value_type> &&mod, Box_t &&box) : module_(std::move(mod)), box_(std::move(box)) {}

  Module_interface(Module<value_type> &&mod, const Box<value_type> &box)
      : module_(std::move(mod)), box_(get_flat_box(box)) {}

  [[nodiscard]] int get_number_of_parameters() const {
    int numParam = module_.get_number_of_parameters();
    if (numParam == get_null_value<int>() && !is_trivial_box(box_)) return box_.size() / 2;
    return numParam;
  }

  [[nodiscard]] int size() const { return module_.size(); }

  [[nodiscard]] int get_max_dimension() const { return module_.get_max_dimension(); }

  iterator begin() { return module_.begin(); }

  iterator end() { return module_.end(); }

  const_iterator begin() const { return module_.begin(); }

  const_iterator end() const { return module_.end(); }

  Summand_of_dimension_range get_summands_of_dimension_range(int degree) const {
    // alternative would be to just warn + return empty range instead of throw
    if (degree < 0) throw std::invalid_argument("Cannot iterate over summands of negative dimension.");
    return module_.get_summands_of_dimension_range(static_cast<Dimension>(degree)) |
           boost::adaptors::type_erased<Summand_t, boost::forward_traversal_tag, const Summand_t &, std::ptrdiff_t>();
  }

  auto get_box_lower_corner_view() const {
    // no transfer of ownership, dies together with the box
    return nanobind::ndarray<const T, nanobind::numpy>(box_.data(), {box_.size() / 2});
  }

  auto get_box_upper_corner_view() const {
    auto shift = box_.size() / 2;
    // no transfer of ownership, dies together with the box
    return nanobind::ndarray<const T, nanobind::numpy>(box_.data() + shift, {shift});
  }

  auto get_box_view_ro() const {
    // no transfer of ownership, dies together with the box
    return nanobind::ndarray<const T, nanobind::numpy>(box_.data(), {2, box_.size() / 2});
  }

  auto get_box_view() {
    // no transfer of ownership, dies together with the box
    return nanobind::ndarray<T, nanobind::numpy>(box_.data(), {2, box_.size() / 2});
  }

  Module_interface &set_box(const Box<value_type> &box) {
    if (module_.get_number_of_parameters() != get_null_value<int>() &&
        module_.get_number_of_parameters() != box.get_number_of_coordinates())
      throw std::invalid_argument(
          "The given box has not the same number of coordinates than parameters in the stored module");
    box_ = get_flat_box(box);
    return *this;
  }

  Module_interface &set_box(Tensor2D box) {
    if (module_.get_number_of_parameters() != get_null_value<int>() &&
        module_.get_number_of_parameters() != box.shape(1))
      throw std::invalid_argument(
          "The given box has not the same number of coordinates than parameters in the stored module");
    box_ = get_flat_box_from_tensor(box);
    return *this;
  }

  Module_interface &set_box(const std::vector<std::vector<T>> &box) {
    if (box.size() != 2) throw std::invalid_argument("Box has to be represented by two corners.");
    if (box[0].size() != box[1].size())
      throw std::invalid_argument("Both corners defining the box must have same dimension.");
    if (module_.get_number_of_parameters() != get_null_value<int>() &&
        module_.get_number_of_parameters() != box[0].size())
      throw std::invalid_argument(
          "The given box has not the same number of coordinates than parameters in the stored module");
    box_ = Box_t(box[0].begin(), box[0].end());
    box_.reserve(box_.size() * 2);
    box_.insert(box_.end(), box[1].begin(), box[1].end());
    return *this;
  }

  [[nodiscard]] nanobind::list get_flat_filtration_values(bool unique) const {
    int numParam = module_.get_number_of_parameters();
    if (numParam == get_null_value<int>()) return {};  // empty module

    std::vector<std::vector<T>> values(numParam);
    {
      nanobind::gil_scoped_release release;
      for (const auto &summand : module_) {
        const auto &births = summand.get_upset();
        GUDHI_CHECK(births.num_parameters() == numParam,
                    std::runtime_error("Upset number of parameters is not coherent."));
        for (int p = 0; p < numParam; ++p) {
          for (std::size_t g = 0; g < summand.get_number_of_birth_corners(); ++g) {
            const T v = births(g, p);
            // in the original, infinite values were never copied
            // but if unique is false, I think it makes more sense to keep a bijection on the indices, no?
            // btw unique == false is never used in multipers right now, so it does not matter too much
            if (!unique || (v != T_inf && v != T_m_inf)) values[p].push_back(v);
          }
        }
        const auto &deaths = summand.get_downset();
        GUDHI_CHECK(deaths.num_parameters() == numParam,
                    std::runtime_error("Downset number of parameters is not coherent."));
        for (int p = 0; p < numParam; ++p) {
          for (std::size_t g = 0; g < summand.get_number_of_death_corners(); ++g) {
            const T v = deaths(g, p);
            // same question then for births
            if (!unique || (v != T_inf && v != T_m_inf)) values[p].push_back(v);
          }
        }
      }
    }

    nanobind::list out;
    for (auto &vals : values) {
      if (unique) {
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      }
      out.append(_wrap_as_numpy_array(std::move(vals), vals.size()));
    }
    return out;
  }

  auto get_all_dimension() const {
    std::vector<std::int32_t> dimensions(module_.size());
    {
      nanobind::gil_scoped_release release;
      for (std::size_t i = 0; i < module_.size(); ++i) {
        dimensions[i] = module_.get_summand(i).get_dimension();
      }
    }
    return _wrap_as_numpy_array(std::move(dimensions), dimensions.size());
  }

  Summand_t &get_summand(int index) { return module_.get_summand(index); }

  const Summand_t &get_summand(int index) const { return module_.get_summand(index); }

  Module_interface &merge(const Module_interface &toMerge) {
    {
      nanobind::gil_scoped_release release;
      module_.merge(toMerge.module_);
    }
    return *this;
  }

  Module_interface &merge(const Module_interface &toMerge, int degree) {
    {
      nanobind::gil_scoped_release release;
      // alternative would be to just warn + return instead of throw
      if (degree < 0) throw std::invalid_argument("Cannot merge summands of negative dimension.");
      module_.merge(toMerge.module_, static_cast<Dimension>(degree));
    }
    return *this;
  }

  Module_interface &merge(nanobind::iterable toMerge) {
    for (nanobind::handle item : toMerge) {
      merge(nanobind::cast<const Module_interface &>(item));
    }
    return *this;
  }

  auto compute_bounds() const {
    Box<T> box;
    {
      nanobind::gil_scoped_release release;
      box = module_.compute_bounds();
    }
    auto &corners = box.get_lower_corner().retrieve_underlying_container();
    auto dim = corners.size();
    {
      nanobind::gil_scoped_release release;
      corners.reserve(dim * 2);
      corners.insert(corners.end(), box.get_upper_corner().begin(), box.get_upper_corner().end());
    }
    return _wrap_as_numpy_array(std::move(corners), 2, dim);
  }

  nanobind::list get_barcode_from_line(Tensor1D basepoint, std::optional<Tensor1D> direction, int degree) const {
    std::vector<std::vector<std::array<double, 2>>> barcode;
    nanobind::list out;
    {
      nanobind::gil_scoped_release release;
      Dimension dim = degree < 0 ? get_null_value<Dimension>() : static_cast<Dimension>(degree);
      Numpy_span baseView(basepoint);
      Line<T> line;
      if (direction.has_value()) {
        Numpy_span dirView(*direction);
        line = Line<T>(baseView.begin(), baseView.end(), dirView.begin(), dirView.end());
      } else {
        line = Line<T>(baseView.begin(), baseView.end());
      }
      barcode = module_.get_barcode_from_line(line, dim);
    }
    for (auto &d : barcode) {
      out.append(_wrap_as_numpy_array(std::move(d)));
    }
    return out;
  }

  nanobind::list get_barcode_from_lines(Tensor2D basepoints,
                                        std::optional<Tensor2D> directions,
                                        int degree,
                                        bool keep_inf) const {
    std::vector<std::vector<std::array<double, 2>>> barcode;
    std::size_t numberOfLines;
    {
      nanobind::gil_scoped_release release;
      Dimension dim = degree < 0 ? get_null_value<Dimension>() : static_cast<Dimension>(degree);
      Numpy_2d_span basesView(basepoints);
      numberOfLines = basesView.size();
      std::vector<Line<T>> lines(numberOfLines);
      if (directions.has_value()) {
        Numpy_2d_span dirsView(*directions);
        if (numberOfLines != dirsView.size())
          throw std::invalid_argument("If directions are specified, there need to be as many as base points.");
        for (std::size_t i = 0; i < lines.size(); ++i) {
          auto baseView = basesView[i];
          auto dirView = dirsView[i];
          lines[i] = Line<T>(baseView.begin(), baseView.end(), dirView.begin(), dirView.end());
        }
      } else {
        for (std::size_t i = 0; i < lines.size(); ++i) {
          auto baseView = basesView[i];
          lines[i] = Line<T>(baseView.begin(), baseView.end());
        }
      }
      barcode = module_.get_barcode_from_range_of_lines(lines, dim);
    }
    return get_numpy_barcode_from_lines(barcode, numberOfLines, keep_inf);
  }

  Module_interface &rescale(const std::vector<T> &rescaleFactors, int degree) {
    {
      nanobind::gil_scoped_release release;
      Dimension dim = degree < 0 ? get_null_value<Dimension>() : static_cast<Dimension>(degree);
      module_.rescale(rescaleFactors, dim);
    }
    return *this;
  }

  Module_interface &rescale(Tensor1D rescaleFactors, int degree) {
    {
      nanobind::gil_scoped_release release;
      Dimension dim = degree < 0 ? get_null_value<Dimension>() : static_cast<Dimension>(degree);
      module_.rescale(Numpy_span(rescaleFactors), dim);
    }
    return *this;
  }

  Module_interface &translate(const std::vector<T> &translation, int degree) {
    {
      nanobind::gil_scoped_release release;
      Dimension dim = degree < 0 ? get_null_value<Dimension>() : static_cast<Dimension>(degree);
      module_.translate(translation, dim);
    }
    return *this;
  }

  Module_interface &translate(Tensor1D translation, int degree) {
    {
      nanobind::gil_scoped_release release;
      Dimension dim = degree < 0 ? get_null_value<Dimension>() : static_cast<Dimension>(degree);
      module_.translate(Numpy_span(translation), dim);
    }
    return *this;
  }

  Module_interface &evaluate_in_grid(const std::vector<std::vector<T>> &grid) {
    {
      nanobind::gil_scoped_release release;
      module_.evaluate_in_grid(grid);
    }
    return *this;
  }

  Module_interface &evaluate_in_grid(const std::vector<Tensor1D> &grid) {
    {
      nanobind::gil_scoped_release release;
      std::vector<Numpy_span<T>> views(grid.begin(), grid.end());
      module_.evaluate_in_grid(views);
    }
    return *this;
  }

  Module_interface &evaluate_in_grid(Tensor2D grid) {
    {
      nanobind::gil_scoped_release release;
      module_.evaluate_in_grid(Numpy_2d_span(grid));
    }
    return *this;
  }

  Module_interface permute_summands(Tensor1D permutation) {
    Module_interface out(box_);
    {
      nanobind::gil_scoped_release release;
      out.module_ = Gudhi::multi_persistence::build_permuted_module(module_, Numpy_span(permutation));
    }
    return out;
  }

  Module_interface get_module_of_degree(int degree) {
    if (degree < 0) throw std::invalid_argument("Cannot get summands of negative dimension.");
    Module_interface out(box_);
    {
      nanobind::gil_scoped_release release;
      out.module_ = Gudhi::multi_persistence::build_module_of_dimension(module_, degree);
    }
    return out;
  }

  Module_interface get_module_of_degrees(IntTensor1D<int> degrees) {
    Module_interface out(box_);
    {
      nanobind::gil_scoped_release release;
      out.module_ = Gudhi::multi_persistence::build_module_of_dimension(module_, Numpy_span(degrees));
    }
    return out;
  }

  Module_interface get_module_of_degrees(const std::vector<int> &degrees) {
    Module_interface out(box_);
    {
      nanobind::gil_scoped_release release;
      out.module_ = Gudhi::multi_persistence::build_module_of_dimension(module_, degrees);
    }
    return out;
  }

  template <typename IntegerType>
  auto compute_landscapes_from_box(int degree,
                                   IntTensor1D<IntegerType> ks,
                                   Tensor2D box,
                                   IntTensor1D<IntegerType> resolution,
                                   int n_jobs) {
    if (degree < 0) throw std::invalid_argument("Landscape dimension has to be positive.");
    if (module_.get_number_of_parameters() != get_null_value<int>() &&
        module_.get_number_of_parameters() != box.shape(1))
      throw std::invalid_argument(
          "The given box has not the same number of coordinates than parameters in the stored module");

    std::vector<maybe_make_signed_t<T>> out;
    Numpy_span resolutionView(resolution);
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_set_of_module_landscapes(
          module_, degree, Numpy_span(ks), get_box_from_tensor(box), resolutionView, n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), ks.shape(0), resolutionView[0], resolutionView[1]);
  }

  template <typename IntegerType>
  auto compute_landscapes_from_grid(int degree,
                                    IntTensor1D<IntegerType> ks,
                                    const std::vector<Tensor1D> &grid,
                                    int n_jobs) {
    if (degree < 0) throw std::invalid_argument("Landscape dimension has to be positive.");

    std::vector<maybe_make_signed_t<T>> out;
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_set_of_module_landscapes(
          module_, degree, Numpy_span(ks), std::vector<Numpy_span<T>>(grid.begin(), grid.end()), n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), ks.shape(0), grid[0].shape(0), grid[1].shape(0));
  }

  template <typename IntegerType>
  auto compute_pixels(Tensor2D coordinates,
                      IntTensor1D<IntegerType> degrees,
                      Tensor2D box,
                      double delta,
                      double p,
                      bool normalize,
                      int n_jobs) {
    if (module_.get_number_of_parameters() != get_null_value<int>() &&
        module_.get_number_of_parameters() != box.shape(1) && box.shape(1) != 0)
      throw std::invalid_argument(
          "The given box is neither trivial nor has not the same number of coordinates than parameters in the stored "
          "module");

    std::vector<double> out;
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_module_pixels(module_,
                                                            Numpy_2d_span(coordinates),
                                                            Numpy_span(degrees),
                                                            get_box_from_tensor(box),
                                                            delta,
                                                            p,
                                                            normalize,
                                                            n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), degrees.shape(0), coordinates.shape(0));
  }

  // nb::overload_cast has a bit of a problem with "auto" returns. The easiest was the rename.
  auto compute_distance_to_iterable(const std::vector<std::vector<T>> &pts, bool signed_distance, int n_jobs) {
    std::vector<maybe_make_signed_t<T>> out;
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_module_distances_to(module_, pts, signed_distance, n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), pts.size(), module_.size());
  }

  auto compute_distance_to_tensor(Tensor2D pts, bool signed_distance, int n_jobs) {
    std::vector<maybe_make_signed_t<T>> out;
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_module_distances_to(module_, Numpy_2d_span(pts), signed_distance, n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), pts.shape(0), module_.size());
  }

  auto compute_interleavings() {
    std::vector<maybe_make_signed_t<T>> interleavings;
    {
      nanobind::gil_scoped_release release;
      interleavings = Gudhi::multi_persistence::compute_module_interleavings(module_, get_box_from_tensor(box_));
    }
    return _wrap_as_numpy_array(std::move(interleavings), interleavings.size());
  }

  auto compute_interleavings_from_box(Tensor2D box) {
    if (module_.get_number_of_parameters() != get_null_value<int>() &&
        module_.get_number_of_parameters() != box.shape(1))
      throw std::invalid_argument(
          "The given box has not the same number of coordinates than parameters in the stored module");

    std::vector<maybe_make_signed_t<T>> interleavings;
    {
      nanobind::gil_scoped_release release;
      interleavings = Gudhi::multi_persistence::compute_module_interleavings(module_, get_box_from_tensor(box));
    }
    return _wrap_as_numpy_array(std::move(interleavings), interleavings.size());
  }

  nanobind::tuple get_flat_indices_in_grid(const std::vector<Tensor1D> &grid) {
    return _get_flat_indices_in_grid(std::vector<Numpy_span<T>>(grid.begin(), grid.end()));
  }

  nanobind::tuple get_flat_indices_in_grid(const std::vector<std::vector<T>> &grid) {
    return _get_flat_indices_in_grid(grid);
  }

  friend bool operator==(const Module_interface &a, const Module_interface &b) {
    bool res;
    {
      // comparing two modules should be expensive enough that it is worth releasing the GIL?
      nanobind::gil_scoped_release release;
      res = (a.module_ == b.module_ && a.box_ == b.box_);
    }
    return res;
  }

  friend char *serialize_value_to_char_buffer(const Module_interface &value, char *start) {
    char *curr = start;
    const std::size_t length = value.box_.size();
    const std::size_t argSize = sizeof(T) * length;
    const std::size_t typeSize = sizeof(std::size_t);
    memcpy(curr, &length, typeSize);
    curr += typeSize;
    memcpy(curr, value.box_.data(), argSize);
    curr += argSize;
    curr = serialize_value_to_char_buffer(value.module_, curr);
    return curr;
  }

  friend const char *deserialize_value_from_char_buffer(Module_interface &value, const char *start) {
    const char *curr = start;
    const std::size_t typeSize = sizeof(std::size_t);
    std::size_t length;
    memcpy(&length, curr, typeSize);
    curr += typeSize;
    std::size_t argSize = sizeof(T) * length;
    value.box_.resize(length);
    memcpy(value.box_.data(), curr, argSize);
    curr += argSize;
    curr = deserialize_value_from_char_buffer(value.module_, curr);
    return curr;
  }

  friend std::size_t get_serialization_size_of(const Module_interface &value) {
    return sizeof(std::size_t) + (sizeof(T) * value.box_.size()) + get_serialization_size_of(value.module_);
  }

  friend void swap(Module_interface &mod1, Module_interface &mod2) noexcept {
    swap(mod1.box_, mod2.box_);
    swap(mod1.module_, mod2.module_);
  }

 private:
  Module<T> module_;
  Box_t box_;

  template <typename Y>
  static constexpr Y get_null_value() {
    return Module<T>::Summand_t::template get_null_value<Y>();
  }

  static bool is_trivial_box(const Box_t &box) {
    if (box.empty()) return true;
    std::size_t size = box.size() / 2;
    std::size_t bstart = 0;
    std::size_t ustart = size;
    // not completely true, as they can be more NaN values, but I just assume that if something is NaN, all is NaN.
    // the loop is faster this way in the non trivial case.
    if (Gudhi::multi_filtration::_is_nan(box[bstart]) || Gudhi::multi_filtration::_is_nan(box[ustart])) return true;
    while (bstart < size) {
      if (box[bstart] != box[ustart]) return false;
      ++bstart;
      ++ustart;
    }
    return true;
  }

  static Box<T> get_box_from_tensor(Tensor2D box) {
    if (box.shape(0) != 2) throw std::invalid_argument("Box has to be represented by two corners.");

    Numpy_2d_span boxView(box);
    auto lowerView = boxView[0];
    auto upperView = boxView[1];
    return {lowerView.begin(), lowerView.end(), upperView.begin(), upperView.end()};
  }

  static Box<T> get_box_from_tensor(const Box_t &box) {
    std::size_t size = box.size() / 2;
    return {box.data(), box.data() + size, box.data() + size, box.data() + (size * 2)};
  }

  static Box_t get_flat_box_from_tensor(Tensor2D box) {
    if (box.shape(0) != 2) throw std::invalid_argument("Box has to be represented by two corners.");

    Numpy_2d_span boxView(box);
    auto lowerView = boxView[0];
    auto upperView = boxView[1];
    Box_t fb(lowerView.begin(), lowerView.end());
    fb.reserve(fb.size() * 2);
    fb.insert(fb.end(), upperView.begin(), upperView.end());
    return fb;
  }

  static Box_t get_flat_box(const Box<T> &box) {
    Box_t fb(box.get_lower_corner().retrieve_underlying_container());
    fb.reserve(fb.size() * 2);
    fb.insert(fb.end(), box.get_upper_corner().begin(), box.get_upper_corner().end());
    return fb;
  }

  static nanobind::list get_numpy_barcode_from_lines_with_inf(std::vector<std::vector<std::array<double, 2>>> &barcode,
                                                              std::size_t numberOfLines) {
    nanobind::list out;
    std::vector<std::uint64_t> splits;

    if (numberOfLines == 0) {
      for (auto &d : barcode) {
        if (d.size() != 0) throw std::logic_error("No lines but the barcode is not empty... ?");
        out.append(nanobind::make_tuple(_wrap_as_numpy_array(std::move(d), 0, 0, 2),
                                        _wrap_as_numpy_array(std::move(splits), 0)));
      }
      return out;
    }

    for (auto &d : barcode) {
      if (d.size() % numberOfLines != 0)
        throw std::logic_error("Barcodes do not have consistent sizes from a line to another.");
      std::size_t numberOfBars = d.size() / numberOfLines;
      out.append(nanobind::make_tuple(_wrap_as_numpy_array(std::move(d), numberOfLines, numberOfBars, 2),
                                      _wrap_as_numpy_array(std::move(splits), 0)));
    }
    return out;
  }

  static nanobind::list get_numpy_barcode_from_lines_without_inf(
      std::vector<std::vector<std::array<double, 2>>> &barcode,
      std::size_t numberOfLines) {
    nanobind::list out;

    if (numberOfLines == 0) {
      for (auto &d : barcode) {
        if (d.size() != 0) throw std::logic_error("No lines but the barcode is not empty... ?");
        std::vector<std::uint64_t> splits;
        out.append(
            nanobind::make_tuple(_wrap_as_numpy_array(std::move(d), 0, 2), _wrap_as_numpy_array(std::move(splits), 0)));
      }
      return out;
    }

    for (auto &d : barcode) {
      if (d.size() % numberOfLines != 0)
        throw std::logic_error("Barcodes do not have consistent sizes from a line to another.");
      std::vector<std::uint64_t> splits(numberOfLines - 1);
      std::size_t barCount = 0;
      std::size_t numberOfBars = d.size() / numberOfLines;
      std::vector<std::array<double, 2>> bars;
      for (std::size_t l = 0; l < numberOfLines; ++l) {
        for (std::size_t i = 0; i < numberOfBars; ++i) {
          auto &b = d[i + (l * numberOfBars)];
          if (b[0] != Module<double>::T_inf) {
            bars.push_back(b);
            ++barCount;
          }
        }
        if (l + 1 < numberOfLines) splits[l] = barCount;
      }
      out.append(nanobind::make_tuple(_wrap_as_numpy_array(std::move(bars), barCount, 2),
                                      _wrap_as_numpy_array(std::move(splits), numberOfLines - 1)));
    }
    return out;
  }

  static nanobind::list get_numpy_barcode_from_lines(std::vector<std::vector<std::array<double, 2>>> &barcode,
                                                     std::size_t numberOfLines,
                                                     bool keepInf) {
    if (keepInf) return get_numpy_barcode_from_lines_with_inf(barcode, numberOfLines);
    return get_numpy_barcode_from_lines_without_inf(barcode, numberOfLines);
  }

  template <typename Index, class GridRow>
  Index _get_grid_index(T value, const GridRow &row) const {
    if (row.empty()) throw std::invalid_argument("Grid axes must be non-empty.");

    // static_cast because of Windows bug not accepting integer types for isnan method...
    if (value == Summand_t::T_inf || std::isnan(static_cast<double>(value))) return row.size() - 1;
    if (value == Summand_t::T_m_inf) return 0;

    if (value <= row[0]) return 0;
    if (value >= row[row.size() - 1]) return row.size() - 1;

    return std::distance(row.begin(), std::lower_bound(row.begin(), row.end(), value));
  }

  template <typename Index, class Corners, class GridRow>
  void _add_corner_coordinates(std::vector<Index> &coordinates,
                               const Corners &corners,
                               const std::vector<GridRow> &grid) const {
    std::size_t startIdx = coordinates.size();
    coordinates.resize(coordinates.size() + (corners.num_generators() * corners.num_parameters()));
    Gudhi::Simple_mdspan coordsView(coordinates.data() + startIdx, corners.num_generators(), corners.num_parameters());
    for (std::size_t g = 0; g < corners.num_generators(); ++g) {
      for (std::size_t p = 0; p < corners.num_parameters(); ++p) {
        coordsView(g, p) = _get_grid_index<Index>(corners(g, p), grid[p]);
      }
    }
  }

  template <class GridRow, typename Index = std::int32_t>
  nanobind::tuple _get_flat_indices_in_grid(const std::vector<GridRow> &grid) const {
    const int numParam = get_number_of_parameters();
    const std::size_t numSummands = module_.size();

    std::vector<Index> sizes;
    std::vector<Index> birthCoordinates;
    std::vector<Index> deathCoordinates;

    if (numParam == get_null_value<int>() || grid.size() == 0 || numSummands == 0) {
      return nanobind::make_tuple(_wrap_as_numpy_array(std::move(sizes), 2, 0),
                                  _wrap_as_numpy_array(std::move(birthCoordinates), 0, grid.size()),
                                  _wrap_as_numpy_array(std::move(deathCoordinates), 0, grid.size()));
    }

    if (grid.size() != numParam)
      throw std::invalid_argument(
          "Given grid does not have the same number of rows as number of parameters in the module.");

    {
      nanobind::gil_scoped_release release;

      sizes.resize(2 * numSummands);
      Gudhi::Simple_mdspan sizesView(sizes.data(), 2, numSummands);
      for (std::size_t s = 0; s < numSummands; ++s) {
        const auto &sum = module_.get_summand(s);
        sizesView(0, s) = sum.get_number_of_birth_corners();
        sizesView(1, s) = sum.get_number_of_death_corners();

        _add_corner_coordinates(birthCoordinates, sum.get_upset(), grid);
        _add_corner_coordinates(deathCoordinates, sum.get_downset(), grid);
      }
    }

    return nanobind::make_tuple(
        _wrap_as_numpy_array(std::move(sizes), 2, numSummands),
        _wrap_as_numpy_array(std::move(birthCoordinates), birthCoordinates.size() / numParam, numParam),
        _wrap_as_numpy_array(std::move(deathCoordinates), deathCoordinates.size() / numParam, numParam));
  }
};

template <typename T>
Module_interface<T> deserialize_module_from_python(
    const nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> &state) {
  Module_interface<T> mod;
  {
    nanobind::gil_scoped_release release;
    deserialize_value_from_char_buffer(mod, state.data());
  }
  return mod;
}

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_MODULE_H_INCLUDED
