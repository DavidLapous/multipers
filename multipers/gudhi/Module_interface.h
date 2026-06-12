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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include <boost/range/any_range.hpp>
#include <boost/range/adaptor/type_erased.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <gudhi/Debug_utils.h>
#include <gudhi/Multi_persistence/Module.h>
#include <gudhi/Multi_persistence/module_helpers.h>
#include <gudhi/Multi_persistence/Line.h>
#include <gudhi/Multi_persistence/Box.h>
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
  using Tensor2D = nanobind::ndarray<const value_type, nanobind::ndim<2>, nanobind::any_contig>;
  template <typename IntegerType>
  using IntTensor1D = nanobind::ndarray<const IntegerType, nanobind::ndim<1>, nanobind::any_contig>;

  static constexpr T T_inf = Summand_t::T_inf;     /**< Infinity. */
  static constexpr T T_m_inf = Summand_t::T_m_inf; /**< Minus infinity. */

  Module_interface() = default;

  Module_interface(const Box<value_type> &box) : module_(), box_(box) {}

  Module_interface(Box<value_type> &&box) : module_(), box_(std::move(box)) {}

  Module_interface(Module<value_type>&& mod, Box<value_type> &&box) : module_(std::move(mod)), box_(std::move(box)) {}

  [[nodiscard]] int get_number_of_parameters() const {
    int numParam = module_.get_number_of_parameters();
    if (numParam == get_null_value<int>() && !box_.is_trivial()) return box_.get_number_of_coordinates();
    return numParam;
  }

  [[nodiscard]] int size() const { return module_.size(); }

  [[nodiscard]] int get_max_dimension() const { return module_.get_max_dimension(); }

  iterator begin() { return module_.begin(); }

  iterator end() { return module_.end(); }

  const_iterator begin() const { return module_.begin(); }

  const_iterator end() const { return module_.end(); }

  // TODO: nanobind binding for multipers
  Summand_of_dimension_range get_summands_of_dimension_range(int degree) const {
    // alternative would be to just warn + return empty range instead of throw
    if (degree < 0) throw std::invalid_argument("Cannot iterate over summands of negative dimension.");
    return module_.get_summands_of_dimension_range(static_cast<Dimension>(degree)) |
           boost::adaptors::type_erased<Summand_t, boost::forward_traversal_tag, const Summand_t &, std::ptrdiff_t>();
  }

  auto get_box_lower_corner_view() const {
    // no transfer of ownership, dies together with the box
    return nanobind::ndarray<const T, nanobind::numpy>(box_.get_lower_corner().data(),
                                                       {box_.get_number_of_coordinates()});
  }

  auto get_box_upper_corner_view() const {
    // no transfer of ownership, dies together with the box
    return nanobind::ndarray<const T, nanobind::numpy>(box_.get_upper_corner().data(),
                                                       {box_.get_number_of_coordinates()});
  }

  auto get_flat_box() const {
    std::vector<T> fb(box_.get_lower_corner().retrieve_underlying_container());
    {
      nanobind::gil_scoped_release release;
      fb.reserve(fb.size() * 2);
      fb.insert(fb.end(), box_.get_upper_corner().begin(), box_.get_upper_corner().end());
    }
    return _wrap_as_numpy_array(std::move(fb), 2, box_.get_number_of_coordinates());
  }

  Module_interface &set_box(const Box<value_type> &box) {
    box_ = box;
    return *this;
  }

  Module_interface &set_box(Tensor2D box) {
    box_ = get_box_from_tensor(box);
    return *this;
  }

  Module_interface &set_box(const std::vector<std::vector<value_type>> &box) {
    if (box.size() != 2) throw std::invalid_argument("Box has to be represented by two corners.");
    if (box[0].size() != box[1].size())
      throw std::invalid_argument("Both corners defining the box must have same dimension.");
    box_ = Box<value_type>(box[0], box[1]);
    return *this;
  }

  [[nodiscard]] nanobind::list get_flat_filtration_values(bool unique) const {
    int numParam = module_.get_number_of_parameters();
    if (numParam == get_null_value<int>()) return {}; // empty module

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
    std::vector<std::vector<std::array<value_type, 2>>> barcode;
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

  Module_interface get_module_of_degrees(Tensor1D degrees) {
    Module_interface out(box_);
    {
      nanobind::gil_scoped_release release;
      out.module_ = Gudhi::multi_persistence::build_module_of_dimension(module_, Numpy_span(degrees));
    }
    return out;
  }

  Module_interface get_module_of_degrees(const std::vector<T> &degrees) {
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

    std::vector<T> out;
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

    std::vector<T> out;
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
                      T delta,
                      T p,
                      bool normalize,
                      int n_jobs) {
    std::vector<T> out;
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

  // for some reasons nanobind cannot deduce an "auto" return type for "compute_distance_to" and
  // "compute_interleavings", even though it has no problems with the other methods ("compute_pixels" etc.)
  // I don't know where this is coming from...
  nanobind::ndarray<nanobind::numpy, T> compute_distance_to(const std::vector<std::vector<T>> &pts,
                                                            bool signed_distance,
                                                            int n_jobs) {
    std::vector<T> out;
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_module_distances_to(module_, pts, signed_distance, n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), pts.size(), module_.size());
  }

  nanobind::ndarray<nanobind::numpy, T> compute_distance_to(Tensor2D pts, bool signed_distance, int n_jobs) {
    std::vector<T> out;
    {
      nanobind::gil_scoped_release release;
      out = Gudhi::multi_persistence::compute_module_distances_to(module_, Numpy_2d_span(pts), signed_distance, n_jobs);
    }
    return _wrap_as_numpy_array(std::move(out), pts.shape(0), module_.size());
  }

  nanobind::ndarray<nanobind::numpy, T> compute_interleavings() {
    std::vector<T> interleavings;
    {
      nanobind::gil_scoped_release release;
      interleavings = Gudhi::multi_persistence::compute_module_interleavings(module_, box_);
    }
    return _wrap_as_numpy_array(std::move(interleavings), interleavings.size());
  }

  nanobind::ndarray<nanobind::numpy, T> compute_interleavings(Tensor2D box) {
    std::vector<T> interleavings;
    {
      nanobind::gil_scoped_release release;
      interleavings = Gudhi::multi_persistence::compute_module_interleavings(module_, get_box_from_tensor(box));
    }
    return _wrap_as_numpy_array(std::move(interleavings), interleavings.size());
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
    curr = serialize_value_to_char_buffer(value.box_, curr);
    curr = serialize_value_to_char_buffer(value.module_, curr);
    return curr;
  }

  friend const char *deserialize_value_from_char_buffer(Module_interface &value, const char *start) {
    const char *curr = start;
    curr = deserialize_value_from_char_buffer(value.box_, curr);
    curr = deserialize_value_from_char_buffer(value.module_, curr);
    return curr;
  }

  friend std::size_t get_serialization_size_of(const Module_interface &value) {
    return get_serialization_size_of(value.box_) + get_serialization_size_of(value.module_);
  }

  friend void swap(Module_interface &mod1, Module_interface &mod2) noexcept {
    swap(mod1.box_, mod2.box_);
    swap(mod1.module_, mod2.module_);
  }

 private:
  Module<T> module_;
  Box<T> box_;

  template <typename Y>
  static constexpr Y get_null_value() {
    return Module<T>::Summand_t::template get_null_value<Y>();
  }

  static Box<T> get_box_from_tensor(Tensor2D box) {
    Numpy_2d_span boxView(box);
    auto lowerView = boxView[0];
    auto upperView = boxView[1];
    return {lowerView.begin(), lowerView.end(), upperView.begin(), upperView.end()};
  }
};

template <typename T>
Module_interface<T> deserialize_from_python(
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
