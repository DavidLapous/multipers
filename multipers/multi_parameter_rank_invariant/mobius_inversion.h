#pragma once

#include <cstddef>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace Gudhi {
namespace multiparameter {
namespace mobius_inversion {

template <typename index_type>
inline std::vector<index_type> compute_strides(const std::vector<index_type> &shape) {
  std::vector<index_type> strides(shape.size());
  if (shape.empty()) return strides;
  strides.back() = 1;
  for (std::size_t axis = shape.size() - 1; axis > 0; --axis) strides[axis - 1] = shape[axis] * strides[axis];
  return strides;
}

template <typename index_type>
inline std::size_t dense_size(const std::vector<index_type> &shape) {
  if (shape.empty()) return 0;
  return static_cast<std::size_t>(
      std::accumulate(shape.begin(), shape.end(), index_type{1}, std::multiplies<index_type>()));
}

template <typename index_type, typename Coordinates>
inline index_type linear_offset(const Coordinates &coordinates, const std::vector<index_type> &strides) {
  index_type offset = 0;
  auto coordinate = coordinates.begin();
  auto stride = strides.begin();
  for (; coordinate != coordinates.end(); ++coordinate, ++stride) offset += (*coordinate) * (*stride);
  return offset;
}

template <typename dtype, typename index_type>
class dense_tensor_view {
 public:
  using sparse_type = std::pair<std::vector<std::vector<index_type>>, std::vector<dtype>>;

  dense_tensor_view(dtype *data, const std::vector<index_type> &shape)
      : data_(data), shape_(shape), strides_(compute_strides(shape)), size_(dense_size(shape)) {}

  template <typename Coordinates>
  inline dtype &operator[](const Coordinates &coordinates) const {
    return data_[linear_offset<index_type>(coordinates, strides_)];
  }

  inline dtype &data_at(index_type offset) const { return data_[offset]; }
  inline dtype *data() const { return data_; }
  inline const std::vector<index_type> &get_resolution() const { return shape_; }
  inline const std::vector<index_type> &get_cum_resolution() const { return strides_; }
  inline std::size_t ndim() const { return shape_.size(); }
  inline std::size_t size() const { return size_; }

  inline std::vector<index_type> data_index_inverse(index_type data_index,
                                                    const std::vector<bool> &flip_axes = {}) const {
    std::vector<index_type> coordinates(shape_.size());
    for (int axis = static_cast<int>(coordinates.size()) - 1; axis >= 0; --axis) {
      const auto q = data_index / shape_[axis];
      const auto r = data_index - q * shape_[axis];
      if (static_cast<int>(flip_axes.size()) > axis && flip_axes[axis])
        coordinates[axis] = shape_[axis] - r;
      else
        coordinates[axis] = r;
      data_index = q;
    }
    return coordinates;
  }

 private:
  dtype *data_;
  std::vector<index_type> shape_;
  std::vector<index_type> strides_;
  std::size_t size_;
};

template <typename dtype, typename index_type>
inline void differentiate(dtype *data, const std::vector<index_type> &shape, index_type axis) {
  const auto strides = compute_strides(shape);
  const std::size_t size = dense_size(shape);
  const auto stride = strides[axis];
  const auto axis_size = shape[axis];
  if (axis_size <= 1) return;

  for (std::size_t linear = size; linear-- > 0;) {
    const auto coordinate = (static_cast<index_type>(linear) / stride) % axis_size;
    if (coordinate == 0) continue;
    data[linear] -= data[linear - stride];
  }
}

template <typename dtype, typename index_type>
inline std::pair<std::vector<std::vector<index_type>>, std::vector<dtype>> sparsify(
    const dense_tensor_view<dtype, index_type> &tensor,
    const std::vector<bool> &flip_axes = {},
    bool verbose = false) {
  std::vector<std::vector<index_type>> coordinates;
  std::vector<dtype> values;
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    const auto value = tensor.data_at(static_cast<index_type>(i));
    if (value == 0) continue;
    coordinates.push_back(tensor.data_index_inverse(static_cast<index_type>(i), flip_axes));
    values.push_back(value);
  }
  if (verbose) {
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
      for (const auto &coordinate : coordinates[i]) std::cout << coordinate << " ";
      std::cout << "| " << values[i] << std::endl;
    }
  }
  return {coordinates, values};
}

}  // namespace mobius_inversion
}  // namespace multiparameter
}  // namespace Gudhi
