#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

// TODO : sparse version, current operator[] is already a hash.
namespace tensor {

template <typename indices_type>
inline std::vector<indices_type> compute_backward_cumprod(const std::vector<indices_type> &resolution) {
  constexpr bool verbose = false;
  std::vector<indices_type> cum_prod_resolution_(resolution.size());
  cum_prod_resolution_.back() = 1;
  for (auto i = resolution.size() - 1; i > 0; i--) {
    // std::cout << i << " " << cum_prod_resolution_.size() << std::endl;
    cum_prod_resolution_[i - 1] = resolution[i] * cum_prod_resolution_[i];
  }
  if constexpr (verbose) {
    std::cout << "Cum resolution ";
    for (auto c : cum_prod_resolution_) std::cout << c << " ";
    std::cout << std::endl;
  }
  return cum_prod_resolution_;
}

template <typename dtype, typename indices_type>
class static_tensor_view {  // Python handles the construction - destruction of
                            // the data,
 public:
  using sparse_type = std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>>;
  static_tensor_view();

  static_tensor_view(dtype *data_ptr, const std::vector<indices_type> &resolution)
      : data_ptr_(data_ptr),
        size_(resolution.size() == 0
                  ? 0
                  : std::accumulate(begin(resolution), end(resolution), 1, std::multiplies<indices_type>())),
        resolution_(resolution)
  // cum_prod_resolution_(compute_backward_cumprod(resolution))
  {
    // cum_prod_resolution_ = std::vector<std::size_t>(resolution.size());
    // std::size_t last = 1;
    // for (auto i = resolution.size() -1; i > 0; i--){
    // 	last *=resolution[i];
    // 	// std::cout << i << " " << cum_prod_resolution_.size() << std::endl;
    // 	cum_prod_resolution_[resolution.size()-1 - i] = last;
    // }
    // cum_prod_resolution_.back() = 1;
    cum_prod_resolution_ = std::move(compute_backward_cumprod(resolution));
  };

  // dtype[]& data_ref(){
  // 	return *data_ptr;
  // }
  inline std::size_t size() const { return size_; }

  inline bool empty() const { return size_ == 0; }

  inline dtype &data_back() const { return *(data_ptr_ + size_ - 1); }

  inline std::size_t ndim() const { return resolution_.size(); }

  template <class oned_array_like = std::initializer_list<indices_type>>
  inline dtype &operator[](const oned_array_like &coordinates) const {
    const bool check = false;
    dtype *data_index = data_ptr_;
    /* 0; // max is 4*10^9, should be fine. just put an assert in python. */

    if constexpr (check) {
      if (coordinates.size() != resolution_.size()) {
        auto it = coordinates.begin();
        for (size_t i = 0u; i < coordinates.size(); i++) std::cerr << *(it++) << "/" << resolution_[i] << ", ";
        std::cerr << ")" << std::endl;
        throw std::invalid_argument("Invalid coordinate dimension.");
      }
      // for (auto [ci, cum_res, res] : std::views::zip(coordinates,
      // cum_prod_resolution_, resolution_)){ // NIK Apple clang for
      // (indices_type i : std::views::iota(0,coordinates.size())){
      auto it = coordinates.begin();
      for (size_t i = 0u; i < coordinates.size(); i++) {
        auto &ci = *(it++);
        auto cum_res = cum_prod_resolution_[i];
        auto res = resolution_[i];
        if (ci >= res) [[unlikely]] {
          std::cerr << "Crash log. Coordinates : (";
          auto it = coordinates.begin();
          for (auto i = 0u; i < coordinates.size(); i++) std::cerr << *(it++) << "/" << resolution_[i] << ", ";
          // for (auto [c, r] : std::views::zip(coordinates, resolution_))
          // std::cerr << c << "/" << r << ", "; // NIK APPLE CLANG
          std::cerr << ")" << std::endl;
          throw std::invalid_argument("Illegal coordinates.");
        }
        data_index += ci * cum_res;
      }
      if (data_index >= this->size()) [[unlikely]] {
        std::cerr << "Crash log. Coordinates : (";
        auto it = coordinates.begin();
        for (size_t i = 0u; i < coordinates.size(); i++) std::cerr << *(it++) << "/" << resolution_[i] << ", ";
        std::cerr << ")" << std::endl;
        throw std::invalid_argument("Internal error : asked data " + std::to_string(data_index) + "/" +
                                    std::to_string(this->size()));
      }
      // std::cout << data_index << " " << this->size() << std::endl;
      // std::cout << data_index << "/" << this->size() << std::endl;
    } else {
      // for (auto [ci, cum_res] : std::views::zip(coordinates,
      // cum_prod_resolution_)){ // small so i'm not sure reduce can be
      // efficient here // NIK Apple clang 	data_index += ci*cum_res;
      // }

      auto coord_ptr = coordinates.begin();
      auto cum_res_ptr = cum_prod_resolution_.begin();
      for (; coord_ptr != coordinates.end(); ++coord_ptr, ++cum_res_ptr) {
        data_index += (*coord_ptr) * (*cum_res_ptr);
      }
    }
    /* return *(data_ptr_ + data_index); */
    return *data_index;
  }

  template <typename idx_type>
  inline dtype &data_at_index(idx_type i) {
    return *(data_ptr_ + i);
  }

  template <typename indice_type_like>
  inline std::vector<indices_type> data_index_inverse(indice_type_like data_index,
                                                      const std::vector<bool> &flip_axes = {}) const {
    std::vector<indices_type> coordinates(resolution_.size());
    int data_index_ = data_index;
    for (int parameter = static_cast<int>(coordinates.size()) - 1; parameter >= 0; parameter--) {
      auto [q, r] = std::div(data_index_, static_cast<int>(resolution_[parameter]));
      if (static_cast<int>(flip_axes.size()) > parameter && flip_axes[parameter])
        coordinates[parameter] = resolution_[parameter] - r;
      else
        coordinates[parameter] = r;
      data_index_ = q;
    }
    return coordinates;
  }

  // friend std::ostream& operator<<(std::ostream& stream, const
  // static_tensor_view<dtype,indices_type>& truc){
  //     stream << "[";
  //     for(indices_type i = 0; i < truc.size()-1; i++){
  //         stream << *(truc.data_ptr_ + i) << ", ";
  //     }
  //     if(!truc.empty()) stream << truc.data_back();
  //     stream << "]";
  // 	stream << "\n resolution : ";
  // 	for(indices_type i = 0; i < truc.resolution_.size(); i++){
  //         stream << truc.resolution_[i] << ", ";
  //     }
  // 	stream << "\n cum resolution : ";
  // 	for(indices_type i = 0; i < truc.cum_prod_resolution_.size(); i++){
  //         stream << truc.cum_prod_resolution_[i] << ", ";
  //     }
  //     return stream;
  // }

  friend std::ostream &operator<<(std::ostream &stream, const static_tensor_view<dtype, indices_type> &truc) {
    // constexpr bool verbose = false;
    for (auto parameter = 0u; parameter < truc.ndim(); parameter++) stream << "[";
    // iterate over data, update coordinates in a vector, and print if in free
    // coords i.e. add one to last coord, modulo if greater, and propagate to
    // the next
    std::vector<indices_type> coordinates(truc.ndim());  /// 0,...,0
    for (auto i = 0u; i < truc.size() - 1; i++) {
      stream << truc.data_at(i);

      // for (indices_type parameter =0; parameter < coordinates.size();
      // parameter++){ 	stream << coordinates[parameter];
      // }
      // stream << "\n";
      coordinates[0]++;
      indices_type parameter = 0;
      for (; parameter < static_cast<int>(coordinates.size()) - 1; ++parameter) {
        if (coordinates[parameter] < truc.get_resolution()[parameter]) {
          // stream << ", ";
          // if (parameter == 1)
          // 	stream << "\n";
          break;
        }
        // for (indices_type i =0; i < parameter; i++)
        // 	stream << ";";
        // for (indices_type i =0; i < parameter+1; i++)
        // 	stream << "]";
        // stream << ", ";
        // for (indices_type i =0; i < parameter; i++)
        // 	stream << "[";
        coordinates[parameter] = 0;  // 1 by 1 so should be fine not doing mods
        coordinates[parameter + 1]++;
      }
      if (parameter == 1)
        stream << "],\n [";
      else {
        for (indices_type i = 0; i < parameter; i++) stream << "]";
        stream << ", ";
        for (indices_type i = 0; i < parameter; i++) stream << "[";
      }
    }

    stream << truc.data_back();
    for (auto parameter = 0u; parameter < truc.ndim(); parameter++) stream << "]";
    return stream;
  }

  // template<class
  // twod_array_like=std::initializer_list<std::initializer_list<indices_type>>>
  // static_tensor_view_view<dtype,indices_type> view(twod_array_like
  // coordinates){ 	auto out = static_tensor_view_view(data_ptr_,
  // resolution_); 	out.free_coordinates = coordinates; 	return out;
  // }
  inline const std::vector<indices_type> &get_resolution() const { return resolution_; }

  inline const std::vector<indices_type> &get_cum_resolution() const { return cum_prod_resolution_; }

  template <typename indice_type_like>
  inline dtype &data_at(indice_type_like i) const {
    return *(data_ptr_ + i);
  }

  void differentiate(indices_type axis);

  inline sparse_type sparsify(const std::vector<bool> &flip_axes = {}, bool verbose = false) const {
    std::vector<std::vector<indices_type>> coordinates;
    std::vector<dtype> values;
    // for (indices_type i = 0; i < static_cast<indices_type>(this->size());
    // i++){
    for (auto i = 0u; i < this->size(); i++) {
      auto stuff = this->data_at(i);
      if (stuff == 0) [[likely]]  // as this is sparse
        continue;
      coordinates.push_back(this->data_index_inverse(i, flip_axes));
      values.push_back(stuff);
    }
    if (verbose) [[unlikely]] {
      // for (auto [pt,w] : std::views::zip(coordinates, values)){ NIK apple
      // clang
      for (auto i = 0u; i < coordinates.size(); i++) {
        for (const auto &v : coordinates[i]) std::cout << v << " ";
        std::cout << "| " << values[i] << std::endl;
      }
    }
    return {coordinates, values};
  }

  // template<class oned_array_like=std::initializer_list<indices_type>>
  void _rec_add_cone(const std::vector<indices_type> &basepoint,
                     dtype value,
                     std::vector<indices_type> &coordinates,
                     int _rec_parameter) const {
    if (_rec_parameter < 0) {
      (*this)[coordinates] += value;
      return;
    }
    for (indices_type c = basepoint[_rec_parameter]; c < this->get_resolution()[_rec_parameter]; c++) {
      coordinates[_rec_parameter] = c;
      this->_rec_add_cone(basepoint, value, coordinates, _rec_parameter - 1);
    }
  }

  inline void add_cone(const std::vector<indices_type> &basepoint, dtype value) const {
    constexpr const bool check = false;
    constexpr const bool verbose = false;
    if constexpr (check) {
      if (basepoint.size() != this->ndim()) throw std::logic_error("Invalid coordinate for cone");
    }
    if constexpr (verbose) {
      std::cout << "Adding cone ";
      for (auto b : basepoint) std::cout << b << " ,";
      std::cout << std::endl;
    }
    std::vector<indices_type> temp_container(this->ndim());
    this->_rec_add_cone(basepoint, value, temp_container, static_cast<int>(this->ndim()) - 1);
  }

  // template<class oned_array_like=std::initializer_list<indices_type>>
  void _rec_add_cone_boundary(const std::vector<indices_type> &basepoint,
                              dtype value,
                              std::vector<indices_type> &coordinates,
                              int _rec_parameter) const {
    if (_rec_parameter < 0) {
      (*this)[coordinates] += value;
      return;
    }

    // for (auto c=basepoint[_rec_parameter];
    // c<this->get_resolution()[_rec_parameter]; c++){
    // 	coordinates[_rec_parameter] = c;
    // 	this->_rec_add_cone(basepoint, value, coordinates, _rec_parameter-1);
    // }

    coordinates[_rec_parameter] = basepoint[_rec_parameter];
    this->_rec_add_cone_boundary(std::vector<indices_type>(basepoint), value, coordinates, _rec_parameter - 1);

    coordinates[_rec_parameter] = this->get_resolution()[_rec_parameter] - 1;
    this->_rec_add_cone_boundary(basepoint, -value, coordinates, _rec_parameter - 1);
  }

  inline void add_cone_boundary(const std::vector<indices_type> &basepoint, dtype value) const {
    const bool check = false;
    if constexpr (check) {
      if (basepoint.size() != this->ndim()) throw std::logic_error("Invalid coordinate for cone boundary");
    }
    std::vector<indices_type> temp_container(this->ndim());
    this->_rec_add_cone_boundary(basepoint, value, temp_container, static_cast<int>(this->ndim()) - 1);
  }

 public:
 private:
  dtype *data_ptr_;
  std::size_t size_;
  std::vector<indices_type> resolution_;
  std::vector<indices_type> cum_prod_resolution_;
  // std::vector<std::vector<indices_types>> fixed_coordinates; // in child
};

template <typename dtype, typename indices_type>
class static_tensor_view_view
    : public static_tensor_view<dtype, indices_type> {  // i'm not sure this class is very efficient.
 public:
  using base = static_tensor_view<dtype, indices_type>;

  static_tensor_view_view(dtype *data_ptr,
                          const std::vector<indices_type> &resolution,
                          const std::vector<std::vector<indices_type>> &free_coordinates,
                          bool use_sparse = true)
      : base(data_ptr, resolution),
        resolution_view(this->compute_resolution(free_coordinates))
  // free_coordinates(free_coordinates)
  {
    this->compute_ptrs(free_coordinates, use_sparse);
  };

  static_tensor_view_view(const static_tensor_view<dtype, indices_type> &parent,
                          const std::vector<std::vector<indices_type>> &free_coordinates,
                          bool use_sparse = true)
      : base(parent),
        resolution_view(this->compute_resolution(free_coordinates))
  // free_coordinates(free_coordinates)
  {
    this->compute_ptrs(free_coordinates, use_sparse);
  };

  inline bool is_float(const std::vector<indices_type> &resolution) const {
    indices_type dim = this->dimension();
    for (indices_type i = 0; i < dim; i++)
      if (resolution[i] > 1) return false;
    return true;
  }

  inline bool is_float() const { return this->is_float(this->resolution_view); }

  template <class oned_array_like = std::initializer_list<indices_type>>
  inline bool is_in_view(const oned_array_like &coordinates,
                         const std::vector<std::vector<indices_type>> &free_coordinates) {
    assert(coordinates.size() == this->ndim());
    auto it = coordinates.begin();
    for (indices_type parameter = 0; parameter < static_cast<indices_type>(this->ndim()); ++parameter) {
      const auto &x = *it;
      it++;
      for (auto stuff : free_coordinates[parameter]) {
        if (stuff < x)
          continue;
        else if (stuff == x)
          break;
        else
          return false;
      }
      if (x > free_coordinates[parameter].back()) return false;
    }
    return true;
  }

  std::size_t _size() const {  // for construction
    std::size_t out = 1;
    for (const auto &r : resolution_view) out *= r;
    return out;
  }

  std::size_t size() const { return ptrs.size(); }

  std::vector<indices_type> compute_resolution(const std::vector<std::vector<indices_type>> &free_coordinates) {
    std::vector<indices_type> out(free_coordinates.size());
    // for (auto [s, stuff] : std::views::zip(out, free_coordinates)) s =
    // stuff.size(); // NIK apple clang
    for (auto i = 0u; i < free_coordinates.size(); i++) out[i] = free_coordinates[i].size();
    return out;
  }

  void compute_ptrs_dense(const std::vector<std::vector<indices_type>> &free_coordinates) {  // todo redo from
    // DO NOT USE
    constexpr bool verbose = false;
    std::vector<dtype *> out(this->_size());
    std::vector<indices_type> coordinates(this->ndim());  /// 0,...,0
    std::size_t count = 0;

    for (int i = 0; i < static_cast<int>(static_tensor_view<dtype, indices_type>::size()) - 1; i++) {
      if constexpr (verbose) {
        std::cout << "Coordinate : ";
        for (auto x : coordinates) std::cout << x << " ";
        if (this->is_in_view(coordinates, free_coordinates))
          std::cout << " in view";
        else
          std::cout << "not in view";
        std::cout << std::endl;
      }

      if (this->is_in_view(coordinates, free_coordinates)) {
        out[count] = &this->data_at(i);
        count++;
      }
      coordinates.back()++;
      for (indices_type parameter = coordinates.size() - 1; parameter > 0; parameter--) {
        if (coordinates[parameter] < this->get_resolution()[parameter]) {
          break;
        }
        for (indices_type i = parameter; i < static_cast<indices_type>(coordinates.size()); i++)
          coordinates[i] = 0;  // 1 by 1 so should be fine not doing mods
        coordinates[parameter - 1]++;
      }
    }
    if (this->is_in_view(coordinates, free_coordinates)) {
      out[count] = &this->data_back();
      count++;
    }
    // assert(count == this->size());
    ptrs.swap(out);
  }

  inline void compute_ptrs_sparse(const std::vector<std::vector<indices_type>> &free_coordinates,
                                  std::vector<indices_type> _rec_coordinates_begin = {}) {  // todo redo from
    constexpr bool verbose = false;
    if (_rec_coordinates_begin.size() == 0) ptrs.reserve(this->_size());
    indices_type parameter = _rec_coordinates_begin.size();
    if (parameter == static_cast<indices_type>(this->ndim())) {
      auto &value = tensor::static_tensor_view<dtype, indices_type>::operator[](
          _rec_coordinates_begin);  // calling [] is not efficient, but not
                                    // bottleneck
      if constexpr (verbose) {
        std::cout << "Adding coordinates ";
        for (auto c : _rec_coordinates_begin) std::cout << c << " ";
        std::cout << " of value " << value;
        std::cout << std::endl;
      }
      ptrs.push_back(&value);
      return;
    }
    _rec_coordinates_begin.reserve(this->ndim());
    _rec_coordinates_begin.resize(parameter + 1);
    for (indices_type coord : free_coordinates[parameter]) {
      _rec_coordinates_begin.back() = coord;
      compute_ptrs_sparse(free_coordinates, _rec_coordinates_begin);
    }
    return;
  }

  inline void compute_ptrs(const std::vector<std::vector<indices_type>> &free_coordinates, bool use_sparse = true) {
    if (use_sparse)
      compute_ptrs_sparse(free_coordinates);
    else
      compute_ptrs_dense(free_coordinates);
  }

  inline void shift_coordinate(indices_type idx, indices_type shift_value) {
    // resolution stays the same,
    auto to_add = this->get_cum_resolution()[idx] * shift_value;
    for (auto &ptr : this->ptrs) ptr += to_add;
  }

  // constant additions
  inline void operator+=(dtype x) {
    // if (ptrs.empty()) this->compute_ptrs_dense();
    for (auto stuff : ptrs) *stuff += x;
    return;
  }

  inline void operator-=(dtype x) {
    // if (ptrs.empty()) this->compute_ptrs_dense();
    for (auto stuff : ptrs) *stuff -= x;
    return;
  }

  inline void operator*=(dtype x) {
    // if (ptrs.empty()) this->compute_ptrs_dense();
    for (auto stuff : ptrs) *stuff *= x;
    return;
  }

  inline void operator/=(dtype x) {
    // if (ptrs.empty()) this->compute_ptrs_dense();
    for (auto stuff : ptrs) *stuff /= x;
    return;
  }

  inline void operator=(dtype x) {
    for (auto stuff : ptrs) *stuff = x;
    return;
  }

  inline void operator=(const static_tensor_view_view<dtype, indices_type> &x) {
    assert(this->size() == x.size());
    this->ptrs = x.ptrs;
    return;
  }

  inline void swap(static_tensor_view_view<dtype, indices_type> &x) {
    this->ptrs.swap(x.ptrs);
    return;
  }

  // retrieves data from ptrs
  inline void operator+=(const static_tensor_view_view<dtype, indices_type> &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] += *x[idx];
    return;
  }

  inline void operator-=(const static_tensor_view_view<dtype, indices_type> &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] -= *x[idx];
    return;
  }

  inline void operator*=(const static_tensor_view_view<dtype, indices_type> &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] *= *x[idx];
    return;
  }

  inline void operator/=(const static_tensor_view_view<dtype, indices_type> &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] /= *x[idx];
    return;
  }

  // Default array_like template
  template <typename array_like = std::initializer_list<dtype>>
  inline void operator+=(const array_like &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] += *(x.begin() + idx);
    return;
  }

  template <typename array_like = std::initializer_list<dtype>>
  inline void operator-=(const array_like &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] -= *(x.begin() + idx);
    return;
  }

  template <typename array_like = std::initializer_list<dtype>>
  inline void operator*=(const array_like &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] *= *(x.begin() + idx);
    return;
  }

  template <typename array_like = std::initializer_list<dtype>>
  inline void operator/=(const array_like &x) {
    std::size_t num_data = this->size();
    assert(num_data == x.size());
    for (auto idx = 0u; idx < num_data; idx++) *ptrs[idx] /= *(x.begin() + idx);
    return;
  }

  // void compute_cum_res(){
  // 	if (cum_resolution_view.size() == 0){

  // 		cum_resolution_view =
  // compute_backward_cumprod(this->resolution_view);
  // 	}
  // }
  template <typename T = std::initializer_list<indices_type>>
  inline dtype &operator[]([[maybe_unused]] T coords) {
    throw std::logic_error("Not yet implemented");
    // this->compute_cum_res();
    // assert(this->cum_resolution_view.size() == coords.size());
    // std::size_t data_index = 0;
    // // for (indices_type i = 0, auto coords_it = coords.begin(); coords_it !=
    // coords.end(); coords_it++, i++)
    // // 	{data_index += (*(coords_it))*cum_resolution_view[i];};
    // for (auto [c, cr] : std::views::zip(coords, cum_resolution_view))
    // 	data_index += c*cr;
    // std::cout << ptrs.size() << " vs " << data_index << std::endl;
    // return *ptrs[data_index];
  }

  void print_data() const {
    std::cout << "[";
    for (auto stuff : ptrs) std::cout << *stuff << " ";
    std::cout << "]\n";
  }

  inline std::vector<dtype> copy_data() {
    std::vector<dtype> out(ptrs.size());
    for (auto i = 0u; i < ptrs.size(); i++) out[i] = *ptrs[i];
    return out;
  }

 public:
  // juste besoin de la resolution, avec les ptrs : ok pour l'affichage
  // const std::vector<std::vector<indices_type>> free_coordinates; // for each
  // parameter, the fixed indices, TODO:REMOVE
  const std::vector<indices_type> resolution_view;

 private:
  std::vector<dtype *> ptrs;
  // std::vector<std::size_t> cum_resolution_view; // not really useful.
};

template <typename dtype, typename indices_type>
void inline static_tensor_view<dtype, indices_type>::differentiate(indices_type axis) {
  std::vector<std::vector<indices_type>> free_coordinates(this->ndim());

  // initialize free_coordinates of the view, full coordinates on each axis
  // exept for axis on which we iterate
  for (auto i = 0u; i < free_coordinates.size(); i++) {
    if (static_cast<indices_type>(i) == axis) continue;
    free_coordinates[i] = std::vector<indices_type>(this->get_resolution()[i]);
    for (auto j = 0u; j < free_coordinates[i].size(); j++) {  // TODO optimize
      free_coordinates[i][j] = j;
    }
  }
  // iterate over coordinate of this axis with ab -> b-a -> ab=b[newslice]
  free_coordinates[axis] = {{0}};
  static_tensor_view_view<dtype, indices_type> x_i(*this, free_coordinates);
  std::vector<dtype> a, b;
  a = x_i.copy_data();
  for (indices_type h = 1; h < this->get_resolution()[axis]; h++) {
    free_coordinates[axis] = {{h}};
    // x_i = static_tensor_view_view<dtype,
    // indices_type>(*this,free_coordinates);
    x_i.shift_coordinate(axis, 1);
    b = std::move(x_i.copy_data());
    x_i -= a;
    a.swap(b);
  }
}

template <typename T>
std::vector<std::vector<T>> cart_product(const std::vector<std::vector<T>> &v) {
  std::vector<std::vector<T>> s = {{}};
  for (const auto &u : v) {
    std::vector<std::vector<T>> r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  for (const auto &truc : s) {
    for (const auto &machin : truc) std::cout << machin << ", ";
    std::cout << "\n";
  }
  return s;
}

}  // namespace tensor
