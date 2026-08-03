#ifndef MULTIPERS_GUDHI_MODULE_LANDSCAPES_HPP
#define MULTIPERS_GUDHI_MODULE_LANDSCAPES_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef GUDHI_USE_TBB
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#endif

#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Module.h>
#include <gudhi/Multi_persistence/summand_helpers.h>
#include <gudhi/Multi_persistence/utils.h>

namespace multipers::detail {

template <typename T>
struct ModuleLandscapeSummandCache {
  const typename Gudhi::multi_persistence::Module<T>::Summand_t* summand;
  std::vector<T> lower;
  std::vector<T> upper;
};

template <typename T>
inline std::vector<ModuleLandscapeSummandCache<T>> build_module_landscape_summand_cache(
    const Gudhi::multi_persistence::Module<T>& module,
    typename Gudhi::multi_persistence::Module<T>::Dimension dimension) {
  using Module = Gudhi::multi_persistence::Module<T>;
  using Summand = typename Module::Summand_t;

  std::vector<ModuleLandscapeSummandCache<T>> summands;
  summands.reserve(module.size());
  for (const auto& summand : module) {
    if (summand.get_dimension() != dimension) continue;
    const auto& births = summand.get_upset();
    const auto& deaths = summand.get_downset();
    if (births.num_generators() == 0 || deaths.num_generators() == 0) continue;

    const auto num_parameters = static_cast<std::size_t>(summand.get_number_of_parameters());
    ModuleLandscapeSummandCache<T> entry{
        &summand,
        std::vector<T>(num_parameters, Summand::T_inf),
        std::vector<T>(num_parameters, Summand::T_m_inf),
    };
    for (std::size_t generator = 0; generator < static_cast<std::size_t>(births.num_generators()); ++generator) {
      for (std::size_t parameter = 0; parameter < num_parameters; ++parameter) {
        entry.lower[parameter] = std::min(entry.lower[parameter], births(generator, parameter));
      }
    }
    for (std::size_t generator = 0; generator < static_cast<std::size_t>(deaths.num_generators()); ++generator) {
      for (std::size_t parameter = 0; parameter < num_parameters; ++parameter) {
        const T death = deaths(generator, parameter);
        if (death == Summand::T_inf) {
          entry.upper[parameter] = Summand::T_inf;
        } else if (entry.upper[parameter] != Summand::T_inf) {
          entry.upper[parameter] = std::max(entry.upper[parameter], death);
        }
      }
    }
    summands.push_back(std::move(entry));
  }
  return summands;
}

template <typename T, class RandomAccessValueRange>
inline bool could_have_positive_module_landscape(const ModuleLandscapeSummandCache<T>& summand,
                                                 const RandomAccessValueRange& x) {
  if (x.size() != summand.lower.size()) return true;
  for (std::size_t parameter = 0; parameter < summand.lower.size(); ++parameter) {
    if (x[parameter] <= summand.lower[parameter] || x[parameter] >= summand.upper[parameter]) return false;
  }
  return true;
}

template <class RandomAccessValueRange>
inline std::size_t module_landscape_top_size(const RandomAccessValueRange& ks) {
  std::size_t top_size = 0;
  for (std::size_t index = 0; index < ks.size(); ++index) {
    const auto k = ks[index];
    if constexpr (std::is_signed_v<std::decay_t<decltype(k)>>) {
      if (k < 0) throw std::invalid_argument("Landscape indices must be non-negative.");
    }
    const auto k_value = static_cast<std::size_t>(k);
    if (k_value == std::numeric_limits<std::size_t>::max()) {
      throw std::length_error("Landscape index is too large.");
    }
    top_size = std::max(top_size, k_value + 1);
  }
  return top_size;
}

template <typename T>
inline void insert_module_landscape_value(T value, std::vector<T>& top) {
  if (top.empty() || !(value > top.back())) return;
  for (std::size_t position = 0; position < top.size(); ++position) {
    if (value > top[position]) {
      for (std::size_t shift = top.size() - 1; shift > position; --shift) {
        top[shift] = top[shift - 1];
      }
      top[position] = value;
      return;
    }
  }
}

template <typename T, class RandomAccessValueRange1, class RandomAccessValueRange2>
inline void set_module_landscape_pixel(std::vector<Gudhi::multi_persistence::maybe_make_signed_t<T>>& images,
                                       std::size_t pixel,
                                       std::size_t plane_size,
                                       const std::vector<ModuleLandscapeSummandCache<T>>& summands,
                                       const RandomAccessValueRange1& x,
                                       const RandomAccessValueRange2& ks,
                                       std::vector<Gudhi::multi_persistence::maybe_make_signed_t<T>>& top) {
  using SignedT = Gudhi::multi_persistence::maybe_make_signed_t<T>;

  std::fill(top.begin(), top.end(), SignedT(0));
  for (const auto& summand : summands) {
    if (!could_have_positive_module_landscape(summand, x)) continue;
    insert_module_landscape_value(Gudhi::multi_persistence::compute_summand_landscape_value(*summand.summand, x), top);
  }
  for (std::size_t index = 0; index < ks.size(); ++index) {
    const auto k = static_cast<std::size_t>(ks[index]);
    images[index * plane_size + pixel] = k < top.size() ? top[k] : SignedT(0);
  }
}

template <typename T, typename U, class RandomAccessValueRange1, class RandomAccessValueRange2>
inline std::vector<Gudhi::multi_persistence::maybe_make_signed_t<T>> compute_module_landscapes(
    const Gudhi::multi_persistence::Module<T>& module,
    typename Gudhi::multi_persistence::Module<T>::Dimension dimension,
    const RandomAccessValueRange1& ks,
    const Gudhi::multi_persistence::Box<U>& box,
    const RandomAccessValueRange2& resolution,
    int n_jobs = 0) {
  static_assert(std::is_same_v<U, T> || std::is_same_v<U, Gudhi::multi_persistence::maybe_make_signed_t<T>>,
                "Box template parameter is not compatible with Summand value type.");
  if (resolution.size() < 2) throw std::invalid_argument("Not enough resolution values.");

  using SignedT = Gudhi::multi_persistence::maybe_make_signed_t<T>;
  const auto nx = static_cast<std::size_t>(resolution[0]);
  const auto ny = static_cast<std::size_t>(resolution[1]);
  const auto plane_size = nx * ny;
  std::vector<SignedT> images(ks.size() * plane_size);
  if (ks.size() == 0 || plane_size == 0) return images;

  const auto summands = build_module_landscape_summand_cache(module, dimension);
  const auto top_size = std::min(module_landscape_top_size(ks), summands.size());
  if (top_size == 0) return images;

  const U step_x = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / static_cast<U>(nx);
  const U step_y = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / static_cast<U>(ny);
  auto set_pixel = [&](std::size_t pixel, std::vector<SignedT>& top) {
    const auto row = pixel / ny;
    const auto column = pixel - row * ny;
    const std::array<U, 2> x{
        box.get_lower_corner()[0] + step_x * static_cast<U>(row),
        box.get_lower_corner()[1] + step_y * static_cast<U>(column),
    };
    set_module_landscape_pixel(images, pixel, plane_size, summands, x, ks, top);
  };

#ifdef GUDHI_USE_TBB
  tbb::enumerable_thread_specific<std::vector<SignedT>> top([&] { return std::vector<SignedT>(top_size); });
  oneapi::tbb::task_arena arena(n_jobs);
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0), plane_size, [&](std::size_t pixel) { set_pixel(pixel, top.local()); });
  });
#else
  std::vector<SignedT> top(top_size);
  for (std::size_t pixel = 0; pixel < plane_size; ++pixel) set_pixel(pixel, top);
#endif

  return images;
}

template <typename T, class RandomAccessValueRange, class RandomAccessArray>
inline std::vector<Gudhi::multi_persistence::maybe_make_signed_t<T>> compute_module_landscapes(
    const Gudhi::multi_persistence::Module<T>& module,
    typename Gudhi::multi_persistence::Module<T>::Dimension dimension,
    const RandomAccessValueRange& ks,
    const std::vector<RandomAccessArray>& grid,
    int n_jobs = 0) {
  if (grid.size() < 2) throw std::invalid_argument("First axis of the grid has not enough values.");

  using SignedT = Gudhi::multi_persistence::maybe_make_signed_t<T>;
  using GridT = std::decay_t<decltype(grid[0][0])>;
  const auto nx = grid[0].size();
  const auto ny = grid[1].size();
  const auto plane_size = nx * ny;
  std::vector<SignedT> images(ks.size() * plane_size);
  if (ks.size() == 0 || plane_size == 0) return images;

  const auto summands = build_module_landscape_summand_cache(module, dimension);
  const auto top_size = std::min(module_landscape_top_size(ks), summands.size());
  if (top_size == 0) return images;

  auto set_pixel = [&](std::size_t pixel, std::vector<SignedT>& top) {
    const auto row = pixel / ny;
    const auto column = pixel - row * ny;
    const std::array<GridT, 2> x{grid[0][row], grid[1][column]};
    set_module_landscape_pixel(images, pixel, plane_size, summands, x, ks, top);
  };

#ifdef GUDHI_USE_TBB
  tbb::enumerable_thread_specific<std::vector<SignedT>> top([&] { return std::vector<SignedT>(top_size); });
  oneapi::tbb::task_arena arena(n_jobs);
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0), plane_size, [&](std::size_t pixel) { set_pixel(pixel, top.local()); });
  });
#else
  std::vector<SignedT> top(top_size);
  for (std::size_t pixel = 0; pixel < plane_size; ++pixel) set_pixel(pixel, top);
#endif

  return images;
}

}  // namespace multipers::detail

#endif
