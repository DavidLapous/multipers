#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace multipers::core {

template <class Boundaries, class Dimensions, class Points>
class RepresentativeCycleIntersection {
 public:
  RepresentativeCycleIntersection(const Boundaries& boundaries, const Dimensions& dimensions, const Points& points)
      : boundaries_(boundaries), dimensions_(dimensions), points_(points), cache_(boundaries.size(), -1) {}

  template <class Cycle>
  bool intersects(const Cycle& cycle) {
    if (points_.empty()) {
      return false;
    }
    for (auto cell : cycle) {
      if (cell_intersects(cell)) {
        return true;
      }
    }
    return false;
  }

 private:
  template <class Index>
  bool cell_intersects(Index cell) {
    auto& cached = cache_[static_cast<std::size_t>(cell)];
    if (cached >= 0) {
      return cached != 0;
    }

    bool intersects = false;
    if (dimensions_[cell] == 0) {
      intersects = points_.find(cell) != points_.end();
    } else {
      for (auto face : boundaries_[cell]) {
        if (cell_intersects(face)) {
          intersects = true;
          break;
        }
      }
    }
    cached = static_cast<int8_t>(intersects);
    return intersects;
  }

  const Boundaries& boundaries_;
  const Dimensions& dimensions_;
  const Points& points_;
  std::vector<int8_t> cache_;
};

template <class VertexBoundaries, class Points>
bool vertex_boundaries_intersect_points(const VertexBoundaries& boundaries, const Points& points) {
  for (const auto& boundary : boundaries) {
    for (auto vertex : boundary) {
      if (points.find(vertex) != points.end()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace multipers::core
