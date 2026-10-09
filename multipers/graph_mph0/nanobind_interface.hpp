#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>

namespace mpnb {

nanobind::object graph_mph0_minimal_presentation(const nanobind::handle& slicer,
                                                 std::int32_t degree,
                                                 bool full_resolution,
                                                 const nanobind::handle& finite_grid_masks);

}  // namespace mpnb
