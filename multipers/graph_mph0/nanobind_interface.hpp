#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

namespace mpnb {

nanobind::tuple graph_mph0_raw(
    nanobind::ndarray<nanobind::numpy, const std::uint64_t, nanobind::ndim<1>, nanobind::c_contig> boundary_indptr,
    nanobind::ndarray<nanobind::numpy, const std::uint32_t, nanobind::ndim<1>, nanobind::c_contig> boundary_indices,
    nanobind::ndarray<nanobind::numpy, const std::int32_t, nanobind::ndim<1>, nanobind::c_contig> dimensions,
    nanobind::ndarray<nanobind::numpy, const double, nanobind::ndim<2>, nanobind::c_contig> grades,
    std::int32_t degree);

nanobind::object graph_mph0_minimal_presentation(const nanobind::handle& slicer,
                                                 std::int32_t degree,
                                                 bool full_resolution);

}  // namespace mpnb
