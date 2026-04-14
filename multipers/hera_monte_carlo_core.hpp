#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "ext_interface/contiguous_slicer_bridge.hpp"

namespace multipers::core {

std::vector<double> monte_carlo_bottleneck_distances_on_lines(
    const contiguous_f64_slicer& left,
    const contiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double delta,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_bottleneck_distances_on_lines(
    const contiguous_f64_slicer& left,
    const kcontiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double delta,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_bottleneck_distances_on_lines(
    const kcontiguous_f64_slicer& left,
    const contiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double delta,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_bottleneck_distances_on_lines(
    const kcontiguous_f64_slicer& left,
    const kcontiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double delta,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_wasserstein_distances_on_lines(
    const contiguous_f64_slicer& left,
    const contiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double order,
    double internal_p = std::numeric_limits<double>::infinity(),
    double delta = 0.01,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_wasserstein_distances_on_lines(
    const contiguous_f64_slicer& left,
    const kcontiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double order,
    double internal_p = std::numeric_limits<double>::infinity(),
    double delta = 0.01,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_wasserstein_distances_on_lines(
    const kcontiguous_f64_slicer& left,
    const contiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double order,
    double internal_p = std::numeric_limits<double>::infinity(),
    double delta = 0.01,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

std::vector<double> monte_carlo_wasserstein_distances_on_lines(
    const kcontiguous_f64_slicer& left,
    const kcontiguous_f64_slicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double order,
    double internal_p = std::numeric_limits<double>::infinity(),
    double delta = 0.01,
    bool ignore_infinite_filtration_values = true,
    int n_jobs = 0);

}  // namespace multipers::core
