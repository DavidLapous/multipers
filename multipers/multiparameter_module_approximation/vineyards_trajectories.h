/*    This file is part of the MMA Library -
 * https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT. See
 * file LICENSE for full license details. Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence
 * class, renaming and cleanup.
 *      - 2022/05 Hannah Schreiber: Addition of Box class.
 */
/**
 * @file vineyards_trajectories.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief This file contains the functions related to trajectories of barcodes
 * via vineyards.
 */

#ifndef VINEYARDS_TRAJECTORIES_H_INCLUDED
#define VINEYARDS_TRAJECTORIES_H_INCLUDED

#include <iostream>
#include <vector>
// #include <iomanip>
#include <cassert>
#include <cmath>
#include <limits>
// #include "gudhi/Simplex_tree.h"

#include "format_python-cpp.h"
#include "structure_higher_dim_barcode.h"
#include "vineyards.h"
// #include "combinatory.h"
#include "debug.h"

#include "ru_matrix.h"
#include "utilities.h"
/*#include "heap_column.h"*/
/*#include "list_column.h"*/
// #include "list_column_2.h"
/*#include "vector_column.h"*/
/*#include "set_column.h"*/
/*#include "unordered_set_column.h"*/
#include <gudhi/Multi_persistence/Box.h>

namespace Gudhi::multiparameter::mma {

using Gudhi::multi_persistence::Box;
std::vector<std::vector<std::vector<interval_type>>>
compute_vineyard_barcode(boundary_matrix &boundaryMatrix,
                         const std::vector<filtration_type> &filtersList,
                         value_type precision, Box<value_type> &box,
                         bool threshold = false, bool multithread = false,
                         const bool verbose = false);

std::vector<std::vector<interval_type>> compute_vineyard_barcode_in_dimension(
    boundary_matrix &boundaryMatrix,
    const std::vector<filtration_type> &filtersList, value_type precision,
    Box<value_type> &box, dimension_type dimension, bool threshold = false,
    const bool verbose = false);

void compute_vineyard_barcode_recursively(
    std::vector<std::vector<std::vector<interval_type>>> &output,
    Vineyard_persistence<Vineyard_matrix_type> &persistence,
    const boundary_matrix &boundaryMatrix, point_type &basepoint,
    std::vector<unsigned int> &position, unsigned int last,
    filtration_type &filter, const std::vector<filtration_type> &filtersList,
    const value_type precision, const Box<value_type> &box,
    const std::vector<unsigned int> &size, bool first = false,
    bool threshold = false, bool multithread = false);

void compute_vineyard_barcode_recursively_in_higher_dimension(
    std::vector<std::vector<std::vector<interval_type>>> &output,
    Vineyard_persistence<Vineyard_matrix_type> &persistence,
    const boundary_matrix &boundaryMatrix, const point_type &basepoint,
    const std::vector<unsigned int> &position, unsigned int last,
    filtration_type &filter, const std::vector<filtration_type> &filtersList,
    const value_type precision, const Box<value_type> &box,
    const std::vector<unsigned int> &size, bool threshold = false,
    bool multithread = false);

void get_filter_from_line(const point_type &lineBasepoint,
                          const std::vector<filtration_type> &filterList,
                          filtration_type &newFilter,
                          const Box<value_type> &box, bool ignoreLast = false);

void threshold_up(point_type &point, const Box<value_type> &box,
                  const point_type &basepoint = point_type(1, negInf));

void threshold_down(point_type &point, const Box<value_type> &box,
                    const point_type &basepoint = point_type(1, negInf));
// template<typename T>
// bool is_smaller(const std::vector<T>& x, const std::vector<T>& y);
// template<typename T>
// bool is_greater(const std::vector<T>& x, const std::vector<T>& y);
// boundary_matrix simplex_tree_to_boundary_matrix(Gudhi::Simplex_tree<>
// &simplexfiltration_value_typeree);

// filtration_value_typeODO improve multithread
// Main function of vineyard computation. It computes the fibered barcodes of
// any multipersistence module, with exact matching.
// Input :
//			B : sparse boundary matrix which is the converted
//simplextree by
//              functions of format_python_cpp
// 			filters_list : [[filtration of dimension i for simplex s
// for s] for i]
//              is the list of filters of each simplex of each filtration
//              dimension
// 			precision : size of the line grid (ie. distance between
// 2 lines) 			box : [min, max] where min and max are points of R^n, and n is the
//              dimension of the filter list.
// 				All of the bars along a line crossing this box
// will be computed 			threshold : If set to true, will intersect the bars with the
// box.
//              Useful for plots / images
// 			multithread : if set to true, will compute the
// trajectories in parallel. 				filtration_value_typehis is a WIP; as this imply
// more memory operations, this is
//              rarely significantly faster than the other implementation.
// OUTPUT :
// 			[[[(birth,death) for line] for summand] for dimension]
/**
 * @brief Main function of vineyard computation. It computes the fibered
 * barcodes of any multipersistence module, with exact matching.
 *
 * @param B Sparse boundary matrix of a chain complex
 * @param filters_list associated filtration of @p B Format :
 * [[filtration of dimension i for simplex s for s] for i]
 * @param precision  precision of the line grid ie. distance between two lines
 * @param box [min, max] where min and max are points of \f$ \mathbb R^n \f$,
 * and n is the dimension of the filter list.
 * All of the bars along a line crossing this box will be computed
 * @param threshold if set to true, will threshold the barcodes with the box
 * @param multithread if set to true, will turn on the multithread flags of the
 * code (WIP)
 * @return vector<vector<vector<interval>>> List of barcodes along the lines
 * intersecting the box. Format : [[[(birth,death) for line] for summand] for
 * dimension]
 */
// Assumes matrix ordered by dimensions

std::vector<std::vector<std::vector<interval_type>>>
compute_vineyard_barcode(boundary_matrix &boundaryMatrix,
                         const std::vector<filtration_type> &filtersList,
                         value_type precision, Box<value_type> &box,
                         bool threshold, bool multithread, bool verbose) {
  Gudhi::multiparameter::mma::verbose = verbose;
  // Checks if dimensions are compatibles
  assert(!filtersList.empty() && "A non trivial filters list is needed !");
  assert(filtersList.size() == box.get_lower_corner().size() &&
         filtersList.size() == box.get_upper_corner().size() &&
         "Filtration and box must be of the same dimension");
  if constexpr (Debug::debug) {
    for (unsigned int i = 1; i < boundaryMatrix.size(); i++)
      assert(boundaryMatrix.at(i - 1).size() <= boundaryMatrix.at(i).size() &&
             "Boundary matrix has to be sorted by dimension!");
  }

  const unsigned int filtrationDimension = filtersList.size();
  if (verbose)
    std::cout << "Filtration dimension : " << filtrationDimension << std::flush
              << std::endl;

  unsigned int numberSimplices = boundaryMatrix.size();
  if (verbose)
    std::cout << "Number of simplices : " << numberSimplices << std::endl;

  filtration_type filter(numberSimplices); // container of filters

  std::vector<unsigned int> sizeLine(filtrationDimension - 1);
  for (unsigned int i = 0; i < filtrationDimension - 1; i++)
    sizeLine[i] = static_cast<unsigned int>(std::ceil(
        std::abs(box.get_upper_corner()[i] - box.get_lower_corner().back() -
                 box.get_lower_corner()[i] + box.get_upper_corner().back()) /
        precision));

  unsigned int numberOfLines = Combinatorics::prod(sizeLine);
  if (verbose)
    std::cout << "Precision : " << precision << std::endl;
  if (verbose)
    std::cout << "Number of lines : " << numberOfLines << std::endl;

  auto basePoint = box.get_lower_corner();
  for (unsigned int i = 0; i < basePoint.size() - 1; i++)
    basePoint[i] -= box.get_upper_corner().back();
  basePoint.back() = 0;

  get_filter_from_line(basePoint, filtersList, filter, box, true);
  // where is the cursor in the output matrix
  std::vector<unsigned int> position(filtrationDimension - 1, 0);

  if (filtersList[0].size() < numberSimplices) {
    filtration_type tmp = filter;
    Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
  }

  Vineyard_persistence<Vineyard_matrix_type> persistence(boundaryMatrix, filter,
                                                         verbose);
  persistence.initialize_barcode();
  auto &firstBarcode = persistence.get_diagram();

  // filtered by dimension so last one is of maximal dimension
  unsigned int maxDimension = firstBarcode.back().dim;
  std::vector<std::vector<std::vector<interval_type>>> output(maxDimension + 1);

  std::vector<unsigned int> numberOfFeaturesByDimension(maxDimension + 1);
  for (unsigned int i = 0; i < firstBarcode.size(); i++) {
    numberOfFeaturesByDimension[firstBarcode[i].dim]++;
  }

  for (unsigned int i = 0; i < maxDimension + 1; i++) {
    output[i] = std::vector<std::vector<interval_type>>(
        numberOfFeaturesByDimension[i],
        std::vector<interval_type>(numberOfLines));
  }

  auto elapsed = clock();
  if (verbose)
    std::cout << "Multithreading status : " << multithread << std::endl;
  if (verbose)
    std::cout << "Starting recursive vineyard loop..." << std::flush;

  compute_vineyard_barcode_recursively(
      output, persistence, boundaryMatrix, basePoint, position, 0, filter,
      filtersList, precision, box, sizeLine, true, threshold, multithread);

  elapsed = clock() - elapsed;
  if (verbose)
    std::cout << " Done ! It took "
              << (static_cast<float>(elapsed) / CLOCKS_PER_SEC) << " seconds."
              << std::endl;
  return output;
}

// Same as vineyard_alt but only returns one dimension
// TODO : reduce computation by only computing this dimension instead of all of
// them
/**
 * @brief Returns only one dimension of the \ref vineyard_alt code.
 *
 * @param B
 * @param filters_list
 * @param precision
 * @param box
 * @param dimension
 * @param threshold
 * @param verbose
 * @param debug
 * @return vector<vector<interval>>
 */

std::vector<std::vector<interval_type>> compute_vineyard_barcode_in_dimension(
    boundary_matrix &boundaryMatrix,
    const std::vector<filtration_type> &filtersList, value_type precision,
    Box<value_type> &box, dimension_type dimension, bool threshold,
    const bool verbose) {
  return compute_vineyard_barcode(boundaryMatrix, filtersList, precision, box,
                                  threshold, false, verbose)[dimension];
}

// This is the core compute function of vineyard_alt.
// It updates and store in `output` the barcodes of a line, and calls itself
// on the next line until reaching the borders of the box
// INPUT :
// 			output : Where to store the barcodes.
// 			persistence : holding previous barcode information.
// 			basepoint : basepoint of the current line on the
// hyperplane {x_n=0}. 			position : index pointer of where to fill the output.
// 			last : which dimensions needs to be increased on this
// trajectory
//              (for recursive trajectories).
// 			filter : container for filer of simplices.
// 			filters_list : holding the filtration value of each
// simplices.
//              Format : [[filtration of simplex s in the kth filtration for s]
//              for k].
// 			precision : line grid scale (ie. distance between two
// consecutive lines). 			box : [min, max] where min and max are points of R^n, and
// n is the
//              dimension of the filter list.
// 				All of the bars along a line crossing this box
// will be computed. 			size : size of the output matrix. 			first : true if it is the
// first barcode. In that case we don't need
//              to call a vineyard update.
// 			threshold : if true, intersects bars with the box.
// 			multithread : if set to true, will compute the
// trajectories in parallel. 				This is a WIP; as this imply more memory
// operations, this is
//              rarely significantly faster than the other implementation.
/**
 * @brief Recursive version of \ref vineyard_alt.
 *
 * @param output
 * @param persistence
 * @param basepoint
 * @param position
 * @param last
 * @param filter
 * @param filters_list
 * @param precision
 * @param box
 * @param size
 * @param first
 * @param threshold
 * @param multithread
 */

void compute_vineyard_barcode_recursively(
    std::vector<std::vector<std::vector<interval_type>>> &output,
    Vineyard_persistence<Vineyard_matrix_type> &persistence,
    const boundary_matrix &boundaryMatrix, point_type &basepoint,
    std::vector<unsigned int> &position, unsigned int last,
    filtration_type &filter, const std::vector<filtration_type> &filtersList,
    const value_type precision, const Box<value_type> &box,
    const std::vector<unsigned int> &size, bool first, bool threshold,
    bool multithread) {
  if (!first) {
    get_filter_from_line(basepoint, filtersList, filter, box, true);
    if (filtersList[0].size() < boundaryMatrix.size()) {
      filtration_type tmp = filter;
      Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp,
                                                    filter);
    }
  }

  // if  (verbose && Debug::debug) Debug::disp_vect(basepoint);

  persistence.update(filter); // Updates the RU decomposition of persistence.
  // Computes the diagram from the RU decomposition
  const diagram_type &dgm = persistence.get_diagram();

  // Fills the barcode of the line having the basepoint basepoint
  unsigned int feature = 0;
  int oldDim = 0;

  // %TODO parallelize this loop, last part is not compatible yet
  for (unsigned int i = 0; i < dgm.size(); i++) {
    dimension_type dim = dgm[i].dim;
    value_type baseBirth = dgm[i].birth;
    value_type baseDeath = dgm[i].death;

    unsigned int index = get_index_from_position_and_size(position, size);
    point_type &birth = output[dim][feature][index].first;
    point_type &death = output[dim][feature][index].second;

    // If the bar is trivial, we skip it
    if (baseBirth != inf && baseBirth != baseDeath) {
      birth.resize(filtersList.size());
      death.resize(filtersList.size());

      // computes birth and death point from the bar and the basepoint of the
      // line
      for (unsigned int j = 0; j < filtersList.size() - 1; j++) {
        birth[j] = basepoint[j] + baseBirth;
        death[j] = basepoint[j] + baseDeath;
      }
      birth.back() = baseBirth;
      death.back() = baseDeath;

      // Threshold birth and death if threshold is set to true
      if (threshold && birth.back() != inf) {
        threshold_down(birth, box, basepoint);
        threshold_up(death, box, basepoint);
      }

      //            if  (verbose) {
      //                std::cout << birth.back() << " " << death.back();
      //                if (threshold) std::cout << ", threshold" << std::endl;
      //                else std::cout << ", no threshold" << std::endl;
      //            }

      // If this threshold has turned this bar to a trivial bar, we skip it
      if (birth.back() >= death.back()) {
        birth.clear();
        death.clear();
      }
    }

    // If next bar is of upper dimension, or we reached the end, then we
    // update the pointer index of where to fill the next bar in output.
    if (i + 1 < dgm.size() && oldDim < dgm[i + 1].dim) {
      oldDim = dgm[i + 1].dim;
      feature = 0;
    } else
      feature++;

    //        if  (verbose)
    //            std::cout <<"Feature : " << feature << " dim : " << oldDim <<
    //            std::endl;
  }

  // recursive calls of bigger dims, minus current dim (to make less copies)
  compute_vineyard_barcode_recursively_in_higher_dimension(
      output, persistence, boundaryMatrix, basepoint, position, last, filter,
      filtersList, precision, box, size, threshold, multithread);

  // We keep -last- on the same thread / memory as the previous call
  // we reached a border and finished this path
  if (size[last] - 1 == position[last])
    return;

  // If we didn't reached the end, go to the next line
  basepoint[last] += precision;
  position[last]++;
  compute_vineyard_barcode_recursively(
      output, persistence, boundaryMatrix, basepoint, position, last, filter,
      filtersList, precision, box, size, false, threshold, multithread);
}

// For persistence dimension higher than 3, this function will be called for
// Tree-like recursion of vineyard_alt.
/**
 * @brief Subfonction of \ref vinyard_alt_recursive to handle dimensions
 * greater than 3.
 *
 * @param output
 * @param persistence
 * @param basepoint
 * @param position
 * @param last
 * @param filter
 * @param filters_list
 * @param precision
 * @param box
 * @param size
 * @param threshold
 * @param multithread
 */

void compute_vineyard_barcode_recursively_in_higher_dimension(
    std::vector<std::vector<std::vector<interval_type>>> &output,
    Vineyard_persistence<Vineyard_matrix_type> &persistence,
    const boundary_matrix &boundaryMatrix, const point_type &basepoint,
    const std::vector<unsigned int> &position, unsigned int last,
    filtration_type &filter, const std::vector<filtration_type> &filtersList,
    const value_type precision, const Box<value_type> &box,
    const std::vector<unsigned int> &size, bool threshold, bool multithread) {
  if (filtersList.size() > 1 && last + 2 < filtersList.size()) {
    //        if  (verbose && Debug::debug) Debug::disp_vect(basepoint);
    //        if  (verbose) std::cout << multithread << std::endl;

    if (multithread) {
      /* #pragma omp parallel for */
      for (unsigned int i = last + 1; i < filtersList.size() - 1; i++) {
        if (size[i] - 1 == position[i])
          continue;
        // TODO check if it get deleted at each loop !! WARNING
        auto copyPersistence = persistence;
        auto copyBasepoint = basepoint;
        auto copyPosition = position;
        copyBasepoint[i] += precision;
        copyPosition[i]++;
        compute_vineyard_barcode_recursively(
            output, copyPersistence, boundaryMatrix, copyBasepoint,
            copyPosition, i, filter, filtersList, precision, box, size, false,
            threshold, multithread);
      }
    } else {
      // No need to copy when not multithreaded.
      // Memory operations are slower than vineyard.
      // %TODO improve trajectory of vineyard
      auto copyPersistence = persistence;
      auto copyBasepoint = basepoint;
      auto copyPosition = position;
      for (unsigned int i = last + 1; i < filtersList.size() - 1; i++) {
        if (size[i] - 1 == position[i])
          continue;
        copyPersistence = persistence;
        copyBasepoint = basepoint;
        copyPosition = position;
        copyBasepoint[i] += precision;
        copyPosition[i]++;
        compute_vineyard_barcode_recursively(
            output, copyPersistence, boundaryMatrix, copyBasepoint,
            copyPosition, i, filter, filtersList, precision, box, size, false,
            threshold, multithread);
      }
    }
  }
}

// INPUT :
//	a slope 1 line is characterized by its intersection with {x_n=0} named
//   line_basepoint.
//	filter_list is : for each coordinate i, and simplex j filter_list[i,j]
//is
//   the filtration value of simplex j on line induced by [0,e_i]
// OUTPUT:
//	filtration value of simplex j on the line.
/**
 * @brief Writes the filters of each simplex on new_filter along the a slope 1
 * line.
 *
 * @param line_basepoint Basepoint of a slope 1 line in \f$\mathbb R^n\f$
 * @param filter_list Multi-filtration of simplices. Format :
 * [[filtration_value for simplex] for dimension]
 * @param new_filter Container of the output.
 * @param ignore_last Ignore this parameter. It is meant for compatibility
 * with old functions.
 */

void get_filter_from_line(const point_type &lineBasepoint,
                          const std::vector<filtration_type> &filterList,
                          filtration_type &newFilter,
                          const Box<value_type> &box, bool ignoreLast) {
  //    if  (verbose && Debug::debug) {
  //        Debug::disp_vect(lineBasepoint);
  //    }

  unsigned int dimension = lineBasepoint.size() + 1 - ignoreLast;

  // 	value_type minLength = box.get_lower_corner().back();
  // 	value_type maxLength = box.get_upper_corner().back();
  // // #pragma omp parallel for reduction(max : minLength)
  // 	for (unsigned int i = 0;  i<dimension-1; i++){
  // 		minLength = std::max(minLength, box.get_lower_corner()[i] -
  // lineBasepoint[i]);
  // 	}
  // // #pragma omp parallel for reduction(min : maxLength)
  // 	for (unsigned int i = 0;  i<dimension-1; i++){
  // 		maxLength = std::min(maxLength, box.get_upper_corner()[i] -
  // lineBasepoint[i]);
  // 	}

  filtration_type relativeFiltrationValues(dimension);
  for (unsigned int i = 0; i < filterList[0].size(); i++) {
    for (unsigned int j = 0; j < dimension - 1; j++) {
      relativeFiltrationValues[j] = filterList[j][i] - lineBasepoint[j];
    }
    relativeFiltrationValues[dimension - 1] = filterList[dimension - 1][i];
    value_type length = *max_element(relativeFiltrationValues.begin(),
                                     relativeFiltrationValues.end());

    //         newFilter[i] = std::min(std::max(length,minLength), maxLength);
    newFilter[i] = length;
  }

  //    if  (verbose && Debug::debug) {
  //        Debug::disp_vect(newFilter);
  //    }
}

/**
 * @brief Threshold a point to the negative cone of d=box.second
 * (ie. the set \f$\{x \in \mathbb R^n \mid x \le d\} \f$)
 * along the slope 1 line crossing this point.
 *
 * @param point The point to threshold.
 * @param box box.second is the point defining where to threshold.
 * @param basepoint Basepoint of the slope 1 line crossing the point.
 * Meant to handle infinite cases (when the point have infinite coordinates,
 * we cannot infer the line).
 */

void threshold_up(point_type &point, const Box<value_type> &box,
                  const point_type &basepoint) {
  Gudhi::multi_filtration::One_critical_filtration
      point_(point);
  // if (is_smaller(point, box.get_upper_corner())) return;
  if (point_ <= box.get_upper_corner())
    return;

  //    if  (verbose && Debug::debug) Debug::disp_vect(point);

  if (basepoint[0] == negInf)
    return;

  // ie. point at infinity, assumes [basepoint,0] is smaller than box.second
  if (point.back() == inf) {
    //        if (verbose) std::cout << " Infinite point" << std::endl;

    value_type threshold = box.get_upper_corner().back();
    for (unsigned int i = 0; i < point.size(); i++) {
      threshold = std::min(threshold, box.get_upper_corner()[i] - basepoint[i]);
    }
    for (unsigned int i = 0; i < point.size() - 1; i++)
      point[i] = basepoint[i] + threshold;
    point.back() = threshold;

    return;
  }

  // if (!is_greater(point, box.get_lower_corner())) {
  if (box.get_lower_corner() <= point_) {
    point[0] = inf; // puts point to infinity
    //        if  (verbose) std::cout << "buggy point" << std::endl;
    return;
  }
  // in this last case, at least 1 coord of point is is_greater than a coord of
  // box.second

  value_type threshold = point[0] - box.get_upper_corner()[0];
  for (std::size_t i = 1; i < point.size(); i++) {
    threshold = std::max(threshold, point[i] - box.get_upper_corner()[i]);
  }

  //    if  (verbose)
  //            std::cout << "Thresholding the point with "<< threshold << " at
  //            ";

  for (std::size_t i = 0; i < point.size(); i++)
    point[i] -= threshold;

  //    if  (verbose && Debug::debug) Debug::disp_vect(point);
}

/**
 * @brief Threshold a point to the positive cone of b=box.first
 * (ie. the set \f$\{x \in \mathbb R^n \mid x \ge b\})
 * along the slope 1 line crossing this point.
 *
 * @param point The point to threshold.
 * @param box box.fist is the point defining where to threshold.
 * @param basepoint Basepoint of the slope 1 line crossing the point.
 * Meant to handle infinite cases (when the point have infinite coordinates,
 * we cannot infer the line).
 */

void threshold_down(point_type &point, const Box<value_type> &box,
                    const point_type &basepoint) {
  if (basepoint[0] == negInf)
    return;

  if (point.back() == inf) { // ie. point at infinity -> feature never appearing
    return;
  }

  // if (is_greater(point, box.get_lower_corner())) return;
  if (point >= box.get_lower_corner())
    return;

  // if (!is_smaller(point, box.get_upper_corner())) {
  if (!(point <= box.get_upper_corner())) {
    point[0] = inf; // puts point to infinity
    return;
  }

  value_type threshold = box.get_lower_corner()[0] - point[0];
  for (unsigned int i = 1; i < point.size(); i++) {
    threshold = std::max(threshold, box.get_lower_corner()[i] - point[i]);
  }
  for (unsigned int i = 0; i < point.size(); i++)
    point[i] += threshold;
}

} // namespace Gudhi::multiparameter::mma

#endif // VINEYARDS_TRAJECTORIES_H_INCLUDED
