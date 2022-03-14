/**
 * @file approximation.h
 * @author David Loiseaux
 * @brief Contains the functions related to the approximation of n-modules.
 * 
 * @copyright Copyright (c) 2022 Inria
 *
 * Modifications: Hannah Schreiber
 * 
 */

#ifndef APPROXIMATION_H_INCLUDED
#define APPROXIMATION_H_INCLUDED

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "vineyards.h"
#include "vineyards_trajectories.h"
#include "combinatory.h"
#include "debug.h"
#include "utilities.h"

namespace Vineyard {

using Debug::Timer;

using corner_type = std::vector<double>;
using corner_list = std::pair<std::vector<corner_type>,
                              std::vector<corner_type> >;///< pair of birth corner list, and death corner list of a summand
using summand_list_type = std::vector<corner_list>;

std::vector<summand_list_type> compute_vineyard_barcode_approximation(
        boundary_matrix& boundaryMatrix,
        const std::vector<filtration_type>& filtersList,
        const double precision,
        const std::pair<corner_type, corner_type>& box,
        const bool threshold = false,
        const bool keepOrder = false,
        const bool complete = true,
        const bool multithread = false,
        const bool verbose = false);
void compute_vineyard_barcode_approximation_recursively(
        std::vector<summand_list_type>& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        corner_type& basepoint,
        std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        double precision,
        const interval_type& box,
        const std::vector<unsigned int>& sizeLine,
        bool first = false,
        const bool threshold = false,
        const bool multithread = false);
void compute_vineyard_barcode_approximation_recursively_for_higher_dimension(
        std::vector<summand_list_type>& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        const corner_type& basepoint,
        const std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        const double precision,
        const std::pair<corner_type, corner_type>& box,
        const std::vector<unsigned int>& size,
        const bool threshold,
        const bool multithread);
void clean(std::vector<summand_list_type>& output,
           const bool keepOrder = false);
void clean(std::vector<corner_type>& list, const bool keepOrder = false);
void fill(std::vector<summand_list_type>& output,
          const double precision);
void add_bar_to_summand(
        double baseBirth,
        double baseDeath,
        corner_list& summand,
        const corner_type& basepoint,
        corner_type& birth,
        corner_type& death,
        const bool threshold,
        const interval_type& box);
bool is_summand_empty(const corner_list &summand);
void complete_birth(std::vector<corner_type>& birthList,
                    const double precision);
void complete_death(std::vector<corner_type>& deathList,
                    const double precision);
void add_birth_to_summand(std::vector<corner_type>& birthList,
                          corner_type& birth);
void add_death_to_summand(std::vector<corner_type>& deathList,
                          corner_type& death);
double get_max_diagonal(const corner_type& a,
                        const corner_type& b);
double get_min_diagonal(const corner_type& a,
                        const corner_type& b);
void factorize_min(corner_type& a, const corner_type& b);
void factorize_max(corner_type& a, const corner_type& b);

/**
 * @brief Appproximate any multipersistence module with an interval
 * decomposable module. If this module is interval decomposable,
 * then the matching is controlled by the precision, and exact under
 * specific circumstances (see TODO: cite paper).
 *
 * @param B p_B: Boundary matrix of the initial simplices.
 * @param filters_list p_filters_list: Filtration of the simplices
 * @param precision p_precision: wanted precision.
 * @param box p_box: Box on which to make the approximation
 * @param threshold p_threshold:... Defaults to false. If set to true, will
 * intersect the computed summands with the box
 * @param kee_order : keeps a natural order of summands at a small
 * computational overhead. See \ref clean .
 * @param complete : gives a more natural output, at a small computational
 * overhead.
 * @param multithread ${p_multithread:...} Defaults to false.
 * WIP, not useful yet.
 * @return std::vector< std::vector< corner_list > >
 */
//Assumes matrix ordered by dimensions
std::vector<summand_list_type> compute_vineyard_barcode_approximation(
        boundary_matrix& boundaryMatrix,
        const std::vector<filtration_type>& filtersList,
        const double precision,
        const std::pair<corner_type, corner_type>& box,
        const bool threshold,
        const bool keepOrder,
        const bool complete,
        const bool multithread,
        const bool verbose)
{
    Vineyard::verbose = verbose;
    // Checks if dimensions are compatibles
    assert(!filtersList.empty() && "A non trivial filters list is needed!");
    assert(filtersList.size() == box.first.size()
           && filtersList.size() == box.second.size()
           && "Filters and box must be of the same dimension!");
    if (Debug::debug){
        for (unsigned int i = 1; i < boundaryMatrix.size(); i++)
            assert(boundaryMatrix.at(i - 1).size() <= boundaryMatrix.at(i).size()
                   && "Boundary matrix has to be sorted by dimension!");
    }

    const unsigned int filtrationDimension = filtersList.size();
    if  (verbose)
            std::cout << "Filtration dimension : " << filtrationDimension
                      << std::flush << std::endl;

    unsigned int numberOfSimplices = boundaryMatrix.size();
    if  (verbose)
            std::cout << "Number of simplices : " << numberOfSimplices
                      << std::flush << std::endl;

    filtration_type filter(numberOfSimplices); // container of filters

    std::vector<unsigned int> size_line(filtrationDimension - 1);
    for (unsigned int i = 0; i < filtrationDimension - 1; i++)
        size_line[i] = static_cast<unsigned int>(
                    std::ceil(
                        std::abs(box.second[i] - box.first.back()
                                 - box.first[i] + box.second.back()) / precision
                        )
                    );

    if  (verbose)
            std::cout << "Precision : " << precision << std::endl;
    if  (verbose)
            std::cout << "Number of lines : "
                      << Combinatorics::prod(size_line) << std::endl;

    auto basepoint = box.first;
    for (unsigned int i = 0; i < basepoint.size() - 1; i++)
        basepoint[i] -= box.second.back();
    basepoint.back() = 0;

    get_filter_from_line(basepoint, filtersList, filter, true);
    // where is the cursor in the output matrix
    std::vector<unsigned int> position(filtrationDimension - 1, 0);

    if (filtersList[0].size() < numberOfSimplices) {
        filtration_type tmp = filter;
        Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
    }

    Vineyard_persistence<Vineyard_matrix_type> persistence(boundaryMatrix, filter, verbose);

    persistence.initialize_barcode();
    const diagram_type& firstBarcode = persistence.get_diagram();

    // filtered by dimension so last one is of maximal dimension
    unsigned int maxDimension = firstBarcode.back().dim;

    // Initialise size of the output.
    std::vector<std::vector<corner_list> > output(maxDimension + 1);
    std::vector<unsigned int> numberOfFeaturesByDimension(maxDimension + 1);
    for (unsigned int i = 0; i < firstBarcode.size(); i++){
        numberOfFeaturesByDimension[firstBarcode[i].dim]++;
    }
    for (unsigned int i = 0; i< maxDimension + 1; i++){
        output[i].resize(numberOfFeaturesByDimension[i]);
    }

    auto elapsed = clock();
    if  (verbose)
            std::cout << "Multithreading status : " <<  multithread << std::endl;
    if  (verbose)
            std::cout << "Starting recursive vineyard loop..." << std::flush;

    // Call the compute recursive function
    compute_vineyard_barcode_approximation_recursively(
                output,
                persistence,
                boundaryMatrix,
                basepoint,
                position,
                0,
                filter,
                filtersList,
                precision,
                box,
                size_line,
                true,
                threshold,
                multithread);

    elapsed = clock() - elapsed;
    if  (verbose)
            std::cout << " Done ! It took "
                      << static_cast<float>(elapsed)/CLOCKS_PER_SEC
                      << " seconds."<< std::endl;

    {//for Timer
        Timer timer("Cleaning output ... ", verbose);
        clean(output, keepOrder);
        if (complete){
            if  (verbose) std::cout << "Completing output ...";
            fill(output, precision);
        }
    }//Timer death
    return output;
}

/**
 * @brief Recursive function of \ref approximation_vineyards.
 * Computes what's on a line, adds the barcode to the module,
 * and goes to the next line.
 *
 * @param output p_output:...
 * @param persistence p_persistence:...
 * @param basepoint p_basepoint:...
 * @param position p_position:...
 * @param last p_last:...
 * @param filter p_filter:...
 * @param filters_list p_filters_list:...
 * @param precision p_precision:...
 * @param box p_box:...
 * @param size_line p_size_line:...
 * @param first p_first:... Defaults to false.
 * @param threshold p_threshold:... Defaults to false.
 * @param multithread p_multithread:... Defaults to false.
 */
void compute_vineyard_barcode_approximation_recursively(
        std::vector<summand_list_type>& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        corner_type& basepoint,
        std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        double precision,
        const interval_type& box,
        const std::vector<unsigned int>& sizeLine,
        bool first,
        const bool threshold,
        const bool multithread)
{
    if (!first) {
        get_filter_from_line(basepoint, filtersList, filter, true);
        if (filtersList[0].size() < boundaryMatrix.size()) {
            filtration_type tmp = filter;
            Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
        }
    }

    if  (verbose && Debug::debug) Debug::disp_vect(basepoint);

    // Updates the RU decomposition of persistence.
    persistence.update(filter);
    // Computes the diagram from the RU decomposition
    const diagram_type& dgm = persistence.get_diagram();

    // Fills the barcode of the line having the basepoint basepoint
    unsigned int feature = 0;
    int oldDim = 0;

    corner_type birthContainer(filtersList.size());
    corner_type deathContainer(filtersList.size());

    for(unsigned int i = 0; i < dgm.size(); i++){
        corner_list &summand = output[dgm[i].dim][feature];
        add_bar_to_summand(
                    dgm[i].birth,
                    dgm[i].death,
                    summand,
                    basepoint,
                    birthContainer,
                    deathContainer,
                    threshold,
                    box);

        // If next bar is of upper dimension, or we reached the end,
        // then we update the pointer index of where to fill the next
        // bar in output.
        if (i + 1 < dgm.size() && oldDim < dgm[i+1].dim) {
            oldDim = dgm[i+1].dim;
            feature = 0;
        }
        else feature++;
//        if  (verbose)
//            std::cout <<"Feature : " << feature << " dim : " << oldDim << std::endl;
    }

    compute_vineyard_barcode_approximation_recursively_for_higher_dimension(
                output,
                persistence,
                boundaryMatrix,
                basepoint,
                position,
                last,
                filter,
                filtersList,
                precision,
                box,
                sizeLine,
                threshold,
                multithread);

    //recursive calls of bigger dims, minus current dim (to make less copies)
    // We keep -last- on the same thread / memory as the previous call
    // we reached a border and finished this path
    if (sizeLine[last] - 1 == position[last]) return;
    // If we didn't reached the end, go to the next line
    basepoint[last] += precision;
    position[last]++;
    compute_vineyard_barcode_approximation_recursively(
                output,
                persistence,
                boundaryMatrix,
                basepoint,
                position,
                last,
                filter,
                filtersList,
                precision,
                box,
                sizeLine,
                false,
                threshold,
                multithread);
}

void compute_vineyard_barcode_approximation_recursively_for_higher_dimension(
        std::vector<summand_list_type>& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        const corner_type& basepoint,
        const std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        const double precision,
        const std::pair<corner_type, corner_type>& box,
        const std::vector<unsigned int>& size,
        const bool threshold,
        const bool multithread)
{
    if (filtersList.size() > 1 && last + 2 < filtersList.size()){
//        if  (verbose && Debug::debug) Debug::disp_vect(basepoint);
//        if  (verbose) std::cout << multithread<< std::endl;

        if (multithread){
//            if  (verbose) std::cout << "Multithreading dimension..." << std::endl;

#pragma omp parallel for
            for (unsigned int i = last + 1; i < filtersList.size() - 1; i++){
                if (size[i] - 1 == position[i]) continue;
                //TODO check if it get deleted at each loop !! WARNING
                auto copyPersistence = persistence;
                auto copyBasepoint = basepoint;
                auto copyPosition = position;
                copyBasepoint[i] += precision;
                copyPosition[i]++;
                compute_vineyard_barcode_approximation_recursively(
                            output, copyPersistence, boundaryMatrix, copyBasepoint, copyPosition,
                            i, filter, filtersList, precision, box, size,false,
                            threshold, multithread
                            );
            }
        } else {
            // No need to copy when not multithreaded.
            // Memory operations are slower than vineyard.
            // %TODO improve trajectory of vineyard
            auto copyPersistence = persistence;
            auto copyBasepoint = basepoint;
            auto copyPosition = position;
            for (unsigned int i = last + 1; i < filtersList.size() - 1; i++){
                if (size[i] - 1 == position[i]) continue;
                copyPersistence = persistence;
                copyBasepoint = basepoint;
                copyPosition = position;
                copyBasepoint[i] += precision;
                copyPosition[i] ++;
                compute_vineyard_barcode_approximation_recursively(
                            output, copyPersistence, boundaryMatrix, copyBasepoint, copyPosition,
                            i, filter, filtersList, precision, box, size, false,
                            threshold, multithread
                            );
            }
        }
    }
}

/**
 * @brief Remove the empty summands of the output
 *
 * @param output p_output:...
 * @param keep_order p_keep_order:... Defaults to false.
 */
void clean(std::vector<summand_list_type>& output, const bool keepOrder)
{
    if (!keepOrder)
        for (unsigned int dim = 0; dim < output.size(); dim++){
            unsigned int i = 0;
            while (i < output[dim].size()){
                while (!output[dim].empty() &&
                       is_summand_empty(*output[dim].rbegin()))
                    output[dim].pop_back();

                if (i >= output[dim].size())
                    break;

                auto &summand = output[dim][i];
                if (is_summand_empty(summand)){
                    summand.swap(*output[dim].rbegin());
                    output[dim].pop_back();
                }
                i++;
            }
        }
    else
        for (unsigned int dim = 0; dim < output.size(); dim++){
            unsigned int i = 0;
            while (i < output[dim].size()){
                if (output[dim][i].first.empty() &&
                        output[dim][i].second.empty())
                    output[dim].erase(output[dim].begin() + i);
                else
                    i++;
            }
        }
}

/**
 * @brief Cleans empty entries of a corner list
 *
 * @param list corner list to clean
 * @param keep_sort If true, will keep the order of the corners,
 * with a computational overhead. Defaults to false.
 */
// WARNING Does permute the output.
void clean(std::vector<corner_type>& list, const bool keepOrder)
{
    unsigned int i = 0;
    if (!keepOrder){
        while (i < list.size()){
            while (!list.empty() && (*(list.rbegin())).empty())
                list.pop_back();
            if (i < list.size() && list[i].empty()){
                list[i].swap(*(list.rbegin()));
                list.pop_back();
            }
            i++;
        }
    } else {
        while (i < list.size()){
            if (list[i].empty())
                list.erase(list.begin() + i);
            else
                i++;
        }
    }
}

void fill(std::vector<summand_list_type>& output, const double precision)
{
    if (output.empty()) return;

    for (unsigned int dim = 0; dim < output.size(); dim++){
        for (corner_list& summand : output[dim]){
            if (is_summand_empty(summand))
                continue;
            complete_birth(summand.first, precision);
            complete_death(summand.second, precision);
        }
    }
}

/**
 * @brief Adds the bar @p bar to the indicator module @p summand if @p bar
 * is non-trivial (ie. not reduced to a point or, if @p threshold is true,
 * its thresholded version should not be reduced to a point) .
 *
 * @param bar p_bar: to add to the support of the summand
 * @param summand p_summand: indicator module which is being completed
 * @param basepoint p_basepoint: basepoint of the line of the bar
 * @param birth p_birth: birth container (for memory optimization purposes).
 * Has to be of the size @p basepoint.size()+1.
 * @param death p_death: death container. Same purpose as @p birth but for
 * deathpoint.
 * @param threshold p_threshold: If true, will threshold the bar with @p box.
 * @param box p_box: Only useful if @p threshold is set to true.
 */
void add_bar_to_summand(
        double baseBirth,
        double baseDeath,
        corner_list& summand,
        const corner_type& basepoint,
        corner_type& birth,
        corner_type& death,
        const bool threshold,
        const interval_type& box)
{
    // bar is trivial in that case
    if (baseBirth >= baseDeath) return;

    for (unsigned int j = 0; j < birth.size() - 1; j++){
        birth[j] = basepoint[j] + baseBirth;
        death[j] = basepoint[j] + baseDeath;
    }
    birth.back() = baseBirth;
    death.back() = baseDeath;

    if (threshold){
        threshold_down(birth, box, basepoint);
        threshold_up(death, box, basepoint);
    }
    add_birth_to_summand(summand.first, birth);
    add_death_to_summand(summand.second, death);
}

/**
 * @brief Returns true if a summand is empty
 *
 * @param summand summand to check.
 * @return bool
 */
bool is_summand_empty(const corner_list &summand)
{
    return summand.first.empty() || summand.second.empty();
}

void complete_birth(std::vector<corner_type>& birthList, const double precision)
{
    if (birthList.empty()) return;

    for (unsigned int i = 0; i < birthList.size(); i++){
        for (unsigned int j = i + 1; j < birthList.size(); j++){
            double dinf = get_max_diagonal(birthList[i], birthList[j]);
            if (dinf < 1.1 * precision){
                factorize_min(birthList[i], birthList[j]);
                birthList[j].clear();
            }
        }
    }
    clean(birthList);
}

void complete_death(std::vector<corner_type>& deathList, const double precision)
{
    if (deathList.empty()) return;

    for (unsigned int i = 0; i < deathList.size(); i++){
        for (unsigned int j = i + 1; j < deathList.size(); j++){
            double d = get_max_diagonal(deathList[i], deathList[j]);
            if (d < 1.1 * precision){
                factorize_max(deathList[i], deathList[j]);
                deathList[j].clear();
            }
        }
    }
    clean(deathList);
}

/**
 * @brief Adds @p birth to the summand's @p birth_list if it is not induced
 * from the @p birth_list (ie. not comparable or smaller than another birth),
 * and removes unnecessary birthpoints (ie. birthpoints that are induced
 * by @p birth).
 *
 * @param birth_list p_birth_list: birthpoint list of a summand
 * @param birth p_birth: birth to add to the summand
 */
void add_birth_to_summand(std::vector<corner_type>& birthList,
                          corner_type &birth)
{
    if (birthList.empty()){
        birthList.push_back(birth);
        return;
    }

    if (birthList.front().front() == negInf) return;

    // when a birth is infinite, we store the summand like this
    if (birth.front() == negInf){
        birthList = {{negInf}};
        return;
    }

    bool isUseful = true;
    for (unsigned int i = 0; i < birthList.size(); i++){
        if (!isUseful)
            continue;
        if (is_greater(birth, birthList[i])) {
            isUseful = false;
            break;
        }
        if (!birthList[i].empty() && is_less(birth, birthList[i])){
            birthList[i].clear();
        }
    }

    clean(birthList);
    if (isUseful)
        birthList.push_back(birth);

    return;
}

/**
 * @brief Adds @p death to the summand's @p death_list if it is not induced
 * from the @p death_list (ie. not comparable or greater than another death),
 * and removes unnecessary deathpoints (ie. deathpoints that are induced
 * by @p death)
 *
 * @param death_list p_death_list: List of deathpoints of a summand
 * @param death p_death: deathpoint to add to this list
 */
void add_death_to_summand(std::vector<corner_type>& deathList,
                          corner_type& death)
{
    if (deathList.empty()){
        deathList.push_back(death);
        return;
    }

    // as drawn in a slope 1 line being equal to -\infty is the same as the
    // first coordinate being equal to -\infty
    if (deathList.front().front() == inf)
        return;

    // when a birth is infinite, we store the summand like this
    if (death.front() == inf){
        deathList = {{inf}};
        return;
    }

    bool isUseful = true;
    for (unsigned int i = 0; i < deathList.size(); i++){
        if (!isUseful)
            continue;
        if (is_less(death, deathList[i])) {
            isUseful = false;
            break;
        }
        if (!deathList[i].empty() && is_greater(death, deathList[i])){
            deathList[i].clear();
        }
    }

    clean(deathList);
    if (isUseful)
        deathList.push_back(death);
}

/**
 * @brief Returns the biggest diagonal length in the rectangle
 * {z : @p a ≤ z ≤ @p b}
 *
 * @param a smallest element of the box
 * @param b biggest element of the box.
 * @return double, length of the diagonal.
 */
double get_max_diagonal(const corner_type& a, const corner_type& b)
{
    if (a.empty() || b.empty() || a.size() != b.size())
        return inf;

    double d = std::abs(a[0] - b[0]);
    for (unsigned int i = 1; i < a.size(); i++)
        d = std::max(d, std::abs(a[i] - b[i]));

    return d;
}

double get_min_diagonal(const corner_type& a, const corner_type& b)
{
    assert(a.size() == b.size() && "Inputs must be of the same size !");
    double s = b[0] - a[0];

    for(unsigned int i = 1; i < a.size(); i++){
        s = std::min(s, b[i] - a[i]);
    }
    return s;
}

void factorize_min(corner_type& a, const corner_type& b)
{
    if (a.size() != b.size()) return;

    for (unsigned int i = 0; i < a.size(); i++)
        a[i] = std::min(a[i], b[i]);
}

void factorize_max(corner_type& a, const corner_type& b)
{
    if (a.size() != b.size()) return;

    for (unsigned int i = 0; i < a.size(); i++)
        a[i] = std::max(a[i], b[i]);
}

}   //namespace Vineyard

#endif // APPROXIMATION_H_INCLUDED
