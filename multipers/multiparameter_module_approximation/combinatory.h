/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence class, renaming and cleanup.
 */
/**
 * @file combinatory.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Combinatorial and sorting functions
 */

#ifndef COMBINATORY_H_INCLUDED
#define COMBINATORY_H_INCLUDED

#include <vector>
#include <iostream>
#include <functional>
#include <algorithm>
#include <climits>
#include <assert.h>
#include "utilities.h"
#include "debug.h"

namespace Gudhi::multiparameter::mma::Combinatorics {

using Gudhi::multiparameter::mma::permutation_type;

template<typename T> void compose(std::vector<T> &p, const permutation_type &q);
unsigned int prod(const std::vector<unsigned int>& toMultiply,
                  unsigned int until = UINT_MAX);
template<typename T>
permutation_type sort_and_return_permutation(
        std::vector<T>& toSort,
        std::function<bool(T&, T&)> lessOrEqualComparator);
template<typename T>
void quicksort_and_record_permutation(
        std::vector<T>& toSort,
        permutation_type& p,
        unsigned int low,
        unsigned int high,
        std::function<bool(T&, T&)> lessOrEqualComparator);

template<typename T>
void compose(std::vector<T> &p,const permutation_type &q){
    unsigned int n = p.size();
//     assert(q.size() == n);
    std::vector<T> r(n);
    for(unsigned int i = 0; i< n; i++){
        r[i] = p[q[i]];
    }
    p.swap(r);
}

template<typename T>
std::vector<T> inverse(const std::vector<T> &p){
    unsigned int n = p.size();
    std::vector<T> inv(n);
    for(unsigned int i = 0; i< n; i++)
        inv[p[i]] = i;
	
    return inv;
}

unsigned int prod(const std::vector<unsigned int>& toMultiply,
                  unsigned int until)
{
    unsigned int output = 1;
    for (unsigned int i = 0; i < toMultiply.size() && i <= until; i++){
        output *= toMultiply[i];
    }
    return output;
}

template<typename T>
permutation_type sort_and_return_permutation(
        std::vector<T>& toSort, std::function<bool(T&, T&)> lessOrEqualComparator)
{
    unsigned int n = toSort.size();

    // initialize p as the identity
    permutation_type p(n);
    for (unsigned int i = 0; i < n ; i++) p[i] = i;

    // call the recursive function doing the job
    quicksort_and_record_permutation<T>(toSort, p, 0, n - 1, lessOrEqualComparator);

    return p;
}

template<typename T>
void quicksort_and_record_permutation(
        std::vector<T>& toSort,
        permutation_type& p,
        unsigned int low,
        unsigned int high,
        std::function<bool(T&, T&)> lessOrEqualComparator)
{
    // compatibility check
    assert(toSort.size() == p.size());
    assert(high < toSort.size());

    if (high <= low) return;

    // take the last element as pivot.
    T pivot = toSort[high];

    int i = low - 1 ;

    for (unsigned int j = low; j < high; j++){
        if (lessOrEqualComparator(toSort[j], pivot)){
            i++;
            std::swap(toSort[i], toSort[j]);
            std::swap(p[i], p[j]);
        }
    }
    std::swap(toSort[i+1], toSort[high]);
    std::swap(p[i+1], p[high]);

    quicksort_and_record_permutation<T>(toSort, p, low, std::max(i, 0), lessOrEqualComparator);
    quicksort_and_record_permutation<T>(toSort, p, i + 2, high, lessOrEqualComparator);
}

} //namespace Combinatorics

#endif // COMBINATORY_H_INCLUDED
