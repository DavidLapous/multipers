/**
 * @file structure_higher_dim_barcode.h
 * @author David Loiseaux
 * @brief Structures to handle higher dimensional persistence=
 * 
 * @copyright Copyright (c) 2021 Inria
 *
 * Modifications: Hannah Schreiber
 * 
 */

#ifndef STRUCTURE_HIGHER_DIM_BARCODE_H_INCLUDED
#define STRUCTURE_HIGHER_DIM_BARCODE_H_INCLUDED

#include <vector>

unsigned int get_index_from_position_and_size(
        const std::vector<unsigned int> &position,
        const std::vector<unsigned int> &size)
{
    unsigned int indice = 0;
    assert(position.size() == size.size() &&
           "Position and Size vector must be of the same size !");
    unsigned int last_product = 1;
    for (unsigned int i = 0; i < position.size(); i++){
        indice += last_product * position[i];
        last_product *= size[i];
    }
    return indice;
}

#endif // STRUCTURE_HIGHER_DIM_BARCODE_H_INCLUDED
