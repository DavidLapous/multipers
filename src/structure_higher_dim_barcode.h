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
 * @file structure_higher_dim_barcode.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Structures to handle higher dimensional persistence.
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
