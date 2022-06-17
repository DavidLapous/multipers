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
 * @file images.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Functions to generate multipersistence images
 */

#ifndef IMAGES_H_INCLUDED
#define IMAGES_H_INCLUDED

#include <vector>

#include "approximation.h"
#include "utilities.h"

using Vineyard::boundary_matrix;
using Vineyard::filtration_type;
using Vineyard::dimension_type;
using Vineyard::Module;
using Vineyard::Box;

std::vector<std::vector<std::vector<double> > > get_2D_image_from_boundary_matrix(
		boundary_matrix &boundaryMatrix,
		std::vector<filtration_type> &filtersList,
		const double precision,
		const Box &box,
		const double delta,
		const std::vector<unsigned int> &resolution,
		const dimension_type dimension,
		const bool complete = true,
		const bool verbose = false)
{
	Box bbox(box);
	bbox.inflate(delta);
	Module approximation =
			Vineyard::compute_vineyard_barcode_approximation(
				boundaryMatrix,
				filtersList,
				precision,
				bbox,
				true,
				complete,
				false,
				verbose);

	if (dimension < 0)
		return approximation.get_vectorization(delta, resolution[0], resolution[1]);

	return {approximation.get_vectorization_in_dimension(dimension, delta, resolution[0], resolution[1])};
}

#endif // IMAGES_H_INCLUDED
