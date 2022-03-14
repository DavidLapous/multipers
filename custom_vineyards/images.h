/**
 * @file images.h
 * @author David Loiseaux
 * @brief Functions to generate multipersistence images
 *
 * @copyright Copyright (c) 2022 Inria
 *
 * Modifications: Hannah Schreiber
 *
 */

#ifndef IMAGES_H_INCLUDED
#define IMAGES_H_INCLUDED

#include <vector>
#include <algorithm>
#include <cmath>

#include "approximation.h"
#include "debug.h"

#include "utilities.h"

using Vineyard::boundary_matrix;
using Vineyard::filtration_type;
using Vineyard::corner_type;
using Vineyard::corner_list;
using Vineyard::summand_list_type;
using Vineyard::dimension_type;
using Vineyard::inf;
using Vineyard::negInf;

template<typename T> class Box;
class Summand;

std::vector<std::vector<std::vector<double> > > get_2D_image_from_boundary_matrix(
		boundary_matrix &boundaryMatrix,
		const std::vector<filtration_type> &filtersList,
		const double precision,
		const std::pair<corner_type, corner_type> &box,
		const double delta,
		const std::vector<unsigned int> &resolution,
		const dimension_type dimension,
		const bool complete = true,
		const bool verbose = false);
std::vector<std::vector<double> > compute_2D_image(
		std::vector<Summand>& module,
		const double delta,
		const std::vector<unsigned int>& resolution,
		const Box<double>& box,
		bool verbose = true);
double get_pixel_value(
		std::vector<Summand>& module,
		const corner_type& x,
		const double delta,
		double moduleWeight = -1);

/**
 * @brief Holds the square box on which to compute.
 */
template<typename T>
class Box
{
public:
	Box(const std::vector<T>& bottomCorner, const std::vector<T>& upperCorner)
		: bottomCorner_(bottomCorner),
		  upperCorner_(upperCorner)
	{
		assert(bottomCorner.size() == upperCorner.size()
			   && Vineyard::is_less(bottomCorner, upperCorner)
			   && "This box is trivial !");
	}

	Box(const std::pair<std::vector<T>, std::vector<T> >& box)
		: bottomCorner_(box.first),
		  upperCorner_(box.second)
	{}

	void inflate(double delta){
#pragma omp simd
		for (unsigned int i = 0; i < bottomCorner_.size(); i++){
			bottomCorner_[i] -= delta;
			upperCorner_[i] += delta;
		}
	}

	const std::vector<T>& getBottomCorner() const{
		return bottomCorner_;
	};

	const std::vector<T>& getUpperCorner() const{
		return upperCorner_;
	};

private:
	std::vector<T> bottomCorner_;
	std::vector<T> upperCorner_;
};

class Summand
{
public:
	Summand() : distanceTo0_(-1), updateDistance_(true)
	{}

	Summand(corner_list &summand)
		: summand_(summand), distanceTo0_(-1), updateDistance_(true)
	{}

	double get_interleaving() {
		if (updateDistance_) _compute_interleaving();
		return distanceTo0_;
	}

	double get_local_weight(const corner_type& x, const double delta){
		if (delta <= 0) return 0;

		double maxDiag = 0;
		std::vector<double> mini(x.size());
		std::vector<double> maxi(x.size());

		// box on which to compute the local weight
#pragma omp simd
		for(unsigned int i = 0; i < x.size(); i++){
			mini[i] = x[i] - delta;
			maxi[i] = x[i] + delta;
		}

		// Pre-allocating
		std::vector<corner_type> birthList(summand_.first.size());
		std::vector<corner_type> deathList(summand_.second.size());
		unsigned int lastEntry = 0;
		for (const corner_type& birth : summand_.first){
			if (Vineyard::is_less(birth, maxi)){
				corner_type tmpBirth(birth.size());
				// WARNING should crash here if birth and x aren't of the same size.
#pragma omp simd
				for (unsigned int i = 0; i < birth.size(); i++)
					tmpBirth[i] = std::max(birth[i], mini[i]);
				birthList[lastEntry].swap(tmpBirth);
				lastEntry++;
			}
		}
		birthList.resize(lastEntry);

		lastEntry = 0;
		for (const corner_type& death : summand_.second){
			if (Vineyard::is_greater(death, mini)){
				corner_type tmpDeath(death.size());
				// WARNING should crash here if birth and x aren't of the same size.
#pragma omp simd
				for (unsigned int i = 0; i < death.size(); i++)
					tmpDeath[i] = std::min(death[i], maxi[i]);
				deathList[lastEntry].swap(tmpDeath);
				lastEntry++;
			}
		}
		deathList.resize(lastEntry);

		for (const corner_type& birth : birthList){
			if (birth.size() == 0 )
				continue;
			for (const corner_type& death : deathList){
				if (death.size() > 0)
					maxDiag = std::max(maxDiag,
									   Vineyard::get_min_diagonal(birth,death));
			}
		}

		return maxDiag; // should be less than delta
	}

	void swapSummand(corner_list& summand){
		summand_.swap(summand);
		updateDistance_ = true;
	}

private:
	corner_list summand_;
	double distanceTo0_;
	bool updateDistance_;

	void _compute_interleaving(){
		distanceTo0_ = 0;
		for (const std::vector<double> &birth : summand_.first){
			for(const std::vector<double> &death : summand_.second){
				distanceTo0_ = std::max(distanceTo0_,
										Vineyard::get_min_diagonal(birth, death));
			}
		}
		updateDistance_ = false;
	}
};

std::vector<std::vector<std::vector<double> > > get_2D_image_from_boundary_matrix(
		boundary_matrix &boundaryMatrix,
		const std::vector<filtration_type> &filtersList,
		const double precision,
		const std::pair<corner_type, corner_type> &box,
		const double delta,
		const std::vector<unsigned int> &resolution,
		const dimension_type dimension,
		const bool complete,
		const bool verbose)
{
	Box<double> bbox(box);
	bbox.inflate(delta);
	std::vector<summand_list_type> approximation =
			Vineyard::compute_vineyard_barcode_approximation(
				boundaryMatrix,
				filtersList,
				precision,
				std::make_pair(bbox.getBottomCorner(), bbox.getUpperCorner()),
				true,
				false,
				complete,
				false,
				verbose);
	if (dimension < 0){
		std::vector<std::vector<std::vector<double>>> image_vector(approximation.size());
// #pragma omp parallel for
		for(unsigned int i = 0; i < approximation.size(); i++){
			std::vector<Summand> module(approximation[i].size());
			for (unsigned int j = 0; j < approximation[i].size(); j++)
				module[j].swapSummand(approximation[i][j]);
			{
				Debug::Timer timer("Computing image of dimension " + std::to_string(i) + " ...", verbose);
				image_vector[i]=compute_2D_image(module, delta, resolution, Box<double>(box), verbose);
			}
		}
		return image_vector;
	}
	std::vector<Summand> module(approximation[dimension].size());
	for (unsigned int i = 0; i < approximation[dimension].size(); i++)
		module[i].swapSummand(approximation[dimension][i]);

	//TODO: verify its not killed too soon
	Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

	return {compute_2D_image(module, delta, resolution, Box<double>(box),verbose)};
}

std::vector<std::vector<double> > compute_2D_image(
		std::vector<Summand>& module,
		const double delta,
		const std::vector<unsigned int>& resolution,
		const Box<double>& box,
		bool verbose)
{
	// Keep dim = 2 here. We ignore other values.
	assert(resolution.size() >= 2);

	std::vector<std::vector<double> > image(resolution[0], std::vector<double>(resolution[1]));
	double moduleWeight = 0;

	{//for Timer
		Debug::Timer timer("Computing module weight ...", verbose);

#pragma omp parallel for reduction(+ : moduleWeight)
		for (Summand& indModule : module){
			moduleWeight += indModule.get_interleaving();
		}
	}//Timer death

	if (verbose) std::cout << "Module weight : " << moduleWeight << "\n";

	if (moduleWeight <= 0){
		if (Debug::debug) std::cout << "!! Negative weight !!" << std::endl;
		return {{0}};
	}

	double stepX = (box.getUpperCorner()[0] - box.getBottomCorner()[0]) / resolution[0];
	double stepY = (box.getUpperCorner()[1] - box.getBottomCorner()[1]) / resolution[1];

	{//for Timer
		Debug::Timer timer("Computing pixel values ...", verbose);

#pragma omp parallel for collapse(2)
		for (unsigned int i = 0; i < resolution[0]; i++){
			for (unsigned int j = 0; j < resolution[1]; j++){
				corner_type x = { box.getBottomCorner()[0] + stepX * i, box.getBottomCorner()[1] + stepY * j };
				image[i][j] = get_pixel_value(module, x, delta, moduleWeight);
			}
		}
	}//Timer death

	return image;
}

double get_pixel_value(
		std::vector<Summand>& module,
		const corner_type& x,
		const double delta,
		double moduleWeight)
{
	double value = 0;

	if (moduleWeight <= 0){
		moduleWeight = 0;

		// Computes the module weight
#pragma omp parallel for reduction(+ : moduleWeight)
		for (Summand &ind_module : module){
			moduleWeight += ind_module.get_interleaving();
		}
	}

#pragma omp parallel for reduction(+ : value)
	for (Summand &indModule : module) {
		double summandWeight = indModule.get_interleaving() / moduleWeight;
		double summandXWeight = indModule.get_local_weight(x, delta) / delta;
		value += summandWeight * summandXWeight;
	}

	return value/2;
}

#endif // IMAGES_H_INCLUDED
