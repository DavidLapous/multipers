/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence class, renaming and cleanup.
 *      - 2022/05 Hannah Schreiber: Addition of Summand class and Module class.
 */
/**
 * @file approximation.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the functions related to the approximation of n-modules.
 */

#ifndef APPROXIMATION_H_INCLUDED
#define APPROXIMATION_H_INCLUDED

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "vineyards.h"
#include "vineyards_trajectories.h"
// #include "combinatory.h"
#include "debug.h"
#include "utilities.h"



namespace Vineyard {

using Debug::Timer;

class Module;
class Summand;

Module compute_vineyard_barcode_approximation(
        boundary_matrix& boundaryMatrix,
        std::vector<filtration_type>& filtersList,
        const double precision,
		Box& box,
        const bool threshold = false,
        const bool complete = true,
        const bool multithread = false,
        const bool verbose = false);
void compute_vineyard_barcode_approximation_recursively(
		Module& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        corner_type& basepoint,
        std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        double precision,
		const Box& box,
        const std::vector<unsigned int>& sizeLine,
        bool first = false,
        const bool threshold = false,
        const bool multithread = false);
void compute_vineyard_barcode_approximation_recursively_for_higher_dimension(
		Module& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        const corner_type& basepoint,
        const std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        const double precision,
		const Box& box,
        const std::vector<unsigned int>& size,
        const bool threshold,
        const bool multithread);
void threshold_filters_list(std::vector<filtration_type>& filtersList, const Box &box);

class Module
{
public:
	using module_type = std::vector<Summand>;
	using image_type = std::vector<std::vector<double> >;
	using get_pixel_value_function_type = std::function<double(const module_type::iterator,const module_type::iterator,double,double)>;

	Module();
	Module(Box &box);

	void resize(unsigned int size);
	Summand& at(unsigned int index);
	module_type::iterator begin();
	module_type::iterator end();

	void clean();
	void fill(const double precision);

	std::vector<image_type> get_vectorization(
			const double delta,
			unsigned int horizontalResolution,
			unsigned int verticalResolution);
	std::vector<image_type> get_vectorization(
			unsigned int horizontalResolution,
			unsigned int verticalResolution,
			get_pixel_value_function_type get_pixel_value);
	image_type get_vectorization_in_dimension(
			const dimension_type dimension,
			const double delta,
			unsigned int horizontalResolution,
			unsigned int verticalResolution);
	image_type get_vectorization_in_dimension(
			const dimension_type dimension,
			unsigned int horizontalResolution,
			unsigned int verticalResolution,
			get_pixel_value_function_type get_pixel_value);
	void add_summand(Summand summand);
	Box get_box() const;
	void set_box(Box box);
	unsigned int size() const;
	void infer_box(std::vector<filtration_type>& filters_list);
	unsigned int get_dimension() const ;
	std::vector<Summand> get_summands_of_dimension(unsigned int dimension) const;
	std::vector<corners_type> get_corners_of_dimension(unsigned int dimension) const;
private:
	module_type module_;
	Box box_;
	void _compute_2D_image(image_type& image,
			const module_type::iterator start,
			const module_type::iterator end,
			const double delta,
			unsigned int horizontalResolution,
			unsigned int verticalResolution);
	void _compute_2D_image(image_type& image,
			const module_type::iterator start,
			const module_type::iterator end,
			unsigned int horizontalResolution,
			unsigned int verticalResolution,
			get_pixel_value_function_type get_pixel_value);
	double _get_pixel_value(const module_type::iterator start,
			const module_type::iterator end,
			const corner_type x,
			const double delta,
			double moduleWeight);
};

class Summand
{
public:
	Summand();
	Summand(std::vector<corner_type>& birth_corners, std::vector<corner_type>& death_corners, dimension_type dimension);

	double get_interleaving();
	double get_local_weight(const corner_type& x, const double delta);

	void add_bar(
			double baseBirth,
			double baseDeath,
			const corner_type& basepoint,
			corner_type& birth,
			corner_type& death,
			const bool threshold,
			const Box& box);
	bool is_empty();

	const std::vector<corner_type>& get_birth_list() const ;
	const std::vector<corner_type>& get_death_list() const ;

	void complete_birth(const double precision);
	void complete_death(const double precision);

	dimension_type get_dimension() const ;
	void set_dimension(dimension_type dimension);

	friend void swap(Summand& sum1, Summand& sum2);

private:
	std::vector<corner_type> birth_corners_;
	std::vector<corner_type> death_corners_;
	double distanceTo0_;
	bool updateDistance_;
	dimension_type dimension_;

	void _compute_interleaving();
	void _add_birth(corner_type& birth);
	void _add_death(corner_type& death);
	double _get_min_diagonal(const corner_type& a, const corner_type& b);
	double _get_max_diagonal(const corner_type& a,
							const corner_type& b);
	void _factorize_min(corner_type& a, const corner_type& b);
	void _factorize_max(corner_type& a, const corner_type& b);
	void _clean(std::vector<corner_type>& list);
};

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
Module compute_vineyard_barcode_approximation(
        boundary_matrix& boundaryMatrix,
        std::vector<filtration_type>& filtersList,
        const double precision,
		Box& box,
        const bool threshold,
        const bool complete,
        const bool multithread,
        const bool verbose)
{
    Vineyard::verbose = verbose;
	if (box.get_bottom_corner().size()<=1){
		std::cout << "#parameter is " << box.get_bottom_corner().size() << ". Infering the box.\n";
		box.infer_from_filters(filtersList);
	}
	Module output(box);
// 	if(threshold)
// 	{
// 		Debug::Timer("Thresholding filtration...", verbose);
// 		threshold_filters_list(filtersList, box);
// 	}
	
	// completes lowerstar filtrations into full filtrations.
	Filtration_creator::complete_lower_star_filters_list(boundaryMatrix, filtersList); 

    // Checks if dimensions are compatibles
    // assert(!filtersList.empty() && "A non trivial filters list is needed!");
	assert(filtersList.size() == box.get_bottom_corner().size()
		   && filtersList.size() == box.get_upper_corner().size()
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
						std::abs(box.get_upper_corner().at(i) - box.get_bottom_corner().back()
								 - box.get_bottom_corner().at(i) + box.get_upper_corner().back()) / precision
                        )
                    );

    if  (verbose)
            std::cout << "Precision : " << precision << std::endl;
    if  (verbose)
            std::cout << "Number of lines : "
                      << Combinatorics::prod(size_line) << std::endl;

	corner_type basepoint = box.get_bottom_corner();
    for (unsigned int i = 0; i < basepoint.size() - 1; i++)
		basepoint[i] -= box.get_upper_corner().back();
    basepoint.back() = 0;

    std::vector<unsigned int> position(filtrationDimension - 1, 0);
    {Timer timer("Computing filtration... ", verbose);
        get_filter_from_line(basepoint, filtersList, filter, box, true);
        // where is the cursor in the output matrix


        if (filtersList[0].size() < numberOfSimplices) {
            filtration_type tmp = filter;
            Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
        }
    }

    Vineyard_persistence<Vineyard_matrix_type> persistence(boundaryMatrix, filter, verbose);
	persistence.initialize_barcode();

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
		output.clean();
        if (complete){
            if  (verbose) std::cout << "Completing output ...";
			output.fill(precision);
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
		Module& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        corner_type& basepoint,
        std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        double precision,
		const Box& box,
        const std::vector<unsigned int>& sizeLine,
        bool first,
        const bool threshold,
        const bool multithread)
{
    if (!first) {
        get_filter_from_line(basepoint, filtersList, filter, box, true);
        if (filtersList[0].size() < boundaryMatrix.size()) {
            filtration_type tmp = filter;
            Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
        }

		// Updates the RU decomposition of persistence.
		persistence.update(filter);
    }

    if  (verbose && Debug::debug) Debug::disp_vect(basepoint);
    if (threshold){
        // TODO if threshold is set to true, we can put a lot of values to 0 / inf so that there is much less swaps to do
    }

    // Computes the diagram from the RU decomposition
    const diagram_type& dgm = persistence.get_diagram();

	if (first){
		output.resize(dgm.size());
	}

    // Fills the barcode of the line having the basepoint basepoint
//     unsigned int feature = 0;
//     int oldDim = 0;

    {
        corner_type birthContainer(filtersList.size());
        corner_type deathContainer(filtersList.size());

//        unsigned int counter = 0;
		for (unsigned int i = 0; i < dgm.size(); i++){
			output.at(i).set_dimension(dgm.at(i).dim);
			output.at(i).add_bar(
						dgm.at(i).birth,
						dgm.at(i).death,
						basepoint,
						birthContainer,
						deathContainer,
						threshold,
						box);
		}
//        for (int dim = 0; dim <= dgm.rbegin()->dim; dim++){
//			module_type &list_of_summands = output[dim];
//            for (unsigned int i = 0; i < output[dim].size(); i++)
//            {
//				list_of_summands[i].add_bar(
//							dgm[counter+i].birth,
//							dgm[counter+i].death,
//							basepoint,
//							birthContainer,
//							deathContainer,
//							threshold,
//							box);
//            }
//            counter += output[dim].size();
//        }
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
		Module& output,
        Vineyard_persistence<Vineyard_matrix_type>& persistence,
        const boundary_matrix& boundaryMatrix,
        const corner_type& basepoint,
        const std::vector<unsigned int>& position,
        unsigned int last,
        filtration_type& filter,
        const std::vector<filtration_type>& filtersList,
        const double precision,
		const Box& box,
        const std::vector<unsigned int>& size,
        const bool threshold,
        const bool multithread)
{
    if (filtersList.size() > 1 && last + 2 < filtersList.size()){
//        if  (verbose && Debug::debug) Debug::disp_vect(basepoint);
//        if  (verbose) std::cout << multithread << std::endl;

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

inline void threshold_filters_list(std::vector<filtration_type>& filtersList, const Box &box){
	return;
	for(unsigned int i =0; i<filtersList.size(); i++){
		for(filtration_value_type &value : filtersList[i]){
			value = std::min(
				std::max(value, box.get_bottom_corner()[i]),
				box.get_upper_corner()[i]
			);
		}
	}
}




inline Module::Module()
{}

inline Module::Module(Box &box) : box_(box)
{}

inline void Module::resize(unsigned int size)
{
	module_.resize(size);
}

inline Summand &Module::at(unsigned int index)
{
	return module_.at(index);
}

inline Module::module_type::iterator Module::begin()
{
	return module_.begin();
}

inline Module::module_type::iterator Module::end()
{
	return module_.end();
}
inline void Module::add_summand(Summand summand){
	module_.push_back(summand);
}

/**
 * @brief Remove the empty summands of the output
 *
 * @param output p_output:...
 * @param keep_order p_keep_order:... Defaults to false.
 */
inline void Module::clean()
{
	module_type tmp;
	for (unsigned int i = 0; i < module_.size(); i++){
		if (!module_.at(i).get_birth_list().empty() || !module_.at(i).get_death_list().empty()){
			tmp.push_back(module_.at(i));
		}
	}
	module_.swap(tmp);
}

inline void Module::fill(const double precision)
{
	if (module_.empty()) return;

	for (Summand& sum : module_){
		if (!sum.is_empty()){
			sum.complete_birth(precision);
			sum.complete_death(precision);
		}
	}
}

inline std::vector<Module::image_type> Module::get_vectorization(
		const double delta,
		unsigned int horizontalResolution,
		unsigned int verticalResolution)
{
	dimension_type maxDim = module_.back().get_dimension();
	std::vector<Module::image_type> image_vector(maxDim + 1);
	module_type::iterator start;
	module_type::iterator end = module_.begin();
	for (dimension_type d = 0; d <= maxDim; d++){
		{//for Timer
			Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
			start = end;
			while (end != module_.end() && end->get_dimension() == d) end++;
			_compute_2D_image(
					image_vector.at(d),
					start,
					end,
					delta,
					horizontalResolution,
					verticalResolution);
		}//Timer death
	}
	return image_vector;
}

inline std::vector<Module::image_type> Module::get_vectorization(
		unsigned int horizontalResolution,
		unsigned int verticalResolution,
		get_pixel_value_function_type get_pixel_value)
{
	dimension_type maxDim = module_.back().get_dimension();
	std::vector<Module::image_type> image_vector(maxDim + 1);
	module_type::iterator start;
	module_type::iterator end = module_.begin();
	for (dimension_type d = 0; d <= maxDim; d++){
		{//for Timer
			Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
			start = end;
			while (end != module_.end() && end->get_dimension() == d) end++;
			_compute_2D_image(
					image_vector.at(d),
					start,
					end,
					horizontalResolution,
					verticalResolution,
					get_pixel_value);
		}//Timer death
	}
	return image_vector;
}

inline Module::image_type Module::get_vectorization_in_dimension(
		const dimension_type dimension,
		const double delta,
		unsigned int horizontalResolution,
		unsigned int verticalResolution)
{
	Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

	Module::image_type image;
	module_type::iterator start = module_.begin();
	while (start != module_.end() && start->get_dimension() < dimension) start++;
	module_type::iterator end = start;
	while (end != module_.end() && end->get_dimension() == dimension) end++;
	_compute_2D_image(
				image,
				start,
				end,
				delta,
				horizontalResolution,
				verticalResolution);

	return image;
}

inline Module::image_type Module::get_vectorization_in_dimension(
		const dimension_type dimension,
		unsigned int horizontalResolution,
		unsigned int verticalResolution,
		get_pixel_value_function_type get_pixel_value)
{
	Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

	Module::image_type image;
	module_type::iterator start = module_.begin();
	while (start != module_.end() && start->get_dimension() < dimension) start++;
	module_type::iterator end = start;
	while (end != module_.end() && end->get_dimension() == dimension) end++;
	_compute_2D_image(
				image,
				start,
				end,
				horizontalResolution,
				verticalResolution,
				get_pixel_value);

	return image;
}

inline Box Module::get_box() const{
	return this->box_;
}
inline void Module::set_box(Box box){
	this->box_ = box;
}

inline unsigned int Module::size() const {
	return this->module_.size();
}
inline void Module::infer_box(std::vector<filtration_type>& f){
	this->box_.infer_from_filters(f);
}

inline unsigned int Module::get_dimension() const {
	return this->module_.back().get_dimension();
}

inline std::vector<Summand> Module::get_summands_of_dimension(const unsigned int dimension) const {
	std::vector<Summand> list;
	for (const Summand &summand : this->module_)
	{
		if (summand.get_dimension() == dimension)
			list.push_back(summand);
	}
	return list;
	
}

inline std::vector<corners_type> Module::get_corners_of_dimension(const unsigned int dimension) const {
	std::vector<corners_type> list;
	for (const Summand &summand : this->module_)
	{
		if (summand.get_dimension() == dimension)
			list.push_back(std::make_pair(summand.get_birth_list(), summand.get_death_list()));
	}
	return list;
	
}



inline void Module::_compute_2D_image(
		Module::image_type &image,
		const module_type::iterator start,
		const module_type::iterator end,
		const double delta,
		unsigned int horizontalResolution,
		unsigned int verticalResolution)
{
	image.resize(horizontalResolution, std::vector<double>(verticalResolution));
	double moduleWeight = 0;
	Box &box = this->box_;
	{//for Timer
		Debug::Timer timer("Computing module weight ...", verbose);

#pragma omp parallel for reduction(+ : moduleWeight)
		for (auto it = start; it != end; it++){
			moduleWeight += it->get_interleaving();
		}
	}//Timer death

	if (verbose) std::cout << "Module weight : " << moduleWeight << "\n";

	if (moduleWeight <= 0){
		if (Debug::debug) std::cout << "!! Negative weight !!" << std::endl;
		image.clear();
		return;
	}

	double stepX = (box.get_upper_corner()[0] - box.get_bottom_corner()[0]) / horizontalResolution;
	double stepY = (box.get_upper_corner()[1] - box.get_bottom_corner()[1]) / verticalResolution;

	{//for Timer
		Debug::Timer timer("Computing pixel values ...", verbose);

#pragma omp parallel for collapse(2)
		for (unsigned int i = 0; i < horizontalResolution; i++){
			for (unsigned int j = 0; j < verticalResolution; j++){
				image[i][j] = _get_pixel_value(
							start,
							end,
							{box.get_bottom_corner()[0] + stepX * i, box.get_bottom_corner()[1] + stepY * j},
							delta,
							moduleWeight);
			}
		}
	}//Timer death
}

inline void Module::_compute_2D_image(
		Module::image_type &image,
		const module_type::iterator start,
		const module_type::iterator end,
		unsigned int horizontalResolution,
		unsigned int verticalResolution,
		get_pixel_value_function_type get_pixel_value)
{
	image.resize(horizontalResolution, std::vector<double>(verticalResolution));
	Box &box = this->box_;
	double stepX = (box.get_upper_corner()[0] - box.get_bottom_corner()[0]) / horizontalResolution;
	double stepY = (box.get_upper_corner()[1] - box.get_bottom_corner()[1]) / verticalResolution;

	{//for Timer
		Debug::Timer timer("Computing pixel values ...", verbose);

#pragma omp parallel for collapse(2)
		for (unsigned int i = 0; i < horizontalResolution; i++){
			for (unsigned int j = 0; j < verticalResolution; j++){
				image[i][j] = get_pixel_value(
							start,
							end,
							box.get_bottom_corner()[0] + stepX * i,
							box.get_bottom_corner()[1] + stepY * j);
			}
		}
	}//Timer death
}

inline double Module::_get_pixel_value(
		const module_type::iterator start,
		const module_type::iterator end,
		const corner_type x,
		const double delta,
		double moduleWeight)
{
	double value = 0;

#pragma omp parallel for reduction(+ : value)
	for (auto it = start; it != end; it++) {
		double summandWeight = it->get_interleaving() / moduleWeight;
		double summandXWeight = it->get_local_weight(x, delta) / delta;
		value += summandWeight * summandXWeight;
	}

	return value/2;
}

inline Summand::Summand() : distanceTo0_(-1), updateDistance_(true), dimension_(-1)
{}

inline Summand::Summand(std::vector<corner_type> &birth_corners,
						std::vector<corner_type> &death_corners,
						dimension_type dimension)
	: birth_corners_(birth_corners),
	  death_corners_(death_corners),
	  distanceTo0_(-1),
	  updateDistance_(true),
	  dimension_(dimension)
{}

//inline Summand::Summand(Summand &summandToCopy)
//	: summand_(summandToCopy.summand_),
//	  distanceTo0_(summandToCopy.distanceTo0_),
//	  updateDistance_(summandToCopy.updateDistance_)
//{}

//inline Summand::Summand(Summand &&other) noexcept
//	: summand_(std::move(other.summand_)),
//	  distanceTo0_(std::exchange(other.distanceTo0_, 0)),
//	  updateDistance_(std::exchange(other.updateDistance_, false))
//{}

inline double Summand::get_interleaving()
{
	if (updateDistance_) _compute_interleaving();
	return distanceTo0_;
}

inline double Summand::get_local_weight(const corner_type &x, const double delta)
{
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
	std::vector<corner_type> birthList(birth_corners_.size());
	std::vector<corner_type> deathList(death_corners_.size());
	unsigned int lastEntry = 0;
	for (const corner_type& birth : birth_corners_){
		if (is_smaller(birth, maxi)){
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
	for (const corner_type& death : death_corners_){
		if (is_greater(death, mini)){
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
								   _get_min_diagonal(birth,death));
		}
	}

	return maxDiag; // should be less than delta
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
inline void Summand::add_bar(
		double baseBirth,
		double baseDeath,
		const corner_type &basepoint,
		corner_type &birth,
		corner_type &death,
		const bool threshold,
		const Box &box)
{
	// bar is trivial in that case
	if (baseBirth >= baseDeath) return;
#pragma omp simd
	for (unsigned int j = 0; j < birth.size() - 1; j++){
		birth[j] = basepoint[j] + baseBirth;
		death[j] = basepoint[j] + baseDeath;
	}
	birth.back() = baseBirth;
	death.back() = baseDeath;

	if (threshold){
		// std::cout << box;
		threshold_down(birth, box, basepoint);
		threshold_up(death, box, basepoint);
	}
	_add_birth(birth);
	_add_death(death);
}

/**
 * @brief Returns true if a summand is empty
 *
 * @param summand summand to check.
 * @return bool
 */
inline bool Summand::is_empty()
{
	return birth_corners_.empty() || death_corners_.empty();
}

inline const std::vector<corner_type> &Summand::get_birth_list() const 
{
	return birth_corners_;
}

inline const std::vector<corner_type> &Summand::get_death_list() const 
{
	return death_corners_;
}

inline void Summand::complete_birth(const double precision)
{
	if (birth_corners_.empty()) return;

	for (unsigned int i = 0; i < birth_corners_.size(); i++){
		for (unsigned int j = i + 1; j < birth_corners_.size(); j++){
			double dinf = _get_max_diagonal(birth_corners_[i], birth_corners_[j]);
			if (dinf < 1.1 * precision){
				_factorize_min(birth_corners_[i], birth_corners_[j]);
				birth_corners_[j].clear();
			}
		}
	}
	_clean(birth_corners_);
}

inline void Summand::complete_death(const double precision)
{
	if (death_corners_.empty()) return;

	for (unsigned int i = 0; i < death_corners_.size(); i++){
		for (unsigned int j = i + 1; j < death_corners_.size(); j++){
			double d = _get_max_diagonal(death_corners_[i], death_corners_[j]);
			if (d < 1.1 * precision){
				_factorize_max(death_corners_[i], death_corners_[j]);
				death_corners_[j].clear();
			}
		}
	}
	_clean(death_corners_);
}

inline dimension_type Summand::get_dimension() const
{
	return dimension_;
}

inline void Summand::set_dimension(dimension_type dimension)
{
	dimension_ = dimension;
}

inline void Summand::_compute_interleaving(){
	distanceTo0_ = 0;
	for (const std::vector<double> &birth : birth_corners_){
		for(const std::vector<double> &death : death_corners_){
			distanceTo0_ = std::max(distanceTo0_,
									_get_min_diagonal(birth, death));
		}
	}
	updateDistance_ = false;
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
inline void Summand::_add_birth(corner_type &birth)
{
	if (birth_corners_.empty()){
		birth_corners_.push_back(birth);
		return;
	}

	if (birth_corners_.front().front() == negInf) return;

	// when a birth is infinite, we store the summand like this
	if (birth.front() == negInf){
		birth_corners_ = {{negInf}};
		return;
	}

	bool isUseful = true;
	for (unsigned int i = 0; i < birth_corners_.size(); i++){
		if (is_greater(birth, birth_corners_[i])) {
			isUseful = false;
			break;
		}
		if (!birth_corners_[i].empty() && is_smaller(birth, birth_corners_[i])){
			birth_corners_[i].clear();
		}
	}

	_clean(birth_corners_);
	if (isUseful)
		birth_corners_.push_back(birth);
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
inline void Summand::_add_death(corner_type &death)
{
	if (death_corners_.empty()){
		death_corners_.push_back(death);
		return;
	}

	// as drawn in a slope 1 line being equal to -\infty is the same as the
	// first coordinate being equal to -\infty
	if (death_corners_.front().front() == inf)
		return;

	// when a birth is infinite, we store the summand like this
	if (death.front() == inf){
		death_corners_ = {{inf}};
		return;
	}

	bool isUseful = true;
	for (unsigned int i = 0; i < death_corners_.size(); i++){
		if (is_smaller(death, death_corners_[i])) {
			isUseful = false;
			break;
		}
		if (!death_corners_[i].empty() && is_greater(death, death_corners_[i])){
			death_corners_[i].clear();
		}
	}

	_clean(death_corners_);
	if (isUseful)
		death_corners_.push_back(death);
}

inline double Summand::_get_min_diagonal(const corner_type &a, const corner_type &b)
{
	assert(a.size() == b.size() && "Inputs must be of the same size !");
	double s = b[0] - a[0];

	for (unsigned int i = 1; i < a.size(); i++){
		s = std::min(s, b[i] - a[i]);
	}
	return s;
}

/**
 * @brief Returns the biggest diagonal length in the rectangle
 * {z : @p a ≤ z ≤ @p b}
 *
 * @param a smallest element of the box
 * @param b biggest element of the box.
 * @return double, length of the diagonal.
 */
inline double Summand::_get_max_diagonal(const corner_type &a, const corner_type &b)
{
	if (a.empty() || b.empty() || a.size() != b.size())
		return inf;

	double d = std::abs(a[0] - b[0]);
	for (unsigned int i = 1; i < a.size(); i++)
		d = std::max(d, std::abs(a[i] - b[i]));

	return d;
}

inline void Summand::_factorize_min(corner_type& a, const corner_type& b)
{
	if (a.size() != b.size()) return;

	for (unsigned int i = 0; i < a.size(); i++)
		a[i] = std::min(a[i], b[i]);
}

inline void Summand::_factorize_max(corner_type &a, const corner_type &b)
{
	if (a.size() != b.size()) return;

	for (unsigned int i = 0; i < a.size(); i++)
		a[i] = std::max(a[i], b[i]);
}

/**
 * @brief Cleans empty entries of a corner list
 *
 * @param list corner list to clean
 * @param keep_sort If true, will keep the order of the corners,
 * with a computational overhead. Defaults to false.
 */
// WARNING Does permute the output.
inline void Summand::_clean(std::vector<corner_type> &list)
{
	unsigned int i = 0;
	while (i < list.size()){
		while (!list.empty() && (*(list.rbegin())).empty())
			list.pop_back();
		if (i < list.size() && list[i].empty()){
			list[i].swap(*(list.rbegin()));
			list.pop_back();
		}
		i++;
	}
}

inline void swap(Summand& sum1, Summand& sum2)
{
	std::swap(sum1.birth_corners_, sum2.birth_corners_);
	std::swap(sum1.death_corners_, sum2.death_corners_);
	std::swap(sum1.distanceTo0_, sum2.distanceTo0_);
	std::swap(sum1.updateDistance_, sum2.updateDistance_);
}

}   //namespace Vineyard

#endif // APPROXIMATION_H_INCLUDED
