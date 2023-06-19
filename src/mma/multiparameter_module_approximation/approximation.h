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

#include "utilities.h"
#include "vineyards.h"

#include "vineyards_trajectories.h"
// #include "combinatory.h"
#include "debug.h"
#include "multi_filtrations/line.h"
#include "multi_filtrations/box.h"
#include "multi_filtrations/finitely_critical_filtrations.h"
#include <tbb/parallel_for.h>


namespace Gudhi::mma
{
	using Debug::Timer;
	using multi_filtrations::Box;
	using multi_filtrations::Line;
	class Module;
	class Summand;

	
	Module compute_vineyard_barcode_approximation(
		boundary_matrix &boundaryMatrix,
		std::vector<filtration_type> &filtersList,
		const value_type precision,
		multi_filtrations::Box<value_type> &box,
		const bool threshold = false,
		const bool complete = true,
		const bool multithread = false,
		const bool verbose = false);
	
	void compute_vineyard_barcode_approximation_recursively(
		Module &output,
		Vineyard_persistence<Vineyard_matrix_type> &persistence,
		const boundary_matrix &boundaryMatrix,
		filtration_type &basepoint,
		std::vector<unsigned int> &position,
		unsigned int last,
		filtration_type &filter,
		const std::vector<filtration_type> &filtersList,
		value_type precision,
		const Box<value_type> &box,
		const std::vector<unsigned int> &sizeLine,
		bool first = false,
		const bool threshold = false,
		const bool multithread = false);
	
	void compute_vineyard_barcode_approximation_recursively_for_higher_dimension(
		Module &output,
		Vineyard_persistence<Vineyard_matrix_type> &persistence,
		const boundary_matrix &boundaryMatrix,
		const filtration_type &basepoint,
		const std::vector<unsigned int> &position,
		unsigned int last,
		filtration_type &filter,
		const std::vector<filtration_type> &filtersList,
		const value_type precision,
		const Box<value_type> &box,
		const std::vector<unsigned int> &size,
		const bool threshold,
		const bool multithread);
	
	void threshold_filters_list(std::vector<value_type> &filtersList, const Box<value_type> &box);

	
	class Module
	{
	public:
		using module_type = std::vector<Summand>;
		using image_type = std::vector<std::vector<value_type>>;
		using get_pixel_value_function_type = std::function<value_type(const module_type::const_iterator, const module_type::const_iterator, value_type, value_type)>;

		
		Module();
		Module(Box<value_type> &box);

		void resize(unsigned int size);
		Summand &at(unsigned int index);
		module_type::iterator begin();
		module_type::iterator end();

		void clean();
		void fill(const value_type precision);

		std::vector<image_type> get_vectorization(
			const value_type delta,
			const value_type p,
			const bool normalize,
			const multi_filtrations::Box<value_type> &box,
			unsigned int horizontalResolution,
			unsigned int verticalResolution);
		std::vector<image_type> get_vectorization(
			unsigned int horizontalResolution,
			unsigned int verticalResolution,
			get_pixel_value_function_type get_pixel_value) const;
		image_type get_vectorization_in_dimension(
			const dimension_type dimension,
			const value_type delta,
			const value_type p,
			const bool normalize,
			const multi_filtrations::Box<value_type> &box,
			unsigned int horizontalResolution,
			unsigned int verticalResolution);
		image_type get_vectorization_in_dimension(
			const dimension_type dimension,
			unsigned int horizontalResolution,
			unsigned int verticalResolution,
			get_pixel_value_function_type get_pixel_value) const;
		std::vector<value_type> get_landscape_values(
			const std::vector<value_type>& x,
			const dimension_type dimension
		) const;
		image_type get_landscape(
			const dimension_type dimension,
			const unsigned int k,
			const Box<value_type>& box,
			const std::vector<unsigned int>& resolution
		)const;
		std::vector<image_type> get_landscapes(
			const dimension_type dimension,
			const std::vector<unsigned int> ks,
			const Box<value_type>& box,
			const std::vector<unsigned int>& resolution
		)const;
		void add_summand(Summand summand);
		Box<value_type> get_box() const;
		void set_box(Box<value_type> box);
		unsigned int size() const;
		void infer_box(std::vector<filtration_type> &filters_list);
		dimension_type get_dimension() const;
		module_type get_summands_of_dimension(const int dimension) const;
		std::vector<std::pair<std::vector<std::vector<value_type>>,std::vector<std::vector<value_type>>>>  get_corners_of_dimension(const int dimension) const;
		MultiDiagram get_barcode(const Line<value_type> &l, const dimension_type dimension = -1, const bool threshold = false) const;
		MultiDiagrams get_barcodes(const std::vector<Line<value_type>> &lines, const dimension_type dimension = -1, const bool threshold = false)const ;
		MultiDiagrams get_barcodes(const std::vector<filtration_type> &basepoints, const dimension_type dimension = -1, const bool threshold = false) const ;
		std::vector<int> euler_curve(const std::vector<filtration_type> &points) const;
	private:
		module_type module_;
		Box<value_type> box_;
		void _compute_2D_image(image_type &image,
							   const module_type::iterator start,
							   const module_type::iterator end,
							   const value_type delta = 0.1,
							   const value_type p = 1,
							   const bool normalize = true,
							   const Box<value_type> &box = Box<value_type>(),
							   const unsigned int horizontalResolution = 100,
							   const unsigned int verticalResolution = 100);
		void _compute_2D_image(image_type &image,
							   const module_type::const_iterator start,
							   const module_type::const_iterator end,
							   unsigned int horizontalResolution,
							   unsigned int verticalResolution,
							   get_pixel_value_function_type get_pixel_value)const;
		value_type _get_pixel_value(
			const module_type::iterator start,
			const module_type::iterator end,
			const filtration_type x, const value_type delta, const value_type p, const bool normalize, const value_type moduleWeight
		) const;
	};
	
	class Summand
	{
	public:
		Summand();
		Summand(std::vector<filtration_type> &birth_corners, std::vector<filtration_type> &death_corners, dimension_type dimension);

		value_type get_interleaving() const;
		value_type get_interleaving(const Box<value_type> &box);
		value_type get_local_weight(const filtration_type &x, const value_type delta) const;
		std::pair<filtration_type, filtration_type> get_bar(const Line<value_type> &line) const;
		void add_bar(
			value_type baseBirth,
			value_type baseDeath,
			const filtration_type &basepoint,
			filtration_type &birth,
			filtration_type &death,
			const bool threshold,
			const Box<value_type> &box);
		bool is_empty() const;

		const std::vector<filtration_type> &get_birth_list() const;
		const std::vector<filtration_type> &get_death_list() const;
		

		void complete_birth(const value_type precision);
		void complete_death(const value_type precision);

		dimension_type get_dimension() const;
		void set_dimension(dimension_type dimension);

		value_type get_landscape_value(const std::vector<value_type>& x) const;

		friend void swap(Summand &sum1, Summand &sum2);

		bool contains(const filtration_type &x) const;

	private:
		std::vector<filtration_type> birth_corners_;
		std::vector<filtration_type> death_corners_;
		value_type distanceTo0_;
		dimension_type dimension_;

		void _compute_interleaving(const Box<value_type> &box);
		void _add_birth(const filtration_type &birth);
		void _add_death(const filtration_type &death);
		value_type _rectangle_volume(const filtration_type &a, const filtration_type &b) const;
		value_type _get_max_diagonal(const filtration_type &a, const filtration_type &b, const Box<value_type> &box) const;
		value_type d_inf(const filtration_type &a, const filtration_type &b) const;
		void _factorize_min(filtration_type &a, const filtration_type &b);
		void _factorize_max(filtration_type &a, const filtration_type &b);
		void _clean(std::vector<filtration_type> &list, bool keep_inf = true);
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
	// Assumes matrix ordered by dimensions
	
	Module compute_vineyard_barcode_approximation(
		boundary_matrix &boundaryMatrix,
		std::vector<filtration_type> &filtersList,
		const value_type precision,
		Box<value_type> &box,
		const bool threshold,
		const bool complete,
		const bool multithread,
		const bool verbose_)
	{

		verbose = verbose_;
		if (box.get_bottom_corner().size() <= 1)
		{
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
		// if constexpr (Debug::debug) assert(!filtersList.empty() && "A non trivial filters list is needed!");
		if constexpr (Debug::debug)
			assert(filtersList.size() == box.get_bottom_corner().size() && filtersList.size() == box.get_upper_corner().size() && "Filters and box must be of the same dimension!");
		if (Debug::debug)
		{
			for (unsigned int i = 1; i < boundaryMatrix.size(); i++)
				if constexpr (Debug::debug)
					assert(boundaryMatrix.at(i - 1).size() <= boundaryMatrix.at(i).size() && "Boundary matrix has to be sorted by dimension!");
		}

		const unsigned int filtrationDimension = filtersList.size();
		if (verbose)
			std::cout << "Filtration dimension : " << filtrationDimension
					  << std::flush << std::endl;

		unsigned int numberOfSimplices = boundaryMatrix.size();
		if (verbose)
			std::cout << "Number of simplices : " << numberOfSimplices
					  << std::flush << std::endl;

		filtration_type filter(numberOfSimplices); // container of filters

		std::vector<unsigned int> size_line(filtrationDimension - 1);
		for (unsigned int i = 0; i < filtrationDimension - 1; i++)
			size_line[i] = static_cast<unsigned int>(
				std::ceil(
					std::abs(box.get_upper_corner().at(i) - box.get_bottom_corner().back() - box.get_bottom_corner().at(i) + box.get_upper_corner().back()) / precision));

		if (verbose)
			std::cout << "Precision : " << precision << std::endl;
		if (verbose)
			std::cout << "Number of lines : "
					  << Combinatorics::prod(size_line) << std::endl;

		filtration_type basepoint = box.get_bottom_corner();
		for (unsigned int i = 0; i < basepoint.size() - 1; i++)
			basepoint[i] -= box.get_upper_corner().back();
		basepoint.back() = 0;

		std::vector<unsigned int> position(filtrationDimension - 1, 0);
		{
			Timer timer("Computing filtration... ", verbose);
			get_filter_from_line(basepoint, filtersList, filter, box, true);
			// where is the cursor in the output matrix

			if (filtersList[0].size() < numberOfSimplices)
			{
				filtration_type tmp = filter;
				Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
			}
		}

		Vineyard_persistence<Vineyard_matrix_type> persistence(boundaryMatrix, filter, verbose);
		persistence.initialize_barcode();

		auto elapsed = clock();
		if (verbose)
			std::cout << "Multithreading status : " << multithread << std::endl;
		if (verbose)
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
		if (verbose)
			std::cout << " Done ! It took "
					  << static_cast<float>(elapsed) / CLOCKS_PER_SEC
					  << " seconds." << std::endl;

		{ // for Timer
			Timer timer("Cleaning output ... ", verbose);
			output.clean();
			if (complete)
			{
				if (verbose)
					std::cout << "Completing output ...";
				output.fill(precision);
			}
		} // Timer death

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
		Module &output,
		Vineyard_persistence<Vineyard_matrix_type> &persistence,
		const boundary_matrix &boundaryMatrix,
		filtration_type &basepoint,
		std::vector<unsigned int> &position,
		unsigned int last,
		filtration_type &filter,
		const std::vector<filtration_type> &filtersList,
		value_type precision,
		const Box<value_type> &box,
		const std::vector<unsigned int> &sizeLine,
		bool first,
		const bool threshold,
		const bool multithread)
	{
		if (!first)
		{
			get_filter_from_line(basepoint, filtersList, filter, box, true);
			if (filtersList[0].size() < boundaryMatrix.size())
			{
				filtration_type tmp = filter;
				Filtration_creator::get_lower_star_filtration(boundaryMatrix, tmp, filter);
			}

			// Updates the RU decomposition of persistence.
			persistence.update(filter);
		}

		if (verbose && Debug::debug)
			Debug::disp_vect(basepoint);
		if (threshold)
		{
			// TODO if threshold is set to true, we can put a lot of values to 0 / inf so that there is much less swaps to do
		}

		// Computes the diagram from the RU decomposition
		const diagram_type &dgm = persistence.get_diagram();

		if (first)
		{
			output.resize(dgm.size());
		}

		// Fills the barcode of the line having the basepoint basepoint
		//     unsigned int feature = 0;
		//     int oldDim = 0;

		{
			filtration_type birthContainer(filtersList.size());
			filtration_type deathContainer(filtersList.size());

			//        unsigned int counter = 0;
			for (unsigned int i = 0; i < dgm.size(); i++)
			{
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

		// recursive calls of bigger dims, minus current dim (to make less copies)
		//  We keep -last- on the same thread / memory as the previous call
		//  we reached a border and finished this path
		if (sizeLine[last] - 1 == position[last])
			return;
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
		Module &output,
		Vineyard_persistence<Vineyard_matrix_type> &persistence,
		const boundary_matrix &boundaryMatrix,
		const filtration_type &basepoint,
		const std::vector<unsigned int> &position,
		unsigned int last,
		filtration_type &filter,
		const std::vector<filtration_type> &filtersList,
		const value_type precision,
		const Box<value_type> &box,
		const std::vector<unsigned int> &size,
		const bool threshold,
		const bool multithread)
	{
		if (filtersList.size() > 1 && last + 2 < filtersList.size())
		{
			//        if  (verbose && Debug::debug) Debug::disp_vect(basepoint);
			//        if  (verbose) std::cout << multithread << std::endl;

			if (multithread)
			{
				//            if  (verbose) std::cout << "Multithreading dimension..." << std::endl;

#pragma omp parallel for
				for (unsigned int i = last + 1; i < filtersList.size() - 1; i++)
				{
					if (size[i] - 1 == position[i])
						continue;
					// TODO check if it get deleted at each loop !! WARNING
					auto copyPersistence = persistence;
					auto copyBasepoint = basepoint;
					auto copyPosition = position;
					copyBasepoint[i] += precision;
					copyPosition[i]++;
					compute_vineyard_barcode_approximation_recursively(
						output, copyPersistence, boundaryMatrix, copyBasepoint, copyPosition,
						i, filter, filtersList, precision, box, size, false,
						threshold, multithread);
				}
			}
			else
			{
				// No need to copy when not multithreaded.
				// Memory operations are slower than vineyard.
				// %TODO improve trajectory of vineyard
				auto copyPersistence = persistence;
				auto copyBasepoint = basepoint;
				auto copyPosition = position;
				for (unsigned int i = last + 1; i < filtersList.size() - 1; i++)
				{
					if (size[i] - 1 == position[i])
						continue;
					copyPersistence = persistence;
					copyBasepoint = basepoint;
					copyPosition = position;
					copyBasepoint[i] += precision;
					copyPosition[i]++;
					compute_vineyard_barcode_approximation_recursively(
						output, copyPersistence, boundaryMatrix, copyBasepoint, copyPosition,
						i, filter, filtersList, precision, box, size, false,
						threshold, multithread);
				}
			}
		}
	}
	
	inline void threshold_filters_list(std::vector<filtration_type> &filtersList, const Box<value_type> &box)
	{
		return;
		for (unsigned int i = 0; i < filtersList.size(); i++)
		{
			for (value_type &value : filtersList[i])
			{
				value = std::min(
					std::max(value, box.get_bottom_corner()[i]),
					box.get_upper_corner()[i]);
			}
		}
	}
	
	inline Module::Module()
	{
	}
	
	inline Module::Module(Box<value_type> &box) : box_(box)
	{
	}
	
	inline void Module::resize(const unsigned int size)
	{
		module_.resize(size);
	}
	
	inline Summand &Module::at(const unsigned int index)
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
	
	inline void Module::add_summand(Summand summand)
	{
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
		for (unsigned int i = 0; i < module_.size(); i++)
		{
			if (!module_.at(i).get_birth_list().empty() || !module_.at(i).get_death_list().empty())
			{
				tmp.push_back(module_.at(i));
			}
		}
		module_.swap(tmp);
	}
	
	inline void Module::fill(const value_type precision)
	{
		if (module_.empty())
			return;

		for (Summand &sum : module_)
		{
			if (!sum.is_empty())
			{
				sum.complete_birth(precision);
				sum.complete_death(precision);
			}
		}
	}
	
	inline typename std::vector<Module::image_type> Module::get_vectorization(
		const value_type delta,
		const value_type p,
		const bool normalize,
		const Box<value_type> &box,
		unsigned int horizontalResolution,
		unsigned int verticalResolution)
	{
		dimension_type maxDim = module_.back().get_dimension();
		std::vector<Module::image_type> image_vector(maxDim + 1);
		module_type::iterator start;
		module_type::iterator end = module_.begin();
		for (dimension_type d = 0; d <= maxDim; d++)
		{
			{ // for Timer
				Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
				start = end;
				while (end != module_.end() && end->get_dimension() == d)
					end++;
				_compute_2D_image(
					image_vector.at(d),
					start,
					end,
					delta,
					p,
					normalize,
					box,
					horizontalResolution,
					verticalResolution);
			} // Timer death
		}
		return image_vector;
	}
	
	inline std::vector<Module::image_type> Module::get_vectorization(
		unsigned int horizontalResolution,
		unsigned int verticalResolution,
		get_pixel_value_function_type get_pixel_value) const
	{
		dimension_type maxDim = module_.back().get_dimension();
		std::vector<Module::image_type> image_vector(maxDim + 1);
		module_type::const_iterator start;
		module_type::const_iterator end = module_.begin();
		for (dimension_type d = 0; d <= maxDim; d++)
		{
			{ // for Timer
				Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
				start = end;
				while (end != module_.end() && end->get_dimension() == d)
					end++;
				_compute_2D_image(
					image_vector.at(d),
					start,
					end,
					horizontalResolution,
					verticalResolution,
					get_pixel_value);
			} // Timer death
		}
		return image_vector;
	}
	
	inline Module::image_type Module::get_vectorization_in_dimension(
		const dimension_type dimension,
		const value_type delta,
		const value_type p,
		const bool normalize,
		const Box<value_type> &box,
		unsigned int horizontalResolution,
		unsigned int verticalResolution)
	{
		Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

		Module::image_type image;
		module_type::iterator start = module_.begin();
		while (start != module_.end() && start->get_dimension() < dimension)
			start++;
		module_type::iterator end = start;
		while (end != module_.end() && end->get_dimension() == dimension)
			end++;
		_compute_2D_image(
			image,
			start,
			end,
			delta,
			p,
			normalize,
			box,
			horizontalResolution,
			verticalResolution);

		return image;
	}
	inline Module::image_type Module::get_vectorization_in_dimension(
		const dimension_type dimension,
		unsigned int horizontalResolution,
		unsigned int verticalResolution,
		get_pixel_value_function_type get_pixel_value) const
	{
		Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

		Module::image_type image;
		module_type::const_iterator start = module_.begin();
		while (start != module_.end() && start->get_dimension() < dimension)
			start++;
		module_type::const_iterator end = start;
		while (end != module_.end() && end->get_dimension() == dimension)
			end++;
		_compute_2D_image(
			image,
			start,
			end,
			horizontalResolution,
			verticalResolution,
			get_pixel_value);

		return image;
	}
	
	std::vector<value_type> Module::get_landscape_values(
		const std::vector<value_type>& x,
		const dimension_type dimension
	) const
	{
		std::vector<value_type> out;
		out.reserve(this->size());
		for (unsigned int i =0; i< this->size(); i++){
			const Summand& summand = this->module_[i];
			if (summand.get_dimension() == dimension)
				out.push_back(summand.get_landscape_value(x));
		}
		std::sort(out.begin(), out.end(), [](const value_type x, const value_type y){ return x>y;});
		return out;
	}
	
	Module::image_type Module::get_landscape(
		const dimension_type dimension,
		const unsigned int k,
		const Box<value_type>& box,
		const std::vector<unsigned int>& resolution
	)const{
		//TODO extend in higher dimension (ie, change the image type to a template class)
		Module::image_type image;
		image.resize(resolution[0], std::vector<value_type>(resolution[1]));
		value_type stepX = (box.get_upper_corner()[0] - box.get_bottom_corner()[0]) / resolution[0];
		value_type stepY = (box.get_upper_corner()[1] - box.get_bottom_corner()[1]) / resolution[1];

		// #pragma omp parallel for collapse(2)
		// for (unsigned int i = 0; i < resolution[0]; i++){
		// 	for (unsigned int j = 0; j < resolution[1]; j++){
		// 		auto landscape = this->get_landscape_values(
		// 			{
		// 				box.get_bottom_corner()[0] + stepX * i,
		// 				box.get_bottom_corner()[1] + stepY * j
		// 			}, dimension);
		// 		image[i][j] = k < landscape.size() ? landscape[k] : 0;
		// 	}
		// }
		tbb::parallel_for(0U, resolution[0], [&](unsigned int i){
			tbb::parallel_for(0U, resolution[1], [&](unsigned int j){
				auto landscape = this->get_landscape_values(
					{
						box.get_bottom_corner()[0] + stepX * i,
						box.get_bottom_corner()[1] + stepY * j
					}, dimension);
				image[i][j] = k < landscape.size() ? landscape[k] : 0;
			});
		});
		return image;
	}
	
	std::vector<Module::image_type> Module::get_landscapes(
		const dimension_type dimension,
		const std::vector<unsigned int> ks,
		const Box<value_type>& box,
		const std::vector<unsigned int>& resolution
	)const{
		std::vector<Module::image_type> images(ks.size());
		for (auto& image : images)
			image.resize(resolution[0], std::vector<value_type>(resolution[1]));
		value_type stepX = (box.get_upper_corner()[0] - box.get_bottom_corner()[0]) / resolution[0];
		value_type stepY = (box.get_upper_corner()[1] - box.get_bottom_corner()[1]) / resolution[1];

		// #pragma omp parallel for collapse(2)
		// for (unsigned int i = 0; i < resolution[0]; i++){
		// 	for (unsigned int j = 0; j < resolution[1]; j++){
		// 		std::vector<value_type> landscapes = this->get_landscape_values(
		// 			{
		// 				box.get_bottom_corner()[0] + stepX * i,
		// 				box.get_bottom_corner()[1] + stepY * j
		// 			}, dimension);
		// 		for (const auto k : ks){
		// 			images[k][i][j] = k < landscapes.size() ? landscapes[k] : 0 ;
		// 		}
		// 	}
		// }
		tbb::parallel_for(0U, resolution[0], [&](unsigned int i){
			tbb::parallel_for(0U, resolution[1], [&](unsigned int j){
				std::vector<value_type> landscapes = this->get_landscape_values(
					{
						box.get_bottom_corner()[0] + stepX * i,
						box.get_bottom_corner()[1] + stepY * j
					}, dimension);
				for (const auto k : ks){
					images[k][i][j] = k < landscapes.size() ? landscapes[k] : 0 ;
				}
			});
		});
		return images;
	}

	
	inline Box<value_type> Module::get_box() const
	{
		return this->box_;
	}
	
	inline void Module::set_box(Box<value_type> box)
	{
		this->box_ = box;
	}
	
	inline unsigned int Module::size() const
	{
		return this->module_.size();
	}
	
	inline void Module::infer_box(std::vector<filtration_type> &f)
	{
		this->box_.infer_from_filters(f);
	}
	
	inline dimension_type Module::get_dimension() const
	{
		return this->module_.empty() ? -1 : this->module_.back().get_dimension();
	}
	
	inline std::vector<Summand> Module::get_summands_of_dimension(const int dimension) const
	{
		std::vector<Summand> list;
		for (const Summand &summand : this->module_)
		{
			if (summand.get_dimension() == dimension)
				list.push_back(summand);
		}
		return list;
	}
	
	inline std::vector<std::pair<std::vector<std::vector<value_type>>,std::vector<std::vector<value_type>>>> Module::get_corners_of_dimension(const int dimension) const
	{
		std::vector<std::pair<std::vector<std::vector<value_type>>,std::vector<std::vector<value_type>>>> list;
		for (const Summand &summand : this->module_)
		{
			if (summand.get_dimension() == dimension)
				list.push_back(std::make_pair(
					Gudhi::multi_filtrations::Finitely_critical_multi_filtration<value_type>().to_python(summand.get_birth_list()), 
					Gudhi::multi_filtrations::Finitely_critical_multi_filtration<value_type>().to_python(summand.get_death_list())
					));
		}
		return list;
	}
	
	MultiDiagram Module::get_barcode(const Line<value_type> &l, const dimension_type dimension, const bool threshold) const
	{
		std::vector<MultiDiagram_point> barcode(this->size());
		std::pair<filtration_type, filtration_type> threshold_bounds;
		if (threshold) threshold_bounds = l.get_bounds(this->box_);
		unsigned int count = 0;
		for (unsigned int i=0; i < this->size(); i++){
			const Summand &summand = this->module_[i];
			if (dimension != -1 && summand.get_dimension() < dimension)
				continue;
			if (dimension != -1 && summand.get_dimension() > dimension)
				break;
			auto pushed_summand = summand.get_bar(l);
			filtration_type &pbirth = pushed_summand.first;
			filtration_type &pdeath = pushed_summand.second;
			if (threshold){
				auto &min = threshold_bounds.first;
				auto &max = threshold_bounds.second;
				if ((pbirth <= min)) pbirth = min;
				else if ((pbirth>= max)) {pbirth = filtration_type(l.get_dim(), inf); pdeath = pbirth;}
				if ((pdeath>= max) && pbirth[0] != inf) pdeath = max;
				else if ((pdeath<= min)) {pbirth = filtration_type(l.get_dim(), negInf); pdeath = pbirth;}
			}
			barcode[count++] = MultiDiagram_point(summand.get_dimension(), pbirth, pdeath);
		}
		barcode.resize(count);
		return MultiDiagram(barcode);
	}
	
	// TODO TBB
	MultiDiagrams Module::get_barcodes(const std::vector<Line<value_type>> &lines, const dimension_type dimension, const bool threshold) const {
		unsigned int nlines = lines.size();
		MultiDiagrams out(nlines);
		tbb::parallel_for(0U, nlines, [&](unsigned int i){
			const Line<value_type> &l = lines[i];
			out[i] = this->get_barcode(l, dimension, threshold);
		});
		return out;
	}
	
	MultiDiagrams Module::get_barcodes(const std::vector<filtration_type> &basepoints, const dimension_type dimension, const bool threshold)const {
		unsigned int nlines = basepoints.size();
		MultiDiagrams out(nlines);
		// for (unsigned int i = 0; i < nlines; i++){
		tbb::parallel_for(0U, nlines, [&](unsigned int i){
			const Line<value_type> &l = Line<value_type>(basepoints[i]);
			out[i] = this->get_barcode(l, dimension, threshold);
		});
		return out;
	}
	
	std::vector<int> Module::euler_curve(const std::vector<filtration_type> &points) const {
		unsigned int npts = points.size();
		std::vector<int> out(npts);
// #pragma omp parallel for
		tbb::parallel_for(0U, static_cast<unsigned int>(out.size()), [&](unsigned int i){
			auto &euler_char = out[i];
			const filtration_type &point = points[i];
#pragma omp parallel for reduction(+:euler_char)
			for (const Summand &I : this->module_){
				if (I.contains(point)){
					int sign = I.get_dimension() %2 ? -1 : 1;
					euler_char += sign;
				}
			}
		});
		return out;
	}





	
	inline void Module::_compute_2D_image(
		Module::image_type &image,
		const module_type::iterator start,
		const module_type::iterator end,
		const value_type delta,
		const value_type p,
		const bool normalize,
		const Box<value_type> &box,
		const unsigned int horizontalResolution,
		const unsigned int verticalResolution)
	{
		image.resize(horizontalResolution, std::vector<value_type>(verticalResolution));
		value_type moduleWeight = 0;
		{ // for Timer
			Debug::Timer timer("Computing module weight ...", verbose);
			for (auto it = start; it != end; it++) //  precomputes interleaving restricted to box for all summands.
				it->get_interleaving(box);
			if(p == 0){
#pragma omp parallel for reduction(+ : moduleWeight)
				for (auto it = start; it != end; it++)
				{
					moduleWeight += it->get_interleaving()>0;
				}
			}
			else if (p != inf)
			{
#pragma omp parallel for reduction(+ : moduleWeight)
				for (auto it = start; it != end; it++)
				{
					// /!\ TODO deal with inf summands (for the moment,  depends on the box ...)
					if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
						moduleWeight += std::pow(it->get_interleaving(), p);
				}
			}
			else
			{
#pragma omp parallel for reduction(std::max : moduleWeight)
				for (auto it = start; it != end; it++)
				{
					if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
						moduleWeight = std::max(moduleWeight, it->get_interleaving());
				}
			}
		} // Timer death
		if (verbose)
			std::cout << "Module " << start->get_dimension() << " has weight : " << moduleWeight << "\n";
		if (!moduleWeight)
			return;

		if constexpr (Debug::debug)
			if (moduleWeight < 0)
			{
				if constexpr (Debug::debug)
					std::cout << "!! Negative weight !!" << std::endl;
				// 		image.clear();
				return;
			}

		value_type stepX = (box.get_upper_corner()[0] - box.get_bottom_corner()[0]) / horizontalResolution;
		value_type stepY = (box.get_upper_corner()[1] - box.get_bottom_corner()[1]) / verticalResolution;

		{ // for Timer
			Debug::Timer timer("Computing pixel values ...", verbose);

			// #pragma omp parallel for collapse(2)
			// for (unsigned int i = 0; i < horizontalResolution; i++)
			// {
			// 	for (unsigned int j = 0; j < verticalResolution; j++)
			// 	{
			// 		image[i][j] = _get_pixel_value(
			// 			start,
			// 			end,
			// 			{box.get_bottom_corner()[0] + stepX * i, box.get_bottom_corner()[1] + stepY * j},
			// 			delta,
			// 			p,
			// 			normalize,
			// 			moduleWeight);
			// 	}
			// }
			tbb::parallel_for(0U, horizontalResolution, [&](unsigned int i){
				tbb::parallel_for(0U, verticalResolution, [&](unsigned int j){
					image[i][j] = _get_pixel_value(
						start,
						end,
						{box.get_bottom_corner()[0] + stepX * i, box.get_bottom_corner()[1] + stepY * j},
						delta,
						p,
						normalize,
						moduleWeight);
				});
			});
		} // Timer death
	}
	
	inline void Module::_compute_2D_image(
		Module::image_type &image,
		const module_type::const_iterator start,
		const module_type::const_iterator end,
		unsigned int horizontalResolution,
		unsigned int verticalResolution,
		get_pixel_value_function_type get_pixel_value)const
	{
		image.resize(horizontalResolution, std::vector<value_type>(verticalResolution));
		const Box<value_type> &box = this->box_;
		value_type stepX = (box.get_upper_corner()[0] - box.get_bottom_corner()[0]) / horizontalResolution;
		value_type stepY = (box.get_upper_corner()[1] - box.get_bottom_corner()[1]) / verticalResolution;

		{ // for Timer
			Debug::Timer timer("Computing pixel values ...", verbose);

// #pragma omp parallel for collapse(2)
// 			for (unsigned int i = 0; i < horizontalResolution; i++)
// 			{
// 				for (unsigned int j = 0; j < verticalResolution; j++)
// 				{
// 					image[i][j] = get_pixel_value(
// 						start,
// 						end,
// 						box.get_bottom_corner()[0] + stepX * i,
// 						box.get_bottom_corner()[1] + stepY * j);
// 				}
// 			}
			tbb::parallel_for(0U, horizontalResolution, [&](unsigned int i){
				tbb::parallel_for(0U, verticalResolution, [&](unsigned int j){
					image[i][j] = get_pixel_value(
						start,
						end,
						box.get_bottom_corner()[0] + stepX * i,
						box.get_bottom_corner()[1] + stepY * j);
				});
			});
			
		} // Timer death
	}
	
	inline value_type Module::_get_pixel_value(
		const module_type::iterator start,
		const module_type::iterator end,
		const filtration_type x,
		const value_type delta,
		const value_type p,
		const bool normalize,
		const value_type moduleWeight)const
	{
		value_type value = 0;
		if (p == 0){
#pragma omp parallel for reduction(+ : value)
				for (auto it = start; it != end; it++) {
					value += it->get_local_weight(x, delta);
				}
				if (normalize) value /= moduleWeight;
				return value;
			}
		if (p != inf)
		{
#pragma omp parallel for reduction(+: value)
			for (auto it = start; it != end; it++)
			{
				value_type summandWeight = it->get_interleaving();
				value_type summandXWeight = it->get_local_weight(x, delta);
				value += std::pow(summandWeight, p) * summandXWeight;
			}
			if (normalize)	value /= moduleWeight;
			return value;
		}

#pragma omp parallel for reduction(std::max : value)
		for (auto it = start; it != end; it++)
		{
			value = std::max(value, it->get_local_weight(x, delta));
		}
		return value;
	}
	/////////////////////////////////////////////////
	
	inline Summand::Summand() : distanceTo0_(-1), dimension_(-1)
	{
	}
	
	inline Summand::Summand(std::vector<filtration_type> &birth_corners,
							std::vector<filtration_type> &death_corners,
							dimension_type dimension)
		: birth_corners_(birth_corners),
		  death_corners_(death_corners),
		  distanceTo0_(-1),
		  dimension_(dimension)
	{
	}
	inline bool Summand::contains(const filtration_type &x) const{
		bool out = false;
		for (const auto &birth : this->birth_corners_){ // checks if there exists a birth smaller than x
			if (birth <= x){
				out = true;
				break;
			}
		}
		if (!out)	return false;
		out = false;
		for (const auto &death : this->death_corners_){
			if (x<=death){
				out = true;
				break;
			}
		}
		return out;
	}


	inline value_type Summand::get_interleaving(const Box<value_type> &box)
	{
		_compute_interleaving(box);
		return distanceTo0_;
	}

	inline value_type Summand::get_interleaving() const
	{
		return distanceTo0_;
	}
	
	inline value_type Summand::get_local_weight(const filtration_type &x, const value_type delta) const
	{
		bool rectangle = delta <= 0;

		filtration_type mini(x.size());
		filtration_type maxi(x.size());

		// box on which to compute the local weight
#pragma omp simd
		for (unsigned int i = 0; i < x.size(); i++)
		{
			mini[i] = delta <= 0 ? x[i] + delta : x[i] - delta;
			maxi[i] = delta <= 0 ? x[i] - delta : x[i] + delta;
		}

		// Pre-allocating
		std::vector<filtration_type> birthList(birth_corners_.size());
		std::vector<filtration_type> deathList(death_corners_.size());
		unsigned int lastEntry = 0;
		for (const filtration_type &birth : birth_corners_)
		{
			if (birth <= maxi)
			{
				filtration_type tmpBirth(birth.size());
				// WARNING should crash here if birth and x aren't of the same size.
				for (unsigned int i = 0; i < birth.size(); i++)
					tmpBirth[i] = std::max(birth[i], mini[i]);
				birthList[lastEntry].swap(tmpBirth);
				lastEntry++;
			}
		}
		birthList.resize(lastEntry);

		// Thresholds birthlist & deathlist to B_inf(x,delta)
		lastEntry = 0;
		for (const filtration_type &death : death_corners_)
		{
			if (death>= mini)
			{
				filtration_type tmpDeath(death.size());
				// WARNING should crash here if birth and x aren't of the same size.
				for (unsigned int i = 0; i < death.size(); i++)
					tmpDeath[i] = std::min(death[i], maxi[i]);
				deathList[lastEntry].swap(tmpDeath);
				lastEntry++;
			}
		}
		deathList.resize(lastEntry);
		value_type local_weight = 0;
		if (!rectangle)
		{
			// Local weight is inteleaving to 0 of module restricted to the square
			// #pragma omp parallel for reduction(std::max: local_weight)
			Box<value_type> trivial_box;
			for (const filtration_type &birth : birthList)
			{
				if (birth.size() == 0)
					continue;
				for (const filtration_type &death : deathList)
				{
					if (death.size() > 0)
						local_weight = std::max(local_weight,
												_get_max_diagonal(birth, death, trivial_box)); // if box is empty, does not thredhold (already done before).
				}
			}
			return local_weight / (2 * std::abs(delta));
		}
		else
		{
			// local weight is the volume of the largest rectangle in the restricted module
			// #pragma omp parallel for reduction(std::max: local_weight)
			for (const filtration_type &birth : birthList)
			{
				if (birth.size() == 0)
					continue;
				for (const filtration_type &death : deathList)
				{
					if (death.size() > 0)
						local_weight = std::max(local_weight,
												_rectangle_volume(birth, death));
				}
			}
			return local_weight / std::pow(2 * std::abs(delta), x.size());
		}
	}
	inline std::pair<filtration_type, filtration_type> Summand::get_bar(const Line<value_type> &l) const{
		filtration_type pushed_birth(this->get_dimension(), inf);
		filtration_type pushed_death(this->get_dimension(), negInf);
		for (filtration_type birth : this->get_birth_list()){
			filtration_type pb = l.push_forward(birth);
			if ((pb<= pushed_birth)) pushed_birth.swap(pb);
		}
		//
		for (const filtration_type &death : this->get_death_list()){
			filtration_type pd = l.push_back(death);
			if ((pd>=pushed_death)) pushed_death.swap(pd);
		}
		if (!(pushed_birth<=pushed_death))
			return {filtration_type(this->get_dimension(), inf), filtration_type(this->get_dimension(), inf)};
		return {pushed_birth, pushed_death};
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
		value_type baseBirth,
		value_type baseDeath,
		const filtration_type &basepoint,
		filtration_type &birth,
		filtration_type &death,
		const bool threshold,
		const Box<value_type> &box)
	{
		// bar is trivial in that case
		if (baseBirth >= baseDeath)
			return;
// #pragma omp simd
// 		for (unsigned int j = 0; j < birth.size() - 1; j++)
// 		{
// 			birth[j] = basepoint[j] + baseBirth;
// 			death[j] = basepoint[j] + baseDeath;
// 		}
// 		birth.back() = baseBirth;
// 		death.back() = baseDeath;

#pragma omp simd
		for (unsigned int j = 0; j < birth.size() - 1; j++)
		{
			value_type temp = basepoint[j] + baseBirth;
			birth[j] = temp < box.get_bottom_corner()[j] ? negInf : temp;
			temp = basepoint[j] + baseDeath;
			death[j] = temp > box.get_upper_corner()[j] ? inf :  temp ;
		}
		birth.back() = baseBirth < box.get_bottom_corner().back() ? negInf : baseBirth;
		death.back() = baseDeath > box.get_upper_corner().back() ? inf : baseDeath;

		if (threshold)
		{
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
	
	inline bool Summand::is_empty() const
	{
		return birth_corners_.empty() || death_corners_.empty();
	}
	
	inline const std::vector<filtration_type> &Summand::get_birth_list() const
	{
		return birth_corners_;
	}
	
	inline const std::vector<filtration_type> &Summand::get_death_list() const
	{
		return death_corners_;
	}
	
	inline void Summand::complete_birth(const value_type precision)
	{
		if (birth_corners_.empty())
			return;

		for (unsigned int i = 0; i < birth_corners_.size(); i++)
		{
			for (unsigned int j = i + 1; j < birth_corners_.size(); j++)
			{
				value_type dinf = d_inf(birth_corners_[i], birth_corners_[j]);
				if (dinf < 1.1 * precision)
				{
					_factorize_min(birth_corners_[i], birth_corners_[j]);
					birth_corners_[j].clear();
				}
			}
		}
		_clean(birth_corners_);
	}
	
	inline void Summand::complete_death(const value_type precision)
	{
		if (death_corners_.empty())
			return;

		for (unsigned int i = 0; i < death_corners_.size(); i++)
		{
			for (unsigned int j = i + 1; j < death_corners_.size(); j++)
			{
				value_type d = d_inf(death_corners_[i], death_corners_[j]);
				if (d < 1.1 * precision)
				{
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
	
	inline value_type Summand::get_landscape_value(const std::vector<value_type>& x) const{
		value_type out = 0;
		Box<value_type> trivial_box;
		for (const filtration_type& b : this->birth_corners_){
			for (const filtration_type& d : this->death_corners_){
				value_type value = std::min(this->_get_max_diagonal(b,x, trivial_box), this->_get_max_diagonal(x,d, trivial_box));
				out = std::max(out, value);
			}
		}
		return out;
	}
	
	inline void Summand::set_dimension(dimension_type dimension)
	{
		dimension_ = dimension;
	}
	
	inline void Summand::_compute_interleaving(const Box<value_type> &box)
	{
		distanceTo0_ = 0;
#pragma omp parallel for reduction(max \
								   : distanceTo0_)
		for (const std::vector<value_type> &birth : birth_corners_)
		{
			for (const std::vector<value_type> &death : death_corners_)
			{
				distanceTo0_ = std::max(distanceTo0_,
										_get_max_diagonal(birth, death, box));
			}
		}
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
	
	inline void Summand::_add_birth(const filtration_type &birth)
	{
		if (birth_corners_.empty())
		{
			birth_corners_.push_back(birth);
			return;
		}

		// if (birth_corners_.front().front() == negInf)
		// 	return;
  //
		// // when a birth is infinite, we store the summand like this
		// if (birth.front() == negInf)
		// {
		// 	birth_corners_ = {{negInf}};
		// 	return;
		// }

		bool isUseful = true;
		for (unsigned int i = 0; i < birth_corners_.size(); i++)
		{
			if ((birth>= birth_corners_[i]))
			{
				isUseful = false;
				break;
			}
			if (!birth_corners_[i].empty() && (birth<= birth_corners_[i]))
			{
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
	
	inline void Summand::_add_death(const filtration_type &death)
	{
		if (death_corners_.empty())
		{
			death_corners_.push_back(death);
			return;
		}

		// // as drawn in a slope 1 line being equal to -\infty is the same as the
		// // first coordinate being equal to -\infty
		// if (death_corners_.front().front() == inf)
		// 	return;
  //
		// // when a birth is infinite, we store the summand like this
		// if (death.front() == inf)
		// {
		// 	death_corners_ = {{inf}};
		// 	return;
		// }

		bool isUseful = true;
		for (unsigned int i = 0; i < death_corners_.size(); i++)
		{
			if ((death<= death_corners_[i]))
			{
				isUseful = false;
				break;
			}
			if (!death_corners_[i].empty() && (death>= death_corners_[i]))
			{
				death_corners_[i].clear();
			}
		}

		_clean(death_corners_);
		if (isUseful)
			death_corners_.push_back(death);
	}
	
	inline value_type Summand::_get_max_diagonal(const filtration_type &birth, const filtration_type &death, const Box<value_type> &box) const
	{
		if constexpr (Debug::debug)
			assert(birth.size() == death.size() && "Inputs must be of the same size !");
		value_type s = inf;
		bool threshold_flag = !box.is_trivial();
		if (threshold_flag)
			for (unsigned int i = 0; i < birth.size(); i++)
			{
				value_type max_i = box.get_upper_corner().size() > i ? box.get_upper_corner()[i] : inf;
				value_type min_i = box.get_upper_corner().size() > i ? box.get_bottom_corner()[i] : negInf;
				value_type t_death = std::min(death[i], max_i);
				value_type t_birth = std::max(birth[i], min_i);
				s = std::min(s, t_death - t_birth);
			}
		else
		{
			for (unsigned int i = 0; i < birth.size(); i++)
				s = std::min(s, death[i] - birth[i]);
		}
		return s;
	}
	
	
	inline value_type Summand::_rectangle_volume(const filtration_type &a, const filtration_type &b) const
	{
		if constexpr (Debug::debug)
			assert(a.size() == b.size() && "Inputs must be of the same size !");
		value_type s = b[0] - a[0];
		for (unsigned int i = 1; i < a.size(); i++)
		{
			s = s * (b[i] - a[i]);
		}
		return s;
	}
	
	inline value_type Summand::d_inf(const filtration_type &a, const filtration_type &b) const
	{
		if (a.empty() || b.empty() || a.size() != b.size())
			return inf;

		value_type d = std::abs(a[0] - b[0]);
		for (unsigned int i = 1; i < a.size(); i++)
			d = std::max(d, std::abs(a[i] - b[i]));

		return d;
	}
	
	inline void Summand::_factorize_min(filtration_type &a, const filtration_type &b)
	{
		if (Debug::debug && (a.empty() || b.empty()))
		{
			std::cout << "Empty corners ??\n";
			return;
		}

		for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++)
			a[i] = std::min(a[i], b[i]);
	}
	
	inline void Summand::_factorize_max(filtration_type &a, const filtration_type &b)
	{

		if (Debug::debug && (a.empty() || b.empty()))
		{
			std::cout << "Empty corners ??\n";
			return;
		}

		for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++)
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
	
	inline void Summand::_clean(std::vector<filtration_type> &list, bool keep_inf)
	{
		unsigned int i = 0;
		while (i < list.size())
		{
			while (!list.empty() && ((*(list.rbegin())).empty() || (!keep_inf && ((*(list.rbegin()))[0] == inf || (*(list.rbegin()))[0] == negInf))))
				list.pop_back();
			if (i < list.size() && (list[i].empty() || (!keep_inf && (list[i][0] == inf || list[i][0] == negInf))))
			{
				list[i].swap(*(list.rbegin()));
				list.pop_back();
			}
			i++;
		}
	}
	
	inline void swap(Summand &sum1, Summand &sum2)
	{
		std::swap(sum1.birth_corners_, sum2.birth_corners_);
		std::swap(sum1.death_corners_, sum2.death_corners_);
		std::swap(sum1.distanceTo0_, sum2.distanceTo0_);
		// 	std::swap(sum1.updateDistance_, sum2.updateDistance_);
	}

} // namespace Vineyard

#endif // APPROXIMATION_H_INCLUDED
