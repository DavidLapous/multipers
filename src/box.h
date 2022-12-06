/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 * 
 */
/**
 * @file box.h
 * @author David Loiseaux
 * @brief BOX.
 */

#ifndef BOX_H_INCLUDED
#define BOX_H_INCLUDED

#include <vector>
#include <ostream>
// #include <iomanip>
#include <cmath>
#include <limits>
#include <cassert>
// #include "gudhi/Simplex_tree.h"



#include "utilities.h"




namespace Vineyard {

/**
 * @brief Holds the square box on which to compute.
 */
class Box
{
public:
	Box();
	Box(const corner_type& bottomCorner, const corner_type& upperCorner);
	Box(const std::pair<corner_type, corner_type>& box);

	void inflate(double delta);
	const corner_type& get_bottom_corner() const;
	const corner_type& get_upper_corner() const;
	bool contains(point_type& point) const;
	void infer_from_filters(std::vector<std::vector<double>> &Filters_list);
    bool is_trivial() const ;

private:
	corner_type bottomCorner_;
	corner_type upperCorner_;
};


inline Box::Box()
{}

inline Box::Box(const corner_type &bottomCorner, const corner_type &upperCorner)
	: bottomCorner_(bottomCorner),
	  upperCorner_(upperCorner)
{
	assert(bottomCorner.size() == upperCorner.size()
		   && is_smaller(bottomCorner, upperCorner)
		   && "This box is trivial !");
}

inline Box::Box(const std::pair<corner_type, corner_type> &box)
	: bottomCorner_(box.first),
	  upperCorner_(box.second)
{}

inline void Box::inflate(double delta)
{
#pragma omp simd
	for (unsigned int i = 0; i < bottomCorner_.size(); i++){
		bottomCorner_[i] -= delta;
		upperCorner_[i] += delta;
	}
}

inline void Box::infer_from_filters(std::vector<std::vector<double>> &Filters_list){
	unsigned int dimension = Filters_list.size();
	unsigned int nsplx = Filters_list[0].size();
	std::vector<double> lower(dimension);
	std::vector<double> upper(dimension);
	for (unsigned int i =0; i < dimension; i++){
		Vineyard::filtration_value_type min = Filters_list[i][0];
		Vineyard::filtration_value_type max = Filters_list[i][0];
		for (unsigned int j=1; j<nsplx; j++){
			min = std::min(min, Filters_list[i][j]);
			max = std::max(max, Filters_list[i][j]);
		}
		lower[i] = min;
		upper[i] = max;
	}
	bottomCorner_.swap(lower);
	upperCorner_.swap(upper);
}
inline bool Box::is_trivial() const {
    return bottomCorner_.empty() || upperCorner_.empty() || bottomCorner_.size() != upperCorner_.size();
}

inline const corner_type &Box::get_bottom_corner() const
{
	return bottomCorner_;
}

inline const corner_type &Box::get_upper_corner() const
{
	return upperCorner_;
}

inline bool Box::contains(point_type &point) const
{
	if (point.size() != bottomCorner_.size()) return false;

	for (unsigned int i = 0; i < point.size(); i++){
		if (point.at(i) < bottomCorner_.at(i)) return false;
		if (point.at(i) > upperCorner_.at(i)) return false;
	}

	return true;
}

std::ostream& operator<<(std::ostream& os, const Box& box)
{
    os << "Box -- Bottom corner : ";
    os << box.get_bottom_corner();
    os << ", Top corner : ";
    os << box.get_upper_corner();
    return os;
}

}   //namespace Vineyard



#endif // BOX_H_INCLUDED
