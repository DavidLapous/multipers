/**
 * @file vineyards_trajectories.h
 * @author David Loiseaux
 * @brief This file contains the functions related to trajectories of barcodes via vineyards.
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */
#ifndef VINEYARDS_TRAJECTORIES_H_INCLUDED
#define VINEYARDS_TRAJECTORIES_H_INCLUDED

#include "vineyards.h"
#include "dependences.h"
#include "structure_higher_dim_barcode.h"
#include "format_python-cpp.h"


// Computes a vineyard update of a permutation
// Input :  vector<int> p permutation
//			RU decomposition RU.
// 			bool verbose : show logs.
// Outputs : vector<barcode> list of barcodes along permutations
// WARNING Does not work
pair<SparseBoundaryMatrix,SparseBoundaryMatrix> vineyard_update(permutation p, RU_decomposition& RU, bool verbose=false){
	vector<int> list_transpositions = coxeter(p);
	for (uint i = 0; i < list_transpositions.size(); i++){
		vineyard_update(list_transpositions[i], RU, verbose);
	}
	return RU;
}



// Computes a list of vineyard update, from a list of permutations
// Input :  vector< vector<int>>  [ permutation for permutation (on a line)], simplextree
//			RU decomposition RU.
//			bool reduced : specify if boundary_matrix is already reduced (and has a computed barcode).
//			bool swapped : specify if there have been row exchanges in boundary_matrix. Only useful if not reduced.
// 			bool verbose : show logs.
// Outputs : vector<barcode> list of barcodes along permutations
vector<barcode> vineyard_trajectory(vector<permutation> list_permutations, RU_decomposition RU, bool verbose=false) //TODO
{
	vector<barcode> list_barcodes={};
	list_barcodes.push_back(RU.first.bc);
	for(uint i=0; i<list_permutations.size(); i++){
		RU = vineyard_update(list_permutations[i], RU, verbose);
		list_barcodes.push_back(RU.first.bc);
	}
	return list_barcodes;
}


// Computes a list of vineyard update, from a list of permutations
// Outputs : vector<barcode> list of barcodes along permutations
vector<barcode> vineyard_trajectory(
	vector<permutation> list_permutations,  //[ permutation for permutation (on a line)]
	SparseBoundaryMatrix boundary_matrix,
	bool swapped=false, // specify if there have been row exchanges in boundary_matrix.
	bool verbose=false // show logs
)
{
	RU_decomposition RU = compute_barcode_sparse(boundary_matrix,swapped);
	return vineyard_trajectory(list_permutations, RU, verbose);
}







vector<barcode> vineyard_trajectory(vector<vector<double>> &filters_list,const boundary_matrix &matrix, bool lower_star = true, bool sorted = false, bool verbose = false){
	vector<barcode> barcode_list;
	uint n = matrix.size();
	vector<int> dimensions(n); for(uint i=0;i<n;i++) dimensions[i] = matrix[i].size();

	VineyardsPersistence persistence(matrix, dimensions, filters_list[0], lower_star, sorted, false);

	persistence.initialize_barcode();
	barcode_list.push_back(persistence.P.first.bc);
	for(uint i = 1; i<filters_list.size(); i++){
		persistence.update(filters_list[i]);
		barcode_list.push_back(persistence.P.first.bc);
	}
	return barcode_list;
}

// Tree<barcode> vineyard_trajectory(Tree<vector<double>> filters_list, boundary_matrix matrix, bool lower_star = true, bool sorted = false, bool verbose = false){
// 	vector<barcode> barcode_list;
// 	uint n = matrix.size();
// 	vector<int> dimensions(n); for(uint i=0;i<n;i++) dimensions[i] = matrix[i].size();
//
// 	VineyardsPersistence persistence(matrix, dimensions, filters_list[0], lower_star, sorted, false);
//
// 	persistence.initialize_barcode();
// 	barcode_list.push_back(persistence.P.first.bc);
// 	for(uint i = 1; i<filters_list.size(); i++){
// 		persistence.update(filters_list[i]);
// 		barcode_list.push_back(persistence.P.first.bc);
// 	}
// 	return barcode_list;
// }
//


//INPUT :
//	a slope 1 line is characterized by its intersection with {x_n=0} named line_basepoint.
//	filter_list is : for each coordinate i, and simplex j filter_list[i,j] is the filtration value of simplex j on line induced by [0,e_i]
//OUTPUT:
//	filtration value of simplex j on the line.
/**
 * @brief Writes the filters of each simplex on new_filter along the a slope 1 line.
 * 
 * @param line_basepoint Basepoint of a slope 1 line in \f$\mathbb R^n\f$ 
 * @param filter_list Multi-filtration of simplices. Format : [[filtration_value for simplex] for dimension]
 * @param new_filter Container of the output.
 * @param ignore_last Ignore this parameter. It is meant for compatibility with old functions.
 */
void line_to_filter(const vector<double>& line_basepoint, const vector<vector<double>>& filter_list, vector<double>& new_filter, bool ignore_last = false){
	const bool verbose = false;
	if(verbose) {
		disp_vect(line_basepoint);
	}
	uint dimension = line_basepoint.size()+1 - ignore_last;
	uint number_of_simplices = filter_list[0].size();
	assert(filter_list.size() == dimension);
// 	#pragma omp parallel for simd
	vector<double> relative_filtration_value(dimension);
	for(uint i=0; i< number_of_simplices; i++){
		for(uint j=0;j<dimension-1;j++){
			relative_filtration_value[j] = filter_list[j][i] - line_basepoint[j];
		}
		relative_filtration_value[dimension-1] = filter_list[dimension-1][i];
		double length = *max_element(relative_filtration_value.begin(), relative_filtration_value.end());

		new_filter[i] = length;
	}
	if(verbose) disp_vect(new_filter);
}

// in dim 2, we only have to specify the coordinate x, allow double instead of vector<double>.
/**
 * @brief Writes the filters of each simplex on new_filter along the a slope 1 line in dimension 2.
 * 
 * @param line_basepoint Basepoint of a slope 1 line in \f$\mathbb R^n\f$ 
 * @param filter_list Multi-filtration of simplices. Format : [[filtration_value for simplex] for dimension]
 * @param new_filter Container of the output.
 */
void line_to_filter(double line_basepoint, const pair<vector<double>,vector<double>> &filter_list, vector<double>& new_filter){
	vector<double> temp = {line_basepoint};
	vector<vector<double>> temp2 = {filter_list.first,filter_list.second};
	line_to_filter( temp, temp2,new_filter, false);
}


// Taken from https://gist.github.com/lorenzoriano/5414671
template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N-1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}





// Generates besepoints of lines from basepoint to basepoint+range. basepoint and range should be of dimension n-1 where n is the dimension.
// INPUT:
// 			basepoint :	basepoint of the first line
// 			range :		range in each coordinate
// 			precision :	distance between 2 lines
//WARNING : not usable in higher dim
// Tree<vector<double>> lines_generator(vector<double> basepoint, vector<double> range, double precision, bool verbose = false){
// 	uint i = range.size(); // dimension of the plane where we generate the lines
// // 	uint dimension = basepoint.size()+1;
// 	assert(i < basepoint.size()+1);
// 	if(i ==0) return Tree<vector<double>>(basepoint,{});
//
// 	list<Tree<vector<double>>> childs ={};
//
// 	double cursor = basepoint[i-1] + precision;
//
// 	while(cursor <= basepoint[i-1] + range.back()){
// 		vector<double> child_node = basepoint;
// 		child_node[i-1] = cursor;
// 		vector<double> child_range(i-1);
// 		if(i>1)  for(uint j =0;j<i-1;j++) child_range[j] = range[j];
// 		childs.push_back(lines_generator(child_node, child_range, precision));
// 		cursor += precision;
// 	}
// 	return Tree<vector<double>>(basepoint, childs);
// }

// same, but with an iterator (for memory efficiency)
// OUTPUT :
//  false if we hit the range
// WARNING : this iterator doesn't go through all possible lines !
bool next_line(vector<double>& basepoint, list<double>& range, double precision, bool debug = false){
	uint i = range.size();
// 	uint dimension = basepoint.size()+1;

	if(debug) assert(i < basepoint.size()+1);
	if(range.empty()) return false;

	if(range.back()>=precision){
		basepoint[i-1]+= precision;
		range.back()-=precision;
		return true;
	}
	range.pop_back();

	return false;
}





barcoded ids_to_filtration(const barcode &bc, const vector<double> &filters){
	uint n = bc.size();
	vector<pair<int,pair<double,double>>> bcd(n);
	for(uint i=0; i<n;i++){
		double birth_time = filters[bc[i].second.first];
		double death_time = filters[bc[i].second.second];
		if(bc[i].second.second == -1) death_time = DBL_MAX;
		pair<double,double> interval = {birth_time,death_time};
		bcd[i] = {bc[i].first, interval};
	}
	return bcd;
}

// vector<barcoded> compute_vineyard_2d(vector<vector<vector<int>>> &B, const pair<vector<double>,vector<double>> &filters_list, double basepoint, double range, double precision){
// // 	uint number_simplices = B.size();
// // 	vector<int> dimensions(number_simplices);
// // 	get_dimensions_from_structure(B, dimensions);
// // 	sparsify_boundary_matrix(B, dimensions);
// 	return compute_vineyard_2d(sparsify_boundary_matrix(B), filters_list, basepoint, range, precision);
//
// }



//TODO
//INPUT:
// simplextree as a sparse boundary matrix
// filters [[filtration on e_j] for basis element e_j]
// basepoint is the intersection of the first line with {y=0}
// precision is the distance between two lines
// spanning lines from basepoint to basepoint + range
//OUTPUT:
// [[feature_dim, [filtration_birth,filtration_death of paired simplices]] for line]
vector<barcoded> compute_vineyard_2d(boundary_matrix &B, const pair<vector<double>,vector<double>> &filters_list, double basepoint, double range, double precision){
	const bool verbose = true;

	uint number_simplices = B.size();
	vector<int> dimensions(number_simplices);
	get_dimensions_from_structure(B, dimensions);


	bool lower_star = false;
	if(filters_list.first.size() < number_simplices ) lower_star = true;
	vector<double> filter(number_simplices); // container of filters
	line_to_filter(basepoint, filters_list, filter);

	VineyardsPersistence persistence(B, dimensions, filter, lower_star, false, false);

	persistence.initialize_barcode();
	persistence.get_diagram();
	uint n = (uint)floor(range / precision)+1;

	vector<barcoded> barcodes(n);
// 	barcodes[0] = ids_to_filtration(persistence.P.first.bc, filter);
	barcodes[0].swap(persistence.dgm);


	// Formatting. Give vectors instead of pairs / double to avoid making copies
	vector<double> basepoint_v ={basepoint};
	vector<vector<double>> filters_list_v = {filters_list.first, filters_list.second};

	if(verbose) cout << "Updating barcode ..." << endl;
	auto elapsed = clock();

	for(uint i=1;i<n;i++){
		if(verbose) cout << "Line " << i << " over " << n << "..." << "\r";
		basepoint_v[0]+=precision;
		line_to_filter(basepoint_v,filters_list_v, filter);
		persistence.update(filter);
		persistence.get_diagram();
		barcodes[i].swap(persistence.dgm);
// 		barcodes[i] = ids_to_filtration(persistence.P.first.bc, new_filter);
	}
	elapsed = clock() - elapsed;
	if (verbose) cout << "Line " << n << " over " << n << "... Done ! It took "<< ((float)elapsed)/CLOCKS_PER_SEC << " seconds."<<endl;
	return barcodes;
}





bool is_less(const pair<double,double> &x, const pair<double,double> &y){
	return (x.first <= y.first && x.second <= y.second);
}


bool is_less(const vector<double> &x, const vector<double> &y){
	for(uint i=0; i<min(x.size(), y.size()); i++)
		if(x[i]>y[i]) return false;
	return true;
}
bool is_greater(const vector<double> &x, const vector<double> &y){
	for(uint i=0; i<min(x.size(), y.size()); i++)
		if(x[i]<y[i]) return false;
	return true;
}

bool is_greater(pair<double,double> x, pair<double,double> y){
	return (x.first >= y.first && x.second >= y.second);
}

bool is_in_box(point_2 point, interval_2 box){
	return is_greater(point, box.first) && is_less(point, box.second);
}

bool is_comparable_to_box(point_2 point, interval_2 box){
	return (is_greater(point, box.first) || is_less(point, box.second));
}



void threshold_up(pair<double,double>& point, interval_2 box, double basepoint = DBL_MIN){
	pair<double,double> x = box.first;
	pair<double,double> y = box.second;
	pair<double,double> z = {basepoint + max(box.second.first, box.second.second) + abs(basepoint) ,max(box.second.first, box.second.second) + abs(basepoint)};
	double threshold = min(max(z.first - y.first, z.second - y.second), min(z.first -x.first, z.second-x.second));
	point = {z.first - threshold, z.second - threshold};
}

void threshold_down(pair<double,double>& point, interval_2 box, double basepoint = DBL_MIN){
	pair<double,double> x = box.first;
	pair<double,double> y = box.second;
	pair<double,double> z = {basepoint - min(box.first.first, box.first.second) - abs(basepoint) ,-min(box.first.first, box.first.second) - abs(basepoint)};
	double threshold = min(max(x.first - z.first, x.second - z.second), min(y.first -z.first, y.second-z.second));
	point = {z.first - threshold, z.second - threshold};
}

// basepoint in necessary to handle essential features.
void threshold_box(pair<double,double>& point, const interval_2& box, double basepoint = DBL_MIN, bool verbose=false){
	if(basepoint == DBL_MIN) basepoint = point.first - point.second;
	if(basepoint < (box.first.first -box.second.second) || basepoint > box.second.first - box.first.second){
		point = {DBL_MAX,DBL_MAX};
		return;
	}
	if(is_in_box(point, box)) { // point is in the box
		if(verbose) cout << "Point in the box"<< endl;
		return ;
	}
	if(is_greater(point, box.first)){
		threshold_up(point, box, basepoint);
		return;
	}

	if(is_less(point, box.second)){
		threshold_down(point, box, basepoint);
		return;
	}
}


/**
 * @brief Threshold a point to the negative cone of d=box.second (ie. the set \f$\{x \in \mathbb R^n \mid x \le d\} \f$) along the slope 1 line crossing this point.
 * 
 * @param point The point to threshold.
 * @param box box.second is the point defining where to threshold.
 * @param basepoint Basepoint of the slope 1 line crossing the point. Meant to handle infinite cases (when the point have infinite coordinates, we cannot infer the line).
 */
void threshold_up(vector<double> &point, const interval &box, const vector<double> &basepoint = vector<double>(1, DBL_MIN)){
	const bool verbose = false;
	if(is_less(point, box.second)) return;
	if constexpr (verbose) disp_vect(point);
	if(basepoint[0] == DBL_MIN) return;

	if( point.back() == DBL_MAX){ // ie. point at infinity, assumes [basepoint,0] is smaller than box.second
		if(verbose) cout << " Infinite point" << endl;
		double threshold = box.second.back();
		for(uint i=0; i< point.size(); i++){
			threshold = min(threshold, box.second[i] - basepoint[i]);
		}
		for(uint i=0; i<point.size()-1; i++) point[i] = basepoint[i] + threshold;
		point.back() = threshold;
		return;
	}

	if(!is_greater(point, box.first)) {
		point[0] = DBL_MAX;// puts point to infinity
		if constexpr (verbose) cout << "buggy point" << endl;
		return;
	}
	//in this last case, at least 1 coord of point is is_greater than a coord of box.second

	double threshold = point[0] - box.second[0];
	for(uint i=1; i< point.size(); i++){
		threshold = max(threshold, point[i]- box.second[i]);
	}
	if constexpr (verbose) cout << "Thresholding the point with "<< threshold << " at ";
	for(uint i=0; i<point.size(); i++) point[i] -= threshold;
	if constexpr (verbose) disp_vect(point);
}

/**
 * @brief Threshold a point to the positive cone of b=box.first (ie. the set \f$\{x \in \mathbb R^n \mid x \ge b\}) along the slope 1 line crossing this point.
 * 
 * @param point The point to threshold.
 * @param box box.fist is the point defining where to threshold.
 * @param basepoint Basepoint of the slope 1 line crossing the point. Meant to handle infinite cases (when the point have infinite coordinates, we cannot infer the line).
 */
void threshold_down(vector<double> &point, const interval &box, const vector<double> &basepoint = vector<double>(1, DBL_MIN)){
	if(basepoint[0] == DBL_MIN) return;

	if( point.back() == DBL_MAX){ // ie. point at infinity -> feature never appearing
		return;
	}

	if (is_greater(point, box.first)) return;
	if(!is_less(point, box.second)) {
		point[0] = DBL_MAX;// puts point to infinity
		return;
	}


	double threshold =  box.first[0] - point[0];
	for(uint i=1; i< point.size(); i++){
		threshold = max(threshold, box.first[i] - point[i]);
	}
// 	#pragma omp simd
	for(uint i=0; i<point.size(); i++) point[i] += threshold;

}



bool is_not_greater(const pair<double,double>& y,const pair<double,double>& x){
	return ((y.first <= x.first) || (y.second <= x.second));
}

// for python plotting.
//OUTPUT:
//			[[[[<birthpoint,deathpoint> thresholded by the box] over line] over matching] over dimension]
vector<vector<vector<interval_2>>> vineyard_2d(
	boundary_matrix& B,
	const pair<vector<double>,vector<double>>& filters_list,
	double basepoint,
	double range,
	double precision,
	const pair<pair<double,double>,pair<double,double>> &box = {{0,0},{0,0}},
	bool verbose = true,
	bool debug = false
){

	// computes vineyard
	vector<barcoded> barcodes = compute_vineyard_2d(B,filters_list, basepoint, range, precision);
	if(verbose) cout << "Formatting output..."<<flush;
	if(barcodes.empty()) return {};
	if(verbose && debug) disp(barcodes);
	uint number_of_line = barcodes.size();
	uint number_of_features = barcodes[0].size();
	uint max_dimension = barcodes[0].back().first;

	vector<uint> number_of_feature_of_dimension(max_dimension+1, 0);
	for(uint i =0; i<number_of_features;i++){
		number_of_feature_of_dimension[barcodes[0][i].first]++;
	}
	if(verbose && debug) {
		cout << "Number of feature of dimension : " ;
		disp_vect(number_of_feature_of_dimension);
		cout << "Number of lines : "<< number_of_line <<endl;
	}

	//initialization of output
	vector<vector<vector<interval_2>>> scaled_barcode(max_dimension+1);
	for(uint i=0; i<max_dimension+1;i++){
		scaled_barcode[i]=vector<vector<pair<pair<double,double>,pair<double,double>>>>(number_of_feature_of_dimension[i], vector<pair<pair<double,double>,pair<double,double>>>(number_of_line));
	}

	// Generates the output in the expected format
	bool threshold = !is_not_greater(box.second, box.first);//ie. box non trivial and we want to threshold.
	for(uint feature=0; feature < number_of_features; feature++){
		uint feature_dim = barcodes[0][feature].first;
		for (uint line=0;line<number_of_line; line++){
			auto bar = barcodes[line][feature].second;

			if(verbose && debug) {
				disp(bar);
				cout << endl;
			}
			pair<double,double> birth = {basepoint + line * precision + bar.first  , bar.first};
			pair<double,double> death = {basepoint + line * precision + bar.second , bar.second};
			if(verbose && debug){
				disp(birth);cout<<" "; disp(death);
			}
			if(threshold){
				threshold_box(birth, box, basepoint + line * precision);
				threshold_box(death, box, basepoint + line * precision);
				if(verbose && debug){disp(birth);cout<<" "; disp(death);}
			}
			if(verbose && debug) cout << endl;

			uint id = feature;
			for (uint i=0;i<feature_dim;i++){
				id-=number_of_feature_of_dimension[i];
			}
			scaled_barcode[feature_dim][id][line] ={birth,death};
		}
	}
	if(verbose) cout <<" Done !"<< endl;
	return scaled_barcode;
}
vector<vector<vector<interval_2>>> vineyard_2d(
	boundary_matrix &B,
	const pair<vector<double>,vector<double>> &filters_list,
	double precision,
	const pair<pair<double,double>,pair<double,double>> &box = {{0,0},{0,0}},
	bool verbose = true,
	bool debug = false
){
	double basepoint = box.first.first - box.first.second - box.second.second;
	double range = (box.second.first - box.first.first) +  (box.first.second + box.second.second) + precision;
	return vineyard_2d(B,filters_list, basepoint, range, precision, box, verbose,debug);
}

/*
INPUT :
	persistence is the current persistence
	basepoint is the current basepoint of the line
	precision is the distance between consecutive lines
	position is the position in the Matrix_nd
	last is the first nonzero coord of position
	filter_list is the list of filter of each simplice in each direction
	output is the matrix on which we write
	filter is the variable containing the current filter
	size is the size of the matrix output
	first is telling if it is the starting line

*/
template<uint dimension>
void vineyard_recursive(VineyardsPersistence &persistence, vector<double> &basepoint, double precision, vector<uint> &position, uint last, const vector<vector<double>> &filters_list, vector<vector<Matrix_nd<dimension-1, interval>>> &output, vector<double> &filter, const vector<uint> &size, bool first = false){
	if(!first) line_to_filter(basepoint, filters_list, filter);
// 	disp_vect(position);
	persistence.update(filter);
	persistence.get_diagram();
	for(uint feature=0; feature<persistence.dgm.size(); feature++){
		// [[[bar of feature, of dimensionon line  for line] for feature] for dimension]
		output[persistence.dgm[feature].first][feature].set(position, persistence.dgm[feature].second);
	}

	// bigger dims
// 	#pragma omp parallel for
	for(uint i=last+1; i<dimension-1;i++){
		cout << "loop "<< i << endl;
		if (size[i]-1 == position[i]) continue;
		auto copy_persistence = persistence;
		auto copy_basepoint = basepoint;
		auto copy_position = position;
		copy_basepoint[i] += precision;
		copy_position[i] ++;
		vineyard_recursive(copy_persistence, copy_basepoint, precision, copy_position, i, filters_list, output, filter, size, false);
	}

	// We keep -last- on the same thread / memory as the previous call
	if (size[last]-1 == position[last]) return; // we reached a border and finished this path
	basepoint[last] += precision;
	position[last] ++;
	cout << "recursion" << endl;
	vineyard_recursive(persistence, basepoint, precision, position, last, filters_list, output, filter, size, false);
	return;
}

/*
INPUT :

OUTPUT :
	[ [ [[bar of feature of dimension, matching on line for line]] for matching] for dimension]

*/

template<uint dimension>
vector<vector<Matrix_nd<dimension-1, interval>>> vineyard(boundary_matrix &B, const vector<vector<double>> &filters_list, double precision, const pair<vector<double>, vector<double>> &box, bool threshold = false, bool verbose = true, bool debug = false){

	uint number_simplices = B.size();
	vector<int> simplicies_dimensions(number_simplices);
	get_dimensions_from_structure(B, simplicies_dimensions);

	bool lower_star = false;
	if(filters_list[0].size() < number_simplices ) lower_star = true;
	vector<double> filter(number_simplices); // container of filters
	line_to_filter(box.first, filters_list, filter);

	VineyardsPersistence persistence(B, simplicies_dimensions, filter, lower_star, false, false);
	persistence.initialize_barcode();
	persistence.get_diagram();

	auto &first_barcode = persistence.dgm;
	uint max_dimension = first_barcode.back().first; // filtered by dimension so last one is of maximal dimension
	uint number_of_features = first_barcode.size();

	vector<uint> number_of_feature_of_dimension(max_dimension+1);
	for(uint i=0; i< number_of_features; i++){
		number_of_feature_of_dimension[first_barcode[i].first]++;
	}

	const uint filtration_dimension = filters_list.size();
	vector<uint> size(filtration_dimension-1);
	for(uint i=0;i<filtration_dimension-1;i++) size[i] = (uint)(ceil(abs(box.second[i] - box.first[i]) / precision));

	vector<vector<Matrix_nd<dimension-1, interval>>> output(max_dimension +1);
	for(uint dim=0;dim<=max_dimension; dim++){
		output[dim] = vector<Matrix_nd<dimension-1, interval>>(number_of_feature_of_dimension[dim], Matrix_nd<dimension-1, interval>(size));
	}

	assert(dimension+1 == filtration_dimension && "Filtration needs to be of dimension " && dimension);
	auto &basepoint = box.first;
	vector<uint> position(filtration_dimension-1, 0); // where is the cursor in the output matrix

	vineyard_recursive<dimension-1>(persistence, basepoint, precision, position, 0, filters_list, output, filter, size, true);

	return output;
}




// signature
void vineyard_higher_dim_recursive(vector<vector<vector<interval>>> &output, VineyardsPersistence &persistence, const vector<double> &basepoint, const vector<uint> &position, uint last, vector<double> &filter, const vector<vector<double>> &filters_list,const double precision, const pair<vector<double>, vector<double>> &box, const vector<uint> &size, bool threshold, bool multithread);




// This is the core compute function of vineyard_alt.
// It updates and store in `output` the barcodes of a line, and calls itself on the next line until reaching the borders of the box
// INPUT :
// 			output : Where to store the barcodes.
// 			persistence : holding previous barcode information.
// 			basepoint : basepoint of the current line on the hyperplane {x_n=0}.
// 			position : index pointer of where to fill the output.
// 			last : which dimensions needs to be increased on this trajectory (for recursive trajectories).
// 			filter : container for filer of simplices.
// 			filters_list : holding the filtration value of each simplices. Format : [[filtration of simplex s in the kth filtration for s] for k].
// 			precision : line grid scale (ie. distance between two consecutive lines).
// 			box : [min, max] where min and max are points of R^n, and n is the dimension of the filter list.
// 				All of the bars along a line crossing this box will be computed.
// 			size : size of the output matrix.
// 			first : true if it is the first barcode. In that case we don't need to call a vineyard update.
// 			threshold : if true, intersects bars with the box.
// 			multithread : if set to true, will compute the trajectories in parallel.
// 							This is a WIP; as this imply more memory operations, this is rarely significantly faster than the other implementation.
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
void vineyard_alt_recursive(vector<vector<vector<interval>>> &output, VineyardsPersistence &persistence, vector<double> &basepoint, vector<uint> &position, uint last, vector<double> &filter, const vector<vector<double>> &filters_list, const double precision, const pair<vector<double>, vector<double>> &box, const vector<uint> &size,  bool first = false, bool threshold = false, bool multithread = false){
	constexpr bool verbose = false;
	if(!first) line_to_filter(basepoint, filters_list, filter, true);
	if constexpr(verbose) disp_vect(basepoint);
	persistence.update(filter); // Updates the RU decomposition of persistence.
	persistence.get_diagram(); // Computes the diagram from the RU decomposition

	// Fills the barcode of the line having the basepoint basepoint
	uint feature = 0;
	int old_dim =0;
// 	vector<double> birth(filters_list.size());
// 	vector<double> death(filters_list.size());
//	%TODO parallelize this loop, last part is not compatible yet
	for(uint i=0; i<persistence.dgm.size(); i++){
		auto &bar = persistence.dgm[i].second;

		uint indice = position_size_to_indice(position, size);
		vector<double> &birth = output[persistence.dgm[i].first][feature][indice].first;
		vector<double> &death = output[persistence.dgm[i].first][feature][indice].second;

		// If the bar is trivial, we skip it
		if(bar.first == DBL_MAX || bar.first == bar.second) goto skip_trivial_label;

		birth.resize(filters_list.size());
		death.resize(filters_list.size());

		// computes birth and death point from the bar and the basepoint of the line
// 		#pragma omp simd
		for(uint j=0; j<filters_list.size()-1;j++){
			birth[j] = basepoint[j] + bar.first;
			death[j] = basepoint[j] + bar.second;
		}
		birth.back() = bar.first;
		death.back() = bar.second;

		// Threshold birth and death if threshold is set to true
		if(threshold && birth.back() != DBL_MAX){
			threshold_down(birth, box, basepoint);
			threshold_up(death, box, basepoint);
		}
		if constexpr (verbose) {
			cout << birth.back() << " " << death.back();
			if(threshold) cout << ", threshold" << endl;
			else cout << ", no threshold" << endl;
		}
		// If this threshold has turned this bar to a trivial bar, we skip it
		if(birth.back() >= death.back()){
// 			goto skip_trivial_label;
			birth = {};
			death = {};
		}

		// Fills this bar to the output
		// [[[bar of feature, of dimension on line  for line] for feature] for dimension]
// 		output[persistence.dgm[i].first][feature][position_size_to_indice(position, size)] = {birth, death};

// 		birth_addr = birth;
// 		death_addr = death;

		skip_trivial_label:
		// If next bar is of upper dimension, or we reached the end, then we update the pointer index of where to fill the next bar in output.
		if(i+1 < persistence.dgm.size() && old_dim < persistence.dgm[i+1].first) {
			old_dim = persistence.dgm[i+1].first;
			feature = 0;
		}
		else feature++;
		if constexpr (verbose)
			cout <<"Feature : " << feature << " dim : " << old_dim << endl;
	}

	//recursive calls of bigger dims, minus current dim (to make less copies)
	vineyard_higher_dim_recursive(output, persistence, basepoint, position, last, filter, filters_list, precision, box, size, threshold, multithread);
	// We keep -last- on the same thread / memory as the previous call
	if (size[last]-1 == position[last]) return; // we reached a border and finished this path
	// If we didn't reached the end, go to the next line
	basepoint[last] += precision;
	position[last] ++;
	vineyard_alt_recursive(output, persistence, basepoint, position, last, filter,  filters_list, precision, box, size, false, threshold, multithread);
	return;
}




// For persistence dimension higher than 3, this function will be called for Tree-like recursion of vineyard_alt.
/**
 * @brief Subfonction of \ref vinyard_alt_recursive to handle dimensions greater than 3.
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
void vineyard_higher_dim_recursive(vector<vector<vector<interval>>> &output, VineyardsPersistence &persistence, const vector<double> &basepoint, const vector<uint> &position, uint last, vector<double> &filter, const vector<vector<double>> &filters_list,const double precision, const pair<vector<double>, vector<double>> &box, const vector<uint> &size, bool threshold = false, bool multithread = false){
// 	const static bool  multithread = false;
	constexpr bool verbose = false;
	if(filters_list.size()>1 && last +2 < filters_list.size()){
		if constexpr(verbose) disp_vect(basepoint);
		if constexpr(verbose) cout << multithread<< endl;
		if(multithread){
			#pragma omp parallel for
			for(uint i=last+1; i<filters_list.size()-1;i++){
				if (size[i]-1 == position[i]) continue;
				auto copy_persistence = persistence; //TODO check if it get deleted at each loop !! WARNING
				auto copy_basepoint = basepoint;
				auto copy_position = position;
				copy_basepoint[i] += precision;
				copy_position[i] ++;
				vineyard_alt_recursive(output, copy_persistence, copy_basepoint, copy_position, i,filter,  filters_list, precision, box, size,false, threshold, multithread);
			}
		}
		else{
			auto copy_persistence = persistence; // No need to copy when not multithreaded. Memory operations are slower than vineyard. %TODO improve trajectory of vineyard
			auto copy_basepoint = basepoint;
			auto copy_position = position;
			for(uint i=last+1; i<filters_list.size()-1;i++){
				if (size[i]-1 == position[i]) continue;
				copy_persistence = persistence;
				copy_basepoint = basepoint;
				copy_position = position;
				copy_basepoint[i] += precision;
				copy_position[i] ++;
				vineyard_alt_recursive(
					output, copy_persistence, copy_basepoint, copy_position, i,filter,
					filters_list, precision, box, size, false, threshold, multithread
				);
			}
		}
	}
}



// TODO improve multithread
// Main function of vineyard computation. It computes the fibered barcodes of any multipersistence module, with exact matching.
// Input :
//			B : sparse boundary matrix which is the converted simplextree by functions of format_python_cpp
// 			filters_list : [[filtration of dimension i for simplex s for s] for i] is the list of filters of each simplex of each filtration dimension
// 			precision : size of the line grid (ie. distance between 2 lines)
// 			box : [min, max] where min and max are points of R^n, and n is the dimension of the filter list.
// 				All of the bars along a line crossing this box will be computed
// 			threshold : If set to true, will intersect the bars with the box. Useful for plots / images
// 			multithread : if set to true, will compute the trajectories in parallel.
// 							This is a WIP; as this imply more memory operations, this is rarely significantly faster than the other implementation.
// OUTPUT :
// 			[[[(birth,death) for line] for summand] for dimension]
/**
 * @brief Main function of vineyard computation. It computes the fibered barcodes of any multipersistence module, with exact matching. 
 * 
 * @param B Sparse boundary matrix of a chain complex 
 * @param filters_list associated filtration of @p B Format :  [[filtration of dimension i for simplex s for s] for i]
 * @param precision  precision of the line grid ie. distance between two lines
 * @param box [min, max] where min and max are points of \f$ \mathbb R^n \f$, and n is the dimension of the filter list. 
 * All of the bars along a line crossing this box will be computed
 * @param threshold if set to true, will threshold the barcodes with the box
 * @param multithread if set to true, will turn on the multithread flags of the code (WIP)
 * @return vector<vector<vector<interval>>> List of barcodes along the lines intersecting the box. Format : [[[(birth,death) for line] for summand] for dimension]
 */
vector<vector<vector<interval>>> vineyard_alt(boundary_matrix &B, const vector<vector<double>> &filters_list, double precision, pair<vector<double>, vector<double>> &box, bool threshold = false, bool multithread = false){
	// Checks if dimensions are compatibles
	assert(!filters_list.empty() && "A non trivial filters list is needed !");
	assert(filters_list[0].size()  == box.first.size() == box.second.size() && "Filtration and box must be of the same dimension");

	constexpr bool verbose = false; // For debug purposes

	const uint filtration_dimension = filters_list.size();
	if constexpr(verbose) cout << "Filtration dimension : " << filtration_dimension << flush << endl;

	uint number_simplices = B.size();
	if constexpr(verbose) cout << "Number of simplices : " << number_simplices << endl;

	vector<int> simplicies_dimensions(number_simplices);

	get_dimensions_from_structure(B, simplicies_dimensions);

	bool lower_star = false;
	if(filters_list[0].size() < number_simplices ) lower_star = true;

	vector<double> filter(number_simplices); // container of filters

	vector<uint> size_line(filtration_dimension-1);
	for(uint i=0;i<filtration_dimension-1;i++)
		size_line[i] = (uint)(ceil((abs( (box.second[i] - box.first.back()) - (box.first[i] - box.second.back()) ) / precision)));


	uint number_of_line = prod(size_line);
	if constexpr(verbose) cout << "Precision : " << precision << endl;
	if constexpr(verbose) cout << "Number of lines : " << number_of_line << endl;

	auto &basepoint = box.first;
	for(uint i=0; i<basepoint.size()-1; i++) basepoint[i] -= box.second.back();
	basepoint.back() = 0;

	line_to_filter(basepoint, filters_list, filter, true);
	vector<uint> position(filtration_dimension-1, 0); // where is the cursor in the output matrix

	VineyardsPersistence persistence(B, simplicies_dimensions, filter, lower_star, false, false);
	persistence.initialize_barcode(verbose, false);
	persistence.get_diagram();

	auto &first_barcode = persistence.dgm;
	uint max_dimension = first_barcode.back().first; // filtered by dimension so last one is of maximal dimension
	uint number_of_features = first_barcode.size();
	vector<vector<vector<interval>>> output(max_dimension+1);

	vector<uint> number_of_feature_of_dimension(max_dimension+1);
	for(uint i=0; i< number_of_features; i++){
		number_of_feature_of_dimension[first_barcode[i].first]++;
	}

// 	#pragma omp parallel for
	for(uint i=0; i<max_dimension+1;i++){
		output[i] = vector<vector<interval>>(number_of_feature_of_dimension[i],vector<interval>(number_of_line
// 															  ,pair<vector<double>,vector<double>>())
		));
	}

	auto elapsed = clock();
	if constexpr(verbose) cout <<"Multithreading status : " <<  multithread << endl;
	if constexpr(verbose) cout << "Starting recursive vineyard loop..." << flush;
	vineyard_alt_recursive(output, persistence, basepoint, position, 0, filter, filters_list,precision,box, size_line, true, threshold, multithread);
	elapsed = clock() - elapsed;
	if constexpr(verbose) cout << " Done ! It took "<< ((float)elapsed)/CLOCKS_PER_SEC << " seconds."<<endl;
	return output;
}


// Same as vineyard_alt but only returns 1 dimension
// TODO : reduce computation by only computing this dimension instead of all of them
/**
 * @brief Returns only 1 dimension of the \ref vineyard_alt code. 
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
vector<vector<interval>> vineyard_alt_dim(boundary_matrix &B, const vector<vector<double>> &filters_list, double precision, pair<vector<double>, vector<double>> &box, uint dimension, bool threshold = false, bool verbose = true, bool debug = false){
	return vineyard_alt(B,filters_list, precision,box, threshold)[dimension];
}









#endif // VINEYARDS_TRAJECTORIES_H_INCLUDED
