#ifndef APPROXIMATION_H_INCLUDED
#define APPROXIMATION_H_INCLUDED


#include "dependences.h"
#include "vineyards.h"
#include "vineyards_trajectories.h"
#include "structure_higher_dim_barcode.h"


typedef pair< vector<vector<double>>, vector<vector<double>> > corner_list; ///< pair of birth corner list, and death corner list of a summand





// signature
void approximation_recursive_higher_dim(vector<vector<corner_list>> &output, VineyardsPersistence &persistence, const vector<double> &basepoint, const vector<uint> &position, uint last, vector<double> &filter, const vector<vector<double>> &filters_list,const double precision, const pair<vector<double>, vector<double>> &box, const vector<uint> &size, const bool threshold, const bool multithread);



double inf_distance(const vector<double> &a, const vector<double> &b){
	if(a.empty() || b.empty() || (a.size() != b.size()))
		return DBL_MAX;
	double d = abs(a[0] - b[0]);
	for(uint i =1; i< a.size(); i++)
		d = max(d,abs(a[i]-b[i]));
	return d;
}



template<typename T>
bool is_vector_empty(const vector<T> &v){
	return v.empty();
}

/**
 * @brief Removes empty vectors of @p list.
 *
 * @param list p_list:...
 */
void clean(vector<vector<double>> &list){ // TODO better use of erase ...
// 	remove_if(list.begin(), list.end(), is_vector_empty<double>);
	uint i=0;
	while(i<list.size()){
		while(!list.empty() && (*(list.rbegin())).empty())
			list.pop_back();
		if(i<list.size() && list[i].empty()){
			list[i].swap(*(list.rbegin()));
			list.pop_back();
		}
		i++;
	}
// 	while(i<list.size()){
// 		if(list[i].empty())
// 			list.erase(list.begin()+i);
// 		else
// 			i++;
// 	}

}



bool is_summand_empty(const corner_list &summand){
	return (summand.first.empty() || summand.second.empty());
}

/**
 * @brief Removes empty summands of @p output. WARNING Does permute the output.
 *
 * @param output p_output:...
 */
void clean(vector<vector<corner_list>> &output){
	for(uint dim=0; dim<output.size(); dim++){
// 		remove_if(output[dim].begin(), output[dim].end(), is_summand_empty);
		uint i =0;
		while(i < output[dim].size()){
			while(!output[dim].empty() && is_summand_empty(*output[dim].rbegin()))
				output[dim].pop_back();
			if(i >= output[dim].size())
				break;
			auto &summand = output[dim][i];
			if(is_summand_empty(summand)){
				summand.swap(*output[dim].rbegin());
				output[dim].pop_back();
			}
			i++;
		}
	}

// 	for(uint dim=0; dim< output.size(); dim++){
// 		uint i=0;
// 		while(i< output[dim].size()){
// 			if(output[dim][i].first.empty() && output[dim][i].second.empty())
// 				output[dim].erase(output[dim].begin()+i);
// 			else
// 				i++;
// 		}
// 	}
}



void factorize_min(vector<double> &a, vector<double> &b){
	if(a.size() != b.size())
		return;
	for(uint i=0; i< a.size(); i++)
		a[i] = min(a[i], b[i]);
}

void factorize_max(vector<double> &a, vector<double> &b){
	if(a.size() != b.size())
		return;
	for(uint i=0; i< a.size(); i++)
		a[i] = max(a[i], b[i]);
}

void complete_birth(vector<vector<double>> &birth_list, const double precision){
	if(birth_list.empty())
		return;
// 	#pragma omp parallel for simd
	for(uint i=0; i< birth_list.size(); i++){
		for(uint j=i+1; j< birth_list.size(); j++){
			double d = inf_distance(birth_list[i], birth_list[j]);
			if (d<=1.1*precision){
				factorize_min(birth_list[i], birth_list[j]);
				birth_list[j].clear();
				break;
			}
		}
	}
	clean(birth_list);
}
void complete_death(vector<vector<double>> &death_list, const double precision){
	if(death_list.empty())
		return;
// 	#pragma omp parallel for simd
	for(uint i=0; i< death_list.size(); i++){
		for(uint j=i+1; j< death_list.size(); j++){
			double d = inf_distance(death_list[i], death_list[j]);
			if (d<=1.1*precision){
				factorize_max(death_list[i], death_list[j]);
				death_list[j].clear();
				break;
			}
		}
	}
	clean(death_list);
}

void fill(vector<vector<corner_list>> &output, const double precision){
	if (output.empty())
		return;
// 	#pragma omp parallel for
	for(uint dim = 0; dim < output.size(); dim++){
		for(uint i =0; i <output[dim].size(); i++){
			auto &summand = output[dim][i];
			if(is_summand_empty(summand))
				continue;
			complete_birth(summand.first, precision);
			complete_death(summand.second, precision);
		}
	}
}








/**
 * @brief Adds @p birth to the summand's @p birth_list if it is not induced from the @p birth_list (ie. not comparable or smaller than another birth), and removes unnecessary birthpoints (ie. birthpoints that are induced by @p birth).
 *
 * @param birth_list p_birth_list: birthpoint list of a summand
 * @param birth p_birth: birth to add to the summand
 */
void add_birth_to_summand(vector<vector<double>> &birth_list, vector<double> &birth){
	if(birth_list.empty()){
		birth_list.push_back(birth);
		return;
	}
	if(birth_list.front().front() == -DBL_MAX)
		return;
	if (birth.front() == -DBL_MAX){ // when a birth is infinite, we store the summand like this
		birth_list = {{-DBL_MAX}};
		return;
	}
	bool is_useful = true;
// 	#pragma omp parallel for shared(is_useful)
	for(uint i=0; i<birth_list.size(); i++){
		if(!is_useful)
			continue;
		if(is_greater(birth, birth_list[i])) {
			is_useful=false;
// 			continue;
			break;
		}
		if(!birth_list[i].empty() && is_less(birth, birth_list[i])){
			birth_list[i].clear();
		}
	}

// 	for(auto birth_iterator = birth_list.begin(); birth_iterator != birth_list.end();){
// 		if(is_greater(birth, *birth_iterator))
// 			return;
// 		if(is_less(birth, *birth_iterator))
// 			birth_iterator = birth_list.erase(birth_iterator);
// 		else
// 			++birth_iterator;
// 	}
	clean(birth_list);
	if(is_useful)
		birth_list.push_back(birth);

	return;
}



/**
 * @brief Adds @p death to the summand's @p death_list if it is not induced from the @p death_list (ie. not comparable or greater than another death), and removes unnecessary deathpoints (ie. deathpoints that are induced by @p death)
 *
 * @param death_list p_death_list: List of deathpoints of a summand
 * @param death p_death: deathpoint to add to this list
 */
void add_death_to_summand(vector<vector<double>> &death_list, vector<double> &death){
	if( death_list.empty()){
		death_list.push_back( death );
		return;
	}
	if ( death_list.front().front() == DBL_MAX) // as drawn in a slope 1 line being equal to -\infty is the same as the first coordinate being equal to -\infty
		return;
	if ( death.front() == DBL_MAX){ // when a birth is infinite, we store the summand like this
		death_list = {{-DBL_MAX}};
		return;
	}
	bool is_useful = true;
// 	#pragma omp parallel for shared(is_useful)
	for(uint i=0; i<death_list.size(); i++){
		if(!is_useful)
			continue;
		if(is_less(death, death_list[i])) {
			is_useful=false;
// 			continue;
			break;
		}
		if(!death_list[i].empty() && is_greater(death, death_list[i])){
			death_list[i].clear();
		}
	}


// 	for(auto death_iterator = death_list.begin(); death_iterator != death_list.end();){
// 		if(is_less(death, *death_iterator))
// 			return;
// 		if(is_greater(death, *death_iterator))
// 			death_iterator = death_list.erase(death_iterator);
// 		else
// 			++death_iterator;
// 	}

	clean(death_list);
	if(is_useful)
		death_list.push_back(death);

	return;
}



/**
 * @brief Adds the bar @p bar to the indicator module @p summand if @p bar is non-trivial (ie. not reduced to a point or, if @p threshold is true, its thresholded version should not be reduced to a point) .
 *
 * @param bar p_bar: to add to the support of the summand
 * @param summand p_summand: indicator module which is being completed
 * @param basepoint p_basepoint: basepoint of the line of the bar
 * @param birth p_birth: birth container (for memory optimization purposes). Has to be of the size @p basepoint.size()+1.
 * @param death p_death: death container. Same purpose as @p birth but for deathpoint.
 * @param threshold p_threshold: If true, will threshold the bar with @p box.
 * @param box p_box: Only useful if @p threshold is set to true.
 */
void add_bar_to_summand(const pair<double,double> &bar, corner_list &summand, const vector<double> &basepoint, vector<double> &birth, vector<double> &death, const bool threshold, const interval &box){
	if(bar.first >= bar.second)
		return; // bar is trivial in that case
// 	#pragma omp simd
	for(uint j=0; j<birth.size()-1;j++){
		birth[j] = basepoint[j] + bar.first;
		death[j] = basepoint[j] + bar.second;
	}
	birth.back() = bar.first;
	death.back() = bar.second;
	if(threshold){
// 		#pragma omp parallel sections
		{
			{threshold_down(birth, box, basepoint);}
// 			#pragma omp section
			{threshold_up(death, box, basepoint);}
		}
		if(is_greater(birth,death))
			return;
	}
// 	#pragma omp parallel sections
	{
	{add_birth_to_summand(summand.first, birth);}
// 	#pragma omp section
	{add_death_to_summand(summand.second, death);}
	}

	return;
}

/**
 * @brief Recursive function of \ref approximation_vineyards. Computes what's on a line, adds the barcode to the module, and goes to the next line.
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
void approximation_recursive(vector<vector<corner_list>> &output, VineyardsPersistence &persistence, vector<double> &basepoint, vector<uint> &position, uint last, vector<double> &filter, const vector<vector<double>> &filters_list, double precision, const  interval &box, const vector<uint> &size_line, bool first = false, const bool threshold = false, const bool multithread=false){
	constexpr bool verbose = false;
	if(!first) line_to_filter(basepoint, filters_list, filter, true);
	if constexpr(verbose) disp_vect(basepoint);
	persistence.update(filter); // Updates the RU decomposition of persistence.
	persistence.get_diagram(); // Computes the diagram from the RU decomposition

	// Fills the barcode of the line having the basepoint basepoint
	uint feature = 0;
	int old_dim =0;

	vector<double> birth_container(filters_list.size());
	vector<double> death_container(filters_list.size());
	for(uint i=0; i<persistence.dgm.size(); i++){
		auto &bar = persistence.dgm[i].second;
		corner_list &summand = output[persistence.dgm[i].first][feature];
		add_bar_to_summand(bar, summand, basepoint, birth_container, death_container, threshold, box);

		// If next bar is of upper dimension, or we reached the end, then we update the pointer index of where to fill the next bar in output.
		if(i+1 < persistence.dgm.size() && old_dim < persistence.dgm[i+1].first) {
			old_dim = persistence.dgm[i+1].first;
			feature = 0;
		}
		else feature++;
		if constexpr (verbose)
			cout <<"Feature : " << feature << " dim : " << old_dim << endl;
	}

	approximation_recursive_higher_dim(output, persistence, basepoint, position, last, filter, filters_list, precision, box, size_line, threshold, multithread);

	//recursive calls of bigger dims, minus current dim (to make less copies)
	// We keep -last- on the same thread / memory as the previous call
	if (size_line[last]-1 == position[last]) return; // we reached a border and finished this path
	// If we didn't reached the end, go to the next line
	basepoint[last] += precision;
	position[last] ++;
	approximation_recursive(output, persistence, basepoint, position, last, filter,  filters_list, precision, box, size_line, false, threshold, multithread);
	return;
}






void approximation_recursive_higher_dim ( std::vector< std::vector< corner_list > >& output, VineyardsPersistence& persistence, const std::vector< double >& basepoint, const std::vector< uint >& position, uint last, std::vector< double >& filter, const std::vector< std::vector< double > >& filters_list, const double precision, const std::pair< std::vector< double >, std::vector< double > >& box, const std::vector< uint >& size, const bool threshold, const bool multithread )
{
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
				approximation_recursive(output, copy_persistence, copy_basepoint, copy_position, i,filter,  filters_list, precision, box, size,false, threshold, multithread);
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
				approximation_recursive(
					output, copy_persistence, copy_basepoint, copy_position, i,filter,
					filters_list, precision, box, size, false,threshold, multithread
				);
			}
		}
	}
}













/**
 * @brief Appproximate any multipersistence module with an interval decomposable module. If this module is interval decomposable, then the matching is controlled by the precision, and exact under specific circumstances (see TODO: cite paper).
 *
 * @param B p_B: Boundary matrix of the initial simplices.
 * @param filters_list p_filters_list: Filtration of the simplices
 * @param precision p_precision: wanted precision.
 * @param box p_box: Box on which to make the approximation
 * @param threshold p_threshold:... Defaults to false. If set to true, will intersect the computed summands with the box
 * @param complete
 * @param multithread ${p_multithread:...} Defaults to false. WIP, not useful yet.
 * @return std::vector< std::vector< corner_list > >
 */
vector<vector<corner_list>> approximation_vineyards(boundary_matrix &B, const vector<vector<double>> &filters_list, const double precision, const pair<vector<double>, vector<double>> &box, const bool threshold = false,const  bool complete=true, const bool multithread = false, const bool verbose = false){
	// Checks if dimensions are compatibles
	assert(!filters_list.empty() && "A non trivial filters list is needed !");

	assert(filters_list.size()  == box.first.size() && "Filters and box must be of the same dimension !");

// 	constexpr bool verbose = false; // For debug purposes

	const uint filtration_dimension = filters_list.size();
	if (verbose) cout << "Filtration dimension : " << filtration_dimension << flush <<  endl;

	uint number_simplices = B.size();
	if (verbose) cout << "Number of simplices : " << number_simplices << flush << endl;

	vector<int> simplicies_dimensions(number_simplices);

	get_dimensions_from_structure(B, simplicies_dimensions);

	bool lower_star = false;
	if(filters_list[0].size() < number_simplices ) lower_star = true;

	vector<double> filter(number_simplices); // container of filters

	vector<uint> size_line(filtration_dimension-1);
// 	#pragma omp simd
	for(uint i=0;i<filtration_dimension-1;i++)
		size_line[i] = (uint)(ceil((abs( (box.second[i] - box.first.back()) - (box.first[i] - box.second.back()) ) / precision)));


	uint number_of_line = prod(size_line);
	if (verbose) cout << "Precision : " << precision << endl;
	if (verbose) cout << "Number of lines : " << number_of_line << endl;

	auto basepoint = box.first;
// 	#pragma omp simd
	for(uint i=0; i<basepoint.size()-1; i++)
		basepoint[i] -= box.second.back();
	basepoint.back() = 0;

	line_to_filter(basepoint, filters_list, filter, true);
	vector<uint> position(filtration_dimension-1, 0); // where is the cursor in the output matrix

	VineyardsPersistence persistence(B, simplicies_dimensions, filter, lower_star, false, false);
	persistence.initialize_barcode(false, false);
	persistence.get_diagram();

	auto &first_barcode = persistence.dgm;
	uint max_dimension = first_barcode.back().first; // filtered by dimension so last one is of maximal dimension
	uint number_of_features = first_barcode.size();

	// Initialise size of the output.
	vector<vector<corner_list>> output(max_dimension+1);
	vector<uint> number_of_feature_of_dimension(max_dimension+1);
	for(uint i=0; i< number_of_features; i++){
		number_of_feature_of_dimension[first_barcode[i].first]++;
	}
// 	#pragma omp simd
	for(uint i=0; i<max_dimension+1;i++){
		output[i].resize(number_of_feature_of_dimension[i]);
	}

	auto elapsed = clock();
	if (verbose) cout <<"Multithreading status : " <<  multithread << endl;
	if (verbose) cout << "Starting recursive vineyard loop..." << flush;
	// Call the compute recursive function
	approximation_recursive(output, persistence, basepoint, position, 0, filter, filters_list,precision,box, size_line, true, threshold, multithread);

	elapsed = clock() - elapsed;
	if (verbose) cout << " Done ! It took "<< ((float)elapsed)/CLOCKS_PER_SEC << " seconds."<<endl;
	clean(output);
	if(complete)
		fill(output, precision);
	return output;
}

#endif // APPROXIMATION_H_INCLUDED
