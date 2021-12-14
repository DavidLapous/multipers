/**
 * @file combinatory.h
 * @author David Loiseaux
 * @brief Combinatorial and sorting functions
 * @copyright Copyright (c) 2021 Inria
 * 
 */

#ifndef COMBINATORY_H_INCLUDED
#define COMBINATORY_H_INCLUDED


#include "dependences.h"
#include "debug.h"
#include <bits/stdc++.h>


void swap(int* a, int* b){
    int t = *a;
    *a = *b;
    *b = t;
}
void swap(double* a, double* b){
    double t = *a;
    *a = *b;
    *b = t;
}

bool any(const vector<bool>& v){
	for( uint i=0; i< v.size(); i++){
		if (v[i]) return true;
	}
	return false;
}


bool all(const vector<bool>& v){
	for( uint i=0; i< v.size(); i++){
		if (!v[i]) return false;
	}
	return true;
}

int first(const vector<bool>& v, bool sign=true){
	for(uint i=0; i< v.size(); i++){
		if (v[i] == sign) return i;
	}
	return -1;
}

void add_t(vector<int>& p, int t){
	for(uint i=0; i<p.size();i++){
		p[i]+=t;
	}
}


vector<vector<int>> disjoint_cycles(const permutation &p){
// 	if (starts_from_1) add_t(p,-1);
	vector<vector<int>> s = {};
	vector<bool> passed_by(p.size(), false);
	int j0=0;
	while(j0 > -1){
		vector<int> cycle={};
		passed_by[j0] = true;
		cycle.push_back(j0);
		int j=j0;
		while(p[j] != j0){
			cycle.push_back(p[j]);
			passed_by[p[j]] = true;
			j = p[j];
		}
// 		if (starts_from_1) add_t(cycle, 1);
		if(cycle.size()>1) s.push_back(cycle);
		j0 = first(passed_by, false);
	};
	return s;
}

vector<pair<int,int>> permutation_to_transpositions(const permutation &s){
	vector<vector<int>> cycles = disjoint_cycles(s);
	vector<pair<int,int>> list_transpositions = {};
	for(const auto& cycle : cycles){
		if(cycle.size()<=1) continue;
		for( uint i =0; i<cycle.size()-1; i++){
			list_transpositions.push_back({cycle[i],cycle[i+1]});
		}
	}
	return list_transpositions;
}


bool check_permutation_to_transpositions(const permutation& p, bool starts_from_1=false){
	auto list_transpositions = permutation_to_transpositions(p);
	uint n= p.size();
	for(uint i=0; i<n;i++){
		int j=i;
		for(int k = list_transpositions.size()-1; k>=0;k--){
			if(list_transpositions[k].first == j) j = list_transpositions[k].second;
			else if(list_transpositions[k].second == j) j = list_transpositions[k].first;
		}
		if(j!=p[i]) return false;
	}
	return true;
}


// from permutation of the form s = 1:n -> [7,3,2,...] returns the coxeter decompo of s = prod (i, i+1) in the form of a vector of index i.
vector<int> coxeter(permutation s, bool inversed_order=false){
	uint n=s.size();
	vector<int> r={};
	while (!is_sorted(s.begin(), s.end())){
		for(uint k=0; k < n-1; k++ ){
			if (s[k] > s[k+1]){
				swap(&s[k], &s[k+1]);
				if(!inversed_order) r.insert(r.begin(),k);
				else r.insert(r.end(),k);
			 }
		}
	}
	return r;
}

// To check if coxeter returns a coxeter decomposition.
bool check_coxeter(vector<int> coxeter_decomposition, vector<int> initial_permutation, bool inversed_order=false){
	uint n = initial_permutation.size();
	for(uint i=0; i<n;i++){
		int j =i;

		uint k;
		if (inversed_order) k=0;
		else k= coxeter_decomposition.size()-1;
		while(0<=k && k <=coxeter_decomposition.size()-1){
			if(coxeter_decomposition[k] == j)  { j++;}
			else if(coxeter_decomposition[k] == j-1){ j--;}

			if (inversed_order) k++;
			else k--;
		}
		if(j!=initial_permutation[i]) return false;
	}
	return true;
}



template<typename T> // type T has to be convertible to double.
void quicksort_and_permutation(vector<T>& to_sort, permutation& p, uint low, uint high){
	// compatibility check
	constexpr bool verbose = false;
	uint n = to_sort.size();
	assert( n == p.size());
	if (verbose) cout << "low : " << low << " high : " << high << " n : " << n << endl;
	assert( high < n);
	if (high <= low) {return ;};

	// take the last element as pivot.
	double pivot = to_sort[high];

	int i = low -1 ;

	for(uint j=low; j< high; j++){
		if(to_sort[j] <= pivot){
			i++;
			swap(&to_sort[i],&to_sort[j]);
			swap(&p[i], &p[j]);
		}
	}
	swap(&to_sort[i+1], &to_sort[high]);
	swap(&p[i+1], &p[high]);

	quicksort_and_permutation<T>(to_sort, p, low, max(i,0));
	quicksort_and_permutation<T>(to_sort, p, i+2, high);
}





// This sorts to_sort and returns the permutation associated to this sorting. ie.
template<typename T>
permutation sort_to_permutation(vector<T>& to_sort){
	uint n = to_sort.size();
	// initialize p as the identity
	permutation p(n); for (uint i =0; i < n ; i++) p[i] = i;
	// call the recursive function doing the job
	if(is_sorted(to_sort.begin(), to_sort.end())) return p;
	quicksort_and_permutation<T>(to_sort, p, 0,n-1);
	return p;
}




template<typename T>
void quicksort_and_permutation(vector<T>& to_sort, vector<uint>& p, uint low, uint high, function<bool(T&, T&)> leq){
	// compatibility check
// 	uint n = to_sort.size();
	assert( to_sort.size() == p.size());
	assert( high < to_sort.size());
	if (high <= low) {return ;};

	// take the last element as pivot.
	T pivot = to_sort[high];

	int i = low -1 ;

	for(uint j=low; j< high; j++){
		if(leq(to_sort[j], pivot)){
			i++;
			swap(to_sort[i],to_sort[j]);
			swap(p[i], p[j]);
		}
	}
	swap(to_sort[i+1], to_sort[high]);
	swap(p[i+1], p[high]);
	quicksort_and_permutation<T>(to_sort, p, low, max(i,0), leq);
	quicksort_and_permutation<T>(to_sort, p, i+2, high, leq);
}

template<typename T>
vector<uint> sort_to_permutation(vector<T>& to_sort, function<bool(T&, T&)> leq){
	uint n = to_sort.size();
	// initialize p as the identity
	vector<uint> p(n); for (uint i =0; i < n ; i++) p[i] = i;
	// call the recursive function doing the job
// 	if(is_sorted(to_sort.begin(), to_sort.end()), leq) return p;
	quicksort_and_permutation<T>(to_sort, p, 0,n-1, leq);
	return p;
}








template<typename T>
vector<T> circ(const vector<T> &p,const  permutation &q){
	uint n = p.size();
	assert(q.size() == n);
	vector<T> r(n);
	for(uint i = 0; i< n; i++){
		r[i] = p[q[i]];
	}
	return r;
}


template<typename T>
vector<T> circ(const vector<T> &p,const  vector<uint> &q){
	uint n = p.size();
	assert(q.size() == n);
	vector<T> r(n);
	for(uint i = 0; i< n; i++){
		r[i] = p[q[i]];
	}
	return r;
}

template<typename T>
void compose(vector<T> &p,const  vector<uint> &q){
	uint n = p.size();
	assert(q.size() == n);
	vector<T> r(n);
	for(uint i = 0; i< n; i++){
		r[i] = p[q[i]];
	}
	p.swap(r);
}



// Returns the last indice having the same dimension as dimension[start].
int find_dimension_indices(uint start, const vector<int>& dimension){
	int dim = dimension[start];
	uint last = start;
	while(last + 1 < dimension.size() && dimension[last+1] == dim){
		last++;
	}
	return last;
}



bool filter_dimension_permutation_update(vector<double>& filters, const vector<int> &dimensions, vector<int> &permutation, bool verbose=false){
	uint start_indice = 0;
	bool is_identity = true;
	while(start_indice < filters.size()){
		uint last_indice = find_dimension_indices(start_indice, dimensions);

		if(verbose)
			cout << "first : " << start_indice << " last : " << last_indice << endl;
		if(!is_sorted(filters.begin()+start_indice, filters.begin()+last_indice+1)){
			quicksort_and_permutation(filters, permutation, start_indice, last_indice);
			is_identity = false;
		}
		start_indice = last_indice+1;
	}
	return is_identity;
}

// sorts filters dimension per dimensions, and returns the permutation
vector<int> filter_dimension_to_permutation(vector<double>& filters, const vector<int>& dimensions, bool verbose = false){
	// We assume dimension is ordered.
	uint n = filters.size();
	assert( n == dimensions.size());
	vector<int> p(n); for(uint i=0;i<n;i++) p[i]=i;
	bool is_identity = filter_dimension_permutation_update(filters, dimensions, p);
	if(is_identity)
		return {};
	return p;
}






permutation new_filter_to_permutation(vector<double>& filter, const vector<int>& dimensions){
	uint n = filter.size();
	assert(dimensions.size() == n);
	permutation p(n); for(uint i=0; i<n;i++) p[i]=i;
	for (int i = n-1; i > 0; i--){
        bool sorted = true;
        for(int j = 0; j < i; j++){
          if((filter[j] > filter[j+1]) || ((filter[j] == filter[j+1]) && (dimensions[j] > dimensions[j+1]))){
            swap(p[j], p[j+1]);
			swap(filter[j], filter[j+1]);
            sorted = false;
          }
        }
        if(sorted)  break; //for(unsigned int i = 0; i < this->filter.size(); i++)  cout << this->filter[i] << " "; cout << endl;
      }
      return p;
}



bool is_sorted(const vector<double>& filtration, const vector<int>& dimensions){
	uint n = filtration.size();
	assert( n == dimensions.size());
	if (!is_sorted(dimensions.begin(), dimensions.end())) return false;
	uint j=0;
	while(j<n){
		int last = find_dimension_indices(j, dimensions);
		if(!is_sorted(filtration.begin()+j, filtration.begin()+last)) return false;
		j=last+1;
	}
	return true;
}


uint prod(const vector<uint> &to_multiply, uint until=UINT_MAX){
	uint output=1;
	for(uint i=0; i<to_multiply.size() && i <= until;i++){
		output *= to_multiply[i];
	}
	return output;
}




/*
permutation ls_filter_update_to_permutation(vector<double> F1, vector<double> F2){
	permutation p1 =
	return vector<int>();
}*/



#endif // COMBINATORY_H_INCLUDED
