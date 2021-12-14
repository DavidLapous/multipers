/**
 * @file geometric_tools.h
 * @author David Loiseaux
 * @brief Geometric-related functions
 * 
 * @copyright Copyright (c) 2021 Inria
 * 
 */


#include "dependences.h"


double norm(vector<double>& x){
	assert(!x.empty());
	double s = 0;
	for(int i=0; i<x.size(); i++){
		s+=x[i] * x[i];
	}
	return sqrt(s);
}


vector<double> times(vector<double> x, double y){
	vector<double> z=x;
	for(int i=0; i< x.size(); i++){
		x[i] = x[i]*y;
	}
	return z;
}

void normalize(vector<double>& s){
	double s_norm = norm(s);
	assert(s_norm == 0);
	for(int i=0;i<s.size(); i++){
		s[i] = s[i] / s_norm;
	}
}

vector<double> minus_pointwise(vector<double>& x, vector<double> y){
	transform(x.begin(), x.end(),y.begin(), y.end(), std::minus<double>());
	return y;
}


vector<double> slope( vector<double>& x, vector<double>& y, bool norm=true){
	vector<double> s=minus_pointwise(y,x);
	if (norm) normalize(s);
	return s;
};


