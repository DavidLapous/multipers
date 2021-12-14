/**
 * @file dependences.h
 * @author David Loiseaux
 * @brief List of external dependences and type macros
 *
 * @copyright Copyright (c) 2021 Inria
 *
 */

#ifndef DEPENDENCES_H_INCLUDED
#define DEPENDENCES_H_INCLUDED

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <climits>
#include <utility>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <set>
#include <list>
#include <cfloat>
#include <unordered_map>
#include <stdlib.h>
#include <sstream>
#include <omp.h>

using namespace std;

typedef vector<int> boundary;
typedef vector<boundary> boundary_matrix;
typedef vector<pair<int,pair<int,int>>> barcode;
typedef vector<pair<int,pair<double,double>>> barcoded;

typedef vector<vector<int>> simplextree;
typedef pair<vector<double>, vector<double>> line;

typedef vector<int> permutation;

typedef pair<double,double> point_2;
typedef pair<pair<double,double>,pair<double,double>> interval_2;
typedef pair<vector<double>, vector<double>> interval;


#endif // DEPENDENCES_H_INCLUDED

