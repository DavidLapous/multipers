#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>

namespace Vineyard {

enum Matrix_types {
    VECTOR_MATRIX,
    RU_MATRIX,
    CHAIN_MATRIX
};

const double inf = std::numeric_limits<double>::infinity();
const double negInf = -1 * inf;

using index = unsigned int;
using filtration_value_type = double;
using filtration_type = std::vector<filtration_value_type>;
using dimension_type = int;
using persistence_pair = std::pair<filtration_value_type, filtration_value_type>;
using boundary_type = std::vector<index>;
using boundary_matrix = std::vector<boundary_type>;
using permutation_type = std::vector<unsigned int>;

struct Bar{
    Bar() : dim(-1), birth(-1), death(-1)
    {}

    Bar(dimension_type dim, int birth, int death)
        : dim(dim), birth(birth), death(death)
    {}

    dimension_type dim;
    int birth;
    int death;
};

using barcode_type = std::vector<Bar>;

struct Diagram_point{
    Diagram_point() : dim(-1), birth(-1), death(-1)
    {}

    Diagram_point(dimension_type dim,
                  filtration_value_type birth,
                  filtration_value_type death)
        : dim(dim), birth(birth), death(death)
    {}

    dimension_type dim;
    filtration_value_type birth;
    filtration_value_type death;
};

using diagram_type = std::vector<Diagram_point>;

}   //namespace Vineyard

#endif // UTILITIES_H
