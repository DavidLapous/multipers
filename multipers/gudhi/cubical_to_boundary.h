#ifndef MULTIPERS_CUBICAL_CONV_H
#define MULTIPERS_CUBICAL_CONV_H

#include <vector>
#include <set>

#include <gudhi/Bitmap_cubical_complex.h>

inline void _to_boundary(const std::vector<unsigned int>& shape,
                         std::vector<std::vector<unsigned int> >& generator_maps,
                         std::vector<int>& generator_dimensions) {
  using Bitmap_cubical_complex_base = Gudhi::cubical_complex::Bitmap_cubical_complex_base<char>;
  using Bitmap_cubical_complex = Gudhi::cubical_complex::Bitmap_cubical_complex<Bitmap_cubical_complex_base>;
  using Simplex_handle = Bitmap_cubical_complex::Simplex_handle;

  if (shape.empty()) return;

  unsigned int size = 1;
  for (auto v : shape) size *= v;

  std::vector<char> vertices(size);
  Bitmap_cubical_complex cub(shape, vertices, false);

  unsigned int dimMax = shape.size();
  std::vector<std::vector<Simplex_handle> > faces(dimMax + 1);
  unsigned int numberOfSimplices = 0;
  for (unsigned int d = 0; d < dimMax + 1; ++d) {
    for ([[maybe_unused]] auto sh : cub.skeleton_simplex_range(d)) {
      ++numberOfSimplices;
    }
  }

  generator_dimensions.resize(numberOfSimplices);
  generator_maps.resize(numberOfSimplices);
  unsigned int i = 0;
  for (unsigned int d = 0; d < dimMax + 1; ++d) {
    for (auto sh : cub.skeleton_simplex_range(d)) {
      cub.assign_key(sh, i);
      generator_dimensions[i] = d;
      auto& col = generator_maps[i];
      for (auto b : cub.boundary_simplex_range(sh)) col.push_back(cub.key(b));
      std::sort(col.begin(), col.end());
      ++i;
    }
  }
};

inline void get_vertices(unsigned int i,
                         std::set<unsigned int>& vertices,
                         const std::vector<std::vector<unsigned int> >& generator_maps) {
  if (generator_maps[i].empty()) {
    vertices.insert(i);
    return;
  }

  for (auto v : generator_maps[i]) get_vertices(v, vertices, generator_maps);
}

#endif  // MULTIPERS_CUBICAL_CONV_H
