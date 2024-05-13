/*    This file is part of the MMA Library -
 * https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT. See
 * file LICENSE for full license details. Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence
 * class, renaming and cleanup.
 */
/**
 * @file debug.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Display functions for debug purposes
 */

#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <chrono>
#include <iostream>
#include <list>
#include <vector>

namespace Gudhi::multiparameter::mma::Debug {

using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;

constexpr bool debug = false;

class Timer {
public:
  Timer() : activated_(false) {}
  Timer(const std::string &string, bool verbose)
      : timer_(clk::now()), activated_(verbose) {
    if (verbose) {
      std::cout << string << std::flush;
    }
  }
  ~Timer() {
    if (activated_) {
      std::chrono::duration<double> elapsed =
          std::chrono::duration_cast<std::chrono::duration<double>>(clk::now() -
                                                                    timer_);
      std::cout << " Done ! (" << elapsed.count() << " s)." << std::endl;
    }
  }

private:
  tp timer_;
  bool activated_;
};

template <typename T> void disp_vect(std::vector<T> v) {
  for (auto i = 0u; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T> void disp_vect(std::list<T> v) {
  while (!v.empty()) {
    std::cout << v.front() << " ";
    v.pop_front();
  }
  std::cout << std::endl;
}

template <typename T> void disp_vect(std::vector<std::pair<T, T>> v) {
  for (unsigned int i = 0; i < v.size(); i++) {
    std::cout << "(" << v[i].first << " " << v[i].second << ")  ";
  }
}

template <typename T>
void disp_vect(std::vector<std::vector<T>> v, bool show_small = true) {
  for (auto i = 0u; i < v.size(); i++) {
    if (v[i].size() <= 1 && !show_small)
      continue;
    std::cout << "(";
    for (auto j = 0u; j < v[i].size(); j++) {
      std::cout << v[i][j];
      if (j < v[i].size() - 1)
        std::cout << " ";
    }
    std::cout << ") ";
  }
  std::cout << std::endl;
}

} // namespace Gudhi::multiparameter::mma::Debug
namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> truc) {
  stream << "[";
  for (unsigned int i = 0; i < truc.size() - 1; i++) {
    stream << truc[i] << ", ";
  }
  if (!truc.empty())
    stream << truc.back();
  stream << "]";
  return stream;
}
} // namespace std

#endif // DEBUG_H_INCLUDED
