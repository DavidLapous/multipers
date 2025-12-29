/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Salinas
 *
 *    Copyright (C) 2014 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */
#ifndef DEBUG_UTILS_H_
#define DEBUG_UTILS_H_

#include <iostream>

#ifndef NDEBUG
  // GUDHI_DEBUG is the Gudhi official flag for debug mode.
  #define GUDHI_DEBUG
#endif

// GUDHI_CHECK throw an exception if expression is false in debug mode, but does nothing in release mode
// Could assert in release mode, but cmake sets NDEBUG (for "NO DEBUG") in this mode, means assert does nothing.
#ifdef GUDHI_DEBUG
  #define GUDHI_CHECK(expression, excpt) ((expression) ? (void) 0 : (throw excpt))
  #define GUDHI_CHECK_code(CODE) CODE
#else
  #if defined(__clang__)
    #define GUDHI_ASSUME(expr) __builtin_assume(expr)
  #elif defined(_MSC_VER)
    #define GUDHI_ASSUME(expr) __assume(expr)
  #else
    #define GUDHI_ASSUME(expr) ((void)0)
  #endif
  #define GUDHI_CHECK(expression, excpt) GUDHI_ASSUME(expression)
  #define GUDHI_CHECK_code(CODE)
#endif

#define PRINT(a) std::clog << #a << ": " << (a) << " (DISP)" << std::endl

// #define DBG_VERBOSE
#ifdef DBG_VERBOSE
  #define DBG(a) std::clog << "DBG: " << (a) << std::endl
  #define DBGMSG(a, b) std::clog << "DBG: " << a << b << std::endl
  #define DBGVALUE(a) std::clog << "DBG: " <<  #a << ": " << a << std::endl
  #define DBGCONT(a) std::clog << "DBG: container " << #a << " -> "; for (auto x : a) std::clog << x << ","; std::clog << std::endl
#else
  #define DBG(a) (void) 0
  #define DBGMSG(a, b) (void) 0
  #define DBGVALUE(a) (void) 0
  #define DBGCONT(a) (void) 0
#endif

#endif  // DEBUG_UTILS_H_
