#pragma once

#ifndef MPFREE_LOGS
#define MPFREE_LOGS 1
#endif

#ifndef TWOPAC_LOGS
#define TWOPAC_LOGS 1
#endif

#ifndef MULTI_CRITICAL_LOGS
#define MULTI_CRITICAL_LOGS 1
#endif

#ifndef FUNCTION_DELAUNAY_LOGS
#define FUNCTION_DELAUNAY_LOGS 1
#endif

namespace multipers::backend_log_flags {

inline constexpr bool mpfree = MPFREE_LOGS != 0;
inline constexpr bool twopac = TWOPAC_LOGS != 0;
inline constexpr bool multi_critical = MULTI_CRITICAL_LOGS != 0;
inline constexpr bool function_delaunay = FUNCTION_DELAUNAY_LOGS != 0;

}  // namespace multipers::backend_log_flags
