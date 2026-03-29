#pragma once

#ifndef MULTIPERS_DISABLE_AIDA_INTERFACE
#if defined(_WIN32)
#define MULTIPERS_DISABLE_AIDA_INTERFACE 1
#else
#define MULTIPERS_DISABLE_AIDA_INTERFACE 0
#endif
#endif

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
#include <aida_interface.hpp>
#include <config.hpp>
#endif
