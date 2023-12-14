////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common shared include file to hide compiler/os specific functions from the rest of the code. 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MOC_SUPPORT_DELAY
#define MOC_SUPPORT_DELAY 1
#endif

#ifndef __MSOC_COMPILER_SPECIFIC__
#define __MSOC_COMPILER_SPECIFIC__

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__clang__)
	#define __MICROSOFT_COMPILER
#endif

#if defined(_WIN32)	&& (defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__clang__)) // Windows: MSVC / Intel compiler / clang
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	#include <arm_neon.h>
#else
	#include <intrin.h>
#endif
	#include <new.h>

#ifndef FORCE_INLINE
	#define FORCE_INLINE __forceinline
#endif

FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
{
	unsigned long idx;
	_BitScanForward(&idx, *mask);
	*mask &= *mask - 1;
	return idx;
}

FORCE_INLINE void *aligned_alloc(size_t alignment, size_t size)
{
	return _aligned_malloc(size, alignment);
}

FORCE_INLINE void aligned_free(void *ptr)
{
	_aligned_free(ptr);
}

FORCE_INLINE void cpuidex(int* cpuinfo, int function, int subfunction)
{
	__cpuidex(cpuinfo, function, subfunction);
}

FORCE_INLINE unsigned long long xgetbv(unsigned int index)
{
	return  _xgetbv(index);
}

#elif defined(__GNUG__)	|| defined(__clang__) // G++ or clang

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#include <mm_malloc.h>
#endif

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
	#include <malloc/malloc.h> // memalign
#else
	#include <malloc.h> // memalign
#endif
	
	#include <new>
	#include <stdlib.h>

#ifndef FORCE_INLINE
	#define FORCE_INLINE inline
#endif

FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
{
	unsigned long idx;
	idx = __builtin_ctzl(*mask);
	*mask &= *mask - 1;
	return idx;
}

FORCE_INLINE void *aligned_alloc(size_t alignment, size_t size)
{
	void* pData = nullptr;
	auto err = posix_memalign((void **)&pData, alignment, size);
	return pData;
	//return memalign(alignment, size);
}

FORCE_INLINE void aligned_free(void *ptr)
{
	free(ptr);
}

#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
#include <cpuid.h>
FORCE_INLINE void cpuidex(int* cpuinfo, int function, int subfunction)
{
	__cpuid_count(function, subfunction, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
}

FORCE_INLINE unsigned long long xgetbv(unsigned int index)
{
	unsigned int eax, edx;
	__asm__ __volatile__(
		"xgetbv;"
		: "=a" (eax), "=d"(edx)
		: "c" (index)
	);
	return ((unsigned long long)edx << 32) | eax;
}
#endif

#else
	#error Unsupported compiler
#endif

#endif
