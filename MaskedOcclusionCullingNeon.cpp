#if USE_SOC
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
#include <string.h>
#include <assert.h>
#include <float.h>
#include "MaskedOcclusionCulling.h"
#include "CompilerSpecific.inl"
#include "CrossDefine.inl"

#if MOC_RECORDER_ENABLE
#include "FrameRecorder.h"
#endif

#if defined(__MICROSOFT_COMPILER) && _MSC_VER < 1900
	// If you remove/comment this error, the code will compile & use the SSE41 version instead. 
	#error Older versions than visual studio 2015 not supported due to compiler bug(s)
#endif

#if (!defined(__MICROSOFT_COMPILER) || _MSC_VER >= 1900) && !(!defined(__ARM_NEON) && !defined(__ARM_NEON__))

// For performance reasons, the MaskedOcclusionCullingAVX2.cpp file should be compiled with VEX encoding for SSE instructions (to avoid 
// AVX-SSE transition penalties, see https://software.intel.com/en-us/articles/avoiding-avx-sse-transition-penalties). However, the SSE
// version in MaskedOcclusionCulling.cpp _must_ be compiled without VEX encoding to allow backwards compatibility. Best practice is to 
// use lowest supported target platform (e.g. /arch:SSE2) as project default, and elevate only the MaskedOcclusionCullingAVX2/512.cpp files.
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
	#error For best performance, MaskedOcclusionCullingAVX2.cpp should be compiled with /armarch64:neon
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific defines and constants
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             4
#define TILE_HEIGHT_SHIFT      2

#define SIMD_LANE_IDX _mms_setr_epi32(0, 1, 2, 3)

#define SIMD_SUB_TILE_COL_OFFSET _mms_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET _mms_setzero_si128()
#define SIMD_SUB_TILE_COL_OFFSET_F _mms_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3)
#define SIMD_SUB_TILE_ROW_OFFSET_F _mms_setzero_ps()

#define SIMD_LANE_YCOORD_I _mms_setr_epi32(128, 384, 640, 896)
#define SIMD_LANE_YCOORD_F _mms_setr_ps(128.0f, 384.0f, 640.0f, 896.0f)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX specific typedefs and functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef __ms __mw;
typedef __msi __mwi;

#define _mmw_storeu_ps				vst1q_f32
#define _mmw_loadu_ps				vld1q_f32
#define _mmw_set1_ps                vdupq_n_f32
FORCE_INLINE __mw _mmw_setzero_ps() {
	return _mmw_set1_ps(0.0f);
}
FORCE_INLINE __mw _mmw_and_ps(__mw a, __mw b) {
	uint32x4_t aInt = vreinterpretq_u32_f32(a);
	uint32x4_t bInt = vreinterpretq_u32_f32(b);
	uint32x4_t resultInt = vandq_u32(aInt, bInt);
	return vreinterpretq_f32_u32(resultInt);
}
FORCE_INLINE __mw _mmw_or_ps(__mw a, __mw b) {
	uint32x4_t aInt = vreinterpretq_u32_f32(a);
	uint32x4_t bInt = vreinterpretq_u32_f32(b);
	uint32x4_t resultInt = vorrq_u32(aInt, bInt);
	return vreinterpretq_f32_u32(resultInt);
}
FORCE_INLINE __mw _mmw_xor_ps(__mw a, __mw b) {
	uint32x4_t aInt = vreinterpretq_u32_f32(a);
	uint32x4_t bInt = vreinterpretq_u32_f32(b);
	uint32x4_t resultInt = veorq_u32(aInt, bInt);
	return vreinterpretq_f32_u32(resultInt);
}
#define _mmw_not_ps(a)              _mmw_xor_ps((a), vreinterpretq_f32_s32(vdupq_n_s32(~0)))
FORCE_INLINE __mw _mmw_andnot_ps(__mw a, __mw b) {
	uint32x4_t bNot = vmvnq_u32(vreinterpretq_u32_f32(b));
	uint32x4_t resultInt = vandq_u32(vreinterpretq_u32_f32(a), bNot);
	return vreinterpretq_f32_u32(resultInt);
}
#define _mmw_neg_ps(a)              _mmw_xor_ps((a), vdupq_n_f32(-0.0f))
#define _mmw_abs_ps(a)              _mmw_and_ps((a), vreinterpretq_f32_s32(vdupq_n_s32(0x7FFFFFFF)))
#define _mmw_add_ps                 vaddq_f32
#define _mmw_sub_ps                 vsubq_f32
#define _mmw_mul_ps                 vmulq_f32
#ifdef __aarch64__
#define _mmw_div_ps                 vdivq_f32
#else
__mw _mmw_div_ps(__mw a, __mw b)
{
    float32x4_t result = { 0, 0, 0, 0 };
    for (int i = 0; i < 4; i++) {
        result[i] = a[i] / b[i];
    }
    return result;
}
#endif
#define _mmw_min_ps                 vminq_f32
#define _mmw_max_ps                 vmaxq_f32
int _mmw_movemask_ps(__ms vec)
{
	uint32x4_t mask = vreinterpretq_u32_f32(vec);
	uint32x4_t signBits = vshrq_n_u32(mask, 31);
	return vgetq_lane_u32(signBits, 0) |
		(vgetq_lane_u32(signBits, 1) << 1) |
		(vgetq_lane_u32(signBits, 2) << 2) |
		(vgetq_lane_u32(signBits, 3) << 3);
} 
#define _mmw_cmpge_ps(a,b)          vreinterpretq_f32_u32(vcgeq_f32(a, b)) //need simd_cast
#define _mmw_cmpgt_ps(a,b)          vreinterpretq_f32_u32(vcgtq_f32(a, b)) //need simd_cast
#define _mmw_cmpeq_ps(a,b)          vreinterpretq_f32_u32(vceqq_f32(a, b)) //need simd_cast
#ifdef __ARM_FEATURE_FMA
#define _mmw_fmadd_ps(a,b,c)        vfmaq_f32(c, a, b)
#else
#define _mmw_fmadd_ps(a,b,c)        vaddq_f32(vmulq_f32(a, b), c)
#endif
#define _mmw_fmsub_ps(a,b,c)        vsubq_f32(vmulq_f32(a, b), c)
//#define _mmw_shuffle_ps             _mm_shuffle_ps vextq_f32 _MM_SHUFFLE
#define _mmw_insertf32x4_ps(a,b,c)  (b)
#define _mmw_cvtepi32_ps            vcvtq_f32_s32
// #define _mmw_blendv_ps(a,b,c)		vbslq_f32(vreinterpretq_u32_f32(c),b,a)
FORCE_INLINE __ms _mmw_blendv_ps(const __ms &a, const __ms &b, const __ms &c)
{   
    __ms cond = vreinterpretq_f32_s32(vshrq_n_s32(vreinterpretq_s32_f32(c), 31));
    return _mmw_or_ps(_mmw_and_ps(a, vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(cond)))), _mmw_and_ps(cond, b));
}
#define _mmw_blendv_epi32(a,b,c)    simd_cast<__mwi>(_mmw_blendv_ps(simd_cast<__mw>(a), simd_cast<__mw>(b), simd_cast<__mw>(c)))

#define _mmw_set1_epi32             vdupq_n_s32
#define _mmw_setzero_epi32()        _mmw_set1_epi32(0)
#define _mmw_and_epi32              vandq_s32
#define _mmw_or_epi32               vorrq_s32
#define _mmw_xor_epi32              veorq_s32
#define _mmw_not_epi32(a)           veorq_s32((a), _mmw_set1_epi32(~0))
#define _mmw_andnot_epi32(a, b)     vbicq_s32(b,a)
#define _mmw_neg_epi32(a)           vsubq_s32(_mmw_set1_epi32(0), (a))
#define _mmw_add_epi32              vaddq_s32
#define _mmw_sub_epi32              vsubq_s32
#define _mmw_subs_epu16(a,b)        vreinterpretq_s32_u16(vqsubq_u16(vreinterpretq_u16_s32(a), vreinterpretq_u16_s32(b)))
#define _mmw_cmpeq_epi32(a,b)       vreinterpretq_s32_u32(vceqq_s32(a,b))
#define _mmw_cmpgt_epi32(a,b)       vreinterpretq_s32_u32(vcgtq_s32(a,b))
#define _mmw_srai_epi32             vshrq_n_s32
#define _mmw_srli_epi32(a,b)        vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), b))
#define _mmw_slli_epi32             vshlq_n_s32
#define _mmw_cvtps_epi32(a)         vcvtq_s32_f32(vrndnq_f32(a)) 
#define _mmw_cvttps_epi32           vcvtq_s32_f32

#define _mmx_fmadd_ps               _mmw_fmadd_ps
#define _mmx_max_epi32              _mmw_max_epi32
#define _mmx_min_epi32              _mmw_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD casting functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Y> FORCE_INLINE T simd_cast(Y A);
template<> FORCE_INLINE __ms  simd_cast<__ms>(float A) { return vdupq_n_f32(A); }
template<> FORCE_INLINE __ms  simd_cast<__ms>(__msi A) { return vreinterpretq_f32_s32(A); }
template<> FORCE_INLINE __ms  simd_cast<__ms>(__ms A) { return A; }
template<> FORCE_INLINE __msi simd_cast<__msi>(int A) { return vdupq_n_s32(A); }
template<> FORCE_INLINE __msi simd_cast<__msi>(__ms A) { return vreinterpretq_s32_f32(A); }
template<> FORCE_INLINE __msi simd_cast<__msi>(__msi A) { return A; }

#define MAKE_ACCESSOR(name, simd_type, base_type, is_const, elements) \
    FORCE_INLINE is_const base_type * name(is_const simd_type &a) { \
        union accessor { simd_type m_native; base_type m_array[elements]; }; \
        is_const accessor *acs = reinterpret_cast<is_const accessor*>(&a); \
        return acs->m_array; \
    }

MAKE_ACCESSOR(simd_f32, __ms, float, , 4)
MAKE_ACCESSOR(simd_f32, __ms, float, const, 4)
MAKE_ACCESSOR(simd_i32, __msi, int, , 4)
MAKE_ACCESSOR(simd_i32, __msi, int, const, 4)

typedef MaskedOcclusionCulling::pfnAlignedAlloc pfnAlignedAlloc;
typedef MaskedOcclusionCulling::pfnAlignedFree  pfnAlignedFree;
typedef MaskedOcclusionCulling::VertexLayout    VertexLayout;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized SSE input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FORCE_INLINE void GatherVertices(__ms* vtxX, __ms* vtxY, __ms* vtxW, const float* inVtx, const unsigned int* inTrisPtr, int numLanes, const VertexLayout& vtxLayout)
{
	for (int lane = 0; lane < numLanes; lane++)
	{
		for (int i = 0; i < 3; i++)
		{
			char* vPtrX = (char*)inVtx + inTrisPtr[lane * 3 + i] * vtxLayout.mStride;
			char* vPtrY = vPtrX + vtxLayout.mOffsetY;
			char* vPtrW = vPtrX + vtxLayout.mOffsetW;

			simd_f32(vtxX[i])[lane] = *((float*)vPtrX);
			simd_f32(vtxY[i])[lane] = *((float*)vPtrY);
			simd_f32(vtxW[i])[lane] = *((float*)vPtrW);
		}
	}
}

namespace MaskedOcclusionCullingNeon
{
	FORCE_INLINE __msi _mmw_mullo_epi32(const __msi& a, const __msi& b) {
		__msi result = vmulq_s32(a, b);
		__msi minInt32 = vdupq_n_s32(INT32_MIN);
		__msi maxInt32 = vdupq_n_s32(INT32_MAX);
		result = vminq_s32(result, maxInt32);
		result = vmaxq_s32(result, minInt32);
		return result;
	}

	FORCE_INLINE __msi _mmw_min_epi32(const __msi& a, const __msi& b) { return vminq_s32(a, b); }
	FORCE_INLINE __msi _mmw_max_epi32(const __msi& a, const __msi& b) { return vmaxq_s32(a, b); }
	FORCE_INLINE __msi _mmw_abs_epi32(const __msi& a) { return vabsq_s32(a); }

	//FORCE_INLINE __ms _mmw_blendv_ps(const __ms& a, const __ms& b, const __ms& c) { return _mm_blendv_ps(a, b, c); }

	FORCE_INLINE int _mmw_testz_epi32(const __msi& a, const __msi& b) {
		__msi result = vandq_s32(a, b);
		auto simd = simd_i32(result);
		return (simd[0] == 0 && simd[1] == 0 && simd[2] == 0 && simd[3] == 0);
	}
	FORCE_INLINE __ms _mmx_dp4_ps(const __ms& a, const __ms& b)
	{
		__ms prod = vmulq_f32(a, b);
		float32_t prodf = prod[0] + prod[1] + prod[2] + prod[3];
		__ms dp = _mms_setr_ps(prodf, prodf, prodf, prodf);
		return dp;
	}
	
	FORCE_INLINE __ms _mmw_floor_ps(const __ms& a) {
#if __ARM_ARCH >= 8 && defined(__ARM_FEATURE_DIRECTED_ROUNDING)
        return vrndmq_f32(a);
#else
        return a;
#endif
    }
	FORCE_INLINE __ms _mmw_ceil_ps(const __ms& a) {
#if __ARM_ARCH >= 8 && defined(__ARM_FEATURE_DIRECTED_ROUNDING)
        return vrndpq_f32(a);
#else
        return a;
#endif
    }
	FORCE_INLINE __msi _mmw_transpose_epi8(const __msi& a)
	{
		uint8x16_t au = vreinterpretq_u8_s32(a);
		au = uint8x16_t{ au[0], au[4], au[8], au[12], au[1], au[5], au[9], au[13], au[2], au[6], au[10], au[14]
			, au[3], au[7], au[11], au[15] };
		return vreinterpretq_s32_u8(au);
	}
	FORCE_INLINE __msi _mmw_sllv_ones(const __msi& ishift)
	{
		__msi shift = vminq_s32(ishift, __msi{ 32, 32, 32, 32 });

		// Uses scalar approach to perform _mm_sllv_epi32(~0, shift)
		static const unsigned int maskLUT[33] = {
			~0U << 0, ~0U << 1, ~0U << 2 ,  ~0U << 3, ~0U << 4, ~0U << 5, ~0U << 6 , ~0U << 7, ~0U << 8, ~0U << 9, ~0U << 10 , ~0U << 11, ~0U << 12, ~0U << 13, ~0U << 14 , ~0U << 15,
			~0U << 16, ~0U << 17, ~0U << 18 , ~0U << 19, ~0U << 20, ~0U << 21, ~0U << 22 , ~0U << 23, ~0U << 24, ~0U << 25, ~0U << 26 , ~0U << 27, ~0U << 28, ~0U << 29, ~0U << 30 , ~0U << 31,
			0U };

		__msi retMask;
		simd_i32(retMask)[0] = (int)maskLUT[simd_i32(shift)[0]];
		simd_i32(retMask)[1] = (int)maskLUT[simd_i32(shift)[1]];
		simd_i32(retMask)[2] = (int)maskLUT[simd_i32(shift)[2]];
		simd_i32(retMask)[3] = (int)maskLUT[simd_i32(shift)[3]];
		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::NEON;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc alignedAlloc, pfnAlignedFree alignedFree)
	{
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)alignedAlloc(64, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(alignedAlloc, alignedFree);
		return object;
	}
};

#else

namespace MaskedOcclusionCullingNeon
{
	typedef MaskedOcclusionCulling::pfnAlignedAlloc            pfnAlignedAlloc;
	typedef MaskedOcclusionCulling::pfnAlignedFree             pfnAlignedFree;

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc alignedAlloc, pfnAlignedFree alignedFree)
	{
		return nullptr;
	}
};

#endif

#endif
