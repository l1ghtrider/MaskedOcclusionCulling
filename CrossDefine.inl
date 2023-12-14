#ifndef __MSOC_CROSS_DEFINE__
#define __MSOC_CROSS_DEFINE__

//#pragma GCC diagnostic ignored "-Wmacro-redefined"

#if defined(__ARM_NEON)
#define _mms_loadu_ps vld1q_f32
#define _mms_load1_ps(a) vdupq_n_f32((a)[0])
#define _mms_storeu_ps vst1q_f32
#define __ms float32x4_t
#define __msi int32x4_t
#define _mms_setr_ps(a,b,c,d) __ms{a,b,c,d}
#define _mms_setzero_ps() __ms{0,0,0,0}
#define _mms_setr_epi32(a,b,c,d) __msi{a,b,c,d}
int _mms_movemask_ps(__ms vec)
{
	uint32x4_t mask = vreinterpretq_u32_f32(vec);
	uint32x4_t signBits = vshrq_n_u32(mask, 31);
	return vgetq_lane_u32(signBits, 0) |
		(vgetq_lane_u32(signBits, 1) << 1) |
		(vgetq_lane_u32(signBits, 2) << 2) |
		(vgetq_lane_u32(signBits, 3) << 3);
}
__ms _mms_xor_ps(__ms a, __ms b)
{
	uint32x4_t aInt = vreinterpretq_u32_f32(a);
	uint32x4_t bInt = vreinterpretq_u32_f32(b);
	uint32x4_t resultInt = veorq_u32(aInt, bInt);
	return vreinterpretq_f32_u32(resultInt);
}
#define _mms_add_ps vaddq_f32
#define _mms_sub_ps vsubq_f32
#define _mms_mul_ps vmulq_f32
__ms _mms_div_ps(__ms a, __ms b)
{
	float32x4_t result = { 0, 0, 0, 0 };
	for (int i = 0; i < 4; i++) {
		result[i] = a[i] / b[i];
	}
	return result;
}
#define _mms_cvttps_epi32 vcvtq_s32_f32
#define _mms_setzero_si128() vdupq_n_s32(0)
#define _mms_and_si128 vandq_s32
#define _mms_add_epi32 vaddq_s32

#else

#define _mms_loadu_ps _mm_loadu_ps
#define _mms_load1_ps _mm_load1_ps
#define _mms_storeu_ps _mm_storeu_ps
#define __ms __m128
#define __msi __m128i
#define _mms_setr_ps _mm_setr_ps
#define _mms_setzero_ps() _mm_setzero_ps()
#define _mms_setr_epi32 _mm_setr_epi32
#define _mms_movemask_ps _mm_movemask_ps
#define _mms_xor_ps _mm_xor_ps
#define _mms_add_ps _mm_add_ps
#define _mms_sub_ps _mm_sub_ps
#define _mms_mul_ps _mm_mul_ps
#define _mms_div_ps _mm_div_ps
#define _mms_cvttps_epi32 _mm_cvttps_epi32
#define _mms_setzero_si128 _mm_setzero_si128
#define _mms_and_si128 _mm_and_si128
#define _mms_add_epi32 _mm_add_epi32
#endif

#endif
