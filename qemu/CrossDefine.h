//intel https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=6406,6234,6264,6423,361
//arm https://developer.arm.com/architectures/instruction-sets/intrinsics/

#include <iostream>
#include <typeindex>
#include <string>

#define FORCE_INLINE inline

#ifdef __ARM_NEON

#include <arm_neon.h>

//load
void load_data(float* input, float32x4_t* output) {
    *output = vld1q_f32(input);
}
//print
void print_simd(float32x4_t vec)
{
    float a = vgetq_lane_f32(vec, 0);
    float b = vgetq_lane_f32(vec, 1);
    float c = vgetq_lane_f32(vec, 2);
    float d = vgetq_lane_f32(vec, 3);
    printf("vec (%f %f %f %f)\n", a, b, c, d);
}
void print_simd(int32x4_t vector)
{
    int32_t result[4];
    vst1q_s32(result, vector);
    printf("vec (%d %d %d %d)\n", result[0], result[1], result[2], result[3]);
}

#else

#ifdef _WIN32
    #include <intrin.h>
#else
    #include <immintrin.h>
#endif

//cv4f
extern "C"
{
    typedef struct { float x, y, z, w; } cv4f;
}
//print
template<typename T, int N, typename P>
void print_impl(T result, void (*func_store)(P*,T))
{
    P results[N];
    func_store((P*)results, result);
    std::cout << "(";
    for (int i = 0; i < N; i++)
    {
        std::cout << results[i];
        if (i != N -1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}
template<typename T>
void print_simd(T result)
{
    std::cout << "error not print supporting simd type: " << (int)(std::type_index(typeid(T)));
}
template<> void print_simd<__m128>(__m128 result) { print_impl<__m128, 4, float>((__m128)result, [](float* p, __m128 t)->void { _mm_storeu_ps(p, t); }); }
template void print_simd<__m128>(__m128);
template<> void print_simd<__m128i>(__m128i result) { print_impl<__m128i, 4, int>((__m128i)result, [](int* p, __m128i t)->void { _mm_store_si128((__m128i*)p, t); }); }

#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define _mms_loadu_ps vld1q_f32
#define _mms_load1_ps(a) vdupq_n_f32(a[0])
#define _mms_storeu_ps vst1q_f32
#define __ms float32x4_t
#define __msi int32x4_t
#define __msu uint32x4_t
#define _mms_setr_ps(a,b,c,d) __ms{a,b,c,d}
#define _mms_setr_epi32(a,b,c,d) __msi{a,b,c,d}
#define _mms_setr_epu32(a,b,c,d) __msu{a,b,c,d}
int _mms_movemask_ps(__ms vec)
{
	uint32x4_t mask = vreinterpretq_u32_f32(vec);
	uint32x4_t signBits = vshrq_n_u32(mask, 31);
    return vgetq_lane_u32(signBits, 0) |
        (vgetq_lane_u32(signBits, 1) << 1) |
        (vgetq_lane_u32(signBits, 2) << 2) |
        (vgetq_lane_u32(signBits, 3) << 3) ;
}
__ms _mms_xor_ps(__ms a, __ms b)
{
    uint32x4_t aInt = vreinterpretq_u32_f32(a);
    uint32x4_t bInt = vreinterpretq_u32_f32(b);
    uint32x4_t resultInt = veorq_u32(aInt, bInt);
    return vreinterpretq_f32_u32(resultInt);
}
// __ms _mms_div_ps(__ms a, __ms b)
// {
//     float32x4_t result;
//     for (int i = 0; i < 4; i++) {
//         result[i] = a[i] / b[i];
//     }
//     return result;
// }
#define _mms_div_ps vdivq_f32
#define _mms_sub_ps vsubq_f32
#define _mms_cvttps_epi32 vcvtq_s32_f32
#define _mms_setzero_si128() vdupq_n_s32(0)
#define _mms_and_si128 vandq_s32
#define _mms_add_epi32 vaddq_s32

#define _mmw_castsi128_ps vreinterpretq_f32_s32
#define _mmw_cmpge_ps(a,b)          vcgeq_f32(a, b)
//#define _mmw_shuffle_ps vextq_f32
//#define _mmw_blendv_ps(a,b,c) vbslq_f32(vreinterpretq_u32_f32(c),b,a)

FORCE_INLINE __ms _mmw_or_ps(__ms a, __ms b) {
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}
FORCE_INLINE __ms _mmw_and_ps(__ms a, __ms b)
{
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}
FORCE_INLINE __ms _mmw_blendv_ps(const __ms &a, const __ms &b, const __ms &c)
{   
    __ms cond = vreinterpretq_f32_s32(vshrq_n_s32(vreinterpretq_s32_f32(c), 31));
    return _mmw_or_ps(_mmw_and_ps(a, vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(cond)))), _mmw_and_ps(cond, b));
}

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

#define _mmw_or_epi32               vorrq_s32
//!!! //vmvnq_s32(vandq_s32(a,b)) 
#define _mmw_andnot_si128(a, b)     vbicq_s32(b,a)
#define _mmw_subs_epu16(a,b)        vreinterpretq_s32_u16(vqsubq_u16(vreinterpretq_u16_s32(a), vreinterpretq_u16_s32(b)))
#define _mmw_srai_epi32             vshrq_n_s32
#define _mmw_srli_epi32(a, shift)   vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), shift))
#define _mmw_slli_epi32             vshlq_n_s32
#define _mmw_cvtps_epi32(a)         vcvtq_s32_f32(vrndnq_f32(a)) 
#define _mmw_cvttps_epi32           vcvtq_s32_f32
FORCE_INLINE __ms _mmw_floor_ps(const __ms& a) { return vrndmq_f32(a); }
FORCE_INLINE __ms _mmw_ceil_ps(const __ms& a) { return vrndpq_f32(a); }
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
    print_simd(shift);

    // Uses scalar approach to perform _mm_sllv_epi32(~0, shift)
    static const unsigned int maskLUT[33] = {
        ~0U << 0, ~0U << 1, ~0U << 2 ,  ~0U << 3, ~0U << 4, ~0U << 5, ~0U << 6 , ~0U << 7, ~0U << 8, ~0U << 9, ~0U << 10 , ~0U << 11, ~0U << 12, ~0U << 13, ~0U << 14 , ~0U << 15,
        ~0U << 16, ~0U << 17, ~0U << 18 , ~0U << 19, ~0U << 20, ~0U << 21, ~0U << 22 , ~0U << 23, ~0U << 24, ~0U << 25, ~0U << 26 , ~0U << 27, ~0U << 28, ~0U << 29, ~0U << 30 , ~0U << 31,
        0U };

    __msi retMask;
    print_simd(*((__msi *)(simd_i32(shift))));
    simd_i32(retMask)[0] = (int)maskLUT[simd_i32(shift)[0]];
    simd_i32(retMask)[1] = (int)maskLUT[simd_i32(shift)[1]];
    simd_i32(retMask)[2] = (int)maskLUT[simd_i32(shift)[2]];
    simd_i32(retMask)[3] = (int)maskLUT[simd_i32(shift)[3]];
    return retMask;
}
#define _mmw_cmpeq_epi32(a,b)          vreinterpretq_s32_u32(vceqq_s32(a,b))
#define _mmw_cmpgt_epi32(a,b)          vreinterpretq_s32_u32(vcgtq_s32(a,b))
#define _mmw_fmsub_ps(a,b,c)           vfmsq_f32(c, a, b)
// #define _mmw_fmsub_ps(a,b,c)           vsubq_f32(vmulq_f32(a, b), c)
#define _mmw_set1_ps(a)                __ms{a, a, a, a}



#else


#define _mms_loadu_ps _mm_loadu_ps
#define _mms_load1_ps _mm_load1_ps
#define _mms_storeu_ps _mm_storeu_ps
#define __ms __m128
#define __msi __m128i
#define __msu __m128i

#define _mms_setr_ps _mm_setr_ps
#define _mms_setr_epi32 _mm_setr_epi32
#define _mms_setr_epu32 _mm_setr_epi32
#define _mms_movemask_ps _mm_movemask_ps
#define _mms_xor_ps _mm_xor_ps
#define _mms_div_ps _mm_div_ps
#define _mms_sub_ps _mm_sub_ps
#define _mms_cvttps_epi32 _mm_cvttps_epi32
#define _mms_setzero_si128 _mm_setzero_si128
#define _mms_and_si128 _mm_and_si128
#define _mms_add_epi32 _mm_add_epi32

#define _mmw_castsi128_ps _mm_castsi128_ps
#define _mmw_cmpge_ps(a,b)          _mm_cmpge_ps(a, b)
#define _mmw_shuffle_ps _mm_shuffle_ps

__m128 _mmw_blendv_ps(const __m128 &a, const __m128 &b, const __m128 &c)
{   
    __m128 cond = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c), 31));
    return _mm_or_ps(_mm_andnot_ps(cond, a), _mm_and_ps(cond, b));
}

#define MAKE_ACCESSOR(name, simd_type, base_type, is_const, elements) \
    FORCE_INLINE is_const base_type * name(is_const simd_type &a) { \
        union accessor { simd_type m_native; base_type m_array[elements]; }; \
        is_const accessor *acs = reinterpret_cast<is_const accessor*>(&a); \
        return acs->m_array; \
    }

MAKE_ACCESSOR(simd_f32, __m128, float, , 4)
MAKE_ACCESSOR(simd_f32, __m128, float, const, 4)
MAKE_ACCESSOR(simd_i32, __m128i, int, , 4)
MAKE_ACCESSOR(simd_i32, __m128i, int, const, 4)

#define _mmw_or_epi32               _mm_or_si128
#define _mmw_andnot_si128           _mm_andnot_si128
#define _mmw_subs_epu16             _mm_subs_epu16
#define _mmw_srai_epi32             _mm_srai_epi32
#define _mmw_srli_epi32             _mm_srli_epi32
#define _mmw_slli_epi32             _mm_slli_epi32
#define _mmw_sll_epi32             _mm_sll_epi32
#define _mmw_cvtps_epi32            _mm_cvtps_epi32
#define _mmw_cvttps_epi32           _mm_cvttps_epi32

//sse4.1
//#define _mmx_dp4_ps                 _mm_dp_ps
FORCE_INLINE __ms _mmx_dp4_ps(const __ms& a, const __ms& b)
{
    // Product and two shuffle/adds pairs (similar to hadd_ps)
    __m128 prod = _mm_mul_ps(a, b);
    __m128 dp = _mm_add_ps(prod, _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1)));
    dp = _mm_add_ps(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(0, 1, 2, 3)));
    return dp;
}
FORCE_INLINE __ms _mmw_floor_ps(const __ms& a) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
FORCE_INLINE __ms _mmw_ceil_ps(const __ms& a) { return _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC); }
FORCE_INLINE __msi _mmw_transpose_epi8(const __msi& a)
{
    const __m128i shuff = _mm_setr_epi8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
    return _mm_shuffle_epi8(a, shuff);
}

FORCE_INLINE __m128i _mmw_sllv_ones(const __m128i &ishift)
{
    __m128i shift = _mm_min_epi32(ishift, _mm_set1_epi32(32));
    print_simd(shift);
    
    // Uses scalar approach to perform _mm_sllv_epi32(~0, shift)
    static const unsigned int maskLUT[33] = {
        ~0U << 0, ~0U << 1, ~0U << 2 ,  ~0U << 3, ~0U << 4, ~0U << 5, ~0U << 6 , ~0U << 7, ~0U << 8, ~0U << 9, ~0U << 10 , ~0U << 11, ~0U << 12, ~0U << 13, ~0U << 14 , ~0U << 15,
        ~0U << 16, ~0U << 17, ~0U << 18 , ~0U << 19, ~0U << 20, ~0U << 21, ~0U << 22 , ~0U << 23, ~0U << 24, ~0U << 25, ~0U << 26 , ~0U << 27, ~0U << 28, ~0U << 29, ~0U << 30 , ~0U << 31,
        0U };

    __m128i retMask;
    print_simd(*((__msi *)(simd_i32(shift))));
    simd_i32(retMask)[0] = (int)maskLUT[simd_i32(shift)[0]];
    simd_i32(retMask)[1] = (int)maskLUT[simd_i32(shift)[1]];
    simd_i32(retMask)[2] = (int)maskLUT[simd_i32(shift)[2]];
    simd_i32(retMask)[3] = (int)maskLUT[simd_i32(shift)[3]];
    return retMask;
}

#define _mmw_cmpeq_epi32            _mm_cmpeq_epi32
#define _mmw_cmpgt_epi32            _mm_cmpgt_epi32
#define _mmw_fmsub_ps(a,b,c)               _mm_sub_ps(_mm_mul_ps(a,b), c)
#define _mmw_set1_ps                _mm_set1_ps

#endif