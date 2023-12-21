#include "CrossDefine.h"
//time
#include <chrono>
#include <ctime>
#include <float.h>

int main()
{
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
	//_mm_setcsr(_mm_getcsr() | 0x8040);
#endif

	//_mm_loadu_ps _mm_setr_epi32
	std::cout << "_mm_loadu_ps, _mm_setr_epi32: -----------------------------------------" << std::endl;
	{
		float floats[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		__ms vec = _mms_loadu_ps(floats);
		print_simd(vec);
		vec = _mms_load1_ps(floats);
		print_simd(vec);
		_mms_storeu_ps(floats, vec);
		printf("floats[3]: %f\n", floats[3]);
		__msi veci = _mms_setr_epi32(11, -22, 33, -44);
		print_simd(veci);
	}
	//_mm_movemask_ps
	std::cout << "_mm_movemask_ps: -----------------------------------------" << std::endl;
	{
		float a = -1.0f, b = 2.0f, c = 3.0f, d = -4.0f;
		__ms vec = _mms_setr_ps(a, 2.0f, 3.0f, d);
	    int vec_mask = _mms_movemask_ps(vec);
		std::cout << "vec mask value: " << vec_mask << std::endl;
	}
	//_mm_xor_ps
	std::cout << "_mm_xor_ps: -----------------------------------------" << std::endl;
	{
		float a = -1.0f, b = 2.0f, c = 3.0f, d = -4.0f;
		__ms vec0 = _mms_setr_ps(a, 2.0f, 3.0f, d);
		__ms vec1 = _mms_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
	    __ms vec = _mms_xor_ps(vec0, vec1);
		print_simd(vec);
	}
	//_mm_div_ps _mm_sub_ps _mm_cvttps_epi32
	std::cout << "_mm_div_ps, _mms_sub_ps, _mm_cvttps_epi32, _mm_setzero_si128, _mms_and_si128, _mms_add_si128: -----------------------------------------" << std::endl;
	{
		__ms vec0 = _mms_setr_ps(8.0f, 2.0f, 3.0f, 1.0f);
		__ms vec1 = _mms_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
	    __ms vec = _mms_div_ps(vec0, vec1);
		print_simd(vec);
		vec = _mms_sub_ps(vec0, vec1);
		print_simd(vec);
		__msi veci = _mms_cvttps_epi32(vec);
		print_simd(veci);
		veci = _mms_setzero_si128();
		print_simd(veci);
		__msi veci0 = _mms_cvttps_epi32(vec0);
		__msi veci1 = _mms_cvttps_epi32(vec1);
		//error: cannot convert ‘<brace-enclosed initializer list>’ to ‘int32x4_t’ <- _mms_setr_epi32(1, 1, 1, 1) <- need to be lvalue
		//veci = _mms_and_si128(_mms_add_epi32(_mms_cvttps_epi32(vec0), _mms_setr_epi32(1, 1, 1, 1)), _mms_cvttps_epi32(vec1));
		__msi vecii = _mms_setr_epi32(1, 1, 1, 1);
		veci = _mms_and_si128(_mms_add_epi32( _mms_cvttps_epi32(vec0), vecii), _mms_cvttps_epi32(vec1));
		print_simd(veci);
	}
	//_mm_castsi128_ps
	std::cout << "_mm_castsi128_ps: -----------------------------------------" << std::endl;
	{
		__ms vec = _mms_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
	    __msi veci0 = _mms_cvttps_epi32(vec);
	    __ms vec0 = _mmw_castsi128_ps(veci0);
	    
	    print_simd(veci0);
		print_simd(vec0);

		__msi veci = _mms_setr_epi32(1 << 30, 1 << 30, 1 << 30, 1 << 30);
		vec0 = _mmw_castsi128_ps(veci);
		print_simd(vec0);
	}
	// //mm_cmpge_ps
	// std::cout << "mm_cmpge_ps: -----------------------------------------" << std::endl;
	// {
	// 	__ms a = {1.0f, 2.0f, 3.0f, 4.0f};
	// 	__ms b = {2.0f, 2.0f, 1.0f, 2.0f};
	// 	__ms result = _mmw_cmpge_ps(a, b);
	// 	print_simd(result);
	// }
	//_mm_shuffle_ps
	std::cout << "_mm_shuffle_ps: -----------------------------------------" << std::endl;
	{
		__ms a = _mms_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
		__ms b = _mms_setr_ps(5.0f, 6.0f, 7.0f, 8.0f);
		//__ms result = _mmw_shuffle_ps(a, b, 0);
		__ms result = _mms_setr_ps(a[1], a[3], b[1], b[3]);
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
		result = _mmw_shuffle_ps(a, b, 0xDD);
#endif
		print_simd(result);

		result = _mms_setr_ps(a[0], a[2], b[0], b[2]);
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
		result = _mmw_shuffle_ps(a, b, 0x88);
#endif
		print_simd(result);

		result = _mms_setr_ps(a[0], a[1], b[0], b[1]);
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
		result = _mmw_shuffle_ps(a, b, 0x44);
#endif
		print_simd(result);

result = _mms_setr_ps(a[2], a[3], b[2], b[3]);
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
		result = _mmw_shuffle_ps(a, b, 0xEE);
#endif
		print_simd(result);
	}
	//_mm_blendv_ps simd_f32
	std::cout << "_mm_blendv_ps simd_f32: -----------------------------------------" << std::endl;
	{
		__ms a = _mms_setr_ps(1.0f, 2.0f, 3.0f, 4.0f);
		__ms b = _mms_setr_ps(5.0f, 6.0f, 7.0f, 8.0f);
#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
		__msu maski = _mms_setr_epu32(-0x7FFFFFFF, 0x00000000, -0x7FFFFFFF, 0x00000000);
		__ms mask = _mm_castsi128_ps(maski);
#else
		__msu maski = _mms_setr_epu32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000);
		__ms mask = vreinterpretq_f32_u32(maski);
#endif
		__ms result = _mmw_blendv_ps(a, b, mask);
		print_simd(result);

		std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++\n" ;
		a = _mms_setr_ps(1922.0f, 1922.0f, 1922.0f, 0.0f);
		b = _mms_setr_ps(1248.0f, 1440.0f, 1922.0f, 0.0f);
		mask = _mms_setr_ps(272.0f, -69376.0f, -270.0f, 0.0f);
		result = _mmw_blendv_ps(a, b, mask);
		print_simd(result);

		printf("%f\n", simd_f32(result)[0]);
	}
	//_mmw_or_epi32 _mm_andnot_epi32 _mm_srai_epi32
	std::cout << "_mm_blendv_ps _mm_andnot_epi32 _mm_srai_epi32 : -----------------------------------------" << std::endl;
	{
		__msi a = _mms_setr_epi32(-6, -1, 0, 4);
		__msi b = _mms_setr_epi32(-5, 1, -7, 7);
		__msi result = _mmw_or_epi32(a, b);
		print_simd(result);
		//~a & b -> if a is negative using two's complement
		printf("_mm_andnot_si128 / vbicq_s32: "); result = _mmw_andnot_si128(a, b);
		print_simd(result);
		result = _mmw_subs_epu16(a,b);
		print_simd(result);

		a = _mms_setr_epi32(100, -200, 2139095039, -400);
		int shift = 1;
		result = _mmw_srai_epi32(a, shift);
		//without sign bit shift
		printf("_mm_srai_epi32 / vshrq_n_s32: "); print_simd(result);
		result = _mmw_srli_epi32(a, shift);
		//with sign bit shift / neon supports uint hence using reinterpretation
		printf("_mm_srli_epi32 / vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), shift)): "); print_simd(result);
		result = _mmw_slli_epi32(a, shift);
		//with sign bit shift
		printf("_mm_slli_epi32 / vshlq_n_s32: "); print_simd(result);

		__ms aa = _mms_setr_ps(1.2f, 2.6f, -3.9f, 4.8f);
		result = _mmw_cvtps_epi32(aa);
		print_simd(result);
		result = _mmw_cvttps_epi32(aa);
		print_simd(result);
		__ms results = _mmw_floor_ps(aa);
		print_simd(results);
		results = _mmw_ceil_ps(aa);
		print_simd(results);
		result = _mmw_transpose_epi8(a);
		print_simd(result);

		a = _mms_setr_epi32(11, 2212, 23, 22);
		result = _mmw_sllv_ones(a);
		printf("_mmw_sllv_ones: "); print_simd(result);
	}
	std::cout << " : -----------------------------------------" << std::endl;
	{
		__ms a = _mmw_set1_ps(10.0f);
		__ms result = _mmw_fmsub_ps(a, _mmw_set1_ps(2.0f), _mmw_set1_ps(-3.0f));
		printf("_mmw_fmsub_ps / vfmsq_f32 :"); print_simd(result);
	}
	std::cout << " : -----------------------------------------" << std::endl;
	//sse4.1
	{
		__msi a = _mms_setr_epi32(-6, 6, 8, 4);
		__msi b = _mms_setr_epi32(5, 6, 7, 7);

		__msi result = _mmw_cmpeq_epi32(a,b);
		printf("_mm_cmpeq_epi32 / vreinterpretq_s32_u32(vceqq_s32(a,b)): "); print_simd(result);
		result = _mmw_cmpgt_epi32(a,b);
		print_simd(result);
	}

	return 0;
}
