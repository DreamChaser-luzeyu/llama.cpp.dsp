#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string.h>
#include <unordered_map>
#include <sys/types.h>

#include "backend_debug.h"

#include "ggml-dsp/MTUtils.hpp"
#include "ggml.h"
#include "ggml-cpu-impl.h"

#include "dsp_utils.h"
#include "MTUtils.hpp"
#include "MTHelper.hpp"


#define NR_CLUSTER 4
#define NR_CORE_PER_CLUSTER 24

static MTDevice mt_dev;

// ------- 声明算子
static DECLARE_DSP_GFUNC_ARR(trans_32to16);
static DECLARE_DSP_GFUNC_ARR(silu_forward_fp16_general);
static DECLARE_DSP_GFUNC_ARR(trans_16to32);
static DECLARE_DSP_GFUNC_ARR(dsp_softmax_forward_fp16);
static DECLARE_DSP_GFUNC_ARR(dsp_softmax_forward);
static DECLARE_DSP_GFUNC_ARR(dsp_gemv_forward_fp16_big_k_sm);
// static DECLARE_DSP_GFUNC_ARR(dsp_gemm_forward_fp16_big_k_sm);
static DECLARE_DSP_GFUNC_ARR(dsp_gemv_forward_fp16_big_k);
// static DECLARE_DSP_GFUNC_ARR(dsp_gemm_forward_fp16_big_k);
static DECLARE_DSP_GFUNC_ARR(general_sgemm_k_self_fp16);
static DECLARE_DSP_GFUNC_ARR(bmm_fp16);
static DECLARE_DSP_GFUNC_ARR(gemm_forward_fp16);
static DECLARE_DSP_GFUNC_ARR(dsp_new_gemm_fp16);
static DECLARE_DSP_GFUNC_ARR(dsp_rms_norm_forward_fp16);
static DECLARE_DSP_GFUNC_ARR(dsp_rms_norm_nodot_forward_fp16);
static DECLARE_DSP_GFUNC_ARR(dsp_rope_forward_fp16_v2);
static DECLARE_DSP_GFUNC_ARR(softmax_forward_fp16_4d_lastdim_xxx_general);

// ------- 可用核信息
static int valid_core_indexes[NR_CLUSTER][NR_CORE_PER_CLUSTER];
#define GET_PHY_CORE_ID(cluster_id, logic_core_index) (valid_core_indexes[(int)(cluster_id)][(int)(logic_core_index)])
static int nr_valid_core[NR_CLUSTER];
static int invalid_core_bits[NR_CLUSTER];
#define IS_BIT_LOW(val, bit_pos) (!(!!(((uint32_t)(val)) & ((uint32_t)(0x1u) << ((uint32_t)(bit_pos))))))

static const int test_cluster_id = TEST_CLUSTER_ID;

static void dsp_test_and_debug() {
    printf("-------Hello debug-------\n");
    // void * src1_data = NULL;
    // src1_data = mt_malloc(test_cluster_id, 32768, 0ul);


}

extern "C" int mt_get_valid_core(int id);

static inline uint64_t __do_get_timestamp_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

uint64_t memcpy_time_ns = 0ul;
#if PROFILING_MEMCPY
#define memcpy(dst, src, size) do { \
    uint64_t bt = __do_get_timestamp_ns(); \
    memcpy(dst, src, size); \
    mt_flush_all(); \
    uint64_t et = __do_get_timestamp_ns(); \
    memcpy_time_ns += (et - bt); \
} while(0)
#endif

FILE * debug_file;
bool is_initialized = false;
void init_dsp() {
    printf("!!!!!!!!!!!!!!!Init DSP !!!!!!!!!!!!!!!!!\n");
    #if !NO_INIT_DEV
    // ----- 注册算子
    for (int cluster_id = 0; cluster_id < 4; cluster_id++) {
        for (int core_id = 0; core_id < 24; core_id++) {
            REG_DSP_GFUNCTION(trans_32to16, cluster_id, core_id);
            REG_DSP_GFUNCTION(silu_forward_fp16_general, cluster_id, core_id);
            REG_DSP_GFUNCTION(trans_16to32, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_softmax_forward_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_softmax_forward, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_gemv_forward_fp16_big_k_sm, cluster_id, core_id);
            // REG_DSP_GFUNCTION(dsp_gemm_forward_fp16_big_k_sm, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_gemv_forward_fp16_big_k, cluster_id, core_id);
            // REG_DSP_GFUNCTION(dsp_gemm_forward_fp16_big_k, cluster_id, core_id);
            REG_DSP_GFUNCTION(general_sgemm_k_self_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(bmm_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(gemm_forward_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_new_gemm_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_rms_norm_forward_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_rms_norm_nodot_forward_fp16, cluster_id, core_id);
            REG_DSP_GFUNCTION(dsp_rope_forward_fp16_v2, cluster_id, core_id);
            REG_DSP_GFUNCTION(softmax_forward_fp16_4d_lastdim_xxx_general, cluster_id, core_id);
        }
    }
    // ----- 获得可用核
    for (int cluster_id = 0; cluster_id < 4; cluster_id++) {        
        int _invalid_core_bits = mt_get_valid_core(cluster_id);
        invalid_core_bits[cluster_id] = _invalid_core_bits;
        int next_valid_core_index = 0;
        nr_valid_core[cluster_id] = 0;
        for (int core_id = 0; core_id < 24; core_id++) {
            if(IS_BIT_LOW(_invalid_core_bits, core_id)) {
                valid_core_indexes[cluster_id][next_valid_core_index] = core_id;
                next_valid_core_index ++;
                nr_valid_core[cluster_id] ++;
            }
        }
    }
    #endif


    static DSPHelper helper(&mt_dev);
    // it is recommended to use `tail -f <path>` to monitor the result
    debug_file = fopen("/home/tju/luzeyu/llama_original/llama.cpp/dump.out", "w+");

    dsp_test_and_debug();

    is_initialized = true;
}

static inline void call_func_on_all_cores(int cluster_id, const multi_core_dsp_func_t & multi_core_func, unsigned long *args, int params_num, int expected_core_num = 22) {
    if(!is_initialized) {
        init_dsp();
    }

    for (int i = 0; i < expected_core_num; i++) {
        MEM_BARRIER_RW;
        (*(multi_core_func[24 * cluster_id + GET_PHY_CORE_ID(cluster_id, i)])).setParamGeneric(params_num, args);
        MEM_BARRIER_RW;
    }
	for(int i = 0; i < expected_core_num; i++) {
		(*(multi_core_func[24 * cluster_id + GET_PHY_CORE_ID(cluster_id, i)])).setCallFlag();
		MEM_BARRIER_RW;
	}
    mt_flush_all();
    for (int i = 0; i < expected_core_num; i++) {
        MEM_BARRIER_RW;
        (*(multi_core_func[24 * cluster_id + GET_PHY_CORE_ID(cluster_id, i)])).queryWait();
        MEM_BARRIER_RW;
    }
}

void rmsnorm_fp32_dsp(
    void * src, void * dst, size_t nr_rows, size_t nr_cols,
    float eps
) {
    assert(dsp_get_cluster_id_from_ptr(src) == dsp_get_main_cluster());
    assert(dsp_get_cluster_id_from_ptr(dst) == dsp_get_main_cluster());
    assert(nr_cols % 32 == 0);

    void * src_fp16_dsp = dsp_malloc_on_cluster(
        sizeof(__fp16) * nr_rows * nr_cols, dsp_get_main_cluster());
    void * dst_fp16_dsp = dsp_malloc_on_cluster(
        sizeof(__fp16) * nr_rows * nr_cols, dsp_get_main_cluster());

    uint64_t trans_32to16_args[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM((uint64_t)nr_rows * nr_cols),
        DSP_PARAM(src),
        DSP_PARAM(src_fp16_dsp)
    };
    call_func_on_all_cores(dsp_get_main_cluster(), trans_32to16_FuncArr, trans_32to16_args, 4);

    // 如果使用原版的 与权重相乘的小融合算子，则需要传入一个全1的权重（乘完相当于没乘）
    // void * _1_val_fp16_vec = dsp_malloc_on_cluster(sizeof(__fp16) * nr_rows * nr_cols, dsp_get_main_cluster());
    // for(size_t i = 0; i < nr_rows * nr_cols; i++) {
    //     ((__fp16*)_1_val_fp16_vec)[i] = (__fp16)1.0;
    // }
    // mt_flush(_1_val_fp16_vec, nr_rows * nr_cols * sizeof(__fp16));
    // mt_flush_all();

    uint64_t rmsn_args[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM(src_fp16_dsp),
        DSP_PARAM(0x66ccfful), // not used, set anything
        DSP_PARAM(dst_fp16_dsp),
        DSP_PARAM((double)(eps)),
        DSP_PARAM(nr_rows),
        DSP_PARAM(nr_cols)
    };
    call_func_on_all_cores(dsp_get_main_cluster(), dsp_rms_norm_nodot_forward_fp16_FuncArr, rmsn_args, 7);

    uint64_t trans_16to32_args[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM((uint64_t)nr_rows * nr_cols),
        DSP_PARAM(dst_fp16_dsp),
        DSP_PARAM(dst)
    };
    call_func_on_all_cores(test_cluster_id, trans_16to32_FuncArr, trans_16to32_args, 4);

    dsp_free(src_fp16_dsp);
    dsp_free(dst_fp16_dsp);
    // dsp_free(_1_val_fp16_vec);
}

void rope_fp16_ref(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
) {
    assert(nr_cols % 2 == 0);

    __fp16 * sin_vals_fp16 = (__fp16 *)sin_vals;
    __fp16 * cos_vals_fp16 = (__fp16 *)cos_vals;
    
    typedef __fp16(*row_ptr_t)[nr_cols];
    row_ptr_t src_fp16_arr = (row_ptr_t)src;
    row_ptr_t dst_fp16_arr = (row_ptr_t)dst;

    for(size_t cr = 0; cr < nr_rows; cr++) {
        for(size_t cc = 0; cc < nr_cols; cc+=2) {
            size_t sin_cos_idx = cc / 2;
            __fp16 sin_val = sin_vals_fp16[sin_cos_idx];
            __fp16 cos_val = cos_vals_fp16[sin_cos_idx];
            __fp16 x0 = src_fp16_arr[cr][cc];
            __fp16 x1 = src_fp16_arr[cr][cc + 1];
            dst_fp16_arr[cr][cc]     = x0 * cos_val - x1 * sin_val;
            dst_fp16_arr[cr][cc + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

void rope_fp32_ref(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
) {
    assert(nr_cols % 2 == 0);
    assert(nr_rows == 1);

    float * sin_vals_fp32 = (float *)sin_vals;
    float * cos_vals_fp32 = (float *)cos_vals;
    
    typedef float(*row_ptr_t)[nr_cols];
    row_ptr_t src_fp32_arr = (row_ptr_t)src;
    row_ptr_t dst_fp32_arr = (row_ptr_t)dst;

    for(size_t cr = 0; cr < nr_rows; cr++) {
        for(size_t cc = 0; cc < nr_cols; cc+=2) {
            size_t sin_cos_idx = cc / 2;
            float sin_val = sin_vals_fp32[sin_cos_idx];
            float cos_val = cos_vals_fp32[sin_cos_idx];
            float x0 = src_fp32_arr[cr][cc];
            float x1 = src_fp32_arr[cr][cc + 1];
            dst_fp32_arr[cr][cc]     = x0 * cos_val - x1 * sin_val;
            dst_fp32_arr[cr][cc + 1] = x0 * sin_val + x1 * cos_val;
        }
    }    
}

void rope_fp32_dsp(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
) {
    assert(dsp_get_cluster_id_from_ptr(src) == dsp_get_main_cluster());
    assert(dsp_get_cluster_id_from_ptr(sin_vals) == dsp_get_main_cluster());
    assert(dsp_get_cluster_id_from_ptr(cos_vals) == dsp_get_main_cluster());
    mt_flush_all();

    // ----- 
    void * src_fp16 = dsp_malloc_on_cluster(sizeof(__fp16) * nr_rows * nr_cols, dsp_get_main_cluster());
    uint64_t trans_32to16_args[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM((uint64_t)nr_rows * nr_cols),
        DSP_PARAM(src),
        DSP_PARAM(src_fp16)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args, 4);

    void * sin_vals_fp16 = dsp_malloc_on_cluster(sizeof(__fp16) * nr_cols, dsp_get_main_cluster());
    uint64_t trans_32to16_args_2[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM((uint64_t)nr_cols),
        DSP_PARAM(sin_vals),
        DSP_PARAM(sin_vals_fp16)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args_2, 4);

    void * cos_vals_fp16 = dsp_malloc_on_cluster(sizeof(__fp16) * nr_cols, dsp_get_main_cluster());
    uint64_t trans_32to16_args_3[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM((uint64_t)nr_cols),
        DSP_PARAM(cos_vals),
        DSP_PARAM(cos_vals_fp16)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args_3, 4);

    void * dst_fp16 = dsp_malloc_on_cluster(sizeof(__fp16) * nr_rows * nr_cols, dsp_get_main_cluster());
    uint64_t args[] = {
        DSP_PARAM(0x000000ul),
        DSP_PARAM(src_fp16),
        DSP_PARAM(sin_vals_fp16),
        DSP_PARAM(cos_vals_fp16),
        DSP_PARAM(dst_fp16),
        DSP_PARAM(nr_rows),
        DSP_PARAM(nr_cols),
        DSP_PARAM(0ul) // unused
    };
    call_func_on_all_cores(dsp_get_main_cluster(), dsp_rope_forward_fp16_v2_FuncArr, args, 8);

    uint64_t trans_16to32_args[] = {
        (size_t)invalid_core_bits[dsp_get_main_cluster()],
        DSP_PARAM((uint64_t)nr_rows * nr_cols),
        DSP_PARAM(dst_fp16),
        DSP_PARAM(dst)
    };
    call_func_on_all_cores(test_cluster_id, trans_16to32_FuncArr, trans_16to32_args, 4);

    dsp_free(src_fp16);
    dsp_free(sin_vals_fp16);
    dsp_free(cos_vals_fp16);
    dsp_free(dst_fp16);
}

void rope_fp16_dsp(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
) {
    assert(dsp_get_cluster_id_from_ptr(src) == dsp_get_main_cluster());
    assert(dsp_get_cluster_id_from_ptr(sin_vals) == dsp_get_main_cluster());
    assert(dsp_get_cluster_id_from_ptr(cos_vals) == dsp_get_main_cluster());

    uint64_t args[] = {
        DSP_PARAM(0x000000ul),
        DSP_PARAM(src),
        DSP_PARAM(sin_vals),
        DSP_PARAM(cos_vals),
        DSP_PARAM(dst),
        DSP_PARAM(nr_rows),
        DSP_PARAM(nr_cols),
        DSP_PARAM(0ul) // unused
    };
    call_func_on_all_cores(dsp_get_main_cluster(), dsp_rope_forward_fp16_v2_FuncArr, args, 8);
}

void softmax_fp32_ref(
    void * src, void * dst, size_t nr_row, size_t nr_col
) {
    size_t nc = nr_col;
    assert(nr_row == 1l);
    float * wp = (float*)src;
    float * dp = (float*)dst;
    
    // 1 - 从wp取最大值
    float max = -INFINITY;
    for(size_t i = 0; i < nc; i++) { max = std::max(max, wp[i]); } // ggml_vec_max_f32(nc, &max, wp);
    // 2 - 进行softmax计算
    typedef double ggml_float;
    ggml_float sum = 0;
    // 2.1 计算dp每个元素 expf(wp[i] - max)
    for (size_t i = 0; i < nc; ++i) {
        float val = expf(wp[i] - max);
        sum += (ggml_float)val;
        dp[i] = val;
    }
    assert(sum > 0.0);
    // 2.2 归一化
    sum = 1.0/sum;
    for(size_t i = 0; i < nc; i++) { ((float*)dp)[i] *= sum; }     // ggml_vec_scale_f32(nc, dp, sum);
}

void softmax_fp32_dsp(
    void * src, void * dst, size_t nr_row, size_t nr_col
) {
    assert(dsp_get_cluster_id_from_ptr(src) == dsp_get_main_cluster());
    assert(dsp_get_cluster_id_from_ptr(dst) == dsp_get_main_cluster());
    uint64_t args[5] = {
        DSP_PARAM(0x000000ul),
        DSP_PARAM(src),
        DSP_PARAM(dst),
        DSP_PARAM(nr_row),
        DSP_PARAM(nr_col)
    };
    call_func_on_all_cores(dsp_get_main_cluster(), dsp_softmax_forward_FuncArr, args, 5);
    // void * inputs_fp16 = dsp_malloc_on_cluster(sizeof(__fp16) * nr_row * nr_col, dsp_get_main_cluster());
    // void * outputs_fp16 = dsp_malloc_on_cluster(sizeof(__fp16) * nr_row * nr_col, dsp_get_main_cluster());
    // uint64_t args[7] = {
    //     DSP_PARAM(0x000000ul),
    //     DSP_PARAM(src),
    //     DSP_PARAM(dst),
    //     DSP_PARAM(inputs_fp16),
    //     DSP_PARAM(outputs_fp16),
    //     DSP_PARAM(nr_row),
    //     DSP_PARAM(nr_col)
    // };
    // mt_flush_all();
    // call_func_on_all_cores(dsp_get_main_cluster(), softmax_forward_fp16_4d_lastdim_xxx_general_FuncArr, args, 7);
    // dsp_free(inputs_fp16);
    // dsp_free(outputs_fp16);
}

void silu_fp32_dsp(const int n, float * dst, const float * src) {
    // test_mutex.lock();
    assert(test_cluster_id == dsp_get_main_cluster());
    
    void * src_fp32 = (void*)src;
    if(dsp_get_cluster_id_from_ptr((void*)src) != test_cluster_id) {
        src_fp32 = dsp_malloc_on_cluster(sizeof(float) * n, dsp_get_main_cluster());
        memcpy(src_fp32, src, sizeof(float) * n);
        mt_flush_all();
        assert(false);
    }
    
    // --- 开辟空间
    void * src_fp16 = mt_malloc(test_cluster_id, sizeof(__fp16) * n, 0ul);
    // void * dst_fp32 = mt_malloc(test_cluster_id, sizeof(float) * n, 0ul);
    void * dst_fp16 = mt_malloc(test_cluster_id, sizeof(__fp16) * n, 0ul);
    
    // --- 量化为fp16
    uint64_t trans_32to16_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)n),
        DSP_PARAM(src_fp32),
        DSP_PARAM(src_fp16)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args, 4);
    
    // --- 进行运算
    uint64_t silu_forward_fp16_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM(src_fp16),
        DSP_PARAM(dst_fp16),
        DSP_PARAM((uint64_t)n)
    };
    call_func_on_all_cores(test_cluster_id, silu_forward_fp16_general_FuncArr, silu_forward_fp16_args, 4);
    
    void * dst_fp32 = dst;
    if(dsp_get_cluster_id_from_ptr(dst) != dsp_get_main_cluster()) {
        // memcpy(dst, dst_fp32, sizeof(float) * n);
        // mt_flush_all();
        dst_fp32 = dsp_malloc_on_cluster(sizeof(float) * n, dsp_get_main_cluster());
        assert(false);
    }

    // --- 转回float
    uint64_t trans_16to32_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)n),
        DSP_PARAM(dst_fp16),
        DSP_PARAM(dst_fp32)
    };
    call_func_on_all_cores(test_cluster_id, trans_16to32_FuncArr, trans_16to32_args, 4);

    

    mt_free(src_fp16, 0);
    if(src_fp32 != src) { mt_free(src_fp32, 0); }
    mt_free(dst_fp16, 0);
    if(dst_fp32 != dst) { 
        memcpy(dst, dst_fp32, sizeof(float) * n);
        mt_free(dst_fp32, 0); 
    }
    mt_flush_all();
}

void trans_fp32_to_fp16_dsp(void * src, void * dst, size_t nr_elem) {
    // test_mutex.lock();

    // if(!is_initialized) { 
    //     init_dsp(); 
    //     is_initialized = true;
    // }
    
    // --- 开辟空间
    void * data_fp32 = mt_malloc(test_cluster_id, sizeof(float) * nr_elem, 0ul);
    void * data_fp16 = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_elem, 0ul);
    
    // --- 量化为fp16
    memcpy(data_fp32, src, sizeof(float) * nr_elem);
    mt_flush_all();
    uint64_t trans_32to16_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)nr_elem),
        DSP_PARAM(data_fp32),
        DSP_PARAM(data_fp16)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args, 4);
    memcpy(dst, data_fp16, sizeof(__fp16) * nr_elem);

    // test_mutex.unlock();
}

void matmul_fp16_ref(
    void * lmat_data_fp16,
    void * rmat_data_fp16,
    void * dst_data_fp16,
    size_t m, size_t k, size_t n
) {

    #define BMAT_ACCESS_FP16(ptr, nb, nr, nc) (*((__fp16(*)[nb][nr][nc])(ptr)))
    #define BMAT_ACCESS_FP32(ptr, nb, nr, nc) (*(( float(*)[nb][nr][nc])(ptr)))

    #define BMAT_ACCESS_LMAT (BMAT_ACCESS_FP16(lmat_data_fp16, 1, m, k))
    #define BMAT_ACCESS_RMAT (BMAT_ACCESS_FP16(rmat_data_fp16, 1, k, n))
    #define BMAT_ACCESS_DST  (BMAT_ACCESS_FP16( dst_data_fp16, 1, m, n))

    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            __fp16 result = 0.0;
            for(size_t curr_line_idx = 0; curr_line_idx < k; curr_line_idx++) {
                result += 
                    (BMAT_ACCESS_LMAT[0][i][curr_line_idx]) * (BMAT_ACCESS_RMAT[0][curr_line_idx][j]);
            }
            (BMAT_ACCESS_DST)[0][i][j] = result;
        }
    }
}

void matmul_fp16_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data,
    size_t m, size_t k, size_t n
) {
    // if(!is_initialized) { 
    //     init_dsp(); 
    //     is_initialized = true;
    // }

    void * lmat_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * k, 0x0);
    memcpy(lmat_data_dsp, lmat_data, sizeof(__fp16) * m * k);
    void * rmat_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * k, 0x0);
    memcpy(rmat_data_dsp, rmat_data, sizeof(__fp16) * k * n);
    mt_flush_all();

    void * dst_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * n, 0x0);

    if(m == 1ul) {
        // m == 1 时使用gemv算子
        uint64_t args[] = {
            0x000000ul,
            DSP_PARAM(lmat_data_dsp),
            DSP_PARAM(rmat_data_dsp),
            DSP_PARAM(dst_data_dsp),
            DSP_PARAM(m),
            DSP_PARAM(k),
            DSP_PARAM(n)
        };
        call_func_on_all_cores(
            test_cluster_id, 
            dsp_gemv_forward_fp16_big_k_sm_FuncArr, 
            args, 
            7
        );
    } else {
        // 否则使用gemm算子
        uint64_t args[] = {
            0x000000ul,
            DSP_PARAM(lmat_data_dsp),
            DSP_PARAM(rmat_data_dsp),
            DSP_PARAM(dst_data_dsp),
            DSP_PARAM(m),
            DSP_PARAM(k),
            DSP_PARAM(n)
        };
        call_func_on_all_cores(
            test_cluster_id, 
            gemm_forward_fp16_FuncArr, 
            args, 
            7
        );
    }
    
    mt_flush_all();
    memcpy(dst_data, dst_data_dsp, sizeof(__fp16) * m * n);
    mt_free(lmat_data_dsp, 0ul);
    mt_free(rmat_data_dsp, 0ul);
    mt_free(dst_data_dsp, 0ul);
}

void matmul_shs_ref(
    void * lmat_data_fp32,
    void * rmat_data_fp16,
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
) {

    #define BMAT_ACCESS_FP16(ptr, nb, nr, nc) (*((__fp16(*)[nb][nr][nc])(ptr)))
    #define BMAT_ACCESS_FP32(ptr, nb, nr, nc) (*(( float(*)[nb][nr][nc])(ptr)))

    #define BMAT_ACCESS_LMAT (BMAT_ACCESS_FP32(lmat_data_fp32, 1, m, k))
    #define BMAT_ACCESS_RMAT (BMAT_ACCESS_FP16(rmat_data_fp16, 1, k, n))
    #define BMAT_ACCESS_DST  (BMAT_ACCESS_FP32( dst_data_fp32, 1, m, n))

    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            __fp16 result = 0.0;
            for(size_t curr_line_idx = 0; curr_line_idx < k; curr_line_idx++) {
                result += 
                    (BMAT_ACCESS_LMAT[0][i][curr_line_idx]) * (BMAT_ACCESS_RMAT[0][curr_line_idx][j]);
            }
            (BMAT_ACCESS_DST)[0][i][j] = result;
        }
    }
}

void matmul_shs_dsp(
    void * lmat_data_fp32,
    void * rmat_data_fp16,
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
) {
    mt_flush_all();

    void * lmat_data_fp32_dsp;
    bool lmat_data_fp32_dsp_needs_free = false;
    if(dsp_get_cluster_id_from_ptr(lmat_data_fp32) != test_cluster_id) {
        lmat_data_fp32_dsp = mt_malloc(test_cluster_id, sizeof(float) * m * k, 0x0);
        assert(lmat_data_fp32_dsp);
        memcpy(lmat_data_fp32_dsp, lmat_data_fp32, sizeof(float) * m * k);
        lmat_data_fp32_dsp_needs_free = true;
        mt_flush_all();
        assert(false);
    } 
    else {
        lmat_data_fp32_dsp = lmat_data_fp32;
    }
    
    // --- 将lmat量化为fp16
    void * lmat_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * k, 0x0);
    assert(lmat_data_fp16_dsp);
    uint64_t trans_32to16_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)(m * k)),
        DSP_PARAM(lmat_data_fp32_dsp),
        DSP_PARAM(lmat_data_fp16_dsp)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args, 4);
    mt_flush_all();

    // --- 计算矩阵乘
    void * rmat_data_fp16_dsp = NULL;
    bool rmat_data_fp16_dsp_needs_free = false;
    if(dsp_get_cluster_id_from_ptr(rmat_data_fp16) != test_cluster_id) {
        rmat_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * k * n, 0x0);
        assert(rmat_data_fp16_dsp);
        memcpy(rmat_data_fp16_dsp, rmat_data_fp16, sizeof(__fp16) * k * n);
        mt_flush_all();
        rmat_data_fp16_dsp_needs_free = true;
        assert(false);
    }
    else {
        rmat_data_fp16_dsp = rmat_data_fp16;
    }
    
    void * dst_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * n, 0x0);
    assert(dst_data_fp16_dsp);
    void * tmp_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * (m * n + n * k), 0x0);
    assert(tmp_data_fp16_dsp);

    uint64_t args[] = {
        DSP_PARAM(lmat_data_fp16_dsp),
        DSP_PARAM(0ul),
        DSP_PARAM(rmat_data_fp16_dsp),
        DSP_PARAM(0ul),
        DSP_PARAM(dst_data_fp16_dsp),
        DSP_PARAM(tmp_data_fp16_dsp),
        DSP_PARAM(m),
        DSP_PARAM(k),
        DSP_PARAM(n)
    };
    call_func_on_all_cores(
        test_cluster_id, 
        dsp_new_gemm_fp16_FuncArr, 
        args, 
        9
    );
    mt_flush_all();

    // --- 将dst反量化为float
    void * dst_data_fp32_dsp = NULL;
    bool dst_data_fp32_dsp_needs_free = false;
    if(dsp_get_cluster_id_from_ptr(dst_data_fp32) == test_cluster_id) {
        dst_data_fp32_dsp = dst_data_fp32;
    }
    else {
        dst_data_fp32_dsp = mt_malloc(test_cluster_id, sizeof(float) * m * n, 0x0);   
        dst_data_fp32_dsp_needs_free = true;
        assert(false);
    }
    assert(dst_data_fp32_dsp);
    uint64_t trans_16to32_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)(m * n)),
        DSP_PARAM(dst_data_fp16_dsp),
        DSP_PARAM(dst_data_fp32_dsp)
    };
    call_func_on_all_cores(test_cluster_id, trans_16to32_FuncArr, trans_16to32_args, 4);
    mt_flush_all();
    
    // memcpy(dst_data_fp32, dst_data_fp32_dsp, sizeof(float) * (m * n));

    if(lmat_data_fp32_dsp_needs_free) { mt_free(lmat_data_fp32_dsp, 0ul); }
    mt_free(lmat_data_fp16_dsp, 0ul);
    if(rmat_data_fp16_dsp_needs_free) { mt_free(rmat_data_fp16_dsp, 0ul); }
    mt_free(dst_data_fp16_dsp, 0ul);
    if(dst_data_fp32_dsp_needs_free) { mt_free(dst_data_fp32_dsp, 0ul); }
    mt_free(tmp_data_fp16_dsp, 0x0ul);
    mt_flush_all();
}

void matmul_shs_rt_dsp(
    void * lmat_data_fp32,
    void * rmat_data_fp16,   // 权重
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
) {
    mt_flush_all();

    void * lmat_data_fp32_dsp;
    bool lmat_data_fp32_dsp_needs_free = false;
    if(dsp_get_cluster_id_from_ptr(lmat_data_fp32) != test_cluster_id) {
        lmat_data_fp32_dsp = mt_malloc(test_cluster_id, sizeof(float) * m * k, 0x0);
        assert(lmat_data_fp32_dsp);
        memcpy(lmat_data_fp32_dsp, lmat_data_fp32, sizeof(float) * m * k);
        lmat_data_fp32_dsp_needs_free = true;
        mt_flush_all();
    } 
    else {
        lmat_data_fp32_dsp = lmat_data_fp32;
    }
    
    // --- 将lmat量化为fp16
    void * lmat_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * k, 0x0);
    assert(lmat_data_fp16_dsp);
    uint64_t trans_32to16_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)(m * k)),
        DSP_PARAM(lmat_data_fp32_dsp),
        DSP_PARAM(lmat_data_fp16_dsp)
    };
    call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args, 4);
    mt_flush_all();

    // --- 计算矩阵乘
    void * rmat_data_fp16_dsp = NULL;
    bool rmat_data_fp16_dsp_needs_free = false;
    if(dsp_get_cluster_id_from_ptr(rmat_data_fp16) != test_cluster_id) {
        rmat_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * k * n, 0x0);
        assert(rmat_data_fp16_dsp);
        memcpy(rmat_data_fp16_dsp, rmat_data_fp16, sizeof(__fp16) * k * n);
        mt_flush_all();
        rmat_data_fp16_dsp_needs_free = true;
    }
    else {
        rmat_data_fp16_dsp = rmat_data_fp16;
    }
    
    void * dst_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * n, 0x0);
    assert(dst_data_fp16_dsp);
    void * tmp_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * (m * n + n * k), 0x0);
    assert(tmp_data_fp16_dsp);

    uint64_t args[] = {
        DSP_PARAM(lmat_data_fp16_dsp),
        DSP_PARAM(0ul),
        DSP_PARAM(rmat_data_fp16_dsp),
        DSP_PARAM(1ul),
        DSP_PARAM(dst_data_fp16_dsp),
        DSP_PARAM(tmp_data_fp16_dsp),
        DSP_PARAM(m),
        DSP_PARAM(k),
        DSP_PARAM(n)
    };
    call_func_on_all_cores(
        test_cluster_id, 
        dsp_new_gemm_fp16_FuncArr, 
        args, 
        9
    );
    mt_flush_all();

    // --- 将dst反量化为float
    void * dst_data_fp32_dsp = NULL;
    bool dst_data_fp32_dsp_needs_free = false;
    if(dsp_get_cluster_id_from_ptr(dst_data_fp32) == test_cluster_id) {
        dst_data_fp32_dsp = dst_data_fp32;
    }
    else {
        dst_data_fp32_dsp = mt_malloc(test_cluster_id, sizeof(float) * m * n, 0x0);   
        dst_data_fp32_dsp_needs_free = true;
    }
    assert(dst_data_fp32_dsp);
    uint64_t trans_16to32_args[] = {
        (size_t)invalid_core_bits[test_cluster_id],
        DSP_PARAM((uint64_t)(m * n)),
        DSP_PARAM(dst_data_fp16_dsp),
        DSP_PARAM(dst_data_fp32_dsp)
    };
    call_func_on_all_cores(test_cluster_id, trans_16to32_FuncArr, trans_16to32_args, 4);
    mt_flush_all();
    
    memcpy(dst_data_fp32, dst_data_fp32_dsp, sizeof(float) * (m * n));

    if(lmat_data_fp32_dsp_needs_free) { mt_free(lmat_data_fp32_dsp, 0ul); }
    mt_free(lmat_data_fp16_dsp, 0ul);
    if(rmat_data_fp16_dsp_needs_free) { mt_free(rmat_data_fp16_dsp, 0ul); }
    mt_free(dst_data_fp16_dsp, 0ul);
    if(dst_data_fp32_dsp_needs_free) { mt_free(dst_data_fp32_dsp, 0ul); }
    mt_free(tmp_data_fp16_dsp, 0x0ul);
    mt_flush_all();
}

// void matmul_shs_rmat_transpose_dsp(
//     void * lmat_data_fp32,
//     void * rmat_data_fp16,
//     void * dst_data_fp32,
//     size_t m, size_t k, size_t n
// ) {

//     // --- 将lmat量化为fp16
//     void * lmat_data_fp32_dsp = mt_malloc(test_cluster_id, sizeof(float) * m * k, 0x0);
//     memcpy(lmat_data_fp32_dsp, lmat_data_fp32, sizeof(float) * m * k);
//     mt_flush_all();
//     void * lmat_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * k, 0x0);
//     uint64_t trans_32to16_args[] = {
//         (size_t)invalid_core_bits[test_cluster_id],
//         DSP_PARAM((uint64_t)(m * k)),
//         DSP_PARAM(lmat_data_fp32_dsp),
//         DSP_PARAM(lmat_data_fp16_dsp)
//     };
//     call_func_on_all_cores(test_cluster_id, trans_32to16_FuncArr, trans_32to16_args, 4);
//     mt_flush_all();

//     // printf("\n!!!\n");
//     // for(int i = 0; i < 32; i++) {
//     //     printf("%.4f ", ((__fp16 *)(lmat_data_fp16_dsp))[i]);
//     // }
//     // printf("\n!!!\n");

//     // --- 计算矩阵乘
//     void * rmat_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * k * n, 0x0);
//     memcpy(rmat_data_fp16_dsp, rmat_data_fp16, sizeof(__fp16) * k * n);
//     mt_flush_all();
//     void * dst_data_fp16_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * m * n, 0x0);
//     void * temp_space_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * 1l * (k * n + m * k), 0x0ul);
//     memset(temp_space_dsp, 0, sizeof(__fp16) * 1 * (k * n + m * k));
//     memset(dst_data_fp16_dsp, 0, sizeof(__fp16) * m * n);

//     // for(size_t i = 0; i < (m * k); i++) {
//     //     ((__fp16*)lmat_data_fp16_dsp)[i] = (__fp16)1.0f;
//     // }
//     // for(size_t i = 0; i < (k * n); i++) {
//     //     ((__fp16*)rmat_data_fp16_dsp)[i] = (__fp16)1.0f;
//     // }
//     // m = 8;
//     // n = 4096;
//     // k = 4096;

//     mt_flush_all();
//     uint64_t bmm_args[] = {
//         DSP_PARAM(0x000000ul), // useless
//         DSP_PARAM(0ul), // 矩阵转置标记
//         DSP_PARAM(1ul), // 矩阵转置标记
//         DSP_PARAM(1ul), // batch数量
//         DSP_PARAM(m),
//         DSP_PARAM(n),
//         DSP_PARAM(k),
//         DSP_PARAM(1.0),
//         DSP_PARAM(lmat_data_fp16_dsp),
//         DSP_PARAM(rmat_data_fp16_dsp),
//         DSP_PARAM(0.0),
//         DSP_PARAM(dst_data_fp16_dsp),
//         DSP_PARAM(temp_space_dsp),
//         DSP_PARAM(0x000000ul), // valid core bits
//     };
//     call_func_on_all_cores(test_cluster_id, bmm_fp16_FuncArr, bmm_args, 14);
//     mt_flush_all();
//     mt_free(temp_space_dsp, 0x0ul);


//     // --- 将dst反量化为float
//     void * dst_data_fp32_dsp = mt_malloc(test_cluster_id, sizeof(float) * m * n, 0x0);    
//     uint64_t trans_16to32_args[] = {
//         (size_t)invalid_core_bits[test_cluster_id],
//         DSP_PARAM((uint64_t)(m * n)),
//         DSP_PARAM(dst_data_fp16_dsp),
//         DSP_PARAM(dst_data_fp32_dsp)
//     };
//     call_func_on_all_cores(test_cluster_id, trans_16to32_FuncArr, trans_16to32_args, 4);
//     mt_flush_all();
    
//     memcpy(dst_data_fp32, dst_data_fp32_dsp, sizeof(float) * (m * n));

//     mt_free(lmat_data_fp32_dsp, 0ul);
//     mt_free(lmat_data_fp16_dsp, 0ul);
//     mt_free(rmat_data_fp16_dsp, 0ul);
//     mt_free(dst_data_fp16_dsp, 0ul);
//     mt_free(dst_data_fp32_dsp, 0ul);
// }

void bmm_fp16_rtranspose_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data, 
    size_t nr_batches,
    size_t m, size_t k, size_t n
) {
    void * lmat_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * m * k, 0x0ul);
    void * rmat_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * k * n, 0x0ul);
    void * dst_data_dsp  = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * m * n, 0x0ul);
    memcpy(lmat_data_dsp, lmat_data, sizeof(__fp16) * nr_batches * m * k);
    memcpy(rmat_data_dsp, rmat_data, sizeof(__fp16) * nr_batches * k * n);
    mt_flush_all();

    void * tmp_data = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * (m * k + k * n), 0x0ul);

    for(size_t i = 0; i < nr_batches; i++) {
        void * lmat_current_batch_ptr = ((uint8_t*)lmat_data_dsp) + (i * (sizeof(__fp16) * m * k));
        void * rmat_current_batch_ptr = ((uint8_t*)rmat_data_dsp) + (i * (sizeof(__fp16) * k * n));
        void * dst_current_batch_ptr  = ((uint8_t*) dst_data_dsp) + (i * (sizeof(__fp16) * m * n));

        uint64_t args[] = {
            DSP_PARAM(lmat_current_batch_ptr),
            DSP_PARAM(0ul),
            DSP_PARAM(rmat_current_batch_ptr),
            DSP_PARAM(1ul),
            DSP_PARAM(dst_current_batch_ptr),
            DSP_PARAM(tmp_data),
            DSP_PARAM(m),
            DSP_PARAM(k),
            DSP_PARAM(n)
        };
        call_func_on_all_cores(
            test_cluster_id, 
            dsp_new_gemm_fp16_FuncArr, 
            args, 
            9
        );
    }
    mt_flush_all();

    memcpy(dst_data, dst_data_dsp, sizeof(__fp16) * nr_batches * m * n);

    mt_free(lmat_data_dsp, 0x0ul);
    mt_free(rmat_data_dsp, 0x0ul);
    mt_free(dst_data_dsp, 0x0ul);
    mt_free(tmp_data, 0x0ul);
}

void bmm_shs_rtranspose_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data, 
    size_t nr_batches,
    size_t m, size_t k, size_t n
) {
    for(size_t i = 0; i < nr_batches; i++) {
        void * lmat_current_batch_ptr = ((uint8_t*)lmat_data) + (i * (sizeof(float ) * m * k));
        void * rmat_current_batch_ptr = ((uint8_t*)rmat_data) + (i * (sizeof(__fp16) * k * n));
        void * dst_current_batch_ptr  = ((uint8_t*) dst_data) + (i * (sizeof(float ) * m * n));
        matmul_shs_rt_dsp(
            lmat_current_batch_ptr, 
            rmat_current_batch_ptr, 
            dst_current_batch_ptr, 
            m, k, n
        );
    }    
}

void bmm_fp16_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data, 
    size_t nr_batches,
    size_t m, size_t k, size_t n
) {
    // if(!is_initialized) { 
    //     init_dsp(); 
    //     is_initialized = true;
    // }

    void * lmat_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * m * k, 0x0ul);
    memcpy(lmat_data_dsp, lmat_data, sizeof(__fp16) * nr_batches * m * k);
    void * rmat_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * k * n, 0x0ul);
    memcpy(rmat_data_dsp, rmat_data, sizeof(__fp16) * nr_batches * k * n);
    mt_flush_all();

    void * dst_data_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * m * n, 0x0ul);
    void * temp_space_dsp = mt_malloc(test_cluster_id, sizeof(__fp16) * nr_batches * (k * n + m * k), 0x0ul);
    uint64_t bmm_args[] = {
        DSP_PARAM(0x000000ul),
        DSP_PARAM(0ul), // 矩阵转置标记
        DSP_PARAM(0ul), // 矩阵转置标记
        DSP_PARAM(nr_batches),
        DSP_PARAM(m),
        DSP_PARAM(k),
        DSP_PARAM(n),
        DSP_PARAM(1.0),
        DSP_PARAM(lmat_data_dsp),
        DSP_PARAM(rmat_data_dsp),
        DSP_PARAM(0.0),
        DSP_PARAM(dst_data_dsp),
        DSP_PARAM(temp_space_dsp),
        DSP_PARAM(0x000000ul),
    };
    call_func_on_all_cores(test_cluster_id, bmm_fp16_FuncArr, bmm_args, 14);
    memcpy(dst_data, dst_data_dsp, sizeof(__fp16) * nr_batches * m * n);

    mt_free(lmat_data_dsp, 0x0ul);
    mt_free(rmat_data_dsp, 0x0ul);
    mt_free(dst_data_dsp, 0x0ul);
    mt_free(temp_space_dsp, 0x0ul);
}
