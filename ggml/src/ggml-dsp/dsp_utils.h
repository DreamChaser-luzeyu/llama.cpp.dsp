#pragma once

#ifdef __cplusplus
extern "C" {
#endif
// using cpp guard because ggml-cpu.c is a C source so we need to export using C convension
#include <stdint.h>
#include <stddef.h>


// ------- Other utils

typedef uint64_t transaction_t;

int dsp_get_main_cluster();

int dsp_get_cluster_id_from_ptr(void * ptr);

void * dsp_get_base_from_ptr(void * ptr);

void * dsp_malloc_on_cluster(size_t size, int cluster_id);

void * dsp_malloc_on_cluster_at(size_t size, int cluster_id, void * fixed_addr);

void dsp_free(void * ptr);

uint64_t dsp_memcpy_async(void * dst, const void * src, size_t size);

void dsp_transc_wait(uint64_t transaction_id);

void dsp_transc_wait_all();

// ------- DSP Kernels
/*
    `shs`标记表示左矩阵为单精度(float)，右矩阵为半精度(__fp16)，结果为单精度(float)
    传入的指针既可以是dsp_malloc分配的，也可以不是；如果不是，将在算子内部进行拷贝
    如果未使用包装过的dsp_malloc，而是直接使用mt_malloc，会导致额外的拷贝

    `rt`标记表示右矩阵转置，其实现方法是在算子里先转置，然后再进行普通矩阵乘
    如无必要，建议手动转置后使用平凡矩阵乘
*/

/**
 * @param sin_vals sin theta的数组，__fp16类型，数量为(nr_cols / 2)
 * @param cos_vals cos theta的数组，__fp16类型，数量为(nr_cols / 2)
 */
void rope_fp16_ref(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
);

void rope_fp32_ref(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
);

void rope_fp32_dsp(
    void * src, void * sin_vals, void * cos_vals,
    void * dst, size_t nr_rows, size_t nr_cols
);

void rmsnorm_fp32_dsp(
    void * src, void * dst, size_t nr_rows, size_t nr_cols,
    float eps
);

void softmax_fp32_dsp(
    void * src, void * dst, size_t nr_row, size_t nr_col
);

/**
 * @param n 要进行silu运算的元素数量（注：不是字节数）
 * @param dst 输出指针
 * @param src 输入指针
 */
void silu_fp32_dsp(const int n, float * dst, const float * src);

void ggml_vec_soft_max_f32_dsp(const int n, float * dst, const float * src, float max);

void trans_fp32_to_fp16_dsp(void * src, void * dst, size_t nr_elem);

void trans_fp16_to_fp32_dsp(void * src, void * dst, size_t nr_elem);

void matmul_fp16_ref(
    void * lmat_data_fp16,
    void * rmat_data_fp16,
    void * dst_data_fp16,
    size_t m, size_t k, size_t n
);

void matmul_fp16_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data,
    size_t m, size_t k, size_t n
);

void bmm_fp16_rtranspose_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data, 
    size_t nr_batches,
    size_t m, size_t k, size_t n
);

void bmm_shs_rtranspose_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data, 
    size_t nr_batches,
    size_t m, size_t k, size_t n
);

/**
 * @brief 矩阵乘，其中lmat和dst是单精度(float)，rmat是半精度(__fp16)
 * @param lmat_data_fp32 此矩阵对应ggml中的src1
 * @param rmat_data_fp16 通常，此矩阵为权重，对应ggml中的src0
 */
void matmul_shs_dsp(
    void * lmat_data_fp32,
    void * rmat_data_fp16,
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
);

void matmul_shs_rmat_transpose_dsp(
    void * lmat_data_fp32,
    void * rmat_data_fp16,
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
);

void matmul_shs_ref(
    void * lmat_data_fp32,
    void * rmat_data_fp16,
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
);

void bmm_fp16_dsp(
    void * lmat_data,
    void * rmat_data,
    void * dst_data, 
    size_t nr_batches,
    size_t m, size_t k, size_t n
);

void matmul_shs_rt_dsp(
    void * lmat_data_fp32,
    void * rmat_data_fp16,
    void * dst_data_fp32,
    size_t m, size_t k, size_t n
);

#ifdef __cplusplus
}
#endif
