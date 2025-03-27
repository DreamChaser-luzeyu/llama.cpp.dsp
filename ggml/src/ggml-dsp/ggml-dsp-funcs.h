#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

typedef double ggml_float;

void * dsp_malloc(size_t size);

void dsp_free(void * ptr);

/**
 * @param n 要进行silu运算的元素数量（注：不是字节数）
 * @param dst 输出指针
 * @param src 输入指针
 */
void ggml_vec_silu_f32_dsp(const int n, float * dst, const float * src);

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
