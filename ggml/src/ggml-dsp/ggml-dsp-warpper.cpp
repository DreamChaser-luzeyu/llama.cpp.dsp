#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string.h>
#include <sys/types.h>
#include <cassert>
#include <chrono>
#include <unistd.h>
#include <unordered_map>

#include "backend_debug.h"
#include "ggml-backend.h"
#include "ggml-dsp/MTUtils.hpp"
#include "ggml.h"
#include "dsp_utils.h"
#include "dsp_tensor_utils.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend-impl.h"



static inline uint64_t __do_get_timestamp_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

uint64_t fallback_time_ns = 0ul;
uint64_t mkcont_time_ns = 0ul;

extern "C" void ggml_compute_forward_rope_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst);

extern "C" void ggml_compute_forward_mul_mat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

extern "C" void ggml_compute_forward_silu_f32(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst);

extern "C" void ggml_compute_forward_silu_f16(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst);

extern "C" void ggml_compute_forward_silu_dsp(
            const struct ggml_compute_params * params,
            struct ggml_tensor * dst);

extern "C" void ggml_compute_forward_mul_mat_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

extern "C" void ggml_compute_forward_soft_max_dsp(
    const struct ggml_compute_params * params,
          struct ggml_tensor * dst);

extern "C" void ggml_compute_forward_rms_norm_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

void ggml_compute_forward_mul_mat_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type_src0 = src0->type;
    const enum ggml_type type_src1 = src1->type;
    const enum ggml_type type_dst = dst->type;

    bool go_on_flag = true;
    go_on_flag &= ggml_is_contiguous(dst);
    if(!go_on_flag) { // dst不连续时fallback到默认实现
        uint64_t bt = __do_get_timestamp_ns();
        ggml_compute_forward_mul_mat(params, dst);
        uint64_t et = __do_get_timestamp_ns();
        fallback_time_ns += (et - bt);
        return;
    }
    
    struct ggml_tensor src0_c;
    if(!ggml_is_contiguous(src0)) {
        src0_c = mkcont_tensor_new(dst->src[0]);
        src0 = &src0_c;
        assert(ggml_is_contiguous(src0));
    }

    struct ggml_tensor src1_c;
    if(!ggml_is_contiguous(src1)) {
        src1_c = mkcont_tensor_new(dst->src[1]);
        src1 = &src1_c;
        assert(ggml_is_contiguous(src1));
    }

    override_buffer_new(src0);
    assert(is_dsp_buffer(src0));
    override_buffer_new(src1);
    assert(is_dsp_buffer(src1));

    assert(is_dsp_buffer(dst)); // ???

    assert(src1->ne[0] == src0->ne[0]);
    assert(src0->ne[2] == src1->ne[2]);
    assert(src0->ne[3] == 1);
    assert(src1->ne[3] == 1);

    size_t m = ne11;
    size_t k = ne10;
    size_t n = ne01;
    dsp_transc_wait_all();
    if(src0->ne[2] == 1) {
        // ----- 进行普通矩阵乘gemm
        if(type_src0 == GGML_TYPE_F16 && type_src1 == GGML_TYPE_F16 && type_dst == GGML_TYPE_F16) {
            assert(false); // should not reach
            uint64_t bt = __do_get_timestamp_ns();
            ggml_compute_forward_mul_mat(params, dst);
            uint64_t et = __do_get_timestamp_ns();
            fallback_time_ns += (et - bt);
            // matmul_fp16_dsp(
            //     src1->data,
            //     src0->data,
            //     dst->data,
            //     m, k, n
            // );
        } 
        else if(type_src0 == GGML_TYPE_F16 && type_src1 == GGML_TYPE_F32 && type_dst == GGML_TYPE_F32){
            if((k % 16 == 0) && (n % 4 == 0)) {
                assert(is_weight_tensor(src0)); // make sure rmat is always weight
                matmul_shs_dsp(
                    src1->data,
                    src0->data,
                    dst->data,
                    m, k, n
                );
                // --- Step 1. emulate multi cluster within one cluster
                // size_t m_per_cluster = (m + 3) / 4;
                // size_t m_left = m;
                // size_t m_c0 = std::max(m_per_cluster, m_left);
                // m_left -= m_c0;
                // size_t m_c1 = std::max(m_per_cluster, m_left);
                // m_left -= m_c1;
                // size_t m_c2 = std::max(m_per_cluster, m_left);
                // m_left -= m_c2;
                // size_t m_c3 = std::max(m_per_cluster, m_left);
                // m_left -= m_c3;
                // assert(m_left == 0);
                



            }
            else {
                assert(false);
                uint64_t bt = __do_get_timestamp_ns();
                ggml_compute_forward_mul_mat(params, dst);
                uint64_t et = __do_get_timestamp_ns();
                fallback_time_ns += (et - bt);
            }
            
        } else {
            assert(false); // Not implemented
        }
    } else {        
        // ----- 进行bmm
        size_t m = ne11;
        size_t k = ne10;
        size_t n = ne01;
        size_t nr_batches = src0->ne[2];

        if((type_src1 == GGML_TYPE_F32) && (type_src0 == GGML_TYPE_F16) && (type_dst == GGML_TYPE_F32)) {
            bmm_shs_rtranspose_dsp(
                src1->data,
                src0->data,
                dst->data,
                nr_batches,
                m, k, n
            );
        }
        else if((type_src1 == GGML_TYPE_F16) && (type_src0 == GGML_TYPE_F16) && (type_dst == GGML_TYPE_F16)) {
            bmm_fp16_rtranspose_dsp(
                src1->data,
                src0->data,
                dst->data,
                nr_batches,
                m, k, n
            );
        }
        else {
            assert(false);
            uint64_t bt = __do_get_timestamp_ns();
            ggml_compute_forward_mul_mat(params, dst);
            uint64_t et = __do_get_timestamp_ns();
            fallback_time_ns += (et - bt);
        }
    }

    if(is_tmp_tensor(src0)) {
        ggml_backend_buffer_free(src0->buffer);
    }
    if(is_tmp_tensor(src1)) {
        ggml_backend_buffer_free(src1->buffer);
    }
}

void ggml_compute_forward_silu_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    struct ggml_tensor * src0 = dst->src[0];
    
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                // ggml_compute_forward_silu_f32(params, dst);

                assert(ggml_is_contiguous_1(src0));
                assert(ggml_is_contiguous_1(dst));
                assert(ggml_are_same_shape(src0, dst));

                override_buffer_new(src0);
                assert(is_dsp_buffer(src0));
                
                const int nr = ggml_nrows(src0);
                const int nc = src0->ne[0];

                for (int cr = 0; cr < nr; cr++) {
                    // ggml_vec_silu_f32(nc,
                    //         (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                    //         (float *) ((char *) src0->data + i1*(src0->nb[1])));
                    silu_fp32_dsp(nc, 
                        (float*)       ((char *)dst->data + cr*(dst->nb[1])),
                        (const float*) ((char *)src0->data + cr*(src0->nb[1]))
                    );
                }

                if(is_tmp_tensor(src0)) {
                    ggml_backend_buffer_free(src0->buffer);
                    assert(false);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(false);
                ggml_compute_forward_silu_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_compute_forward_soft_max_dsp(
    const struct ggml_compute_params * params,
          struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                const struct ggml_tensor * src1 = dst->src[1];
                assert(ggml_is_contiguous(dst));
                assert(ggml_are_same_shape(src0, dst));
                // ggml_compute_forward_soft_max_f32_dsp(params,dst);

                // --- 读取超参数
                float scale    = 1.0f;
                float max_bias = 0.0f;
                memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
                memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
                
                // TODO: handle transposed/permuted matrices
                
                const int ith = params->ith;
                const int nth = params->nth;
                
                GGML_TENSOR_UNARY_OP_LOCALS
                
                // TODO: is this supposed to be ceil instead of floor?
                //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
                const uint32_t n_head      = ne02;
                const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
                
                const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
                const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
                
                const int nc = src0->ne[0];
                const int nr = ggml_nrows(src0);
                
                // rows per thread
                const int dr = (nr + nth - 1)/nth;
                
                // row range for this thread
                const int ir0 = dr*ith;
                const int ir1 = MIN(ir0 + dr, nr);
                
                #define CACHE_LINE_SIZE 64
                static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE / sizeof(float);
            
                float * wp = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;
                
                const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);
                
                for (int i1 = ir0; i1 < ir1; i1++) {
                    // ----- 按行迭代
                    // ALiBi
                    const uint32_t h = (i1/ne01)%ne02; // head
                    const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;
                
                    float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);        // 指向src0的每一行
                    float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);        // 指向dst的每一行
                
                    // broadcast the mask across rows
                    // src1有可能为NULL
                    ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;
                    float       * mp_f32 = src1 ? (float       *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;
                                                                                         // 指向src1的第一个batch的每一行
                
                    memcpy(wp, sp, sizeof(float) * nc); // ggml_vec_cpy_f32  (nc, wp, sp);
                    for(size_t i = 0; i < nc; i++) { ((float*)wp)[i] *= scale; } // ggml_vec_scale_f32(nc, wp, scale);
                                                                                         // 将src0的当前行存入wp并进行缩放
                    // 如果src1存在，则再给当前行加上 src1的当前行乘一个系数
                    if (mp_f32) {
                        if (use_f16) {
                            for (int i = 0; i < nc; ++i) {
                                wp[i] += slope*GGML_FP16_TO_FP32(mp_f16[i]);
                                // workaround: DSP上exp(-inf)有些问题，因此使用-10代替-inf
                                if(wp[i] == -INFINITY) { wp[i] = (__fp16)-10.0; }
                            }
                        } else {
                            for (int i = 0; i < nc; ++i) {
                                wp[i] += slope*mp_f32[i];
                                if(std::isinf(wp[i])) { wp[i] = -10.0; }
                            }
                        }
                    }
                
                    #ifndef NDEBUG
                    for (int i = 0; i < nc; ++i) {
                        //printf("p[%d] = %f\n", i, p[i]);
                        assert(!isnan(wp[i]));
                    }
                    #endif

                    // --- 此处替换为DSP的softmax
                    void * wp_dsp = dsp_malloc_on_cluster(sizeof(float) * nc, dsp_get_main_cluster());
                    dsp_memcpy_async(wp_dsp, wp, sizeof(float) * nc);
                    dsp_transc_wait_all();
                    softmax_fp32_dsp(wp_dsp, dp, 1l, nc);
                    // mt_flush_all();
                    dsp_free(wp_dsp);

                    #ifndef NDEBUG
                    for (int i = 0; i < nc; ++i) {
                        assert(!isnan(dp[i]));
                        assert(!isinf(dp[i]));
                    }
                    #endif
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void cmp_fp32(void * ref, void * data, size_t nr_elem) {
    typedef float type_t;
    double err_factor = 0.0;
    for(size_t i = 0; i < nr_elem; i++) {
        type_t diff = ((type_t *)ref)[i] - ((type_t *)data)[i];
        if(std::isnan(diff)) { 
            assert(false);
            continue; 
        }
        // assert(!std::isnan(diff));
        double err = std::abs(((double)diff) / (double)(((type_t *)ref)[i]));
        // printf("<%06f, %06f> ", ((type_t *)ref)[i], ((type_t *)data)[i]);
        // if((i + 1) % 8 == 0) {
        //     printf("\n");
        // }
        if(std::isnan(err)) { continue; }
        err_factor += err;
        assert(!std::isnan(err_factor));
    }
    // printf("\n");
    fflush(stdout);
    err_factor = err_factor / (double)(nr_elem);
    if(err_factor > 0.3) {
        assert(false);
    }
    // std::cout << "Relative error: " << err_factor << std::endl;
    // printf("Relative error: %05f %% \n", err_factor * 100);
}

static void rmsn_row_ref_fp32(void * src, void * dst, size_t nr_col, float eps) {
    float * x = (float *)src;
    float * y = (float *)dst;
    size_t ne00 = nr_col;
    // 1-- 计算当前行平方和
    double sum = 0.0;
    for (int64_t i00 = 0; i00 < ne00; i00++) {
        sum += (double)(x[i00] * x[i00]);
    }
    // 2-- 计算当前行平均值
    const float mean = sum/ne00;
    // 3-- 将当前行拷贝进dst
    memcpy(y, x, ne00 * sizeof(float));
    // 4-- 计算均值
    const float scale = 1.0f/sqrtf(mean + eps);
    for(size_t i = 0; i < ne00; i++) { ((float*)y)[i] *= scale; }
}

void ggml_compute_forward_rms_norm_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                const struct ggml_tensor * src0 = dst->src[0];

                GGML_ASSERT(ggml_are_same_shape(src0, dst));
            
                GGML_ASSERT(src0->nb[0] == sizeof(float));
            
                const int ith = params->ith;
                const int nth = params->nth;
            
                GGML_TENSOR_UNARY_OP_LOCALS
            
                float eps;
                memcpy(&eps, dst->op_params, sizeof(float));
            
                GGML_ASSERT(eps >= 0.0f);
                mt_flush_all();

                // TODO: optimize
                typedef double ggml_float;
                for (int64_t i03 = 0; i03 < ne03; i03++) {
                    for (int64_t i02 = 0; i02 < ne02; i02++) { // 按batch迭代
                        float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
                        float * y = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
                        rmsnorm_fp32_dsp(x, y, ne01, ne00, eps);
                        // for (int64_t i01 = ith; i01 < ne01; i01 += nth) { // 按行迭代
                            // float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            // float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
                            // ----- 参考实现
                            // 1-- 计算当前行平方和
                            // ggml_float sum = 0.0;
                            // for (int64_t i00 = 0; i00 < ne00; i00++) {
                            //     sum += (ggml_float)(x[i00] * x[i00]);
                            // }
                            // // 2-- 计算当前行平均值
                            // const float mean = sum/ne00;
                            // // 3-- 将当前行拷贝进dst
                            // memcpy(y, x, ne00 * sizeof(float));
                            // // 4-- 计算均值
                            // const float scale = 1.0f/sqrtf(mean + eps);
                            // for(size_t i = 0; i < ne00; i++) { ((float*)y)[i] *= scale; } // ggml_vec_scale_f32(ne00, y, scale);
                            // ----- DSP实现
                            // mt_flush_all();
                            // void * ref_y = malloc(sizeof(float) * ne00);
                            // rmsnorm_fp32_dsp(x, y, 1, ne00, eps);
                            // rmsn_row_ref_fp32(x, ref_y, ne00, eps);
                            // cmp_fp32(ref_y, y, ne00);
                            // free(ref_y);
                            // dump_mat(y, __fp16, 1, 1, ne00);
                        // }
                    }
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
    float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
    float * cache, float sin_sign, float theta_scale) {
   // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
   float theta = theta_base;
   for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
       const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
       rope_yarn(
           theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
       );
       cache[i0 + 1] *= sin_sign;

       theta *= theta_scale;
   }
}

static void ggml_mrope_cache_init(
    float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool indep_sects,
    float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
    float * cache, float sin_sign, float theta_scale) {
   // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
   float theta_t = theta_base_t;
   float theta_h = theta_base_h;
   float theta_w = theta_base_w;
   float theta_e = theta_base_e;  // extra position id for vision encoder
   int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
   int sec_w = sections[1] + sections[0];
   int sec_e = sections[2] + sec_w;
   GGML_ASSERT(sect_dims <= ne0);

   for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
       const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

       int sector = (i0 / 2) % sect_dims;
       if (indep_sects) {
           // compute theta independently for each dim sections
           // (i.e. reset corresponding theta when `i0` go from one section to another)
           if (sector == 0) {
               theta_t = theta_base_t;
           }
           else if (sector == sections[0]) {
               theta_h = theta_base_h;;
           }
           else if (sector == sec_w) {
               theta_w = theta_base_w;
           }
           else if (sector == sec_e) {
               theta_e = theta_base_e;
           }
       }

       float theta = theta_t;
       if (sector >= sections[0] && sector < sec_w) {
           theta = theta_h;
       }
       else if (sector >= sec_w && sector < sec_w + sections[2]) {
           theta = theta_w;
       }
       else if (sector >= sec_w + sections[2]) {
           theta = theta_e;
       }

       rope_yarn(
           theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
       );
       cache[i0 + 1] *= sin_sign;

       theta_t *= theta_scale;
       theta_w *= theta_scale;
       theta_h *= theta_scale;
       theta_e *= theta_scale;
   }
}

// rope is fp32
void ggml_compute_forward_rope_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    static const size_t CACHE_LINE_SIZE_F32 = 16;
    
    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                // ggml_compute_forward_rope_f16(params, dst, true);
                bool forward = true;
                const struct ggml_tensor * src0 = dst->src[0];
                const struct ggml_tensor * src1 = dst->src[1];
                const struct ggml_tensor * src2 = dst->src[2];

                float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                int sections[4];

                //const int n_past     = ((int32_t *) dst->op_params)[0];
                const int n_dims     = ((int32_t *) dst->op_params)[1];
                const int mode       = ((int32_t *) dst->op_params)[2];
                //const int n_ctx      = ((int32_t *) dst->op_params)[3];
                const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
                memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
                memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
                memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
                memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
                memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
                memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
                memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);


                GGML_TENSOR_UNARY_OP_LOCALS

                //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
                //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

                GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));

                const int ith = params->ith;
                const int nth = params->nth;

                const int nr = ggml_nrows(dst);

                GGML_ASSERT(n_dims <= ne0);
                GGML_ASSERT(n_dims % 2 == 0);

                // rows per thread
                const int dr = (nr + nth - 1)/nth;

                // row range for this thread
                const int ir0 = dr*ith;
                const int ir1 = MIN(ir0 + dr, nr);

                // row index used to determine which thread to use
                int ir = 0;

                const float theta_scale = powf(freq_base, -2.0f/n_dims);

                float corr_dims[2];
                ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

                const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
                const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
                const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

                if (is_mrope) {
                    GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
                }

                if (is_vision) {
                    GGML_ASSERT(n_dims == ne0/2);
                }

                const float * freq_factors = NULL;
                if (src2 != NULL) {
                    GGML_ASSERT(src2->type == GGML_TYPE_F32);
                    GGML_ASSERT(src2->ne[0] >= n_dims / 2);
                    freq_factors = (const float *) src2->data;
                }

                // backward process uses inverse rotation by cos and sin.
                // cos and sin build a rotation matrix, where the inverse is the transpose.
                // this essentially just switches the sign of sin.
                const float sin_sign = forward ? 1.0f : -1.0f;

                const int32_t * pos = (const int32_t *) src1->data;

                for (int64_t i3 = 0; i3 < ne3; i3++) {
                    for (int64_t i2 = 0; i2 < ne2; i2++) {

                        // --- 计算sin theta & cos theta
                        float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
                        if (!is_mrope) {
                            const int64_t p = pos[i2];
                            ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                        }
                        else {
                            const int64_t p_t = pos[i2];
                            const int64_t p_h = pos[i2 + ne2];
                            const int64_t p_w = pos[i2 + ne2 * 2];
                            const int64_t p_e = pos[i2 + ne2 * 3];
                            ggml_mrope_cache_init(
                                p_t, p_h, p_w, p_e, sections, is_vision,
                                freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                        }

                        for (int64_t i1 = 0; i1 < ne1; i1++) {
                            if (ir++ < ir0) continue;
                            if (ir   > ir1) break;
                            // --- 应该不是我们用到的
                            if (is_neox || is_mrope) {
                                if (is_vision) {
                                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                        const int64_t ic = i0/2;

                                        const float cos_theta = cache[i0 + 0];
                                        const float sin_theta = cache[i0 + 1];

                                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                                        const float x1 = GGML_FP16_TO_FP32(src[n_dims]);

                                        dst_data[0]      = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                                        dst_data[n_dims] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                                    }
                                } else {
                                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                        const int64_t ic = i0/2;

                                        const float cos_theta = cache[i0 + 0];
                                        const float sin_theta = cache[i0 + 1];

                                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                                        const float x1 = GGML_FP16_TO_FP32(src[n_dims/2]);

                                        dst_data[0]        = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                                        dst_data[n_dims/2] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                                    }
                                }
                            } else {
                                for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                    const float cos_theta = cache[i0 + 0];
                                    const float sin_theta = cache[i0 + 1];

                                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                                          ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                                    const float x0 = GGML_FP16_TO_FP32(src[0]);
                                    const float x1 = GGML_FP16_TO_FP32(src[1]);

                                    dst_data[0] = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                                    dst_data[1] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                                }
                            }

                            // --- 关注这一部分
                            if (is_vision) {
                                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                                    const int64_t ic = i0/2;
                                    // - 取出当前“组”的sin和cos数值
                                    const float cos_theta = cache[i0 + 0];
                                    const float sin_theta = cache[i0 + 1];

                                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                                    ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                                    const float x0 = GGML_FP16_TO_FP32(src[0]);
                                    const float x1 = GGML_FP16_TO_FP32(src[n_dims]);

                                    dst_data[0]      = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                                    dst_data[n_dims] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                                }
                            } else {
                                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                                    ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                                    dst_data[0] = src[0];
                                    dst_data[1] = src[1];
                                }
                            }
                        }
                    }
                }
            } break;
        case GGML_TYPE_F32:
            {
                // ggml_compute_forward_rope_f32(params, dst, true);
                bool forward = true;
                const struct ggml_tensor * src0 = dst->src[0];
                const struct ggml_tensor * src1 = dst->src[1];
                const struct ggml_tensor * src2 = dst->src[2];
            
                float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                int sections[4];
            
                //const int n_past     = ((int32_t *) dst->op_params)[0];
                const int n_dims     = ((int32_t *) dst->op_params)[1];
                const int mode       = ((int32_t *) dst->op_params)[2];
                //const int n_ctx      = ((int32_t *) dst->op_params)[3];
                const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
            
                memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
                memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
                memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
                memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
                memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
                memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
                memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
            
                GGML_TENSOR_UNARY_OP_LOCALS
            
                //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
                //printf("n_past = %d, ne2 = %d\n", n_past, ne2);
            
                GGML_ASSERT(nb00 == sizeof(float));
            
                const int ith = params->ith;
                const int nth = params->nth;
            
                const int nr = ggml_nrows(dst);
            
                GGML_ASSERT(n_dims <= ne0);
                GGML_ASSERT(n_dims % 2 == 0);
            
                // rows per thread
                const int dr = (nr + nth - 1)/nth;
            
                // row range for this thread
                const int ir0 = dr*ith;
                const int ir1 = MIN(ir0 + dr, nr);
            
                // row index used to determine which thread to use
                int ir = 0;
            
                const float theta_scale = powf(freq_base, -2.0f/n_dims);
            
                float corr_dims[2];
                ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            
                const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
                const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
                const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
            
                if (is_mrope) {
                    GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
                }
            
                if (is_vision) {
                    GGML_ASSERT(n_dims == ne0/2);
                }
            
                const float * freq_factors = NULL;
                if (src2 != NULL) {
                    GGML_ASSERT(src2->type == GGML_TYPE_F32);
                    GGML_ASSERT(src2->ne[0] >= n_dims / 2);
                    freq_factors = (const float *) src2->data;
                }
            
                // backward process uses inverse rotation by cos and sin.
                // cos and sin build a rotation matrix, where the inverse is the transpose.
                // this essentially just switches the sign of sin.
                const float sin_sign = forward ? 1.0f : -1.0f;
            
                const int32_t * pos = (const int32_t *) src1->data;
            
                for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
                    for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            
                        float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
                        if (!is_mrope) {
                            const int64_t p = pos[i2];
                            ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                        }
                        else {
                            const int64_t p_t = pos[i2];
                            const int64_t p_h = pos[i2 + ne2];
                            const int64_t p_w = pos[i2 + ne2 * 2];
                            const int64_t p_e = pos[i2 + ne2 * 3];
                            ggml_mrope_cache_init(
                                p_t, p_h, p_w, p_e, sections, is_vision,
                                freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                        }
            
                        for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                            if (ir++ < ir0) continue;
                            if (ir   > ir1) break;
            
                            if (is_neox || is_mrope) {
                                if (is_vision){
                                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                        const int64_t ic = i0/2;
            
                                        const float cos_theta = cache[i0 + 0];
                                        const float sin_theta = cache[i0 + 1];
            
                                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
            
                                        const float x0 = src[0];
                                        const float x1 = src[n_dims];
            
                                        dst_data[0]      = x0*cos_theta - x1*sin_theta;
                                        dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                                    }
                                } else {
                                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                        const int64_t ic = i0/2;
            
                                        const float cos_theta = cache[i0 + 0];
                                        const float sin_theta = cache[i0 + 1];
            
                                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
            
                                        const float x0 = src[0];
                                        const float x1 = src[n_dims/2];
            
                                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
                                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                                    }
                                }
                            } else {
                                // --- prepare sin cos val arrs
                                assert(n_dims == 128);
                                // float sin_vals[64];
                                // float cos_vals[64];
                                float * sin_vals = (float*)dsp_malloc_on_cluster(sizeof(float) * n_dims / 2, dsp_get_main_cluster());
                                float * cos_vals = (float*)dsp_malloc_on_cluster(sizeof(float) * n_dims / 2, dsp_get_main_cluster());
                                for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                    sin_vals[i0 / 2] = cache[i0 + 1];
                                    cos_vals[i0 / 2] = cache[i0];
                                }
                                // for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                                //     // ----- here
                                //     // const float cos_theta = cache[i0 + 0];
                                //     // const float sin_theta = cache[i0 + 1];
                                    
                                //     const float cos_theta = cos_vals[i0 / 2];
                                //     const float sin_theta = sin_vals[i0 / 2];

                                //     float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                                //     float *  dst_data = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                                    
                                //     const float x0 = src[0];
                                //     const float x1 = src[1];
                                    
                                //     dst_data[0] = x0*cos_theta - x1*sin_theta;
                                //     dst_data[1] = x0*sin_theta + x1*cos_theta;
                                    
                                // }
                                int64_t i0 = 0;
                                float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                                float *  dst_data = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
                                rope_fp32_ref(src, sin_vals, cos_vals, dst_data, 1l, n_dims);
                                dsp_free(sin_vals);
                                dsp_free(cos_vals);
                            }
                            
                            // ----- not used
                            if (is_vision) {
                                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                                    const int64_t ic = i0/2;
            
                                    const float cos_theta = cache[i0 + 0];
                                    const float sin_theta = cache[i0 + 1];
            
                                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                                    float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
            
                                    const float x0 = src[0];
                                    const float x1 = src[n_dims];
            
                                    dst_data[0]      = x0*cos_theta - x1*sin_theta;
                                    dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                                }
                            } else {
                                // fill the remain channels with data from src tensor
                                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                                    float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
            
                                    dst_data[0] = src[0];
                                    dst_data[1] = src[1];
                                }
                            }
                        }
                    }
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}
