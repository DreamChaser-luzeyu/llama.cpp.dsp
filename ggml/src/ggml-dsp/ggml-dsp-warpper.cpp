#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string.h>
#include <sys/types.h>
#include <cassert>
#include <chrono>
#include <unistd.h>
#include <unordered_map>

#include "ggml-dsp/MTUtils.hpp"
#include "ggml.h"
#include "ggml-dsp-funcs.h"
#include "ggml-cpu-impl.h"

#define dump_file(...) do { \
    extern FILE * debug_file; \
    fprintf(debug_file, __VA_ARGS__); \
    fflush(debug_file); \
} while(0)

#define dump_mat(ptr, type, nb, m, n) do { \
    for(size_t bb = 0; bb < (nb); bb++) { \
        dump_file("batch %ld\n", bb); \
        for(size_t i = 0; i < (m); i++) { \
            for(size_t j = 0; j < (n); j++) { \
                typedef type(*mat_arr_ptr)[nb][m][n]; \
                dump_file("%.4f ", (*(mat_arr_ptr)(ptr))[bb][i][j]); \
            } \
            dump_file("\n"); \
        } \
    } \
} while(0)

static inline uint64_t __do_get_timestamp_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

uint64_t fallback_time_ns = 0ul;
uint64_t mkcont_time_ns = 0ul;

extern "C" void ggml_compute_forward_mul_mat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

extern "C" void ggml_compute_forward_mul_mat_test(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

extern "C" void ggml_compute_forward_mul_mat_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

extern "C" void ggml_compute_forward_cpy(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
);

static struct ggml_tensor mkcont_tensor(const struct ggml_compute_params * params, struct ggml_tensor * t) {
    assert(!ggml_is_contiguous(t));

    size_t nr_elem = (t->ne[0]) * (t->ne[1]) * (t->ne[2]) * (t->ne[3]);

    struct ggml_tensor ts;
    ts.ne[0] = t->ne[0];
    ts.ne[1] = t->ne[1];
    ts.ne[2] = t->ne[2];
    ts.ne[3] = t->ne[3];
    
    ts.nb[0] = ggml_type_size(t->type);
    ts.nb[1] = ts.nb[0] * ts.ne[0];
    ts.nb[2] = ts.nb[1] * ts.ne[1];
    ts.nb[3] = ts.nb[2] * ts.ne[2];

    ts.src[0] = t;
    ts.type = t->type;

    uint64_t bt = __do_get_timestamp_ns();
    ts.data = malloc(nr_elem * ggml_type_size(t->type));
    ggml_compute_forward_cpy(params, &ts);
    uint64_t et = __do_get_timestamp_ns();
    mkcont_time_ns += (et - bt);             // 还好，没多大开销
    mt_flush_all();
    // t->data = ts.data;
    // t->nb[0] = ts.nb[0];
    // t->nb[1] = ts.nb[1];
    // t->nb[2] = ts.nb[2];
    // t->nb[3] = ts.nb[3];

    return ts;
}

static const int test_cluster_id = 1;

int get_cluster_id_from_buffer(void * ptr);

void ggml_compute_forward_mul_mat_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
) {
    mt_flush_all();
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type_src0 = src0->type;
    const enum ggml_type type_src1 = src1->type;
    const enum ggml_type type_dst = dst->type;

    bool go_on_flag = true;
    go_on_flag &= ggml_is_contiguous(dst);
    if(!go_on_flag) { // 数据不连续时fallback到默认实现
        uint64_t bt = __do_get_timestamp_ns();
        ggml_compute_forward_mul_mat(params, dst);
        uint64_t et = __do_get_timestamp_ns();
        fallback_time_ns += (et - bt);
        return;
    }

    
    bool src0_need_free = false;
    struct ggml_tensor src0_c;
    if(!ggml_is_contiguous(src0)) {
        src0_c = mkcont_tensor(params, dst->src[0]);
        src0 = &src0_c;
        assert(ggml_is_contiguous(src0));
        src0_need_free = true;
    }

    bool src1_need_free = false;
    struct ggml_tensor src1_c;
    if(!ggml_is_contiguous(src1)) {
        src1_c = mkcont_tensor(params, dst->src[1]);
        src1 = &src1_c;
        assert(ggml_is_contiguous(src1));
        src1_need_free = true;
    }

    assert(src1->ne[0] == src0->ne[0]);
    assert(src0->ne[2] == src1->ne[2]);
    assert(src0->ne[3] == 1);
    assert(src1->ne[3] == 1);

    size_t m = ne11;
    size_t k = ne10;
    size_t n = ne01;

    // uint64_t bt = __do_get_timestamp_ns();
    // ggml_compute_forward_mul_mat(params, dst);
    // uint64_t et = __do_get_timestamp_ns();
    // fallback_time_ns += (et - bt);
    // return;

    void * src1_data = NULL;
    bool src1_cpy_need_free = false;
    if(get_cluster_id_from_buffer(src1->data) != test_cluster_id) {
        size_t size = (src1->ne[0]) * (src1->ne[1]) * (src1->ne[2]) * (src1->ne[3]) * ggml_type_size(src1->type);
        src1_data = dsp_malloc(size);
        assert(src1_data);
        memcpy(src1_data, src1->data, size);
        mt_flush_all();
        src1_cpy_need_free = true;
    }
    else {
        src1_data = src1->data;
        printf("src1 hit! \n");
    }

    void * src0_data = NULL;
    bool src0_cpy_need_free = false;
    if(get_cluster_id_from_buffer(src0->data) != test_cluster_id) {
        size_t size = (src0->ne[0]) * (src0->ne[1]) * (src0->ne[2]) * (src0->ne[3]) * ggml_type_size(src0->type);
        src0_data = dsp_malloc(size);
        assert(src0_data);
        memcpy(src0_data, src0->data, size);
        mt_flush_all();
        src0_cpy_need_free = true;
    }
    else {
        src0_data = src0->data;
        printf("src0 hit! \n");
    }

    void * bbbug = NULL;
    bbbug = mt_malloc(test_cluster_id, 32768, 0ul);
    assert(bbbug);
    // dump_file("m");

    if(src0->ne[2] == 1) {
        // ----- 进行普通矩阵乘gemm

        if(type_src0 == GGML_TYPE_F16 && type_src1 == GGML_TYPE_F16 && type_dst == GGML_TYPE_F16) {
            assert(false);
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
            
            if((m % 4 == 0) && (n % 4 == 0) && (k % 4 == 0)) {  // m == 1时使用gemv
                
                // dump_file("%016lx %016lx %016lx \n", src1->data, src0->data, dst->data);
                // void * bbbug = NULL;
                // bbbug = mt_malloc(test_cluster_id, 32768, 0ul);
                // assert(bbbug);
                matmul_shs_rt_dsp(
                    src1_data, // src1->data,
                    src0_data, // src0->data,
                    dst->data,
                    m, k, n
                );
                // mt_free(bbbug, 0ul);

                // dump_mat(dst->data, float, 1, m, n);
                // uint64_t bt = __do_get_timestamp_ns();
                // ggml_compute_forward_mul_mat(params, dst);
                // uint64_t et = __do_get_timestamp_ns();
                // fallback_time_ns += (et - bt);
            }
            else if((m == 1) && (n % 4 == 0) && (k % 4 == 0)) {
                // void * src1_data = NULL;
                // src1_data = mt_malloc(test_cluster_id, 32768, 0ul);
                // assert(src1_data);
                // void * bbbug = NULL;
                // bbbug = mt_malloc(test_cluster_id, 32768, 0ul);
                // assert(bbbug); // wtf??? gemv可能仍有问题，考虑替代gemv再试
                matmul_shs_rt_dsp(
                    src1->data,
                    src0->data,
                    dst->data,
                    m, k, n
                );
                // uint64_t bt = __do_get_timestamp_ns();
                // ggml_compute_forward_mul_mat(params, dst);
                // uint64_t et = __do_get_timestamp_ns();
                // fallback_time_ns += (et - bt);
                // dump_mat(dst->data, float, 1, m, n);
                // mt_free(bbbug, 0ul);
                
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
        // assert(false);

        
        // ----- 进行bmm
        size_t m = ne11;
        size_t k = ne10;
        size_t n = ne01;
        size_t nr_batches = src0->ne[2];

        uint64_t bt = __do_get_timestamp_ns();
        ggml_compute_forward_mul_mat(params, dst);
        uint64_t et = __do_get_timestamp_ns();
        fallback_time_ns += (et - bt);
        // if((type_src1 == GGML_TYPE_F32) && (type_src0 == GGML_TYPE_F16) && (type_dst == GGML_TYPE_F32)) {
            
        //     bmm_shs_rtranspose_dsp(
        //         src1->data,
        //         src0->data,
        //         dst->data,
        //         nr_batches,
        //         m, k, n
        //     );
        // }
        // else if((type_src1 == GGML_TYPE_F16) && (type_src0 == GGML_TYPE_F16) && (type_dst == GGML_TYPE_F16)) {
        //     bmm_fp16_rtranspose_dsp(
        //         src1->data,
        //         src0->data,
        //         dst->data,
        //         nr_batches,
        //         m, k, n
        //     );
        // }
        // else {
        //     assert(false);
        //     uint64_t bt = __do_get_timestamp_ns();
        //     ggml_compute_forward_mul_mat(params, dst);
        //     uint64_t et = __do_get_timestamp_ns();
        //     fallback_time_ns += (et - bt);
        // }
    }




    if(src0_need_free) {
        free(src0->data);
    }
    if(src1_need_free) {
        free(src1->data);
    }
    
    if(src0_cpy_need_free) {
        dsp_free(src0_data);
    }
    if(src1_cpy_need_free) {
        dsp_free(src1_data);
    }

    mt_free(bbbug, 0ul);
    // dump_file("f");
}
