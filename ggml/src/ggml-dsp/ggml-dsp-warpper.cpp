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
#include "ggml-dsp-funcs.h"
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
    ts.buffer = NULL;
    // ts.view_src = NULL;

    uint64_t bt = __do_get_timestamp_ns();
    ts.data = malloc(nr_elem * ggml_type_size(t->type));
    ggml_compute_forward_cpy(params, &ts);
    uint64_t et = __do_get_timestamp_ns();
    mkcont_time_ns += (et - bt);             // 还好，没多大开销

    return ts;
}

static const int test_cluster_id = TEST_CLUSTER_ID;

int get_cluster_id_from_buffer(void * ptr);

bool is_dsp_buffer(struct ggml_tensor * ts) {
    if(!ts->buffer) { return false; }
    if(strcmp(ggml_backend_buffer_name(ts->buffer), "MT_MALLOC") == 0) {
        return true;
    }
    return false;
}

bool is_cpu_mapped_buffer(struct ggml_backend_buffer * buffer) {
    if(!buffer) { return false; }
    if(strcmp(ggml_backend_buffer_name(buffer), "CPU_Mapped") == 0) {
        return true;
    }
    return false;
}

void override_buffer(const struct ggml_compute_params * params, struct ggml_tensor * ts) {
    bool data_need_free = false;
    if(!ggml_is_contiguous(ts)) {
        assert(false);

        struct ggml_tensor cont_tensor = mkcont_tensor(params, ts);
        
        ts->data = cont_tensor.data;
        data_need_free = true;
        ts->buffer = NULL;

        ts->nb[0] = cont_tensor.nb[0];
        ts->nb[1] = cont_tensor.nb[1];
        ts->nb[2] = cont_tensor.nb[2];
        ts->nb[3] = cont_tensor.nb[3];

        ts->ne[0] = cont_tensor.ne[0];
        ts->ne[1] = cont_tensor.ne[1];
        ts->ne[2] = cont_tensor.ne[2];
        ts->ne[3] = cont_tensor.ne[3];

        assert(ggml_is_contiguous(ts));
    }
    if(!ts->buffer) { 
        // do not cope with tensors made by mkcont
        return; 
    }
    // ----- override buffer
    if(!is_dsp_buffer(ts)) {
        printf("tensor %s is not dsp buffer, is %s buffer \n", 
            ts->name, 
            ts->buffer ? ggml_backend_buffer_name(ts->buffer) : "Unknown"
        );
        // --- alloc new dsp buffer
        size_t size;
        // size = (ts->ne[0]) * (ts->ne[1]) * (ts->ne[2]) * (ts->ne[3]) * ggml_type_size(ts->type);
        // should be same
        size = ggml_backend_buft_get_alloc_size(
            ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("DSP")),
            ts
        );
        ggml_backend_buffer_t dsp_buf = ggml_backend_buft_alloc_buffer(
            ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("DSP")), 
            size
        );

        // MEM_BARRIER_RW;

        // --- set dsp buffer data
        void * old_data = ts->data;
        ggml_backend_buffer_t old_buffer = ts->buffer;
        ts->buffer = dsp_buf;
        ts->data = ggml_backend_buffer_get_base(dsp_buf);
        ggml_backend_tensor_set(ts, old_data, 0, size);
        // --- free old buffer
        if(!is_cpu_mapped_buffer(old_buffer) && old_buffer) {
            ggml_backend_buffer_free(old_buffer);
            assert(false); // should not reach
        }
        if(data_need_free) { free(old_data); }
    }
    else {
        size_t buffer_size = ts->buffer->size;
        void * buffer_base = ts->buffer->context;
        // make sure the tensor->data matches tensor->buffer
        assert(
            (buffer_base <= ts->data) &&
            (ts->data < ((uint8_t*)buffer_base) + buffer_size)
        );
    }
}

void ggml_compute_forward_mul_mat_dsp(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
) {
    // mt_flush_all();
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type_src0 = src0->type;
    const enum ggml_type type_src1 = src1->type;
    const enum ggml_type type_dst = dst->type;

    bool go_on_flag = true;
    go_on_flag &= ggml_is_contiguous(dst);
    // go_on_flag &= src0->op != GGML_OP_MUL_MAT
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
        assert(src0->buffer == NULL);
    }

    bool src1_need_free = false;
    struct ggml_tensor src1_c;
    if(!ggml_is_contiguous(src1)) {
        src1_c = mkcont_tensor(params, dst->src[1]);
        src1 = &src1_c;
        assert(ggml_is_contiguous(src1));
        src1_need_free = true;
        assert(src1->buffer == NULL);
    }

    override_buffer(params, src0);
    assert(is_dsp_buffer(src0) || src0->buffer == NULL);
    override_buffer(params, src1);
    assert(is_dsp_buffer(src1) || src1->buffer == NULL);

    assert(src1->ne[0] == src0->ne[0]);
    assert(src0->ne[2] == src1->ne[2]);
    assert(src0->ne[3] == 1);
    assert(src1->ne[3] == 1);

    size_t m = ne11;
    size_t k = ne10;
    size_t n = ne01;

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
            if((m != 1) && (n % 4 == 0) && (k % 4 == 0)) {
                matmul_shs_rt_dsp(
                    src1->data,
                    src0->data,
                    dst->data,
                    m, k, n
                );
            }
            else if((m == 1) && (n % 4 == 0) && (k % 4 == 0)) {
                // dsp上的gemv确实有问题，目前在dsp上重新实现了简易的gemv，推理结果正确
                matmul_shs_rt_dsp(
                    src1->data,
                    src0->data,
                    dst->data,
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
            
        } else {
            assert(false); // Not implemented
        }
        

    } else {        
        // ----- 进行bmm
        size_t m = ne11;
        size_t k = ne10;
        size_t n = ne01;
        size_t nr_batches = src0->ne[2];

        // uint64_t bt = __do_get_timestamp_ns();
        // ggml_compute_forward_mul_mat(params, dst);
        // uint64_t et = __do_get_timestamp_ns();
        // fallback_time_ns += (et - bt);
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

    if(src0_need_free) {
        free(src0->data);
    }
    if(src1_need_free) {
        free(src1->data);
    }
}
