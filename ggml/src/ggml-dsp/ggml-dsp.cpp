#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-aarch64.h"
#include "ggml-cpu-traits.h"
#include "ggml-dsp/MTUtils.hpp"
#include "ggml-impl.h"
#include "ggml-dsp-funcs.h"
#include "ggml.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "backend_debug.h"

#if NO_INIT_DEV
#define mt_flush_all()
#endif

static ggml_backend_device ggml_backend_cpu_device;

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t>& ggml_backend_cpu_get_extra_buffers_type() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#ifdef GGML_USE_CPU_AARCH64
        if (ggml_backend_cpu_aarch64_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_aarch64_buffer_type());
        }
#endif

        bufts.push_back(NULL);

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    return ggml_backend_cpu_get_extra_buffers_type().data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra && extra == buft) return true;
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

static const char * ggml_backend_dspp_get_name(ggml_backend_t backend) {
    return "DSP";

    GGML_UNUSED(backend);
}

static void ggml_backend_dspp_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_dspp_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return cpu_plan;
}

static void ggml_backend_dspp_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

extern "C" enum ggml_status ggml_graph_compute_dsp(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
static enum ggml_status ggml_backend_dspp_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute_dsp(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

extern "C" enum ggml_status ggml_graph_compute_dsp(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
static enum ggml_status ggml_backend_dspp_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return ggml_graph_compute_dsp(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_dspp_i = {
    /* .get_name                = */ ggml_backend_dspp_get_name,
    /* .free                    = */ ggml_backend_dspp_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_dspp_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_dspp_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_dspp_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_dspp_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_dspp_guid(void) {
    // static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    static ggml_guid guid = { 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa };
    return &guid;
}

extern "C" void ggml_dsp_init(void);
ggml_backend_t ggml_backend_dspp_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_dsp_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;

    ggml_backend_reg_t ggml_backend_dspp_reg(void);
    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_dspp_guid(),
        /* .interface = */ ggml_backend_dspp_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_dspp_reg(), 0),
        /* .context   = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    void init_dsp();
    init_dsp();

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_dspp_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "DSPP";

    ggml_backend_cpu_device_context() {
    }
};

static const char * ggml_backend_dspp_device_get_name(ggml_backend_dev_t dev) {
    return "DSPP";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_dspp_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_dspp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_dspp_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
    // return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_dspp_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_dspp_device_get_name(dev);
    props->description = ggml_backend_dspp_device_get_description(dev);
    props->type        = ggml_backend_dspp_device_get_type(dev);
    ggml_backend_dspp_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_dspp_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_dspp_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static void ggml_backend_dspp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    // if(buffer->size == 4620288 || buffer->size == 6586368) {
    //     ggml_aligned_free(buffer->context, buffer->size);
    // }
    // else {
        dsp_free(buffer->context); 
    // }
    // dsp_free(buffer->context);
}

static void * ggml_backend_dspp_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_dspp_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);
    mt_flush_all();
    GGML_UNUSED(buffer);
}

static void ggml_backend_dspp_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);
    mt_flush_all();
    GGML_UNUSED(buffer);
}

static void ggml_backend_dspp_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);
    mt_flush_all();
    GGML_UNUSED(buffer);
}

static bool ggml_backend_dspp_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        mt_flush_all();
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_dspp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
    mt_flush_all();
}

int get_cluster_id_from_buffer(void * ptr);
static enum ggml_status ggml_backend_dspp_buffer_init(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // Maybe need to cope with align issue when co-computing with CPU
    // size_t size = ggml_backend_buffer_get_alloc_size(buffer, tensor);
    // void * dsp_data = dsp_malloc(size);
    // memcpy(dsp_data, tensor->data, size);
    // tensor->data = dsp_data;
    // mt_flush_all();
    // printf("***Init buffer with %lu bytes*** \n", size);

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_buffer_i ggml_backend_dspp_buffer_i = {
    /* .free_buffer     = */ ggml_backend_dspp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_dspp_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_dspp_buffer_init, // no initialization required
    /* .memset_tensor   = */ ggml_backend_dspp_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_dspp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_dspp_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_dspp_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_dspp_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_dspp_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "MT_MALLOC";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_dspp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // 4620288 6586368
    // FIXME: workaround
    void * data;
    // if(size == 4620288 || size == 6586368) {
        data = ggml_aligned_malloc(size);
    // } 
    // else {
    //     data = dsp_malloc(size);
    // }
    
    // void * data = dsp_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_dspp_buffer_i, data, size);
}

static size_t ggml_backend_dspp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_dspp_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_dspp_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_dspp_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_dspp_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_dspp_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_dspp_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_dspp_buffer_type_is_host,
        },
        /* .device  = */ &ggml_backend_cpu_device, // FIXED!
        /* .context = */ NULL,
    };

    return &ggml_backend_dspp_buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_dspp_device_get_buffer_type(ggml_backend_dev_t dev) {
    // return ggml_backend_cpu_buffer_type();
    return ggml_backend_dspp_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_dspp_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);

    // void * dsp_data;
    // dsp_data = dsp_malloc(size);
    // dsp_data = malloc(size); // for test
    // assert(dsp_data);

    // memcpy(dsp_data, ptr, size);
    // mt_flush_all();
    void * host_data;
    host_data = ptr;

    return ggml_backend_buffer_init(
        ggml_backend_dspp_buffer_type(), 
        ggml_backend_dspp_buffer_i,
        host_data, 
        size
    );
}

static bool ggml_backend_dspp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (
        op->op == GGML_OP_NONE 
        || op->op == GGML_OP_RESHAPE 
        || op->op == GGML_OP_VIEW 
        || op->op == GGML_OP_PERMUTE 
        || op->op == GGML_OP_TRANSPOSE
    ) {
        return true;
    }

    // extra_buffer_op?
    for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
        if (extra) {
            auto buf_extra = (ggml::cpu::extra_buffer_type*) extra->context;
            if (buf_extra && buf_extra->supports_op(dev, op)) {
                return true;
            }
        }
    }

    // the other case need host buffer.
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && !ggml_backend_buft_is_host(op->src[i]->buffer->buft)) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 || (ggml_is_quantized(src0->type) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        default:
            return true;
    }
}

static bool ggml_backend_dspp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_dspp_device_i = {
    /* .get_name             = */ ggml_backend_dspp_device_get_name,
    /* .get_description      = */ ggml_backend_dspp_device_get_description,
    /* .get_memory           = */ ggml_backend_dspp_device_get_memory,
    /* .get_type             = */ ggml_backend_dspp_device_get_type,
    /* .get_props            = */ ggml_backend_dspp_device_get_props,
    /* .init_backend         = */ ggml_backend_dspp_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_dspp_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_dspp_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_dspp_device_supports_op,
    /* .supports_buft        = */ ggml_backend_dspp_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_dspp_reg_get_name(ggml_backend_reg_t reg) {
    return "DSPP";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_dspp_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}


static ggml_backend_dev_t ggml_backend_dspp_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_dspp_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

extern "C" void ggml_dsp_init(void);
// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_dsp_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_AARCH64
        features.push_back({ "AARCH64_REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_dspp_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_dspp_reg_i = {
    /* .get_name         = */ ggml_backend_dspp_reg_get_name,
    /* .get_device_count = */ ggml_backend_dspp_reg_get_device_count,
    /* .get_device       = */ ggml_backend_dspp_reg_get_device,
    /* .get_proc_address = */ ggml_backend_dspp_get_proc_address,
};

extern "C" void ggml_dsp_init(void);
ggml_backend_reg_t ggml_backend_dspp_reg(void) {
    // init CPU feature detection
    ggml_dsp_init();

    static struct ggml_backend_reg ggml_backend_dspp_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_dspp_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_dspp_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_dspp_reg)
