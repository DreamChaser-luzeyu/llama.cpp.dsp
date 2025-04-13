#include "dsp_tensor_utils.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend-impl.h"

extern "C" void ggml_compute_forward_dup(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst);

struct ggml_tensor mkcont_tensor_new(struct ggml_tensor * t) {
    assert(!ggml_is_contiguous(t));

    struct ggml_compute_params params;
    params.ith = 0;
    params.nth = 1;

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
    ts.view_src = NULL;
    // ts.view_src = NULL;

    // uint64_t bt = __do_get_timestamp_ns();
    ts.data = malloc(nr_elem * ggml_type_size(t->type));
    ggml_compute_forward_dup(&params, &ts);
    // uint64_t et = __do_get_timestamp_ns();
    // mkcont_time_ns += (et - bt);             // 还好，没多大开销

    strcpy(ts.name, "__TEMP__");

    return ts;
}

void override_buffer_new(struct ggml_tensor * ts) {
    assert(ggml_is_contiguous(ts));
    // ----- override buffer
    if(!is_dsp_buffer(ts)) {
        // printf("tensor %s is not dsp buffer, is %s buffer \n", 
        //     ts->name, 
        //     ts->buffer ? ggml_backend_buffer_name(ts->buffer) : "Unknown"
        // );
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
        // --- set dsp buffer data
        void * old_data = ts->data;
        ggml_backend_buffer_t old_buffer = ts->buffer;
        ts->buffer = dsp_buf;
        ts->data = ggml_backend_buffer_get_base(dsp_buf);
        // - do the actual memcpy here
        ggml_backend_buffer_init_tensor(ts->buffer, ts);
        ggml_backend_tensor_set(ts, old_data, 0, size);
        // --- free old buffer
        if(!is_cpu_mapped_buffer(old_buffer) && old_buffer) {
            ggml_backend_buffer_free(old_buffer);
            assert(false); // should not reach
        }
        if(is_tmp_tensor(ts)) { 
            // assert(ts->buffer == NULL);
            free(old_data); 
        }
    }
    else {
        // --- actually does nothing
        size_t buffer_size = ts->buffer->size;
        void * buffer_base = ts->buffer->context;
        // make sure the tensor->data matches tensor->buffer
        assert(
            (buffer_base <= ts->data) &&
            (ts->data < ((uint8_t*)buffer_base) + buffer_size)
        );
    }
}


