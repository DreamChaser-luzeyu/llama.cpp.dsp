#pragma once

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"

struct ggml_tensor mkcont_tensor_new(struct ggml_tensor * t);

/**
 * @note This will free the old buffer!
 */
void override_buffer_new(struct ggml_tensor * ts);

static bool is_dsp_buffer(struct ggml_tensor * ts) {
    if(!ts->buffer) { return false; }
    if(strcmp(ggml_backend_buffer_name(ts->buffer), "MT_MALLOC") == 0) {
        return true;
    }
    return false;
}

static bool is_tmp_tensor(struct ggml_tensor * ts) {
    if(strcmp(ts->name, "__TEMP__") == 0) {
        return true;
    }
    return false;
}

static bool is_cpu_mapped_buffer(struct ggml_backend_buffer * buffer) {
    if(!buffer) { return false; }
    if(strcmp(ggml_backend_buffer_name(buffer), "CPU_Mapped") == 0) {
        return true;
    }
    return false;
}

static bool is_ro_tensor(struct ggml_tensor * ts) {
    if(!ts) { return false; }
    if(strcasestr(ts->name, "weight")) {
        return true;
    }
    return false;
}

static bool is_weight_tensor(struct ggml_tensor * ts) {
    if(!ts) { return false; }
    if(strcasestr(ts->name, "weight")) {
        return true;
    }
    return false;
}
