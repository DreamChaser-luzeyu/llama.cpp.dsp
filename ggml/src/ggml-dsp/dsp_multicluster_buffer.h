#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct dsp_multi_cluster_desc {
    void * cluster_0_data;
    size_t c0_b, c0_m, c0_n;
    void * cluster_1_data;
    size_t c1_b, c1_m, c1_n;
    void * cluster_2_data;
    size_t c2_b, c2_m, c2_n;
    void * cluster_3_data;
    size_t c3_b, c3_m, c3_n;

    uint8_t alloc_ready;
    uint8_t data_ready;

    void * cont_data;
} dsp_multi_cluster_desc_t;

#define OFFSET_OF(struct_t, member) ((uint64_t)(&(((struct_t*)0ul)->member)))
#define DESC_FROM_DATA(data_ptr) \
    (dsp_multi_cluster_desc_t*)( \
        ((uint8_t*)(data_ptr)) - OFFSET_OF(dsp_multi_cluster_desc_t, cont_data) \
    )

#define ALIGN_UPBOUND(align_factor, var) ( ((var) % (align_factor)) ? \
    ((var) - ((var) % (align_factor)) + (align_factor)) : (var) )

#define PAGE_SIZE 4096
