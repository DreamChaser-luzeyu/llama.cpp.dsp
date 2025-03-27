#pragma once

#define NO_INIT_DEV      1
#define PROFILING_MEMCPY 1
#define TEST_CLUSTER_ID  1

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
