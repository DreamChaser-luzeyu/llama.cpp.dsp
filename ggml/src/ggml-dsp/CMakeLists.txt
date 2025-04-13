set(GGML_DSP_NAME ggml-dsp)

list (APPEND GGML_DSP_SOURCES
    ggml-cpu.c
    ggml-dsp.cpp
    ggml-cpu-aarch64.cpp
    ggml-cpu-aarch64.h
    ggml-cpu-quants.c
    ggml-cpu-quants.h
    ggml-cpu-traits.cpp
    ggml-cpu-traits.h
    ggml-cpu-impl.h
    ggml-dsp-funcs.cpp
    ggml-dsp-warpper.cpp

    dsp_buffer_utils.cpp
    dsp_tensor_utils.cpp
    )
ggml_add_backend_library(${GGML_DSP_NAME} ${GGML_DSP_SOURCES})