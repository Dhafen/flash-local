#include "flash_quants.h"
#include <assert.h>
#include <limits.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_RESTRICT restrict
#include "vendor/ggml-common.h"

static float flash_half_to_float(uint16_t bits) {
    _Float16 half;
    memcpy(&half, &bits, sizeof(half));
    return (float) half;
}
#define GGML_FP16_TO_FP32(x) flash_half_to_float(x)
#include "vendor/dequant.inc"

int flash_dequantize(uint32_t type, const void * packed, size_t packed_bytes,
                     float * output, size_t output_elements, size_t elements) {
    size_t block, bytes, alignment = 2;
    switch (type) {
        case 0: block = 1; bytes = 4; alignment = 1; break;
        case 1: case 30: block = 1; bytes = 2; alignment = 1; break;
        case 2: block = 32; bytes = 18; break;
        case 6: block = 32; bytes = 22; break;
        case 8: block = 32; bytes = 34; break;
        case 12: block = 256; bytes = 144; alignment = 4; break;
        case 14: block = 256; bytes = 210; break;
        default: return 1;
    }
    // Bound a single tile to 16 MiB FP32, including the canonical routines'
    // narrower loop counters. No caller can accidentally request a whole head.
    if (!packed || !output || elements == 0 || elements > 4 * 1024 * 1024 ||
        elements % block || output_elements < elements ||
        packed_bytes != elements / block * bytes ||
        (uintptr_t) packed % alignment || (uintptr_t) output % _Alignof(float)) return 2;
    const uintptr_t in = (uintptr_t) packed, out = (uintptr_t) output;
    if (in > UINTPTR_MAX - packed_bytes || out > UINTPTR_MAX - elements * sizeof(float)) return 2;
    if (in < out + elements * sizeof(float) && out < in + packed_bytes) return 2;
    switch (type) {
        case 0: memcpy(output, packed, packed_bytes); break;
        case 1:
            for (size_t i = 0; i < elements; ++i) {
                uint16_t bits; memcpy(&bits, (const uint8_t *) packed + i * 2, 2);
                output[i] = flash_half_to_float(bits);
            }
            break;
        case 30:
            for (size_t i = 0; i < elements; ++i) {
                uint16_t half; memcpy(&half, (const uint8_t *) packed + i * 2, 2);
                uint32_t bits = (uint32_t) half << 16; memcpy(output + i, &bits, 4);
            }
            break;
        case 2: dequantize_row_q4_0(packed, output, (int64_t) elements); break;
        case 6: dequantize_row_q5_0(packed, output, (int64_t) elements); break;
        case 8: dequantize_row_q8_0(packed, output, (int64_t) elements); break;
        case 12: dequantize_row_q4_K(packed, output, (int64_t) elements); break;
        case 14: dequantize_row_q6_K(packed, output, (int64_t) elements); break;
    }
    return 0;
}
