#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// No allocation. Exact packed byte count and output capacity are mandatory.
// Returns zero on success; rejects unsupported types/geometry/buffer sizes.
int flash_dequantize(uint32_t type, const void * packed, size_t packed_bytes,
                     float * output, size_t output_elements, size_t elements);
#ifdef __cplusplus
}
#endif
