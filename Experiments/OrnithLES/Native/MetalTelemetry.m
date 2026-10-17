#import <Metal/Metal.h>
#include <stdatomic.h>
#include <stdint.h>
static _Atomic(uint64_t) dispatches, buffers, gpu_ns, timestamped_buffers;

void flash_omni_record_dispatch(void) {
    atomic_fetch_add_explicit(&dispatches, 1, memory_order_relaxed);
}
void flash_omni_commit(id<MTLCommandBuffer> buffer) {
    atomic_fetch_add_explicit(&buffers, 1, memory_order_relaxed);
    [buffer addCompletedHandler:^(id<MTLCommandBuffer> done) {
        double seconds = done.GPUEndTime - done.GPUStartTime;
        if (done.status == MTLCommandBufferStatusCompleted && done.GPUStartTime > 0 && seconds >= 0 && seconds < 3600) {
            atomic_fetch_add_explicit(&gpu_ns, (uint64_t)(seconds * 1e9), memory_order_relaxed);
            atomic_fetch_add_explicit(&timestamped_buffers, 1, memory_order_relaxed);
        }
    }];
    [buffer commit];
}
void flash_omni_metal_snapshot(uint64_t * d, uint64_t * b, uint64_t * n, uint64_t * t) {
    *d = atomic_load_explicit(&dispatches, memory_order_relaxed);
    *b = atomic_load_explicit(&buffers, memory_order_relaxed);
    *n = atomic_load_explicit(&gpu_ns, memory_order_relaxed);
    *t = atomic_load_explicit(&timestamped_buffers, memory_order_relaxed);
}
