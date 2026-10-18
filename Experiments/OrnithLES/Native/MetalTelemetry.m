#import <Metal/Metal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
static _Atomic(uint64_t) dispatches, buffers, gpu_ns, timestamped_buffers;
static _Atomic(uint64_t) live_buffers, owned_bytes, mapped_bytes;
// The pinned ggml device singleton outlives every context and snapshot. Do not
// create a device merely to query telemetry (CPU controls have no Metal work).
static _Atomic(uintptr_t) observed_device;

void flash_omni_buffer_lifetime(id<MTLDevice> device, bool allocated, bool owned, size_t bytes) {
    atomic_store_explicit(&observed_device, (uintptr_t)device, memory_order_relaxed);
    if (allocated) {
        atomic_fetch_add_explicit(&live_buffers, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(owned ? &owned_bytes : &mapped_bytes, bytes, memory_order_relaxed);
    } else {
        atomic_fetch_sub_explicit(&live_buffers, 1, memory_order_relaxed);
        atomic_fetch_sub_explicit(owned ? &owned_bytes : &mapped_bytes, bytes, memory_order_relaxed);
    }
}
void flash_omni_memory_snapshot(uint64_t * count, uint64_t * owned, uint64_t * mapped,
                                uint64_t * device_bytes, bool * observed) {
    *count = atomic_load_explicit(&live_buffers, memory_order_relaxed);
    *owned = atomic_load_explicit(&owned_bytes, memory_order_relaxed);
    *mapped = atomic_load_explicit(&mapped_bytes, memory_order_relaxed);
    id<MTLDevice> device = (id<MTLDevice>)atomic_load_explicit(&observed_device, memory_order_relaxed);
    *observed = device != nil;
    *device_bytes = device ? device.currentAllocatedSize : 0;
}

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
