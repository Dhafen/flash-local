#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static _Atomic(uint64_t) dispatches, buffers, gpu_ns, timestamped_buffers;

// The hosted Mac exposes an Apple5 virtual device even when its emulator
// implements SIMD reduction. Never infer support from a requested flag: run
// and verify the exact instruction first. This path cannot run on iPhone.
bool flash_omni_ci_simd_probe(id<MTLDevice> device) {
#if TARGET_OS_OSX
    const char * requested = getenv("ORNITH_LES_CI_SIMD_PROBE");
    if (!requested || strcmp(requested, "1") || ![device.name containsString:@"Paravirtual"]) return false;
    static bool passed = false;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        @autoreleasepool {
            NSString * source = @"#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void probe(device float * out [[buffer(0)]], uint gid [[thread_position_in_grid]], "
                "uint lane [[thread_index_in_simdgroup]], uint3 group [[threadgroup_position_in_grid]]) { "
                "float value = simd_sum(float(gid + 1)); if (lane == 0) out[group.x] = value; }";
            NSError * error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
            id<MTLFunction> function = [library newFunctionWithName:@"probe"];
            id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
            id<MTLBuffer> result = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (pipeline && result && queue && pipeline.threadExecutionWidth == 32) {
                memset(result.contents, 0, 16);
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline]; [encoder setBuffer:result offset:0 atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
                passed = command.status == MTLCommandBufferStatusCompleted;
                for (int i = 0; i < 4; ++i) passed = passed && ((float *)result.contents)[i] == 528.0f + 1024.0f * i;
            }
            [queue release]; [result release]; [pipeline release]; [function release]; [library release];
            fprintf(stderr, "ORNITH_LES_VIRTUAL_GPU reductionProbe=%s physicalPhone=false\n", passed ? "passed" : "failed");
        }
    });
    return passed;
#else
    (void)device; return false;
#endif
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
