"""Pinned experimental LES source. No weights, upstream writes or model downloads.

The local patch preserves router selection and all mathematical operators. It
adds a tensor-major source reader, aggregate instrumentation and an accessor.
Q4_K decoding promotes the stored half scale before dividing by sixteen, so
small scales retain the values represented by the original packed weights.
CPU fused attention honors requested F32 value accumulation with F16 KV.
Interleaved SWA KV injection skips unallocated, unused attention masks.
The upstream checkout is an ignored build input, never the production runtime.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

REVISION = "96bbf61710d44421ea54e63f5194f9b17c20d877"
REPOSITORY = "https://github.com/tc-mb/llama.cpp-omni.git"


def patched_files(source):
    result = {}

    def replace(file, old, new, count=1):
        text = result.get(file, (source / file).read_text(encoding="utf-8"))
        if text.count(old) != count:
            raise ValueError(f"Pinned LES patch context changed: {file}")
        result[file] = text.replace(old, new)

    replace("src/llama-moe-experts.h", "    int32_t layer = -1;",
            "    bool tensor_major = false; // Flash Local: byte-identical GGUF source spans\n    int32_t layer = -1;")
    replace("src/llama-moe-experts.cpp", '#include "llama-moe-experts.h"',
            '#include "llama-moe-experts.h"\n#include "VerifiedDescriptor.h"')
    replace("src/llama-moe-experts.cpp", "fs::file_size(entry.path, error)",
            "flash_local_expert_file_size(entry.path, error)")
    replace("src/llama-moe-slot.cpp", '#include "llama-moe-slot.h"',
            '#include "llama-moe-slot.h"\n#include "VerifiedDescriptor.h"')
    replace("src/llama-moe-experts.cpp",
            'layout != "layer_major_expert" && layout != "expert_major"',
            'layout != "layer_major_expert" && layout != "expert_major" && layout != "gguf_tensor_major"')
    replace("src/llama-moe-experts.cpp", "        llama_moe_experts_entry entry;",
            '        llama_moe_experts_entry entry;\n        entry.tensor_major = layout == "gguf_tensor_major";')
    replace("src/llama-moe-experts.cpp", "        if (entry.expert_stride < family_end) {",
            "        if (entry.tensor_major ? entry.expert_stride != entry.bytes_per_expert : entry.expert_stride < family_end) {")
    replace("src/llama-moe-slot.cpp", "        const size_t st = bindings[0].entry->expert_stride;", """        if (bindings[0].entry->tensor_major) {
            // Keep separate original GGUF family extents. Destination slots
            // remain GPU-visible; each family is read directly, without staging.
            base = 0; stride = 0; path = p; order.clear();
            for (size_t i = 0; i < bindings.size(); ++i) {
                const auto * entry = bindings[i].entry;
                if (!entry->tensor_major || entry->path != p ||
                    entry->expert_stride != entry->bytes_per_expert) return;
                order.push_back(i); stride += entry->bytes_per_expert;
            }
            ok = true; return;
        }
        const size_t st = bindings[0].entry->expert_stride;""")
    replace("src/llama-moe-slot.cpp", "        readv_span(fd, offset, iov, n);", """        if (bindings[order[0]].entry->tensor_major) {
            for (int i = 0; i < n; ++i) {
                const auto * entry = bindings[order[(size_t)i]].entry;
                readv_span(fd, entry->offset + (size_t)expert * entry->expert_stride, &iov[i], 1);
            }
        } else {
            readv_span(fd, offset, iov, n);
        }""")
    replace("src/llama-moe-slot.h", "        int64_t process;", """        uint64_t all_bytes, all_calls, predicted, prediction_hits, prefetch_bytes;
        uint64_t demand_hits, demand_misses;
        int64_t process;""")
    replace("src/llama-moe-slot.cpp", "    std::atomic<uint64_t> io_bytes_all{0};",
            "    std::atomic<uint64_t> demand_hits{0}, demand_misses{0};\n    std::atomic<uint64_t> io_bytes_all{0};")
    replace("src/llama-moe-slot.cpp", "    const int n_hit = n_unique - n_miss;", """    const int n_hit = n_unique - n_miss;
    pimpl->demand_hits.fetch_add((uint64_t)n_hit, std::memory_order_relaxed);
    pimpl->demand_misses.fetch_add((uint64_t)n_miss, std::memory_order_relaxed);""")
    replace("src/llama-moe-slot.cpp", "    counters c;", """    counters c;
    c.all_bytes = pimpl->io_bytes_all.load(std::memory_order_relaxed);
    c.all_calls = pimpl->io_calls_all.load(std::memory_order_relaxed);
    c.predicted = pimpl->lookahead.n_armed.load(std::memory_order_relaxed);
    c.prediction_hits = pimpl->lookahead.n_hits.load(std::memory_order_relaxed);
    c.prefetch_bytes = pimpl->lookahead.n_pred_bytes.load(std::memory_order_relaxed);""")
    replace("src/llama-moe-slot.cpp", "    c.process  = pimpl->process_ns.load(std::memory_order_relaxed);", """    c.demand_hits = pimpl->demand_hits.load(std::memory_order_relaxed);
    c.demand_misses = pimpl->demand_misses.load(std::memory_order_relaxed);
    c.process  = pimpl->process_ns.load(std::memory_order_relaxed);""")
    # Cache mode is a per-context experiment, not a process-first-call setting.
    replace("src/llama-moe-slot.cpp", "        static const int m = [] {", "        const int m = [] {")
    replace("src/llama-moe-slot.cpp", """        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (experts_nocache_desired() && fcntl(fd, F_NOCACHE, 1) != 0) {
            moe_log("F_NOCACHE 设置失败: %s", std::strerror(errno));
        }""", """        const int fd = flash_local_open_experts(path);
        // A preceding production TileReader may have set F_NOCACHE on a dup
        // of this same source. Explicitly clear it for page-cache trials.
        if (fd >= 0 && fcntl(fd, F_NOCACHE, experts_nocache_desired() ? 1 : 0) != 0) {
            const int saved = errno;
            close(fd);
            throw std::runtime_error(format("Cannot apply expert page-cache policy: %s", std::strerror(saved)));
        }""")
    replace("src/llama-context.h", "struct llama_context {", """struct llama_context {
    llama_moe_slot_runtime * flash_local_slots() const { return moe_slots.get(); }""")
    replace("src/llama-model.cpp", '#include "llama-model.h"',
            '#include "llama-model.h"\n#if defined(__APPLE__)\n#include <TargetConditionals.h>\n#endif')
    replace("src/llama-model.cpp", "    // calculate the split points", """#if defined(__APPLE__) && TARGET_OS_OSX
    // Host-only synthetic control: no mmap, no trained-model fallback. CPU
    // devices are not in the GPU-device loop that registers alloc-only bufts.
    const char * control = std::getenv("ORNITH_LES_CPU_CONTROL");
    if (control && std::strcmp(control, "1") == 0 && n_gpu_layers == 0) {
        pimpl->alloc_only_buft.emplace(cpu_dev, ggml_backend_cpu_buffer_type());
        pimpl->cpu_buft_list = {{cpu_dev, ggml_backend_cpu_buffer_type()}};
    }
#endif
    // calculate the split points""")
    replace("src/llama-model.cpp", """    const auto it = dev ? pimpl->alloc_only_buft.find(dev) : pimpl->alloc_only_buft.end();
    if (it == pimpl->alloc_only_buft.end()) {
        throw std::runtime_error(format(""", """#if defined(__APPLE__) && TARGET_OS_OSX
    // ggml's plain CPU buffer intentionally has device=nullptr. Only this
    // bounded host control uses the assigned CPU device instead of that buft.
    const char * control = std::getenv("ORNITH_LES_CPU_CONTROL");
    if (!dev && tn.bid >= 0 && params.n_gpu_layers == 0 &&
        control && std::strcmp(control, "1") == 0) {
        dev = pimpl->dev_layer.at(tn.bid).dev;
    }
#endif
    const auto it = dev ? pimpl->alloc_only_buft.find(dev) : pimpl->alloc_only_buft.end();
    if (it == pimpl->alloc_only_buft.end()) {
        throw std::runtime_error(format(""")
    replace("src/models/qwen35moe.cpp", "il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp != nullptr",
            "il + 1 < n_transformer_layers && model.layers[il + 1].ffn_gate_inp != nullptr")
    # DFlash KV injection writes K/V without performing attention. Its mask
    # tensors are consequently pruned and have no buffer. The plain KV input
    # already handles that case; the interleaved SWA input did not, aborting in
    # ggml_backend_buffer_is_host(nullptr) on the very first injection pass.
    # Keep all allocated-mask behavior (ordinary attention) exactly unchanged.
    file = "src/llama-graph.cpp"
    for suffix, cache in [("", "base"), ("_swa", "swa")]:
        call = f"mctx->get_{cache}()->set_input_kq_mask(self_kq_mask{suffix}, ubatch, cparams.causal_attn);"
        replace(file, "        " + call,
                f"        if (self_kq_mask{suffix} && self_kq_mask{suffix}->buffer) {{\n            {call}\n        }}")
        call = f"res &= can_reuse_kq_mask(self_kq_mask{suffix}, mctx->get_{cache}(), params.ubatch, params.cparams);"
        replace(file, "        " + call,
                f"        if (self_kq_mask{suffix} && self_kq_mask{suffix}->buffer) {{\n            {call}\n        }}")
    file = "ggml/src/ggml-metal/ggml-metal-context.m"
    replace(file, '#import <Metal/Metal.h>', '#import <Metal/Metal.h>\nextern void flash_omni_commit(id<MTLCommandBuffer>);')
    text = result[file]
    # All live commit sites; leave commented diagnostics untouched.
    pattern = r"(?m)^(\s*)\[(cmd_buf|next_buffer) commit\];"
    text, count = re.subn(pattern, r"\1flash_omni_commit(\2);", text)
    if count != 7:
        raise ValueError(f"Pinned Metal commit sites changed: {count}")
    result[file] = text
    replace("ggml/src/ggml-metal/ggml-metal-device.m",
            "    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];",
            "    extern void flash_omni_record_dispatch(void);\n    flash_omni_record_dispatch();\n"
            "    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];")
    file = "ggml/src/ggml-metal/ggml-metal-device.m"
    replace(file, '#include <Metal/Metal.h>', '#include <Metal/Metal.h>\nextern void flash_omni_commit(id<MTLCommandBuffer>);')
    result[file], count = re.subn(pattern, r"\1flash_omni_commit(\2);", result[file])
    if count != 5:
        raise ValueError(f"Pinned Metal copy/fill commit sites changed: {count}")
    # Keep allocation/free symmetric. iOS 27 can leave freed malloc-backed
    # page buffers charged to phys_footprint even after allocator relief.
    # Reuse ggml's existing macOS VM path, including CPU aligned buffers.
    # See ggml-org/llama.cpp#30556; phone confirmation remains a release gate.
    replace("ggml/src/ggml.c", "  #elif TARGET_OS_OSX\n    GGML_UNUSED(alignment);",
            "  #elif defined(__APPLE__)\n    GGML_UNUSED(alignment);")
    replace("ggml/src/ggml.c", "#elif TARGET_OS_OSX\n    if (ptr != NULL) {\n        vm_deallocate",
            "#elif defined(__APPLE__)\n    if (ptr != NULL) {\n        vm_deallocate")
    replace(file, "#if TARGET_OS_OSX\n    kern_return_t err = vm_allocate",
            "#if defined(__APPLE__)\n    kern_return_t err = vm_allocate")
    replace(file, "#if TARGET_OS_OSX\n        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);",
            "#if defined(__APPLE__)\n        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);")
    replace(file, 'extern void flash_omni_commit(id<MTLCommandBuffer>);',
            'extern void flash_omni_commit(id<MTLCommandBuffer>);\n'
            'extern void flash_omni_buffer_lifetime(id<MTLDevice>, bool, bool, size_t);')
    # Count only successfully created buffers, and pair the free before the
    # wrapper disappears. Mapped bytes are address spans, not physical RAM.
    replace(file, "    ggml_metal_device_rsets_add(dev, res->rset);",
            "    flash_omni_buffer_lifetime(dev->mtl_device, true, res->owned, res->all_size);\n"
            "    ggml_metal_device_rsets_add(dev, res->rset);", count=2)
    replace(file, "    free(buf);\n}\n\nbool ggml_metal_buffer_park",
            "    flash_omni_buffer_lifetime(buf->dev->mtl_device, false, buf->owned, buf->all_size);\n"
            "    free(buf);\n}\n\nbool ggml_metal_buffer_park")
    # The shared dequantizer feeds GET_ROWS and small-batch matvec kernels.
    # Half division loses scale bits before the subsequent float assignment.
    # Preserve the existing bytes/levels, doing the scale adjustment in F32.
    replace("ggml/src/ggml-metal/ggml-metal.metal",
            "    const float d   = il < 2 ? xb->d : xb->d / 16.h;",
            "    const float d   = il < 2 ? xb->d : float(xb->d) / 16.f;")
    # The scalar CPU flash-attention path previously accumulated V in half
    # even when its graph explicitly requested GGML_PREC_F32. Reuse its
    # existing F32 accumulator/conversion path, keeping F16 storage and the
    # default-precision branch unchanged. No additional sequence-sized buffer.
    file = "ggml/src/ggml-cpu/ops.cpp"
    old = "    ggml_to_float_t   const v_to_float     = ggml_get_type_traits(v->type)->to_float;"
    replace(file, old, old + "\n    const bool f16_accumulator = v->type == GGML_TYPE_F16 && dst->op_params[3] != GGML_PREC_F32;")
    replace(file, "if (v->type == GGML_TYPE_F16) {", "if (f16_accumulator) {", count=3)
    return result


def prepare(root, run=None):
    source = root / ".work/ornith-les-source"
    if run is None:
        def run(args):
            subprocess.run([str(x) for x in args], cwd=root, check=True)
    if not source.exists():
        # The upstream branch is mutable. Fetch the reviewed object directly
        # so a fresh build keeps working after that branch advances. Existing
        # inputs still fail closed below; never reset a caller's checkout.
        source.mkdir(parents=True)
        run(["git", "-C", source, "init"])
        run(["git", "-C", source, "remote", "add", "origin", REPOSITORY])
        run(["git", "-C", source, "fetch", "--depth", "1", "origin", REVISION])
        run(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError("LES source moved; refusing an unreviewed revision")
    record = source / "flash-local-source.json"
    script_hash = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    if record.exists():
        saved = json.loads(record.read_text())
        if saved["patchScriptSHA256"] != script_hash or any(
                hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
                for name, digest in saved["patchedSHA256"].items()):
            raise ValueError("LES local patch changed; use a new clean build-input directory")
        return source
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("LES input has unrelated tracked modifications")
    files = patched_files(source)
    for name, text in files.items():
        (source / name).write_text(text, encoding="utf-8", newline="\n")
    record.write_text(json.dumps(dict(revision=REVISION, repository=REPOSITORY,
        patchScriptSHA256=script_hash,
        patchedSHA256={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files},
        changes=["tensor-major direct GGUF reads", "aggregate I/O and Metal telemetry", "exclude unused MTP prediction",
                 "Apple VM-backed page buffers with symmetric deallocation", "native buffer lifetime accounting",
                 "Q4_K GPU nibble-scale adjustment in F32", "CPU Flash Attention requested F32 accumulation honored",
                 "DFlash interleaved SWA KV injection skips unallocated attention masks"],
        routingChanged=False, quantizationChanged=False), indent=2) + "\n")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(prepare(pathlib.Path(__file__).resolve().parents[1]))
