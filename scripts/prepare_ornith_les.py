"""Pinned experimental LES source. No weights, upstream writes or model downloads.

The local patch preserves router selection and all mathematical operators. It
adds a tensor-major source reader, aggregate instrumentation and an accessor.
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
    return result


def prepare(root, run=None):
    source = root / ".work/ornith-les-source"
    if run is None:
        def run(args):
            subprocess.run([str(x) for x in args], cwd=root, check=True)
    if not source.exists():
        run(["git", "clone", "--filter=blob:none", "--depth", "1", "--single-branch", "--branch", "ios-omni",
             "--no-recurse-submodules", REPOSITORY, source])
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
        changes=["tensor-major direct GGUF reads", "aggregate I/O and Metal telemetry", "exclude unused MTP prediction"],
        routingChanged=False, quantizationChanged=False), indent=2) + "\n")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(prepare(pathlib.Path(__file__).resolve().parents[1]))
