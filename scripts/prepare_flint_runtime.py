"""Build an isolated, pinned Flint runtime; never modify the Ornith checkout.

The official graph and vocabulary are vendored under Experiments/Flint/Native.
This backport retains the reviewed LES descriptor/precision patches and changes
only Flint-specific architecture dispatch, graph and metadata. No weights.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

import prepare_ornith_les as les

ROOT = pathlib.Path(__file__).resolve().parents[1]
NATIVE = ROOT / "Experiments/Flint/Native"
SOURCE = ROOT / ".work/flint-source"
UPSTREAM_FLINT = "a37cc46104eb50900d5ab4a9ba02a0abdac039db"


def patched_files(source):
    result = les.patched_files(source)

    def edit(file, before, after, count=1):
        text = result.get(file, (source / file).read_text(encoding="utf-8"))
        if text.count(before) != count:
            raise ValueError(f"Pinned Flint patch context changed: {file}: {before[:70]}")
        result[file] = text.replace(before, after)

    edit("src/llama-arch.h", "    LLM_ARCH_QWEN3,", "    LLM_ARCH_QWEN3,\n    LLM_ARCH_FLINT,")
    edit("src/llama-arch.cpp", '    { LLM_ARCH_QWEN3,', '    { LLM_ARCH_FLINT, "flint" },\n    { LLM_ARCH_QWEN3,')
    edit("src/llama-arch.h", "    LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,",
         "    LLM_KV_ATTENTION_ROPE_PATTERN,\n    LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,")
    edit("src/llama-arch.cpp", '    { LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,',
         '    { LLM_KV_ATTENTION_ROPE_PATTERN, "%s.attention.rope_pattern" },\n    { LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,')
    edit("src/llama-hparams.h", "    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_layers;",
         "    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_layers;\n"
         "    std::array<uint32_t, LLAMA_MAX_LAYERS> flint_rope_layers{};")
    edit("src/models/models.h", "//\n// base classes\n//", """// Flint: official graph backported to this pinned runtime.
struct llama_model_flint : public llama_model_base {
    llama_model_flint(const llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
    struct graph : llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };
};
//
// base classes
//""")
    edit("src/llama-model.cpp", "        case LLM_ARCH_QWEN3:\n            return new llama_model_qwen3(params);",
         "        case LLM_ARCH_FLINT:\n            return new llama_model_flint(params);\n"
         "        case LLM_ARCH_QWEN3:\n            return new llama_model_qwen3(params);")
    # The second QWEN3 case belongs to the normal-RoPE model list.
    edit("src/llama-model.cpp", "        case LLM_ARCH_QWEN3:\n        case LLM_ARCH_QWEN3MOE:",
         "        case LLM_ARCH_FLINT:\n        case LLM_ARCH_QWEN3:\n        case LLM_ARCH_QWEN3MOE:")
    edit("src/llama-kv-cache.cpp", "if (model.arch == LLM_ARCH_STEP35)",
         "if (model.arch == LLM_ARCH_STEP35 || model.arch == LLM_ARCH_FLINT)")
    edit("src/llama-model.cpp", 'if (arch != LLM_ARCH_QWEN35MOE) {\n            throw std::runtime_error("MoE experts currently supports only qwen35moe");',
         'if (arch != LLM_ARCH_QWEN35MOE && arch != LLM_ARCH_FLINT) {\n            throw std::runtime_error("MoE experts requires qwen35moe or flint");')
    # The source GGUF interleaves dense and routed tensors across 26 GB. Do
    # not wrap gaps between dense tensors into enormous Metal buffers or
    # madvise(WILLNEED) the whole file. Keep CPU embeddings file-backed and
    # allocate only the fixed GPU weights plus expert banks in shared buffers.
    edit("src/llama-model.cpp", "    std::map<ggml_backend_dev_t, ggml_backend_buffer_type_t> alloc_only_buft;",
         "    std::map<ggml_backend_dev_t, ggml_backend_buffer_type_t> alloc_only_buft;\n"
         "    std::map<ggml_backend_dev_t, buft_list_t> flint_owned_buft_lists;")
    edit("src/llama-model.cpp", "    pimpl->dev_output = get_layer_buft_list(n_layer);", """    pimpl->dev_output = get_layer_buft_list(n_layer);
    if (arch == LLM_ARCH_FLINT) {
        auto bounded_gpu_weights = [&](impl::layer_dev & target) {
            if (!target.dev || target.dev == cpu_dev) return;
            const auto owned = pimpl->alloc_only_buft.find(target.dev);
            if (owned == pimpl->alloc_only_buft.end()) {
                throw std::runtime_error("Flint streaming requires owned host-visible GPU buffers");
            }
            auto & list = pimpl->flint_owned_buft_lists[target.dev];
            if (list.empty()) list.emplace_back(target.dev, owned->second);
            target.buft_list = &list;
        };
        for (auto & target : pimpl->dev_layer) bounded_gpu_weights(target);
        bounded_gpu_weights(pimpl->dev_output);
    }""")
    edit("src/llama-model.cpp", "    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);",
         "    ml.init_mappings(arch != LLM_ARCH_FLINT, use_mlock ? &pimpl->mlock_mmaps : nullptr);")
    # CPU alloc-only expert banks for bounded synthetic host controls. This is
    # never a trained full-model fallback and is not enabled in iOS builds.
    edit("src/llama-model.cpp", "#if defined(__APPLE__) && TARGET_OS_OSX\n    // Host-only synthetic control:",
         "#if !defined(__APPLE__) || TARGET_OS_OSX\n    // Host-only synthetic control:")
    edit("src/llama-model.cpp", "#if defined(__APPLE__) && TARGET_OS_OSX\n    // ggml's plain CPU buffer",
         "#if !defined(__APPLE__) || TARGET_OS_OSX\n    // ggml's plain CPU buffer")
    # The policy-changing fcntl only exists on Darwin. Linux controls exercise
    # the normal page cache; deliberately do not claim cold NAND benchmarks.
    edit("src/llama-moe-slot.cpp", "        if (fd >= 0 && fcntl(fd, F_NOCACHE, experts_nocache_desired() ? 1 : 0) != 0) {",
         "#if defined(__APPLE__)\n        if (fd >= 0 && fcntl(fd, F_NOCACHE, experts_nocache_desired() ? 1 : 0) != 0) {")
    edit("src/llama-moe-slot.cpp", '"Cannot apply expert page-cache policy: %s", std::strerror(saved)));\n        }',
         '"Cannot apply expert page-cache policy: %s", std::strerror(saved)));\n        }\n#endif')
    edit("src/llama-moe-slot.cpp", "#ifndef _WIN32\n        std::lock_guard<std::mutex> lock(fd_mutex);\n        for (const auto & kv : fds)",
         "#if defined(__APPLE__)\n        std::lock_guard<std::mutex> lock(fd_mutex);\n        for (const auto & kv : fds)")
    edit("src/llama-graph.cpp", "    if (moe_decode_phase &&\n        moe_slots && moe_slots->uses_layer(il) && moe_slots->matmul_split() > 0 &&",
         "    if (arch != LLM_ARCH_FLINT && moe_decode_phase &&\n        moe_slots && moe_slots->uses_layer(il) && moe_slots->matmul_split() > 0 &&")
    vocab = (NATIVE / "FlintVocabulary.inc").read_text(encoding="utf-8")
    edit("src/llama-vocab.h", "enum llama_vocab_pre_type {", "enum llama_vocab_pre_type {\n    LLAMA_VOCAB_PRE_TYPE_FLINT = 1000,")
    edit("src/llama-vocab.cpp", "        switch (vocab.get_pre_type()) {",
         "        switch (vocab.get_pre_type()) {\n" + vocab)
    edit("src/llama-vocab.cpp", '            } else if (tokenizer_pre == "default") {',
         '            } else if (tokenizer_pre == "flint") {\n'
         '                pre_type = LLAMA_VOCAB_PRE_TYPE_FLINT;\n'
         '                clean_spaces = false;\n'
         '            } else if (tokenizer_pre == "default") {')
    edit("src/llama-vocab.cpp", '\n                    || t.first == "<|end_of_text|>"\n',
         '\n                    || t.first == "<|end_of_text|>"\n'
         '                    || (pre_type == LLAMA_VOCAB_PRE_TYPE_FLINT && (t.first == "<|user|>" || t.first == "<|observation|>"))\n')
    # Official Flint down-expert overflow protection. Scale only the product
    # AFTER SwiGLU, undo AFTER the linear down projection, before bias/weights.
    # A power of two preserves the real-valued model function.
    edit("src/llama-graph.cpp", "    experts = build_lora_mm_id(down_exps, cur, selected_experts_mm); // [n_embd, n_expert_used, n_tokens]", """    ggml_tensor * flint_scale = nullptr;
    if (arch == LLM_ARCH_FLINT) {
        auto * amax = ggml_pool_1d(ctx0, ggml_abs(ctx0, cur), GGML_OP_POOL_MAX, cur->ne[0], cur->ne[0], 0);
        auto * e = ggml_scale(ctx0, amax, 1.0f / 1024.0f);
        e = ggml_scale(ctx0, ggml_log(ctx0, e), 1.4426950408889634f);
        e = ggml_clamp(ctx0, ggml_ceil(ctx0, e), 0.0f, 127.0f);
        flint_scale = ggml_exp(ctx0, ggml_scale(ctx0, e, 0.6931471805599453f));
        cur = ggml_div(ctx0, cur, flint_scale);
    }
    experts = build_lora_mm_id(down_exps, cur, selected_experts_mm); // [n_embd, n_expert_used, n_tokens]
    if (flint_scale) experts = ggml_mul(ctx0, experts, flint_scale);""")
    result["src/models/flint.cpp"] = (NATIVE / "FlintModel.cpp").read_text(encoding="utf-8")
    return result


def prepare(source=SOURCE):
    source = source.resolve()
    if not (source / ".git").exists():
        if source.exists() and any(source.iterdir()):
            raise ValueError("Refusing to replace an existing non-checkout source directory")
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", les.REPOSITORY, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", les.REVISION], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", les.REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != les.REVISION:
        raise ValueError(f"Expected detached source at {les.REVISION}, got {revision}; fetch/check out the pin in a fresh build directory")
    # Compute patches from immutable git blobs, even after a prior preparation.
    class PinnedFile:
        def __init__(self, name): self.name = name
        def read_text(self, encoding):
            return subprocess.check_output(["git", "-C", str(source), "show", f"{les.REVISION}:{self.name}"], encoding=encoding)
    class PinnedSource:
        def __truediv__(self, name): return PinnedFile(name)
    files = patched_files(PinnedSource())
    hashes = {}
    for name, text in files.items():
        data = text.encode("utf-8")
        target = source / name
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"schemaVersion": 1, "repository": les.REPOSITORY, "baseRevision": les.REVISION,
                "flintGraphRevision": UPSTREAM_FLINT, "patches": hashes,
                "trainedModelNumericsVerified": False, "iPhoneVerified": False}
    (source / "flash-local-flint-source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=SOURCE)
    args = parser.parse_args()
    print(prepare(args.source))
