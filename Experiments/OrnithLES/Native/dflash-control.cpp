// This host-only executable includes the exact implementation to exercise its
// internal draft graph with tiny fixtures. No alternate exported production API
// or relaxed production geometry/tokenizer checks are compiled into the app.
#include "FlashOrnithLES.cpp"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache.h"
#include <iostream>
#include <set>

int main(int argc, char ** argv) {
    try {
        require(argc == 4, "target fixture, draft fixture and scratch directory required");
        const fs::path root(argv[3]); fs::create_directories(root);
        const fs::path prepared = root / "prepared"; fs::create_directories(prepared);
        fd_owner source{open(argv[1], O_RDONLY)}, draft_fd{open(argv[2], O_RDONLY)};
        require(source.value >= 0 && draft_fd.value >= 0, "Missing fixtures");
        require(file_size(source.value) < (2 << 20) && file_size(draft_fd.value) < (1 << 20), "Not tiny fixtures");
        require(flo_prepare(source.value, prepared.c_str(), nullptr, nullptr) == 0, flo_last_error());
        flo_options o{};
        o.context_tokens = 8192; o.prefill_slots = 32; o.cache_slots_per_layer = 8;
        o.cpu_threads = o.io_threads = 2; o.memory_ceiling_bytes = 5500000000ULL;
        o.experimental_dflash = true; o.additional_reserve_bytes = 1000000000ULL;
        std::set<int> phases;
        o.diagnostic = [](int32_t phase, void * data) { static_cast<std::set<int> *>(data)->insert(phase); return true; };
        o.diagnostic_user = &phases;
        std::vector<int32_t> ids;
        {
            std::unique_ptr<flo_context> reference(flo_open(source.value, prepared.c_str(), o, nullptr, nullptr));
            require(bool(reference), flo_last_error());
            const int32_t prompt[] = {1, 2, 3, 4, 1, 2, 3, 4, 1}; std::vector<float> logits(reference->vocabulary);
            require(flo_decode(reference.get(), prompt, 9, logits.data(), int(logits.size())) == 0, flo_last_error());
            for (int i = 0; i < 128; ++i) {
                ids.push_back(flo_argmax(logits.data(), int(logits.size())));
                require(flo_decode(reference.get(), &ids.back(), 1, logits.data(), int(logits.size())) == 0, flo_last_error());
            }
        }
        {
            std::unique_ptr<flo_context> c(flo_open(source.value, prepared.c_str(), o, nullptr, nullptr));
            require(bool(c), flo_last_error());
            auto d = std::make_unique<flo_draft>();
            d->file = fdopen(dup(draft_fd.value), "rb"); require(d->file, "Draft stream failed");
            auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.use_mmap = false; mp.use_extra_bufts = false;
            d->model = llama_model_load_from_file_ptr(d->file, mp); require(d->model, "Tiny DFlash load failed");
            require(d->model->arch == LLM_ARCH_DFLASH && d->model->hparams.n_layer == 6 &&
                !d->model->tok_embd && !d->model->output, "Fixture must share target embedding/head");
            d->layers = {1, 2, 3}; d->mask_token = 15;
            initialize_draft_context(c.get(), *d);
            const auto * cache = dynamic_cast<const llama_kv_cache_iswa *>(llama_get_memory(d->context));
            require(cache && cache->get_base()->get_size() == 8192 && cache->get_swa()->get_size() == 4352,
                "Draft cache allocation disagrees with sliding-window admission");
            require(draft_kv_bytes(8192) == 122683392 && draft_kv_bytes(32768) == 223346688,
                "Production draft KV budget differs from the padded native allocation");
            c->draft = std::move(d);
            const int32_t prompt[] = {1, 2, 3, 4, 1, 2, 3, 4, 1}; std::vector<float> logits(c->vocabulary);
            require(flo_ingest(c.get(), prompt, 4) == 0 && flo_ingest(c.get(), prompt + 4, 4) == 0, flo_last_error());
            require(flo_decode(c.get(), prompt + 8, 1, logits.data(), int(logits.size())) == 0, flo_last_error());
            require(flo_snapshot(c.get()).state_only_prefill_tokens == 8 &&
                flo_snapshot(c.get()).state_only_prefill_calls == 2, "Draft state-only prefill was not executed");
            size_t offset = 0;
            while (offset < ids.size()) {
                int32_t block[4];
                const int n = flo_dflash_step(c.get(), logits.data(), int(logits.size()), block,
                    std::min<int>(4, ids.size() - offset));
                require(n > 0, flo_last_error());
                require(std::equal(block, block + n, ids.begin() + offset), "Draft leaked a rejected or changed target token");
                offset += n;
            }
            require(c->draft_blocks > 0 && c->draft_proposals > 0 && c->feature_peak > 0, "Draft graph did not execute");
            for (int phase : {11, 12, 13, 14, 15, 16, 17, 19}) require(phases.count(phase), "Missing native phase diagnostic");
        }
        require(live_contexts.load() == 0, "Control leaked context ownership");
        std::cout << "ORNITH_DFLASH_GRAPH encoder=true injection=true proposals=true targetVerification=true rollback=true "
            "greedyTokens=128 slidingCells=4352 baseCells=8192 bf16=true syntheticOnly=true physicalPhone=false\n";
        std::cout << "ORNITH_DFLASH_PREFILL stateOnlyTokens=8 ordinaryReference=true greedyTokens=128 syntheticOnly=true physicalPhone=false\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
