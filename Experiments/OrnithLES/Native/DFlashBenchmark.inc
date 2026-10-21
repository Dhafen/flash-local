// Isolated greedy DFlash benchmark, using the pinned runtime's encoder and
// cache-injection graph. Algorithm follows common/speculative.cpp (MIT;
// retained upstream notice in OrnithLES-LICENSE.txt). No Python/common server
// dependency, target requantization or changed expert routing.
struct flo_batch_owner {
    llama_batch value;
    explicit flo_batch_owner(int count, int embedding = 0, bool mrope = false)
        : value(llama_batch_init(count, embedding, 1)) {
        if (mrope) {
            free(value.pos); value.pos = (llama_pos *)calloc(size_t(count) * 4, sizeof(llama_pos));
            require(value.pos, "Draft position allocation failed");
        }
        value.n_tokens = count;
    }
    ~flo_batch_owner() { llama_batch_free(value); }
    flo_batch_owner(const flo_batch_owner &) = delete;
};
static int flo_argmax(const float * logits, int vocabulary) {
    require(logits && vocabulary > 0, "Missing speculative logits");
    int best = 0;
    for (int i = 0; i < vocabulary; ++i) {
        require(std::isfinite(logits[i]), "Non-finite speculative logit");
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}
static void flo_positions(llama_batch & b, int start, bool mrope, bool logits) {
    for (int i = 0; i < b.n_tokens; ++i) {
        b.pos[i] = start + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = logits;
        if (mrope) {
            b.pos[b.n_tokens + i] = start + i; b.pos[2 * b.n_tokens + i] = start + i;
            b.pos[3 * b.n_tokens + i] = 0;
        }
    }
}
static void draft_process(flo_context * c, int position, int count) {
    auto & d = *c->draft;
    check_abort(c->abort, c->user);
    require(count > 0 && count <= 16, "Draft feature microbatch bound");
    const int hidden = llama_model_n_embd(c->model), width = hidden * int(d.layers.size());
    d.features.resize(size_t(count) * width);
    c->feature_peak = std::max(c->feature_peak, uint64_t(d.features.size() * sizeof(float)));
    for (size_t k = 0; k < d.layers.size(); ++k) {
        const float * source = llama_get_embeddings_layer_inp(c->context, d.layers[k]);
        require(source, "Missing extracted target feature stream");
        for (int i = 0; i < count; ++i)
            std::copy(source + size_t(i) * hidden, source + size_t(i + 1) * hidden,
                d.features.begin() + size_t(i) * width + k * hidden);
    }
    // Drop draft-only noise positions before injecting actual target features.
    require(llama_memory_seq_rm(llama_get_memory(d.context), 0, position, -1), "Draft suffix trim failed");
    flo_batch_owner encoded(count, width, d.mrope);
    std::copy(d.features.begin(), d.features.end(), encoded.value.embd);
    flo_positions(encoded.value, position, d.mrope, false);
    require(llama_encode(d.context, encoded.value) == 0, "Draft feature fusion failed");
    const float * fused = llama_get_embeddings_pre_norm(d.context);
    require(fused, "Draft feature encoder returned no rows");
    flo_batch_owner injected(count, hidden, d.mrope);
    std::copy(fused, fused + size_t(count) * hidden, injected.value.embd);
    flo_positions(injected.value, position, d.mrope, false);
    require(llama_decode(d.context, injected.value) == 0, "Draft KV injection failed");
    check_abort(c->abort, c->user);
}
int flo_attach_dflash(flo_context * c, int source) {
    try {
        require(c && !c->poisoned && c->experimental_dflash && c->metal && !c->draft &&
            c->position == 0 && c->batch >= 4, "DFlash needs a fresh benchmark-only Metal context, batch four");
        const uint64_t bytes = file_size(source);
        const uint64_t draft_kv = 4096ULL * (c->capacity + 5ULL * std::min(c->capacity, 4096));
        require(bytes <= 800000000ULL && c->additional_reserve >= bytes + draft_kv + (64ULL << 20),
            "DFlash weights/KV/scratch not admitted");
        auto d = std::make_unique<flo_draft>(); d->source.value = dup(source);
        require(d->source.value >= 0, "Cannot retain draft descriptor");
        int fd = dup(d->source.value); require(fd >= 0, "Cannot open draft stream");
        d->file = fdopen(fd, "rb"); if (!d->file) { close(fd); throw std::runtime_error("Draft stream failed"); }
        require(fseek(d->file, 0, SEEK_SET) == 0, "Cannot rewind draft stream");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999; mp.use_mmap = true; mp.use_mlock = false; mp.use_extra_bufts = false;
        mp.progress_callback = [](float, void * data) {
            auto * c = static_cast<flo_context *>(data); return !c->abort || !c->abort(c->user);
        };
        mp.progress_callback_user_data = c;
        d->model = llama_model_load_from_file_ptr(d->file, mp); require(d->model, "Draft model load failed");
        require(d->model->arch == LLM_ARCH_DFLASH && d->model->hparams.n_layer == 6 &&
            llama_model_n_embd(d->model) == 2048 && llama_model_n_embd(c->model) == 2048 &&
            c->model->hparams.n_layer - c->model->hparams.nextn_predict_layers == 40,
            "Unsupported Ornith draft/target geometry");
        require(!d->model->tok_embd && !d->model->output, "Draft must share target embedding and output weights");
        const auto * ids = llama_model_target_layer_ids(d->model);
        const int32_t expected[] = {2, 7, 12, 17, 23, 28, 33, 38};
        require(llama_model_target_layer_ids_n(d->model) == 8 && ids && std::equal(ids, ids + 8, expected),
            "Draft target-layer offset mismatch");
        d->layers.assign(ids, ids + 8);
        const auto * a = llama_model_get_vocab(c->model); const auto * b = llama_model_get_vocab(d->model);
        require(llama_vocab_n_tokens(b) == c->vocabulary && c->vocabulary == 248320,
            "Draft vocabulary shape mismatch");
        for (int i = 0; i < c->vocabulary; ++i) {
            if (i % 4096 == 0) check_abort(c->abort, c->user);
            const char * pa = llama_vocab_get_text(a, i), * pb = llama_vocab_get_text(b, i);
            require(pa && pb && std::strcmp(pa, pb) == 0, "Draft token IDs differ from installed target");
        }
        require(llama_vocab_mask(b) == 248077, "Draft mask ID mismatch");
        d->mrope = llama_model_rope_type(d->model) == LLAMA_ROPE_TYPE_MROPE;
        auto cp = llama_context_default_params();
        cp.n_ctx = c->capacity; cp.n_batch = cp.n_ubatch = 16; cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.ctx_other = c->context;
        cp.no_perf = false; cp.abort_callback = c->abort; cp.abort_callback_data = c->user;
        d->context = llama_init_from_model(d->model, cp); require(d->context, "Draft context allocation failed");
        llama_set_embeddings_pre_norm(d->context, true, false); llama_set_causal_attn(d->context, false);
        for (int layer : d->layers) llama_set_embeddings_layer_inp(c->context, layer, true);
        check_abort(c->abort, c->user); c->draft = std::move(d); return 0;
    } catch (const std::exception & e) { if (c) c->poisoned = true; last_error = e.what(); return -1; }
}
int flo_verify_greedy(flo_context * c, const int32_t * block, int32_t count, float * output, int32_t vocabulary) {
    try {
        require(c && !c->poisoned && block && output && vocabulary == c->vocabulary &&
            count >= 1 && count <= 4 && count <= c->batch && count <= c->capacity - c->position,
            "Invalid greedy verification block");
        for (int i = 0; i < count; ++i) require(block[i] >= 0 && block[i] < vocabulary, "Invalid proposal token");
        require(flo_capture_prefix(c) > 0, "Cannot capture speculative recurrent state");
        const int start = c->position;
        flo_batch_owner batch(count); std::copy(block, block + count, batch.value.token);
        flo_positions(batch.value, start, false, true);
        require(llama_decode(c->context, batch.value) == 0 && !c->routing_failed, "Target block verification failed");
        c->target_tokens += count; c->position += count;
        c->maximum_physical_batch = std::max(c->maximum_physical_batch, count);
        int accepted = 1; // the caller chose the anchor from ordinary target logits
        for (int i = 0; i < count; ++i) {
            const float * row = llama_get_logits_ith(c->context, i);
            const int next = flo_argmax(row, vocabulary);
            if (i + 1 < count && accepted == i + 1 && next == block[i + 1]) ++accepted;
        }
        if (accepted < count) {
            require(flo_restore_prefix(c) == 0, "Speculative state rollback failed");
            ++c->draft_replays;
            require(flo_decode(c, block, accepted, output, vocabulary) == 0, "Accepted-prefix replay failed");
        } else {
            std::copy(llama_get_logits_ith(c->context, count - 1), llama_get_logits_ith(c->context, count - 1) + vocabulary, output);
            if (c->draft) draft_process(c, start, count);
        }
        check_abort(c->abort, c->user); return accepted;
    } catch (const std::exception & e) { if (c) c->poisoned = true; last_error = e.what(); return -1; }
}
int flo_dflash_step(flo_context * c, float * logits, int32_t vocabulary, int32_t * output, int32_t maximum) {
    try {
        require(c && c->draft && !c->poisoned && logits && output && vocabulary == c->vocabulary &&
            maximum >= 1 && maximum <= 4 && maximum <= c->capacity - c->position, "Invalid DFlash benchmark step");
        check_abort(c->abort, c->user);
        auto & d = *c->draft;
        const int anchor = flo_argmax(logits, vocabulary);
        flo_batch_owner noise(maximum, 0, d.mrope); flo_positions(noise.value, c->position, d.mrope, true);
        noise.value.token[0] = anchor;
        for (int i = 1; i < maximum; ++i) noise.value.token[i] = 248077;
        require(llama_decode(d.context, noise.value) == 0, "Draft proposal forward failed");
        std::vector<int32_t> block(maximum); block[0] = anchor;
        for (int i = 1; i < maximum; ++i) block[i] = flo_argmax(llama_get_logits_ith(d.context, i), vocabulary);
        const int accepted = flo_verify_greedy(c, block.data(), maximum, logits, vocabulary);
        require(accepted > 0, "DFlash verification failed");
        std::copy(block.begin(), block.begin() + accepted, output);
        ++c->draft_blocks; c->draft_proposals += maximum - 1; c->draft_accepted += accepted - 1;
        return accepted;
    } catch (const std::exception & e) { if (c) c->poisoned = true; last_error = e.what(); return -1; }
}
