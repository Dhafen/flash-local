// Flint graph derived from ggml-org/llama.cpp a37cc46104eb50900d5ab4a9ba02a0abdac039db (MIT).
// Bounded expert staging is adapted from tc-mb/llama.cpp-omni 96bbf617.
// See ../PROVENANCE.md. No routing, quantization or RoPE changes.
#include "models.h"
#include "llama-moe-experts.h"
#include <map>
#include <tuple>

void llama_model_flint::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.flint_rope_layers, hparams.n_layer);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    if (hparams.n_layer_dense_lead >= hparams.n_layer || hparams.n_expert == 0 ||
        hparams.n_expert_used == 0 || hparams.n_expert_used > hparams.n_expert ||
        hparams.n_swa < 2 || hparams.n_ff_exp == 0 || hparams.n_ff_shexp == 0) {
        throw std::runtime_error("Invalid Flint attention/expert geometry");
    }
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        if (hparams.swa_layers[i] > 1 || hparams.flint_rope_layers[i] > 1) {
            throw std::runtime_error("Invalid Flint per-layer attention/RoPE pattern");
        }
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_flint::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp        = hparams.n_ff_exp;

    // One bounded staging bank per (projection, type, expert stride), shared
    // across all MoE layers. Never alias different quantization strides.
    // This first port intentionally has no lookahead or per-layer cache.
    if (moe_slot_bank_enabled() && moe_decode_window() != 0) {
        throw std::runtime_error("Flint initially requires shared expert staging (decode window 0)");
    }
    using bank_key = std::tuple<std::string, int, size_t>;
    std::map<bank_key, ggml_tensor *> banks;
    auto create_routed = [&](const LLM_TN_IMPL & name,
                             std::initializer_list<int64_t> dims, const char * family) {
        if (!moe_slot_bank_enabled()) return create_tensor(name, dims, 0);
        const auto * entry = moe_experts_entry(name.str().c_str());
        if (!entry || entry->bytes_per_expert == 0) {
            throw std::runtime_error("Missing Flint expert manifest entry: " + name.str());
        }
        ggml_tensor * tensor = create_tensor_own_alloc(ml, name, dims, TENSOR_NOT_REQUIRED);
        if (!tensor) tensor = create_expert_tensor_from_manifest(ml, name, dims, entry->type);
        if (!tensor || ggml_nbytes(tensor) != entry->bytes_per_expert * size_t(n_expert) ||
            tensor->type != entry->type || moe_slot_bank_size() < n_expert_used) {
            throw std::runtime_error("Flint expert type/shape/slot capacity mismatch: " + name.str());
        }
        ml.virtualize_expert_tensor(tensor, n_expert, moe_slot_bank_size(),
                                    entry->type, entry->bytes_per_expert);
        bank_key key{family, int(entry->type), entry->bytes_per_expert};
        auto existing = banks.find(key);
        if (existing != banks.end()) {
            ml.alias_expert_slot_tensor(tensor, existing->second);
            return tensor;
        }
        banks.emplace(key, tensor);
        const std::string original = name.str();
        ggml_set_name(tensor, (original + ".base").c_str());
        return ml.window_expert_slot_tensor(original.c_str(), tensor, 0,
                                            moe_slot_bank_size(), entry->bytes_per_expert);
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_routed(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, "gate");
            layer.ffn_down_exps = create_routed(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, "down");
            layer.ffn_up_exps   = create_routed(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, "up");

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        hparams.n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_flint::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_flint::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            if (hparams.flint_rope_layers[il] != 0) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, nullptr, nullptr,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Intermediate prompt chunks need attention KV, not vocabulary logits.
    // Keep the whole layer dependency chain; only omit the output projection.
    if (params.n_outputs == 0) {
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
    // This older runtime requires each model graph to expand its own output.
    ggml_build_forward_expand(gf, cur);
}
