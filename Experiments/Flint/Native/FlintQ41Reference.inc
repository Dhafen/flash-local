// CPU-only Flint diagnostic reference for the original GGML Q4_1 bytes.
// GGML's ordinary CPU Q4_1 path quantizes F32 activations to Q8_1, whereas
// Metal's vector path consumes F32. Keep the ordinary trait as the default.
// The exclusive Flint owner may select this reference while comparing full
// logits; it must reset the switch after destroying that context.
static atomic_int flint_q4_reference_enabled = 0;

void ggml_flint_set_q4_reference(bool enabled);
void ggml_flint_set_q4_reference(bool enabled) {
    atomic_store_explicit(&flint_q4_reference_enabled, enabled ? 1 : 0, memory_order_release);
}

static void flint_q4_1_f32_reference(int n, float * GGML_RESTRICT out, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by);
    GGML_ASSERT(n >= 0 && n % 32 == 0 && nrc == 1);
    const uint8_t * packed = (const uint8_t *) vx;
    const float * input = (const float *) vy;
    double sum = 0.0;
    for (int block = 0; block < n / 32; ++block) {
        const uint8_t * b = packed + 20 * block;
        const uint16_t dbits = (uint16_t) b[0] | ((uint16_t) b[1] << 8);
        const uint16_t mbits = (uint16_t) b[2] | ((uint16_t) b[3] << 8);
        const float scale = ggml_fp16_to_fp32(dbits);
        const float minimum = ggml_fp16_to_fp32(mbits);
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = b[4 + j];
            const float low = (q & 15) * scale + minimum;
            const float high = (q >> 4) * scale + minimum;
            sum += (double) low * (double) input[block * 32 + j];
            sum += (double) high * (double) input[block * 32 + j + 16];
        }
    }
    *out = (float) sum;
}

static const struct ggml_type_traits_cpu flint_q4_1_f32_traits = {
    .from_float = quantize_row_q4_1,
    .vec_dot = flint_q4_1_f32_reference,
    .vec_dot_type = GGML_TYPE_F32,
    .nrows = 1,
};
