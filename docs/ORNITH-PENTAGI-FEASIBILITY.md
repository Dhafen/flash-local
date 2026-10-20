# Ornith iPhone → PentAGI: feasibility and model decision (2026-10-20)

## Decision
**Not ready for PentAGI.** Keep Ornith as an experimental baseline, not the sole PentAGI provider. Prefer **Qwen3.5-9B** as the first alternative to benchmark for function calling and context; **Qwen3-4B-Instruct-2507** as a lower-memory fallback. No speed or 65K claims are made for either on this iPhone until real-device tests pass.

## PentAGI requirements (verified)
- PentAGI custom OpenAI provider: `LLM_SERVER_URL`, `LLM_SERVER_KEY`, `LLM_SERVER_MODEL`, optional per-agent provider config.
- PentAGI uses function calling and multi-step agent flows; docs recommend stronger models for primary/pentester/adviser and lighter models for searcher/summarization.
- A PentAGI README documents approximately 64K typical context and 110K recommended margin for Ollama deployment. This is a workload guideline, not proof every request requires 110K.
- Sources: https://pentagi.com/docs/configuration/llm-providers/ and https://github.com/vxcontrol/pentagi/blob/main/backend/docs/config.md and https://github.com/vxcontrol/pentagi

## Actual Ornith repo status
- `OrnithLESChatGeneration.contextTokens = 8192`; its memory plan also requests 8192. Long-context correctness and real-device performance are not yet verified.
- `LocalGenerationOptions` now permits up to 4096 generated tokens, which *reduces the prompt budget* to at most 4096 at that maximum.
- `LocalChatProtocol` explicitly rejects nonempty `tools`, `tool_calls`, `tool_choice` other than none, and non-text response formats. No PentAGI tool-call compatibility.
- iOS local API has bearer authentication but uses HTTP over LAN, not TLS. Never expose publicly or transmit sensitive scope information on untrusted networks.
- Existing benchmark on iPhone (build 145, 4096 context) showed `omniPageCache` about 4.33 generated tokens/s warm, with ~0.51 GB *process* footprint; OS page cache excluded and long-duration performance unknown. Full-model numerics verification was false. This is **not** an 8K, 65K or PentAGI result.
- No demonstrated MTP speedup for Ornith; do not implement speculative decoding without compatible head/draft model and measured acceptance.

## Candidate comparison
| Candidate | Evidence for PentAGI suitability | Constraints | Verdict |
| --- | --- | --- | --- |
| Ornith 1.5 35B-A3B Q4_K_M (existing) | 4K short-chat measured ~4.33 tok/s; iPhone page-cache engine available | 8K unverified; API rejects tool calls; large model on flash with I/O cost; full numerics not verified | Keep as baseline only |
| Qwen3.5-9B | Official model card explicitly documents tool-call parser, agentic usage, and native 262144 context; official MTP serving example | Model-card context is not an iPhone RAM guarantee; different architecture needs a new supported inference backend; quantization impacts quality | **First alternative to prototype**, not proven faster on iPhone |
| Qwen3-4B-Instruct-2507 | Official native 262144 context and tool use; smaller weight footprint; non-thinking can avoid reasoning token overhead | Lower capacity for complex pentest reasoning; still needs real iOS backend and long-context testing | Fallback for memory/speed |

Official model sources:
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja

## Engineering acceptance gates
1. **Model support**: model weights/tokenizer/architecture work with a *real* iOS inference backend under 5.5 GB process ceiling, including OS cache pressure. No model-name-only substitution.
2. **API contract**: OpenAI chat-completions tool schemas, tool_choice, assistant tool_calls, tool-role replies, correct call IDs, multi-turn, streaming deltas, stop reasons and token usage; reject malformed tool calls. Tests using captured *synthetic* PentAGI request fixtures.
3. **Context**: test 8K, 16K, 32K, 65K on device, measure effective prompt + tool-schema + history + output budget, quality/retrieval, KV size, peak memory and throttling. Do not claim 65K until passing.
4. **Speed**: cold/warm prefill, 128+ decode tokens, sustained 20-turn tool loops, first-token latency, tok/s, I/O bytes, temperature and process/OS memory pressure. Compare same tasks and prompts.
5. **Reliability**: cancellation, timeout, API reconnect, app foreground and phone sleep, prompt injection in tool results, no hallucinated tool success; ROE enforced by PentAGI independently of model.
6. **Security**: isolate LAN, protect bearer key, add TLS or trusted encrypted tunnel for real engagements, avoid cloud fallback without authorization.

## Recommendation
Prioritize a genuine PentAGI protocol compatibility test **before** optimizing Ornith MTP. In parallel, implement an iPhone Qwen3.5-9B backend proof of concept and compare against Qwen3-4B-Instruct-2507. Select by verified *successful agent tasks per minute* and bounded memory, not raw tokens/s or parameter count alone.

**Status:** research and acceptance plan only; neither alternative is integrated or benchmarked on the user's iPhone. The custom Kali TCP-probe prototype is not a substitute for PentAGI.
