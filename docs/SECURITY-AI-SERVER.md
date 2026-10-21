# Flash Local security-agent server

Development is isolated on `codex/security-ai-server`, based on `95ac8b0`.
The installed Ornith Q4_K_M, REAM, LES, QPack and conversations are retained.
No inference mathematics, model quantization, main CI, signing or TestFlight
configuration is changed by this integration. Host controls and physical
acceptance are separate gates; this document is not a delivery receipt.

Test-only capacity workaround: runs [39465263876](https://github.com/Dhafen/flash-local/actions/runs/39465263876)
and [39465270366](https://github.com/Dhafen/flash-local/actions/runs/39465270366)
failed before executing any step. GitHub annotations report that hosted macOS
arm64 runners were not acquired. The isolated API controls now have an explicit
Intel fallback with the same Xcode/source/tests; a separate Linux job validates
the portable original-template oracle and Python client. Neither bypasses
Apple-silicon LES/QPack, integrated app, visual or archive gates. Only standard
runners in this public repository are used.

Delivery run [179](https://github.com/Dhafen/flash-local/actions/runs/39471916861)
stopped in the Swift agent-template parity test: its oracle rendered tool
definitions in client insertion order, while the adapter already emits sorted
JSON keys. Jinja preserves that order, so the rendered bytes/token IDs differed.
The fixture now supplies the same ordered tools to the independent original
template and explicitly checks the adapter's serialized bytes. Exact rendered
text and token-ID assertions remain for all four cases and both thinking modes;
the client fixture keeps its different key order. Runtime code, model mathematics
and delivery configuration are unchanged. The repaired [delivery 180](https://github.com/Dhafen/flash-local/actions/runs/39481746775)
passes all full-app gates and independent Apple availability verification.

## Delivered checkpoint: 0.1.0 (185)

[Build 185](https://github.com/Dhafen/flash-local/actions/runs/39574507501),
app source `ebd97a99665d6b6c5affd7324754c94ff7a5e990`, passes LES/QPack,
585 app, 488 package (nine environment skips), 321 streaming and 18 reference
controls, including 23 actual TCP/Keychain API functions and 38 native Swift
controls. All 64 actual host/Simulator captures were reviewed in both
appearances. Narrow macOS control previews elide some explanatory copy;
iPhone controls/help are visible. Archive, signature and source-bound upload
pass. Independent Apple verification confirms `VALID`, `INTERNAL_ONLY`,
`IN_BETA_TESTING`, nonexpired, with the unchanged private group and sole owner.
IPA SHA256: `dfe16c0a2a3eef8ffe452379120461bd05bea2eb9d1ee7f473f0780c5825bda5`.

Client tools at `fe0a106` pass 125 Python controls. Installed OpenCode 1.19.3
and Codex 0.160.0 complete text and actual owned file/echo-tool round trips
against explicitly synthetic upstream replies. No trained model was run on
the computer. These are not real phone LAN/tool-quality or long-context scores.
16K/32K/65K remain experimental; 8K is the default. No new phone performance,
full-model numerics or three-model benchmark is claimed.

### Historical baseline: 0.1.0 (180)

Source `e8d25318510262568b3b81fbc8100e36751ee383` passes 585 app controls,
487 package controls (nine environment skips), 321 streaming controls,
18 reference-adapter controls and 38 selected native/chat/tokenizer controls.
All 22 actual TCP/Keychain API controls pass with a synthetic generation
provider. LES/QPack and the iOS build pass; 64 real host/Simulator captures were
reviewed in both appearances. The original-template token comparison now passes.
These are host controls, not trained-model phone/tool or long-context evidence.

Archive, Apple signature and private upload match that exact source. A fresh
read-only Apple audit confirms `VALID`, `INTERNAL_ONLY`, `IN_BETA_TESTING`,
nonexpired, the unchanged private group and sole owner. The IPA SHA256 is
`7241567009bc3eebec3ad98c8dc860b275c0bdc281d63ce3568520816d10ac8e`.
The user reports installing the update. A short phone screenshot reports
5.54 decode token/s, 6.50 prompt token/s and 145 reused prompt tokens; it does
not establish sustained throughput or actual use of the full 8K context.

The delivered display correction reuses the existing assistant Markdown
renderer in the shared GGUF/LES message row. Bold text, lists and code blocks
render while the raw model response and API payload stay intact. User messages
remain literal. Its reviewed preview contains only public synthetic sample text.
Model-generated spelling or factual
errors are separate from Markdown formatting and are not silently rewritten.

## Protocol and execution boundary

The context/client implementation is documented in
[Phone agent launcher](PHONE-AGENT-LAUNCHER.md): live session capacities,
experimental context profiles, acceptance harness and launch configurations
for OpenCode, Codex, Hermes and PentAGI. Its Swift/iOS checks pass; physical
phone acceptance remains separate from the verified delivery checkpoint above.

The foreground server implements authenticated `/v1/models`,
`/v1/chat/completions`, JSON and SSE. Function definitions, `tool_choice`
(`auto`, `none`, `required`, named), `parallel_tool_calls`, assistant
`tool_calls` and matching `role: tool` results are supported. The installed
Ornith template's XML envelopes become OpenAI function calls with JSON-string
arguments. History arguments become mappings only for the original Jinja
renderer. The tokenizer, weights, routing and graph are unchanged.

Arguments must match a bounded JSON Schema subset. Local references, common
unions, exact JSON types, required fields, enums and size/numeric bounds are
checked. Unsupported validation keywords fail with 400. This is validation of
completed output, not grammar-constrained sampling or a guarantee that the
model always chooses a correct tool. Undeclared, malformed, incomplete or
token-limited calls fail with 502 and emit no executable call. Tool-enabled
streams withhold answer/call payloads until validation; reasoning still
streams separately. Results must resolve each outstanding call ID once.
Structural tokenizer control strings in message/tool data are rejected.

The phone never executes tools. `scripts/flash_local_agent.py` is an optional
Kali-compatible standard-library client. Other OpenAI-compatible frameworks
can supply their own function schemas. `/v1/responses`, image input and
general constrained `response_format` are not implemented.

The key is configurable and stored in the device-local Keychain. A blank
configuration field reuses the saved key; rotating it requires stopping the
server. HTTP is intended for the trusted LAN. iOS foreground/locking,
IPv4/Wi-Fi/Ethernet and USB forwarding limits remain as documented in
[the API guide](ORNITH-LOCAL-API.md). The same admitted inference context serves
chat and API serially. Disconnect cancels the owner before context recovery.
OpenAI `reasoning_effort` maps to the four local modes. Inherited thinking
budgets fit shorter client-supplied totals; explicit incompatible budgets fail.

## First phone acceptance test

After a source-matched build passes and becomes available in TestFlight,
update without uninstalling. Activate Ornith, open **API**, start the server
and keep the app in the foreground. On Kali, enter the displayed key privately:

```sh
read -r -s -p 'Local API key: ' FLASH_LOCAL_API_KEY
export FLASH_LOCAL_API_KEY
printf '\n'
python3 scripts/flash_local_verify.py \
  --base-url http://IPHONE_LAN_IP:8080/v1 --output phone-acceptance.json --cancellation
```

This performs authenticated model discovery, a wrong-key rejection and real
JSON/SSE function-call round trips. The computer adds 50+5 and returns a new
random receipt after the model calls its function; the model must read that
receipt from the result. No target is scanned. Missing native measurements
remain missing. Server-reported hardware is distinguished from independent
physical verification. `--cancellation` closes a reserved live stream and
checks that a subsequent inference request is accepted after bounded recovery;
it does not claim independent observation of GPU teardown.

Add `--benchmark` for six real runs: Off/Small/Medium, twice each with the same
prompt and model. Records include actual token counts, native prefill/decode,
I/O, peak footprint, thermal state, exact model SHA and app/source identity.
Runs below 128 output tokens are flagged. These are three configurations of
one model, not the required comparison of three models. OS-cache coldness is
not controlled and thermal state is not a temperature in degrees.

## Scoped Kali workflow

Create a private scope file for an authorized lab or engagement. Replace the
documentation-only address with an explicitly authorized target:

```json
{"authorized":true,"authorization_id":"your-ROE-reference","ips":["192.0.2.10"],"ports":[80,443],"allowed_tools":["tcp_connect","http_headers"],"allowed_http_paths":["/"]}
```

An optional `expires_at` is an ISO-8601 timestamp with a timezone. Validate
without making any API or target connection:

```sh
python3 scripts/flash_local_agent.py --scope scope.json --log dry.jsonl --dry-run
```

Then run with the actual phone address:

```sh
python3 scripts/flash_local_agent.py --scope scope.json \
  --base-url http://IPHONE_LAN_IP:8080/v1 --log assessment.jsonl
```

Each tool needs approval by default. `--approval scope` explicitly admits
the file's limited read-only tools/targets; `--approval deny` executes none.
Only bounded TCP connect and HTTP HEAD are implemented. No shell, arbitrary
command, exploit, credentials, redirects or target discovery is supplied.
Scope is independently checked after model output and again after approval.
HTTP HEAD is restricted to the exact `allowed_http_paths` list (root only
when omitted), including the query string; paths are not discovered or expanded.
Steps and duration are bounded, repeated calls are blocked, Ctrl-C stops the
loop, complete tool groups are compacted together, and a new local JSONL log
records proposals, observations, usage and outcome without the API key.
The old text-only prototype remains available separately.

## Candidate models and remaining gates

The first comparison set is the existing **Ornith 35B-A3B Q4_K_M**, a
**Qwen3.5-4B Q4** candidate and a **Qwen3.5-2B Q4** candidate. Dense candidates
may avoid expert I/O if their complete working set fits the measured budget;
that is an engineering hypothesis, not a phone benchmark. Their exact GGUF
artifacts, template/quantization validation and production runner admission
must precede inference. General model import and these additional runners
are not yet implemented. Do not download replacement weights for this branch.

Metadata inspected on 2026-10-20, without downloading any tensor payload:

| Candidate artifact | Exact publisher file bytes | Inspected revision |
| --- | ---: | --- |
| Ornith Q4_K_M (installed) | 21,713,463,040 | `12393612fd4f730ff5aadc23e9b8f9648aa49ceb` |
| [Unsloth Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523) | 2,740,937,888 | `e87f176479d0855a907a41277aca2f8ee7a09523` |
| [Unsloth Qwen3.5-2B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | 1,280,835,840 | `f6d5376be1edb4d416d56da11e5397a961aca8ae` |

Candidate LFS SHA-256 values are respectively
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
and `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.
These are repository declarations, not locally verified complete artifacts.
Original Qwen config revisions inspected: 4B
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 2B
`15852e8c16360a2fea060d615a32b45270f8a8fc`. They describe dense hybrid
graphs with 32/24 layers and 8/6 full-attention layers, respectively. They
cannot enter the existing MoE-only wrapper by changing a name.

The [4B publisher card](https://huggingface.co/Qwen/Qwen3.5-4B) and
[2B publisher card](https://huggingface.co/Qwen/Qwen3.5-2B) document tool use;
the 2B publisher also warns about thinking loops. Evaluate bounded modes and
completion reliability explicitly. [Ornith's publisher](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
documents the XML tool parser. [Liquid's 1.2B card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
supports agent/tool tasks but advises against programming and knowledge-heavy
work, so it is not the primary coding candidate. Publisher scores are not
measurements of these phone quantizations.

Current chat capacity is configured to 8,192 total tokens with dynamic native
memory admission. Actual 8K reliability, 16K/32K/65K operation, full trained
numerical validation, cross-model quality/speed, sustained phone LAN sessions
and real Kali interoperability remain required physical milestones. The
last supplied controlled phone performance remains the build-123 evidence;
the server integration itself establishes no new token/s result.
