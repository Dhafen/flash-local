# Connect agent clients to the iPhone

Python 3.11+ is required. The phone performs all model inference; the computer
runs the selected client and its tools. No weights are downloaded to the
computer. Start Flash Local's API on trusted Wi-Fi and keep the app foreground.
Use the actual address/key shown in the app. The launcher requires the new
authenticated `flash_local` capabilities in `/v1/models`; older builds need an
update. It never invents the available context window.

From this repository on Kali/macOS:

```sh
./scripts/flashlocal --base-url http://192.168.1.12:8080/v1 connect
./scripts/flashlocal status
./scripts/flashlocal run opencode --project /path/to/authorized/project
./scripts/flashlocal run codex --project /path/to/authorized/project
./scripts/flashlocal run hermes --project /path/to/authorized/project
./scripts/flashlocal run pentagi --project /path/to/existing/pentagi
```

On Windows use `./scripts/flashlocal.ps1` with the same arguments. Without an
environment key, the launcher asks for it privately. Alternatively supply
`FLASH_LOCAL_API_KEY` through your password manager/environment. The saved
connection contains the URL and model ID only. Scripts never install clients,
pull images, build containers or silently switch inference providers.

## Client differences and current verification

| Client | Configuration | Verified / limitation |
| --- | --- | --- |
| OpenCode | Explicit custom OpenAI-compatible provider, phone main/helper model, live limits | Configuration and owned loopback tests. Actual OpenCode/phone session still required. Default tool permissions ask; shell/web fetch/external directories denied. |
| Codex CLI | Session-only custom provider via authenticated loopback Responses adapter | Installed Codex 0.160.0 completed text and a real harmless echo-tool round trip with synthetic upstream replies on Windows. This proves transport, not Ornith tool quality. Read-only sandbox, on-request approvals, hosted search/apps/plugins/delegation disabled. Configured MCP servers disabled per invocation. |
| Hermes | New private `HERMES_HOME` profile; phone model and auxiliary tasks, no fallback chain | Installed Hermes rejected a real 16K window before inference: current minimum is 64,000 tokens. Launcher explicitly stops below this floor. Do not fake 64K to bypass it. 65K phone quality/admission still unverified. |
| PentAGI | New Compose override/provider file for an existing installation; all 13 model roles use the phone | Configuration tests only; Docker/PentAGI unavailable on this development host. No full PentAGI/phone success claim. `ASK_USER=true`; select its custom/local provider. Structured JSON is prompt/parser based, not guaranteed constrained generation. |

Hermes profiles retain their session directories and contain no API key.
PentAGI's generated files reference an environment key; resolved Compose output
is validated privately. Docker necessarily makes container environment visible
to the local Docker administrator. Existing Compose data volumes are retained.
Concurrent PentAGI inference may receive the phone's explicit busy response;
no unbounded inference queue or hidden cloud fallback is added.

Codex uses its existing user configuration plus explicit session overrides;
the launcher does not rewrite it or reuse the app's running daemon. Existing
client versions/configuration may impose additional requirements. Unknown MCP
server names that cannot be disabled safely stop launch. The adapter supports
stateless text, function calls/results and custom text tool input/results.
Text streams as it arrives; tool arguments are emitted only after the phone
finishes validating them. It rejects images, hosted/namespace tools, stored
response IDs, opaque reasoning history, compaction and JSON-schema generation.
It does not execute tools. Disconnect closes the phone request; busy returns
429. Its ephemeral key is distinct from the phone key and it binds loopback only.

Client permissions are not rules-of-engagement enforcement. For bounded
authorized network probes, use [the scoped client](ORNITH-KALI-AGENT.md).
Do not assume third-party shell tools inherit that client's scope validator.

## Context and quality acceptance

8K remains the default. In **Promptinläsning och batch**, 16K/32K/65K are
explicit experimental profiles. Reactivation performs the existing memory
admission including native KV/state/workspace. The active profile, not a
pending selection, is advertised to clients. No weights, routing, RoPE or KV
precision are altered. Successful allocation is not a long-context quality test.

Run a quick real-phone protocol/logic/tool check first:

```sh
./scripts/flashlocal test --quick --output quick-phone.json
```

For a full 8K baseline, choose an already-installed digest-pinned Python
container image. Generated code runs only in a networkless, read-only,
resource-limited Docker/Podman container; no host execution or automatic pulls.

```sh
./scripts/flashlocal test --full --expected-context 8192 \
  --code-runner docker --code-image python@sha256:YOUR_INSTALLED_IMAGE_DIGEST \
  --output baseline-8k.json
```

Then select/reactivate 16K on the phone and run the same suite:

```sh
./scripts/flashlocal test --full --expected-context 16384 \
  --baseline baseline-8k.json --long-context --endurance \
  --code-runner docker --code-image python@sha256:YOUR_INSTALLED_IMAGE_DIGEST \
  --output candidate-16k.json
```

The deterministic bank has 40 bilingual logic tasks, 100 tool tasks (including
missing arguments and exact string escaping), ten fresh-receipt multistep
fixture tasks and 20 executable Python tasks. Passing thresholds are 34/40,
95/100 correct tools with 99/100 valid formats, 8/10 agent tasks and 16/20 code.
These are bounded fixtures, not evidence of broad security expertise.
Existing scoped-client tests separately cover approval/scope/cancellation
guards; the model harness does not certify third-party tool execution policies.

Near-full context uses the phone tokenizer's actual counts and five distributed
fresh facts, with at least 70% of the selected total window exercised. Endurance
requires at least ten minutes, at least 128 output tokens per measured run,
at least 3 decode token/s, reported peak <=5.5 GB and thermal state below critical.
Compare identical source/model SHA, reasoning mode and cases against the 8K
baseline; losing more than one previously passed logic case fails promotion.
Missing native measurements or container execution are blocking, not a pass.
Reports checkpoint after each case and never overwrite an existing output file.

Long prefill can take a long time. The API's bounded authenticated-request
deadline scales with the admitted context, up to three hours for the largest
experiment. No client or allocation test establishes 16K/32K/65K reliability
without the actual phone report. No new phone benchmark has been fabricated.

Implementation references: [OpenCode providers](https://opencode.ai/docs/providers/),
[Hermes configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration),
[Codex configuration](https://developers.openai.com/codex/config-reference/),
[Responses events](https://developers.openai.com/api/reference/resources/responses/streaming-events),
[PentAGI provider](https://github.com/vxcontrol/pentagi/blob/main/examples/configs/custom-openai.provider.yml).
