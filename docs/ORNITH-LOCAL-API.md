# Local Ornith API and generation controls

Built and privately delivered as TestFlight **0.1.0 (125)**, runtime source
`fdb7801742a3de0015e3047a055f555db41d9b98`. Integrated host, archive/signing
and independent Apple availability checks pass. Physical-phone network/USB
reachability and explicit-budget answer quality remain unverified.
The installed Q4_K_M remains unchanged. One admitted native context serves
quick chat and API requests serially. No model or request goes to OpenAI.

## From the phone

Open **Ornith → API**, prepare the installed model and select **Starta lokalt
API**. Copy the displayed **Base URL** and local API key to your client. The
server creates a new random key each start, requires `Authorization: Bearer`,
does not persist the key and does not record API conversations. An authenticated
`GET /v1/models` lists `ornith-1.5-35b-a3b`; `GET /health` checks the listener.
The key is separate from an OpenAI account or paid API key.

Use the same Wi-Fi or a supported USB-C Ethernet adapter. The listener accepts
private/link-local IPv4 peers and prohibits cellular interfaces. Both Wi-Fi and
wired addresses can be displayed. A plain USB cable requires an additional
networking or TCP-forwarding setup on the computer; see the experimental USB
path below. Apple lists [USB-to-Ethernet adapters as supported iPhone
USB-C devices](https://support.apple.com/en-us/105099).

Keep the app and Ornith page open in the foreground. **Behåll skärmen vaken**
prevents idle locking while the server runs. Closing the page, backgrounding,
locking or memory pressure stops the listener and cancels active inference.
This is a foreground service, not an iOS background daemon. The protocol uses
HTTP on a trusted local network, without TLS or browser CORS access.

## Generation controls

Quick chat's **Generering** panel accepts numeric values. Valid settings persist
locally. The API snapshots those defaults at start; individual requests can
override them. Stop the API to edit its default settings.
An inactive saved thinking budget does not constrain a non-thinking request.

| Control / API parameter | Default | Meaning |
| --- | --- | --- |
| Max token / `max_tokens` or `max_completion_tokens` | 512 | Total new tokens, including reasoning and budget-closing tokens; 1–2048 |
| Tänkande / `enable_thinking` | true | Uses the model's canonical thinking or non-thinking chat template |
| Tänkbudget / `thinking_budget` | no separate limit | Optional cap on sampled reasoning tokens; 0 disables thinking |
| Temperatur / `temperature` | 0 | Greedy at 0; seeded sampling above 0, up to 2 |
| Top-p / `top_p` | 1 | Nucleus sampling, greater than 0 and at most 1; applies above temperature 0 |
| Slumpfrö / `seed` | 42 | Per-request deterministic random seed, 0–2147483647 |

An explicit reasoning cap feeds the canonical `</think>\n\n` token sequence
through the original model, then generates its answer. The inserted IDs count
toward the total/context limits and are disclosed in diagnostics/usage. They
do not substitute hidden text or alter top-eight routing. Leave at least nine
tokens between reasoning and total limits. A shorter reasoning budget can
reduce answer quality; no separate cap is enabled by default. Default greedy
sampling and unbounded-within-total thinking retain the previous policy.

Context capacity remains 4096, including prompt and total new-token allowance.
Prompts that do not fit are rejected rather than silently truncated. This
release does not establish 8K, 32K or 65K operation.

## Standard Python client

With the OpenAI Python client installed on the **computer**, configure only
the phone's local address and local key:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["FLASH_LOCAL_BASE_URL"],  # e.g. http://192.168.1.25:8080/v1
    api_key=os.environ["FLASH_LOCAL_API_KEY"],
    timeout=1800,
)
for chunk in client.chat.completions.create(
    model="ornith-1.5-35b-a3b",
    messages=[{"role": "user", "content": "Skriv en Python-funktion som testar primtal."}],
    max_tokens=512,
    temperature=0,
    stream=True,
    extra_body={"enable_thinking": True, "thinking_budget": 128},
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

`POST /v1/chat/completions` also supports nonstreaming JSON and
`stream_options: {"include_usage": true}`. Stream chunks keep one completion
ID, send separate `delta.content` and optional `delta.reasoning_content`, then
finish reason and `data: [DONE]`. `reasoning_content`, `thinking_budget` and
`forced_thinking_end_tokens` are local extensions. Base text and streaming
formats follow the [official Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [streaming event schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).

Only one inference request is admitted at a time; other requests receive 429
and `Retry-After`. A disconnected client cancels its inference owner; the
listener remains available and recreates the context after native teardown,
reusing only the existing in-process numerical admission. No simultaneous
model copy is allocated. Request bodies are limited to 256 KiB, headers to
16 KiB and live connections to four. Slow or oversized clients fail boundedly.

This initial text API explicitly rejects structured tools/tool-call history,
images, custom stop sequences, nonzero penalties and JSON-constrained output.
It does not implement `/v1/responses`. A client requiring those features needs
further integration; generic text compatibility is not a verified Hermes
agent workflow.

## Validation boundary

Host controls exercise real loopback TCP, authentication, HTTP framing,
streamed/nonstreamed JSON, reasoning separation, busy responses, cancellation,
seeded sampling and canonical closing-token decoding. These controls contain
no trained inference. Phone-side checks still govern native model admission;
end-to-end iPhone LAN/adapter, explicit-budget answer quality and sustained
server workloads require physical measurements. The accepted build-123 phone
comparison remains the performance checkpoint; adding an API is not evidence
of higher decode throughput.

## First physical acceptance check

1. Update the installed app without uninstalling or downloading weights again.
   Open **Ornith → Snabbchatt**, activate the model and expand **Generering**.
   Set total tokens to 512 and the optional thinking budget to 128. Send a new
   prompt and check that reasoning finishes before the visible answer continues.
   Export its measurement locally; it records the selected settings, sampled
   reasoning count and any forced closing tokens. Compare quality with the cap
   disabled before choosing a permanent cap.
2. Open **API**, start the service and allow iOS local-network access if asked.
   Keep this page open. On the same Wi-Fi/Ethernet network, configure the client
   with the displayed base URL, model ID and newly generated local key. Start
   with authenticated `/v1/models`, then the streaming example above. Do not
   paste the key into an issue or measurement report.
3. Stop or disconnect an active client, then issue another request. The old
   inference must stop and the service must admit the next request after native
   teardown. A concurrent generation receives 429. Background the phone app and
   confirm that the service stops; return and start it again with its new key.

These checks establish phone integration only. Use **Jämförelse** for the fixed
128-token engine benchmark; a short API reply is not a comparable speed run.

## Experimental direct USB path

Direct cable access may use a computer-side TCP forward rather than Ethernet.
The [libusbmuxd implementation](https://github.com/libimobiledevice/libusbmuxd/blob/master/tools/iproxy.c)
supports USB-only forwarding and an explicit loopback listener:

```sh
iproxy -l -s 127.0.0.1 18080:8080
```

With the phone API running on port 8080, the computer client would then use
`http://127.0.0.1:18080/v1` and the same phone-generated key. Keep the forwarding
process running; do not expose its local port on a public interface. With
multiple phones, select the intended device using `iproxy -u DEVICE_UDID`.

The library documents [AppleMobileDeviceSupport/usbmuxd as a Windows prerequisite](https://github.com/libimobiledevice/libusbmuxd#prerequisites).
The phone also needs an accepted [Trust This Computer pairing](https://support.apple.com/en-us/109054).
The current development PC has no discovered forwarding utility or Apple mobile
device service. No driver installation, cable pairing, physical USB connection
or end-to-end USB inference has been verified. The app accepts loopback TCP,
but that host test alone does not establish the actual USB peer/path behavior.
This is a reproducible candidate to test, not delivered plug-and-play USB support.
