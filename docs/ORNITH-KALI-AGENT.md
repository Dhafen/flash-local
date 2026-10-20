# Ornith Kali assessment agent (early prototype)

This **bounded prototype** uses the iPhone Ornith OpenAI-compatible text API
as a planning engine and executes only scoped TCP connection probes on Kali.
It is **not** Pentagi, does not implement structured OpenAI tool calls,
and cannot autonomously exploit vulnerabilities or run arbitrary commands.

## Run

1. Start the Ornith API in the iPhone app. Keep the phone awake and connected
   to the same trusted local network as Kali.
2. Create a JSON scope file containing only IPs and TCP ports you are
   explicitly authorized to assess. Example (documentation-only TEST-NET IP):

```json
{"authorized": true, "ips": ["192.0.2.10"], "ports": [80, 443]}
```

3. On Kali, first validate the scope without any network probes:

```sh
python3 scripts/ornith_kali_agent.py --scope scope.json --report dry-run.json --dry-run
```

4. For a real approved assessment, use your iPhone's displayed LAN API URL
   (including /v1), **not** the default 127.0.0.1 endpoint. Export the API
   bearer key in your shell without committing or logging it:

```sh
export ORNITH_API_KEY='YOUR_PRIVATE_API_KEY'
python3 scripts/ornith_kali_agent.py --scope scope.json \
  --endpoint http://IPHONE_LAN_IP:8080/v1 --report authorized-run.json --max-steps 8
```

The agent supports only `tcp_connect` and `finish` actions, a maximum of
12 steps, no repeated target probes, a 2-second connection timeout, and a
half-second delay between probes. Scope is checked *after* model output and
*before* any socket connection. Reports contain only tested endpoints and
connection outcomes; an open port is not a verified vulnerability.

**Security caveats:** The current iPhone API is HTTP with bearer authentication,
not encrypted TLS. Use only a trusted, isolated LAN and avoid sending secrets,
credentials, or private target data over untrusted Wi-Fi. The agent is not
ready for unsupervised production bug bounty use. Add per-program ROE policy,
explicit approval for intrusive actions, robust tool-call formatting,
recovery, audit logs, and end-to-end device tests before expanding scope.

Run local unit tests with:
```sh
python3 -m unittest discover -s scripts/tests -p 'test_ornith_kali_agent.py' -v
```
