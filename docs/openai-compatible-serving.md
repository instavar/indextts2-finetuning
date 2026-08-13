# Experimental OpenAI-Compatible Speech Serving

## Scope

`tools/openai_speech_server.py` is a dependency-light reference server for one
operator-selected IndexTTS2 checkpoint and one reviewed speaker prompt. It is a
runtime bridge for local evaluation and controlled integration testing. It is
not a production gateway, a multi-tenant voice service, or evidence of real
checkpoint serving quality.

The compatible surface is intentionally smaller than the full OpenAI API:

| Route | Contract |
|---|---|
| `GET /healthz` | Process is accepting HTTP requests. |
| `GET /readyz` | Startup model loading completed and the configured model and voice IDs are available. |
| `POST /v1/audio/speech` | JSON input and PCM WAV output for the one fixed model and voice. |

Accepted speech fields are `model`, `input`, `voice`, `response_format`, and
`speed`. The last two default to `wav` and `1.0`. No other response format or
speed is claimed.

## Fixed-artifact boundary

Only the operator can choose these values at startup:

- YAML config
- base model directory
- selected fine-tuned GPT checkpoint
- optional tokenizer override
- speaker reference audio
- device and precision
- seed and sampling controls
- model and voice IDs
- optional artifact-set identity and startup receipt destination

The HTTP request cannot name any path or swap any of those assets. Generated
audio is written to a server-owned temporary directory, validated as a bounded,
positive-duration PCM WAV, read into the response, and removed with the
temporary directory.

This prevents a request from turning the server into a local-file oracle or
selecting an unreviewed voice. It does not make the startup assets immutable.
For a reproducible deployment, separately bind the complete base dependency,
config, tokenizer, speaker prompt, and fine-tuned checkpoint to a reviewed
content-addressed artifact manifest.

For qualification runs, pass `--startup-receipt` together with
`--artifact-set-id` and `--artifact-set-sha256`. The server writes the receipt
with exclusive creation after model loading and exposes its byte SHA-256 from
`/readyz`. The receipt hashes the config, selected GPT checkpoint, speaker
prompt, and optional tokenizer without retaining their local paths. It does not
walk or hash every inherited dependency in `model_dir`, so the artifact-set
manifest must state that boundary explicitly.

## Concurrency and resource behavior

IndexTTS2 is treated as a mutable, non-thread-safe engine. The HTTP listener can
serve lightweight requests concurrently, but synthesis uses a nonblocking lock.
If generation is already active, a second synthesis request receives HTTP 429
instead of entering the engine or accumulating an unbounded generation queue.

The server also applies:

- a maximum JSON body size, default 16 KiB
- a maximum input length, default 4,000 characters
- a maximum generated WAV size, default 100 MiB
- a per-connection timeout, default 30 seconds
- required `Content-Length` and `application/json`
- strict rejection of duplicate or unknown fields and unsupported values
- fixed seeding immediately before each serialized generation

These are local process guards. They do not replace an ingress proxy with
connection limits, TLS, request-rate policy, observability, cancellation, and
GPU worker supervision.

## Authentication and network exposure

The default listener is `127.0.0.1`. Loopback can run without authentication for
local integration testing. Any non-loopback host requires `--api-key-env`, whose
value is read from the named environment variable. The server never accepts the
secret as a command-line value.

Literal IPv6 hosts select an IPv6 listener. Temporary model configuration is
closed before IndexTTS2 reopens it, so the startup path does not depend on Unix
open-file deletion behavior.

Bearer authentication is a minimal safety gate, not a complete production
identity system. Put a production deployment behind TLS and a reviewed gateway.
Do not expose this reference process directly to the public internet.

## Error contract

Client errors return JSON with a stable code and a bounded message. Unexpected
engine exceptions are logged server-side with a generated request ID, while the
client receives only `synthesis_failed`. This keeps checkpoint paths and model
internals out of HTTP responses. Operational logs can still contain sensitive
local details and require normal access controls and retention policy.

## Verification

Run the dependency-free contract tests on a machine without IndexTTS2 model
dependencies:

```bash
python3 -m unittest tests/openai_speech_server_test.py -v
```

The tests cover fixed model and voice selection, request-controlled path
rejection, strict field behavior, body and text limits, bearer authentication,
loopback binding, non-loopback fail-closed behavior, generated WAV validation,
temporary output cleanup, concurrency rejection, live HTTP headers, and bounded
engine errors. They also cover no-overwrite startup receipts, live receipt
binding, an exact frozen-row client, and HTTP versus CLI parity validation.

The focused qualification tools are dependency-light after the server starts:

```bash
python tools/qualify_openai_speech_runtime.py \
  --endpoint http://127.0.0.1:8000 \
  --model-id indextts2-finetuned \
  --voice-id instavar-reference \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id indextts2-step14000 \
  --sample-id indextts2-step14000--neutral-brief--seed-42 \
  --expected-startup-receipt-sha256 <sha256> \
  --output-dir evaluation/http/seed-42/neutral-brief

python tools/validate_http_cli_parity.py \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id indextts2-step14000 \
  --sample-id indextts2-step14000--neutral-brief--seed-42 \
  --cli-observations evaluation/cli/generation-observations.json \
  --http-observation evaluation/http/seed-42/neutral-brief/http-generation-observation.json \
  --startup-receipt evaluation/http/seed-42/startup-receipt.json \
  --output evaluation/http/seed-42/neutral-brief/parity.json

python tools/probe_openai_speech_runtime.py \
  --endpoint http://127.0.0.1:8000 \
  --model-id indextts2-finetuned \
  --voice-id instavar-reference \
  --input "A long request keeps synthesis active for the overlap probe." \
  --expected-startup-receipt-sha256 <sha256> \
  --include-concurrency \
  --output evaluation/http/contract-probes.json
```

The qualification client rejects instruction-bearing rows because this HTTP
subset cannot apply the CLI engine's emotion-text controls. The parity validator
requires exact WAV equality and returns nonzero when the bytes differ. Exact
equality proves only one deterministic matched row under the frozen artifacts
and settings.

Successful speech responses also return `X-Generation-Seconds` and, on CUDA,
`X-Peak-Memory-Bytes`. The server measures both inside the serialized synthesis
boundary after seed reset. The client records server generation time separately
from end-to-end request time so the evaluator does not mistake network and HTTP
overhead for model execution.

A real-runtime qualification still requires all of the following:

1. Start from a clean checkout and a content-bound artifact set.
2. Load the selected checkpoint in a fresh process on the declared device.
3. Call `/readyz`, then generate every frozen prompt and seed through HTTP.
4. Record latency, real-time factor, peak memory, invalid-output rate, and all
   evaluator objective metrics.
5. Compare HTTP outputs against the CLI runtime under matched artifacts and
   settings.
6. Complete blinded listening for speaker identity, Singapore English accent,
   pronunciation, cadence, monotony, naturalness, artifacts, and fatigue.
7. Exercise disconnects, timeouts, malformed requests, concurrent load, GPU OOM,
   and worker restart behavior.

Until that evidence exists, the runtime status remains `experimental` and its
conformance status remains `not_run`.
