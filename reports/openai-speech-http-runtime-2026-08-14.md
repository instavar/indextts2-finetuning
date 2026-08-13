# OpenAI-compatible HTTP runtime qualification, 2026-08-14

## Result

The experimental IndexTTS2 OpenAI-compatible speech subset passed a focused,
preregistered CUDA smoke qualification for the selected step-14000 full-SFT
checkpoint. Nine HTTP rows covered three non-instruction prompts and three
seeds. Every HTTP WAV was byte-identical to its matched CLI control. A fresh
seed-42 server restart reproduced the same neutral-brief WAV.

The runtime remains `experimental`. This result establishes one fixed-artifact
local HTTP bridge on one host. It does not establish production readiness,
quality improvement, public exposure safety, sustained load, cancellation, GPU
OOM recovery, multi-worker behavior, or blind-listening quality.

## Frozen inputs

- runner revision: `c712a483d525a98b2151e400e55a317927ffeee8`
- clean remote checkout:
  `/mnt/work/chee-wei-jie/voice-models/instavar-index-http-clean-20260814`
- evidence root:
  `/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260814_index_http_runtime_v4`
- protocol SHA-256:
  `d70c104129ba07ee306323614ad3446077a4cd656353f61046b2dd8c64df234d`
- generation plan SHA-256:
  `33f7ac75ed069d79a82c8bc77bb867f05a0e88876474d940598b9473d0463b72`
- focused artifact manifest SHA-256:
  `3a094e1422a7d7834b10db076f076b0ca80748ee60d8efebd07a9497f4317f60`
- evaluator revision: `2812e200233804fde685c35ea1da1cbf9fe8ef4b`
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- Torch: `2.8.0+cu128`

The focused artifact manifest binds the config, selected GPT checkpoint,
tokenizer, and retained speaker prompt. It does not enumerate every inherited
codec, acoustic, speaker, and vocoder dependency under the model directory.

## Runtime and OOD probes

All nine planned CLI rows and all nine planned HTTP rows produced valid PCM
WAVs. The ten parity reports comprise the nine planned rows plus the repeated
seed-42 restart row:

- exact HTTP versus CLI rows: 10 of 10
- parity summary SHA-256:
  `4637d0ed6f38e629113e17a0a7bc81df442f788c67dbbc35aa871b4f3e590297`
- seed-42 startup receipt SHA-256:
  `74786e85e6921ff947bc899e1d13dc1b5f1a59861c668aba4f47bc89155732df`
- seed-314159 startup receipt SHA-256:
  `02df4a0d23d664d0c845924ad1716d22686c5c4a962fb616ebd0574ec13be5d7`
- seed-20260812 startup receipt SHA-256:
  `409d12a884b0a13f65b780c4b57952380260725a0ffd99ef6d37a61d20946bfb`

The restarted seed-42 server produced the same startup-receipt digest and the
same neutral-brief WAV SHA-256,
`d53db1de94f1e195acf9451cb7476e2a49e4bca15c55fce5559900b18bbf0b39`.

The live negative and overlap probe passed all four frozen cases:

- request-controlled speaker path: HTTP 400 `unsupported_field`
- wrong model: HTTP 400 `unsupported_model`
- unsupported MP3 format: HTTP 400 `unsupported_response_format`
- overlapping long-form synthesis: one HTTP 200 and one HTTP 429 `server_busy`

The successful overlapping request reproduced the matched structured-long-form
WAV. The probe report SHA-256 is
`49f99c80c5401d813139500f29119dafdb4e58a890d70bf6754a04bfc61970e4`.

## Objective diagnostics

The evaluator recorded complete coverage for all nine planned HTTP rows and all
nine required objective metrics. Selected aggregate observations were:

| Measurement | Result |
| --- | ---: |
| Invalid-output rate | 0 |
| Mean ASR word error rate | 0.027294 |
| Mean ECAPA cosine similarity | 0.824936 |
| Mean server generation time | 7.607366 seconds |
| Mean real-time factor | 0.481311 |
| Mean peak allocated CUDA memory | 7,800,936,334 bytes |
| Mean silence fraction | 0.358821 |
| Clipping fraction | 0 |
| Sample rate | 22,050 Hz |

The ASR content diagnostic evaluated all nine rows and flagged none for high
word error, repeated n-gram excess, retained-reference overlap, or spoken
instruction overlap. Its mean WER was 0.027294. `not_flagged` is not proof of
content faithfulness, pronunciation, accent fidelity, or naturalness.

Evidence hashes:

- generation observations:
  `f1760eb939939d5b38a53f47c306165f524b08a80736e661bf847a8999aeb5b2`
- generation-attempt receipt:
  `9072c820467ea35b3ff9fd4da5dbf83bea12ba0c20b5a60a5b79806fbd0e87dd`
- audio probes:
  `8d6d26dfa3f5bce6ac5bc6d34a71bcb6d4bf140e11429d6a416ee45a5fde6681`
- prosody proxies:
  `ace9911c844b9d38a16a6d6dd855d0480038261f9a096e5c2ed12bbd1ca6344c`
- faster-whisper results:
  `c8f64fdb6c3624407a921762f2548b55789a483c281e6aefbb652e4e0a2cedd2`
- SpeechBrain ECAPA results:
  `780283fe549e75ca3c79c2bc76764f8bbcfb32dd57d36f7eed73d66aa3297608`
- complete observations:
  `ad6ca02300127369595c8dafe4ea56002992bbf492be94a9220cfa4dbff2ba31`
- content-faithfulness report:
  `8f84193e81f37293631965bdf17d64b656182a1d4df775abec59a917ff461e51`
- objective report:
  `19da16605285b0a4a92bf52f8de84fca20f27339bae8ef54736aceec06e4f1f1`
- complete suite coverage:
  `d19b00ac05f3e0989f6260b7b0598bcae175a78a43c0f4e14a6a105298030d45`

The speaker-reference assignment was made after generation and uses the same
single retained studio reference for every row. It is suitable for symmetric
diagnostics, not for a claim that the reference policy was preregistered.

## Failures retained during hardening

Three implementation and environment failures were retained instead of being
silently discarded:

1. The first overlap probe attempted to decode a successful binary WAV as JSON.
   The fixed decoder now treats non-UTF-8 bodies as having no JSON error code.
2. The first evaluator application rejected sample-rate fields owned by the
   audio-probe extractor. The HTTP client no longer writes extractor-owned
   fields.
3. CUDA faster-whisper could not initialize in the available host environments:
   one had an OpenSSL-linked PyAV mismatch and another lacked the required
   cuDNN runtime. The final run used the same frozen ASR model and decoding on
   CPU int8. These are host-environment failures, not TTS runtime failures.

## Remaining scope

The next HTTP-runtime evidence should cover sustained repeated requests,
client disconnect and timeout behavior during generation, GPU OOM and worker
restart recovery, non-loopback deployment behind a real TLS gateway, and a
pre-generation speaker-reference plan. Blind listening remains required for
speaker identity, Singapore English accent, `paiseh` pronunciation, cadence,
monotony, naturalness, artifacts, and fatigue. This run did not evaluate
instruction-bearing prompts because the fixed HTTP subset does not implement
emotion-text controls.
