# Live-conditioned full-SFT resume evidence

Date: 2026-08-14, Asia/Singapore

## Result

IndexTTS2 full SFT reached evaluator 0.45 claim tier
`byte_exact_live_conditioned_artifact_set` for one bounded RTX 3090 Ti pair.
The uninterrupted condition and a real process-group interruption followed by
a separate resume produced byte-identical final model, optimizer, scheduler,
trainer, and RNG role files.

This result required two implementation corrections that dependency-free tests
had not exposed:

1. opt-in deterministic CUDA controls for byte-exact evidence; and
2. canonical CPU tensor storage for the optimizer evidence sidecar.

The executed producer revision was
`94e1ba1bb2105ff5a5655b538f6ce3fbfe366cd5`. The evaluator revision was
`29c38cfd86b889abc8b79df063c817dd8f684903`.

## Bound configuration

- upstream source revision:
  `ddeb0ae15e411b3db9f31132bdb5d0f819f0d847`
- Python: 3.10.16
- PyTorch: 2.8.0+cu128
- GPU: NVIDIA GeForce RTX 3090 Ti
- train rows: 1
- validation rows: 1
- batch size: 1
- gradient accumulation: 1
- target epochs and optimizer updates: 2
- AMP: enabled
- seed: 1234
- deterministic algorithms: enabled
- CUDA BLAS workspace: `:4096:8`
- TF32: disabled
- Base and loaded initial-state SHA-256:
  `baaaeb8b56328da81731dc540a85a7dee32eca9da28f174b05757cb651c602a4`

The original Base file and the loaded initial-state copy had distinct inodes.
Both training conditions loaded the copy, while evaluator receipts separately
bound the original Base artifact and the live initial-state file.

## Negative control 1: ordinary CUDA was not byte-exact

The first real pair used public main
`34455b95dda8ba90506842451cb3a8d7b16a0a70` without deterministic mode. Model
and optimizer artifacts differed, while scheduler, trainer, and RNG artifacts
matched. The epoch-one role hashes already differed before any resume, so the
primary cause was independent CUDA execution rather than restoration order.

Tensor-level diagnosis found real numerical drift across 301 of 665 model
tensors and 602 of 903 optimizer tensors. Maximum absolute differences were
approximately `2.44e-05` for model state and `1.56e-05` for optimizer state.
The retained evaluator report has internal SHA-256
`4484a155aa1f79a616492a3faabe9965ffe09294c006ea1aa70549fd17f17530`
and file SHA-256
`ebbe7defd8b37c060d81d093d7971644cfca55d7c53eb308b764f2c9fb52c8d0`.

Retained root:
`/mnt/work/chee-wei-jie/voice-models/instavar-indextts2-resume-live-045-20260814-v2`.

## Negative control 2: equivalent optimizer values serialized differently

Revision `a035b0c3d5893c9d141a8838277c4b6b23be4c8a` added strict deterministic
controls. The two epoch-one artifact sets became byte-identical, and the final
model, scheduler, trainer, and RNG roles also matched. Only the final optimizer
sidecar bytes differed.

Recursive comparison found all 903 optimizer tensors, scalar fields, parameter
groups, dictionary keys, and key order equal. Loading optimizer state in a new
process had reconstructed equivalent tensors with a different storage layout.
A controlled republish that cloned every tensor leaf to contiguous CPU storage
produced the same SHA-256 for both conditions:
`9373713399c3b2bd989636acd5c15c57bc68f7adbfec77d857e4131601d4ac3c`.

That result motivated the repository-owned canonical optimizer sidecar
publisher in `94e1ba1bb2105ff5a5655b538f6ce3fbfe366cd5`. The retained negative report
has internal SHA-256
`59ab5545c17c68961ed6f14a6c79dcbde9b550ac303064d40bc13d66a33b5355`
and file SHA-256
`b2eec1e211c5952c3f61e05de6f1529ddef3d176349efa6a7c03e2f046c0c09a`.

Retained root:
`/mnt/work/chee-wei-jie/voice-models/instavar-indextts2-resume-live-045-20260814-v3`.

## Positive interruption and resume

The successful run observed the complete epoch-one checkpoint, its schema 1.1
metadata, and all five immutable role files before sending `SIGTERM` to the
process group. The process exited `143`. The harness confirmed that the target
epoch-two checkpoint, metadata, artifact directory, and partial files were all
absent before a new process started.

The separate process resumed from `model_epoch01_step1.pth` and reached update
two. Logged metrics matched the uninterrupted condition at the compared
boundaries:

- update one: text loss `4.4786`, mel loss `6.1733`, mel top-1 `0.0349`
- validation one: text loss `3.5401`, mel loss `5.7665`, mel top-1 `0.0433`
- update two: text loss `3.7563`, mel loss `5.7514`, mel top-1 `0.0474`
- validation two: text loss `3.5162`, mel loss `5.7675`, mel top-1 `0.0433`

Final role hashes were identical:

| Role | SHA-256 | Bytes |
| --- | --- | ---: |
| model | `f8d4f560977f18fb32ba78383fffc7aae080517dab066cb01fe1398e99b61a70` | 3,484,661,257 |
| optimizer | `9373713399c3b2bd989636acd5c15c57bc68f7adbfec77d857e4131601d4ac3c` | 4,216,915,993 |
| scheduler | `b4c10a0b94d1cd21c85fd5cec4fac86da5416199331d5baff4c3ce5329eb6da3` | 1,501 |
| trainer | `071579beac3a590d54b19255d21f9c16bb14b335a256ed78eea6ea63e9c756b1` | 77 |
| RNG | `f20be36128d2aa096571f56e6fbe2c78f383a8e312eb32421dd2718d5d3ea2f7` | 14,709 |

Evaluator report hashes:

- internal report SHA-256:
  `78a497e1f2f849b63c5d4b9f68719a23fb4eadb9d142d041abcdbff62fe5262f`
- report file SHA-256:
  `984d43b83c063130cd9812bc68e64b4fc455d3b4c5ead85244ef42b2b155cf86`
- uninterrupted receipt SHA-256:
  `56df4e5e1e4b07d04653c22382905084e5a0d2af676d1334aa03be36b198da64`
- resumed receipt SHA-256:
  `9ef10abea708cacd728e15ddd4083276f38739cb89b69e56ebb3ab9d96b01692`
- interruption receipt SHA-256:
  `ca7bc9d4e87de6977534c02a1165c65ed79e2fa12b79d1777a1f74654112df40`

Full evidence root:
`/mnt/ext4_4tb/chee-wei-jie/voice-models/instavar-indextts2-resume-live-045-20260814-v4`.

## Storage and clean-source findings

A first preparation attempt stopped because one upstream example-audio Git LFS
object was unavailable. Code-only qualification therefore cloned the exact
upstream revision with LFS smudging disabled and verified a clean tracked source
tree before execution. Runtime-generated text-normalization caches stayed
inside the isolated evidence checkout.

The executed producer still wrote a 7.7 GB `latest.pth` compatibility copy at
every epoch in addition to the combined checkpoint and decomposed sidecars. On
the slower evidence disk, that redundant copy dominated wall time. Subsequent
revision `3a6298a` added `WRITE_LATEST=0` for bounded evidence runs and made the
enabled compatibility write atomic and fsynced. That later I/O path passed 53
local contract and runtime tests plus hosted contract checks, but it was not the
producer revision of the GPU pair described above.

## Evidence boundary

This is one one-row, two-update, deterministic AMP, single-process,
world-size-one epoch-boundary pair on one GPU and dependency stack. It proves
byte equality only for the five declared final role files under the rehashed
conditioning receipts.

It does not prove:

- training semantics or hidden state completeness;
- perceptual quality, adaptation benefit, speaker similarity, or accent;
- arbitrary seeds, datasets, batch schedules, accumulation settings, or epoch
  counts;
- mid-epoch or mid-accumulation resume;
- multi-worker, shuffled multi-row, distributed, or multi-GPU resume;
- cross-version, cross-device, or cross-runtime continuation;
- convergence or long-run stability; or
- backup durability, disaster recovery, or model and dataset rights.

No audio was generated in this resume experiment. Quality remains governed by
the separate matched Base-versus-adapted and blind-listening evidence paths.
