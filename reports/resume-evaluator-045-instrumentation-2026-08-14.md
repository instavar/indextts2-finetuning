# Evaluator 0.45 resume instrumentation

Date: 2026-08-14, Asia/Singapore

## Change

Future IndexTTS2 epoch-boundary checkpoints publish a metadata-bound
`*.resume-artifacts` directory next to the original combined `.pth` loader
checkpoint. The five evaluator 0.45 roles map as follows:

| Evaluator role | IndexTTS2 resume artifact |
| --- | --- |
| `model_state` | `model-state.pt` |
| `optimizer_state` | `optimizer-state.pt` |
| `scheduler_state` | `scheduler-state.pt` |
| `trainer_state` | `trainer-state.json` |
| `rng_state` | `rng-state.pt` |

Schema 1.1 metadata binds the complete artifact tree. The mapper rehashes that
tree, requires all five exact files, and rejects symlinks and hardlink aliases.
The original monolithic checkpoint remains the loader source, preserving schema
1.0 resume compatibility.

The decomposed model state duplicates bytes already present in the combined
checkpoint. This storage cost is explicit and applies only to resumable epoch
boundaries. A future checkpoint schema can remove the duplication only with a
backward-compatible loader migration and real restore evidence.

## OOD and compatibility controls

Dependency-free tests cover:

- one complete schema 1.1 role mapping;
- artifact mutation after metadata publication;
- cross-role hardlink rejection;
- legacy schema 1.0 resume;
- checkpoint, contract, and feature drift;
- unsafe checkpoint and feature symlinks;
- completed-target rejection; and
- source-level confirmation that the trainer publishes and binds decomposed
  artifacts.

The public contract workflow pins evaluator revision
`29c38cfd86b889abc8b79df063c817dd8f684903` and verifies its schema 1.1 receipt
builder and comparison APIs.

## Evidence boundary

No model training or GPU test was run for this instrumentation change. The
historical selected checkpoint and runtime qualifications predate both the new
artifact directory and evaluator live-conditioning receipts. They are not
upgraded.

A stronger comparison must preregister and fingerprint the Base artifact,
dataset-lineage receipt, training controls, and initial state. It then needs an
independent uninterrupted run and an observed interrupted-resumed run that both
reach the same target update. Evaluator 0.45 rehashes those four conditioning
artifacts and the five final-state files.

Even a passing report proves only byte equality for the declared files. It does
not prove trainer semantics, hidden floating-point equivalence, quality,
adaptation benefit, cross-version resume, or distributed resume.
