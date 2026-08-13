# Matched long-form base and full-SFT evidence, 2026-08-13

## Result

One exact IndexTTS2 base versus step-14000 full-SFT pair completed the focused
long-form objective and blind-pack pipeline. The comparison passed all nine
required objective metric coverage checks and reported
`proves_adaptation_benefit: false`. No listening ratings or quality winner are
recorded.

Both candidates used prompt `cadence-two-minute`, seed `20260812`, the same
retained FEMALE_01 reference, the same IndexTTS2 model dependencies, CUDA
device, FP16 mode, segmentation settings, and evaluator revision. The runner
loaded the original `gpt.pth` for the control and the explicit selected
`model_step14000.pth` for full SFT.

## Reproducibility anchors

- runner repository revision: `8acc35332bebc0e2bfcf0ed30cef492ba0abd7e2`
- clean detached runner checkout:
  `/mnt/work/chee-wei-jie/voice-models/instavar-index-matched-clean-20260813T1245SGT`
- evaluator revision: `982367abc7837cb6da5ebb94192c9642dea62fce`
- prompt pack: `instavar-singapore-english` version 1.2.0,
  SHA-256 `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- generation plan file SHA-256:
  `f6c67522448e6c9fbbd7d76a084fa8358d7c360234950e65cf11d60e3ae9ff72`
- canonical generation plan SHA-256:
  `06daee10006580dbc15e080d338469e97dfb268716964c3ced197c25a0a01b77`
- original base GPT SHA-256:
  `baaaeb8b56328da81731dc540a85a7dee32eca9da28f174b05757cb651c602a4`
- selected step-14000 GPT SHA-256:
  `3db5cb0b9c00d0025933599bf3133896fb45ad6c37d5855f4dafc5f03ed40676`
- shared source config SHA-256:
  `ea9c2815ecc3874577c7ac158b97248c027250e5b05972fbbb1216b6d6539081`
- retained reference audio SHA-256:
  `2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`
- evidence directory:
  `/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260813_index_matched_long_form_v1`

The candidate artifact-set digests are canonical hashes over the shared source
config record and the candidate-specific GPT checkpoint record. The base set is
`8ef2475d1e7a7230b39cb3628c13d7bd59154d665788ed993696acc763970b23`.
The full-SFT set is
`47f8cc3775a286b3537734221bf5f79a7a1d8904bc351f71d7813ae0ab33975c`.
These declarations bind the intended artifacts but do not independently prove
host trust or model-loader honesty. The generation logs separately record the
resolved GPT path restored by IndexTTS2.

Hosted contract runs passed for both the feature revision and merged main:

- feature run: `31666718905`
- main run: `31667177064`

The clean runner checkout remained unmodified after generation.

## Objective observations

| Measurement | Base | Full SFT | Full SFT minus base |
| --- | ---: | ---: | ---: |
| Audio duration, seconds | 79.0138 | 74.6252 | -4.3886 |
| Generation time, seconds | 36.5572 | 34.1217 | -2.4355 |
| Real-time factor | 0.462668 | 0.457241 | -0.005428 |
| Peak allocated CUDA memory, bytes | 8,337,923,584 | 8,261,996,544 | -75,927,040 |
| ASR word error rate | 0.012987 | 0.025974 | +0.012987 |
| ECAPA cosine similarity | 0.873532 | 0.858753 | -0.014780 |
| Sample rate, Hz | 22,050 | 22,050 | 0 |
| Silence fraction | 0.346047 | 0.357260 | +0.011214 |
| Clipping fraction | 0 | 0 | 0 |

The faster-whisper extractor used revision
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` with artifact-set SHA-256
`3433b5ac25f4b005aadfcde370f3615a5d2883fe40d251e823b80204071115d6`.
The SpeechBrain ECAPA extractor used revision
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` with artifact-set SHA-256
`5a8cd13222e7edf1c932b8695e34c6537c15230e8e47aabe9af454284906dd7c`.
The speaker assignment used one exact shared reference and was frozen after
generation but before speaker scoring. It is symmetric, but it is not evidence
of preregistration.

The matched objective report SHA-256 is
`5e32e1c0b3d6e66c4758db94ad55bb1ba2b7ced734cf6ad0da809ee230918192`.
The complete objective report SHA-256 is
`8783e25d256f0fef474ef763858ee031488fafa0c2b44fa2d44008f3ff822593`.

One pair cannot support throughput, memory, intelligibility, speaker-identity,
or adaptation-effect conclusions. The WER and ECAPA differences are
extractor-specific observations, not perceptual judgments or evidence that
full SFT helped or harmed the voice.

## Non-directional prosody proxies

Both outputs were eligible for the long-form proxy comparison. Selected signed
full-SFT-minus-base deltas were:

- phrase-duration coefficient of variation: `+0.069016`
- pause-duration coefficient of variation: `+0.033070`
- active RMS dB standard deviation: `+0.125398`
- window RMS dB standard deviation: `+0.132448`
- zero-crossing-rate standard deviation: `+434.159850 Hz`
- pause rate: `-7.906803` per minute

The matched prosody comparison SHA-256 is
`c86a1ef7de7d6e6b516ce24707b950de6b738dda87e123ab44a36a48fd361e33`.
Its directions are explicitly not established. These signal proxies do not
prove cadence naturalness, reduced monotony, accent fidelity, preference, or
causation.

## Blind listening status

Two identity-neutral audio files were staged with a private reveal document.
The focused assignment includes speaker identity, cadence variation, long-form
monotony, naturalness, artifact severity, and listening fatigue. It explicitly
excludes Singapore English accent fidelity, lexical pronunciation, and emotion
obedience because the cadence-only prompt does not route to those criteria.

- listening assignment SHA-256:
  `ce9219773a9a7bc11cf17ce29e7618a92f3f585a1eb00704715be318d0edbdfb`
- identity-neutral review SHA-256:
  `18f84e8c6c0fc64277def10ab9fd05acb323856a06f095cef716917300961295`
- staged blind-audio manifest SHA-256:
  `4188a60fe7cc6d10f1a9db6331a52e9c8ee1b8f75e7ed5d8d5082668029e7e12`

No ratings were invented. The private reveal must remain closed until the
scheduled listening review is complete.

## Scope and remaining work

This run closes the missing exact-base execution path and establishes one
content-addressed, plan-matched long-form pair. It does not replace:

1. multiple seeds and more long-form structures;
2. prompt slices that actually evaluate accent, lexical pronunciation, and
   emotion control;
3. completed blinded ratings from multiple reviewers;
4. a pre-generation server-stamped speaker-reference assignment;
5. repeated warm and cold runtime measurements;
6. current IndexTTS 2.5 adaptation revalidation.
