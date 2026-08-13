from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from tools import openai_speech_server as server
from tools import probe_openai_speech_runtime as probe
from tools import qualify_openai_speech_runtime as qualify
from tools import validate_http_cli_parity as parity


class FakeEngine:
    def generate(self, text: str, output_path: Path) -> None:
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\x00\x00" * 160)


class RunningServer:
    def __init__(self, receipt_sha256: str) -> None:
        config = server.SpeechServerConfig(
            model_id="fixed-model",
            voice_id="fixed-voice",
            startup_receipt_sha256=receipt_sha256,
        )
        service = server.SpeechService(FakeEngine(), config)
        self.httpd = server.SpeechHTTPServer(("127.0.0.1", 0), server.build_handler(service))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> RunningServer:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"


class QualificationTests(unittest.TestCase):
    def test_binary_success_body_has_no_json_error_code(self) -> None:
        self.assertIsNone(probe.error_code(b"RIFF\xba\x00\x00\x00WAVE"))

    def test_live_negative_request_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_sha = "b" * 64
            output = root / "probes.json"
            with RunningServer(receipt_sha) as running, patch.object(
                sys,
                "argv",
                [
                    "probe",
                    "--endpoint",
                    running.endpoint,
                    "--model-id",
                    "fixed-model",
                    "--voice-id",
                    "fixed-voice",
                    "--input",
                    "long enough for a live probe",
                    "--expected-startup-receipt-sha256",
                    receipt_sha,
                    "--output",
                    str(output),
                ],
            ):
                self.assertEqual(probe.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["results"]), 3)

    def test_live_qualification_and_exact_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "1.1.0",
                        "samples": [
                            {
                                "sample_id": "sample-42",
                                "candidate_id": "candidate",
                                "prompt_id": "prompt",
                                "category": "general",
                                "seed": 42,
                                "text": "Hello from Singapore.",
                                "expected_audio_path": "audio/sample-42.wav",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            receipt = root / "startup.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "runtime_id": "indextts2_openai_compatible_http",
                        "seed": 42,
                        "artifact_set_id": "frozen-set",
                        "artifact_set_sha256": "a" * 64,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
            output_dir = root / "http"
            with RunningServer(receipt_sha) as running, patch.object(
                sys,
                "argv",
                [
                    "qualify",
                    "--endpoint",
                    running.endpoint,
                    "--model-id",
                    "fixed-model",
                    "--voice-id",
                    "fixed-voice",
                    "--generation-plan",
                    str(plan),
                    "--candidate-id",
                    "candidate",
                    "--sample-id",
                    "sample-42",
                    "--expected-startup-receipt-sha256",
                    receipt_sha,
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                self.assertEqual(qualify.main(), 0)

            http_observation = json.loads(
                (output_dir / "http-generation-observation.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("sample_rate_hz", http_observation)
            self.assertNotIn("channels", http_observation)
            self.assertNotIn("sample_width_bytes", http_observation)
            cli_observations = root / "cli.json"
            cli_observations.write_text(
                json.dumps([{**http_observation, "runtime": "indextts2_pytorch_cuda_full_sft"}]),
                encoding="utf-8",
            )
            report = root / "parity.json"
            with patch.object(
                sys,
                "argv",
                [
                    "parity",
                    "--generation-plan",
                    str(plan),
                    "--candidate-id",
                    "candidate",
                    "--sample-id",
                    "sample-42",
                    "--cli-observations",
                    str(cli_observations),
                    "--http-observation",
                    str(output_dir / "http-generation-observation.json"),
                    "--startup-receipt",
                    str(receipt),
                    "--output",
                    str(report),
                ],
            ):
                self.assertEqual(parity.main(), 0)
            self.assertTrue(json.loads(report.read_text(encoding="utf-8"))["exact_wav_equivalent"])

    def test_instruction_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "samples": [
                            {
                                "sample_id": "sample",
                                "candidate_id": "candidate",
                                "instruction": "sound happy",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot apply instruction"):
                qualify.load_plan_row(plan, "candidate", "sample")


if __name__ == "__main__":
    unittest.main()
