from __future__ import annotations

import http.client
import io
import json
import socket
import threading
import unittest
import wave
from pathlib import Path
from typing import Self

from tools import openai_speech_server as server


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


class FakeEngine:
    def __init__(self, *, invalid: bool = False, error: Exception | None = None) -> None:
        self.invalid = invalid
        self.error = error
        self.texts: list[str] = []
        self.output_paths: list[Path] = []

    def generate(self, text: str, output_path: Path) -> None:
        self.texts.append(text)
        self.output_paths.append(output_path)
        if self.error:
            raise self.error
        output_path.write_bytes(b"not-wave" if self.invalid else wav_bytes())


class RunningServer:
    def __init__(self, service: server.SpeechService) -> None:
        self.httpd = server.SpeechHTTPServer(("127.0.0.1", 0), server.build_handler(service))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])


class SpeechRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice")

    def test_accepts_only_the_supported_fixed_artifact_request(self) -> None:
        request = server.parse_speech_request(
            {
                "input": "Hello from Singapore.",
                "model": "fixed-model",
                "response_format": "wav",
                "speed": 1.0,
                "voice": "fixed-voice",
            },
            self.config,
        )
        self.assertEqual(request.text, "Hello from Singapore.")

    def test_rejects_request_controlled_paths_and_unknown_fields(self) -> None:
        for field in ("checkpoint", "config", "emotion_audio", "output", "speaker", "tokenizer"):
            with self.subTest(field=field), self.assertRaisesRegex(server.ApiError, "unsupported request field"):
                server.parse_speech_request(
                    {"input": "hello", "model": "fixed-model", "voice": "fixed-voice", field: "/tmp/x"},
                    self.config,
                )

    def test_rejects_wrong_model_voice_format_and_speed(self) -> None:
        cases = (
            ({"input": "x", "model": "other", "voice": "fixed-voice"}, "model"),
            ({"input": "x", "model": "fixed-model", "voice": "other"}, "voice"),
            (
                {"input": "x", "model": "fixed-model", "voice": "fixed-voice", "response_format": "mp3"},
                "response_format",
            ),
            ({"input": "x", "model": "fixed-model", "voice": "fixed-voice", "speed": 1.01}, "speed"),
            ({"input": "x", "model": "fixed-model", "voice": "fixed-voice", "speed": float("nan")}, "speed"),
            ({"input": "x", "model": "fixed-model", "voice": "fixed-voice", "speed": True}, "speed"),
        )
        for payload, message in cases:
            with self.subTest(payload=payload), self.assertRaisesRegex(server.ApiError, message):
                server.parse_speech_request(payload, self.config)

    def test_rejects_blank_non_string_and_oversized_input(self) -> None:
        config = server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice", max_input_chars=3)
        for value in (None, 7, "", "   ", "four"):
            with self.subTest(value=value), self.assertRaises(server.ApiError):
                server.parse_speech_request(
                    {"input": value, "model": "fixed-model", "voice": "fixed-voice"}, config
                )

    def test_rejects_control_characters_and_unpaired_unicode(self) -> None:
        for value in ("hello\x00world", "\ud800"):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(server.ApiError, "input"):
                server.parse_speech_request(
                    {"input": value, "model": "fixed-model", "voice": "fixed-voice"}, self.config
                )

    def test_rejects_duplicate_json_fields(self) -> None:
        with self.assertRaisesRegex(server.ApiError, "duplicate JSON field") as raised:
            server.decode_json_body(
                b'{"model":"fixed-model","voice":"fixed-voice","voice":"other","input":"hello"}'
            )
        self.assertEqual(raised.exception.code, "duplicate_json_field")


class StartupValidationTests(unittest.TestCase):
    def test_public_ids_are_bounded_and_log_safe(self) -> None:
        for value in ("", " leading", "line\nbreak", "x" * 129):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, "public identifier"):
                server.SpeechServerConfig(model_id=value, voice_id="fixed-voice").validate()

    def test_sampling_controls_reject_invalid_domains(self) -> None:
        cases = (
            {"top_k": 0},
            {"num_beams": True},
            {"top_p": 0},
            {"top_p": 1.1},
            {"temperature": float("inf")},
            {"temperature": 0},
            {"unknown": 1},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                server.validate_generation_kwargs(values)

    def test_sampling_controls_preserve_valid_values(self) -> None:
        values = {"top_k": 20, "top_p": 0.8, "temperature": 0.9, "num_beams": 2}
        self.assertEqual(server.validate_generation_kwargs(values), values)


class SpeechServiceTests(unittest.TestCase):
    def test_generates_valid_wav_in_a_server_owned_temporary_directory(self) -> None:
        engine = FakeEngine()
        service = server.SpeechService(
            engine,
            server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice"),
        )
        audio = service.synthesize(server.SpeechRequest("hello"))
        self.assertEqual(audio, wav_bytes())
        self.assertEqual(engine.texts, ["hello"])
        self.assertFalse(engine.output_paths[0].exists())
        self.assertEqual(engine.output_paths[0].name, "response.wav")

    def test_rejects_invalid_wav(self) -> None:
        service = server.SpeechService(
            FakeEngine(invalid=True),
            server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice"),
        )
        with self.assertRaisesRegex(RuntimeError, "invalid PCM WAV"):
            service.synthesize(server.SpeechRequest("hello"))

    def test_rejects_concurrent_generation_without_queueing(self) -> None:
        service = server.SpeechService(
            FakeEngine(),
            server.SpeechServerConfig(model_id="fixed-model", voice_id="fixed-voice"),
        )
        service._generation_lock.acquire()
        try:
            with self.assertRaisesRegex(server.ApiError, "another synthesis request") as raised:
                service.synthesize(server.SpeechRequest("hello"))
            self.assertEqual(raised.exception.status, 429)
        finally:
            service._generation_lock.release()


class SpeechHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.config = server.SpeechServerConfig(
            model_id="fixed-model",
            voice_id="fixed-voice",
            api_key="test-only-key",
        )

    def request(
        self,
        port: int,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        authorized: bool = True,
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {}
        body = None
        if authorized:
            headers["Authorization"] = "Bearer test-only-key"
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = response.status, {key.casefold(): value for key, value in response.getheaders()}, response.read()
        connection.close()
        return result

    def test_live_success_returns_wav_and_security_headers(self) -> None:
        service = server.SpeechService(self.engine, self.config)
        with RunningServer(service) as running:
            status, headers, body = self.request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {"input": "hello", "model": "fixed-model", "voice": "fixed-voice"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "audio/wav")
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertTrue(headers["x-request-id"].startswith("req_"))
        self.assertEqual(body, wav_bytes())

    def test_live_request_requires_authentication(self) -> None:
        service = server.SpeechService(self.engine, self.config)
        with RunningServer(service) as running:
            status, _, body = self.request(running.port, "GET", "/readyz", authorized=False)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["code"], "invalid_api_key")

    def test_live_error_does_not_expose_engine_exception(self) -> None:
        service = server.SpeechService(
            FakeEngine(error=RuntimeError("secret internal path /sensitive/checkpoint.pth")),
            self.config,
        )
        with RunningServer(service) as running:
            status, _, body = self.request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {"input": "hello", "model": "fixed-model", "voice": "fixed-voice"},
            )
        self.assertEqual(status, 500)
        self.assertNotIn(b"sensitive", body)
        self.assertEqual(json.loads(body)["error"]["code"], "synthesis_failed")

    def test_live_request_rejects_oversized_body_before_generation(self) -> None:
        config = server.SpeechServerConfig(
            model_id="fixed-model", voice_id="fixed-voice", max_body_bytes=8, api_key="test-only-key"
        )
        service = server.SpeechService(self.engine, config)
        with RunningServer(service) as running:
            status, _, body = self.request(
                running.port,
                "POST",
                "/v1/audio/speech",
                {"input": "hello", "model": "fixed-model", "voice": "fixed-voice"},
            )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body)["error"]["code"], "body_too_large")
        self.assertEqual(self.engine.texts, [])

    def test_live_request_rejects_duplicate_fields_before_generation(self) -> None:
        service = server.SpeechService(self.engine, self.config)
        body = b'{"model":"fixed-model","voice":"fixed-voice","input":"first","input":"second"}'
        with RunningServer(service) as running:
            connection = http.client.HTTPConnection("127.0.0.1", running.port, timeout=2)
            connection.request(
                "POST",
                "/v1/audio/speech",
                body=body,
                headers={
                    "Authorization": "Bearer test-only-key",
                    "Content-Length": str(len(body)),
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
        self.assertEqual(response.status, 400)
        self.assertEqual(payload["error"]["code"], "duplicate_json_field")
        self.assertEqual(self.engine.texts, [])


class BindingSecurityTests(unittest.TestCase):
    def test_loopback_can_run_without_authentication(self) -> None:
        self.assertIsNone(server.read_api_key("127.0.0.1", None))
        self.assertIsNone(server.read_api_key("::1", None))
        self.assertIsNone(server.read_api_key("localhost", None))

    def test_non_loopback_requires_environment_delivered_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            server.read_api_key("0.0.0.0", None)
        with self.assertRaisesRegex(ValueError, "unset or empty"):
            server.read_api_key("0.0.0.0", "ABSENT_TEST_KEY")

    def test_ipv6_listener_uses_the_ipv6_socket_family(self) -> None:
        self.assertEqual(server.IPv6SpeechHTTPServer.address_family, socket.AF_INET6)


if __name__ == "__main__":
    unittest.main()
