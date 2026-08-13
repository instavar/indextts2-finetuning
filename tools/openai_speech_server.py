#!/usr/bin/env python3
"""Serve one fixed IndexTTS2 artifact through a strict OpenAI speech subset."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import ipaddress
import json
import logging
import math
import os
import random
import re
import socket
import tempfile
import threading
import time
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

LOG = logging.getLogger("indextts2.openai_speech_server")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
ALLOWED_SPEECH_FIELDS = frozenset({"input", "model", "response_format", "speed", "voice"})


class ApiError(Exception):
    """A bounded error that is safe to return to an HTTP client."""

    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class DuplicateJsonKey(ValueError):
    """Raised when a JSON object repeats a field name."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result


def decode_json_body(body: bytes) -> Any:
    try:
        return json.loads(body, object_pairs_hook=_strict_json_object)
    except DuplicateJsonKey as error:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "duplicate_json_field",
            "request body contains a duplicate JSON field",
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_json", "request body is not valid JSON") from error


class SpeechEngine(Protocol):
    """Minimal engine boundary used by the HTTP service and dependency-free tests."""

    def generate(self, text: str, output_path: Path) -> Mapping[str, int | float] | None: ...


@dataclass(frozen=True)
class SynthesisResult:
    audio: bytes
    generation_seconds: float
    peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class SpeechServerConfig:
    model_id: str
    voice_id: str
    max_input_chars: int = 4_000
    max_body_bytes: int = 16_384
    max_audio_bytes: int = 100 * 1024 * 1024
    request_timeout_seconds: float = 30.0
    api_key: str | None = None
    startup_receipt_sha256: str | None = None

    def validate(self) -> None:
        for label, value in (("model id", self.model_id), ("voice id", self.voice_id)):
            if not isinstance(value, str) or not PUBLIC_ID_RE.fullmatch(value):
                raise ValueError(f"{label} must be a bounded public identifier")
        for label, value in (
            ("max input chars", self.max_input_chars),
            ("max body bytes", self.max_body_bytes),
            ("max audio bytes", self.max_audio_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not math.isfinite(float(self.request_timeout_seconds))
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request timeout seconds must be finite and positive")
        if self.api_key is not None and not self.api_key:
            raise ValueError("API key must be nonempty when configured")
        if self.startup_receipt_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.startup_receipt_sha256
        ):
            raise ValueError("startup receipt sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SpeechRequest:
    text: str


def parse_speech_request(payload: Any, config: SpeechServerConfig) -> SpeechRequest:
    """Validate the supported OpenAI speech subset without coercing caller values."""

    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_request", "JSON body must be an object")
    unknown = sorted(set(payload) - ALLOWED_SPEECH_FIELDS)
    if unknown:
        joined = ", ".join(unknown)
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "unsupported_field",
            f"unsupported request field(s): {joined}",
        )

    model = payload.get("model")
    if model != config.model_id:
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_model", "model is not available")
    voice = payload.get("voice")
    if voice != config.voice_id:
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_voice", "voice is not available")

    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_input", "input must be a nonempty string")
    if len(text) > config.max_input_chars:
        raise ApiError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "input_too_large",
            f"input exceeds the {config.max_input_chars}-character limit",
        )
    if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_input",
            "input contains an unsupported control character",
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_input", "input is not valid Unicode") from error

    response_format = payload.get("response_format", "wav")
    if response_format != "wav":
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "unsupported_response_format",
            "only response_format='wav' is supported",
        )

    speed = payload.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_speed", "speed must equal 1.0")
    if not math.isfinite(float(speed)) or float(speed) != 1.0:
        raise ApiError(HTTPStatus.BAD_REQUEST, "unsupported_speed", "only speed=1.0 is supported")
    return SpeechRequest(text=text)


def validate_wav(path: Path, max_audio_bytes: int) -> bytes:
    """Read a bounded, positive-duration PCM WAV generated in a server-owned directory."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("synthesis did not produce a regular WAV file")
    size = path.stat().st_size
    if size <= 0 or size > max_audio_bytes:
        raise RuntimeError("synthesis produced an empty or oversized WAV file")
    try:
        with wave.open(str(path), "rb") as wav_file:
            if (
                wav_file.getnchannels() <= 0
                or wav_file.getsampwidth() <= 0
                or wav_file.getframerate() <= 0
                or wav_file.getnframes() <= 0
            ):
                raise RuntimeError("synthesis produced a zero-duration WAV file")
    except (EOFError, wave.Error) as error:
        raise RuntimeError("synthesis produced an invalid PCM WAV file") from error
    return path.read_bytes()


class SpeechService:
    """Serialize a mutable model engine and reject overlapping synthesis requests."""

    def __init__(self, engine: SpeechEngine, config: SpeechServerConfig) -> None:
        config.validate()
        self.engine = engine
        self.config = config
        self._generation_lock = threading.Lock()

    def synthesize(self, request: SpeechRequest) -> SynthesisResult:
        if not self._generation_lock.acquire(blocking=False):
            raise ApiError(
                HTTPStatus.TOO_MANY_REQUESTS,
                "server_busy",
                "another synthesis request is in progress",
            )
        try:
            with tempfile.TemporaryDirectory(prefix="indextts2-speech-") as temporary:
                output = Path(temporary) / "response.wav"
                started = time.perf_counter()
                metrics = self.engine.generate(request.text, output) or {}
                elapsed = time.perf_counter() - started
                generation_seconds = metrics.get("generation_seconds", elapsed)
                peak_memory_bytes = metrics.get("peak_memory_bytes")
                if (
                    isinstance(generation_seconds, bool)
                    or not isinstance(generation_seconds, (int, float))
                    or not math.isfinite(float(generation_seconds))
                    or generation_seconds <= 0
                ):
                    raise RuntimeError("engine returned invalid generation timing")
                if peak_memory_bytes is not None and (
                    isinstance(peak_memory_bytes, bool)
                    or not isinstance(peak_memory_bytes, int)
                    or peak_memory_bytes < 0
                ):
                    raise RuntimeError("engine returned invalid peak memory")
                return SynthesisResult(
                    audio=validate_wav(output, self.config.max_audio_bytes),
                    generation_seconds=float(generation_seconds),
                    peak_memory_bytes=peak_memory_bytes,
                )
        finally:
            self._generation_lock.release()


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file")
    return resolved


def _resolve_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_startup_receipt(
    *,
    config_path: Path,
    gpt_checkpoint: Path,
    speaker: Path,
    tokenizer: Path | None,
    model_id: str,
    voice_id: str,
    device: str | None,
    fp16: bool,
    seed: int,
    max_text_tokens: int,
    interval_silence: int,
    generation_kwargs: Mapping[str, Any],
    artifact_set_id: str | None,
    artifact_set_sha256: str | None,
) -> dict[str, Any]:
    """Describe the fixed live configuration without retaining local paths."""

    if bool(artifact_set_id) != bool(artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    if artifact_set_id is not None and not PUBLIC_ID_RE.fullmatch(artifact_set_id):
        raise ValueError("artifact set id must be a bounded public identifier")
    if artifact_set_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", artifact_set_sha256):
        raise ValueError("artifact set sha256 must be a lowercase SHA-256 digest")

    def artifact(path: Path) -> dict[str, Any]:
        resolved = _resolve_file(path, "startup artifact")
        return {"sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}

    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "runtime_id": "indextts2_openai_compatible_http",
        "model_id": model_id,
        "voice_id": voice_id,
        "device": device,
        "fp16": fp16,
        "seed": seed,
        "max_text_tokens": max_text_tokens,
        "interval_silence": interval_silence,
        "generation_kwargs": dict(sorted(generation_kwargs.items())),
        "artifacts": {
            "config": artifact(config_path),
            "gpt_checkpoint": artifact(gpt_checkpoint),
            "speaker": artifact(speaker),
            **({"tokenizer": artifact(tokenizer)} if tokenizer is not None else {}),
        },
        "boundary": (
            "This receipt proves the fixed startup inputs observed by the reference server. "
            "It does not hash every dependency under model_dir or prove loader behavior, host trust, quality, or rights."
        ),
    }
    if artifact_set_id is not None:
        receipt["artifact_set_id"] = artifact_set_id
        receipt["artifact_set_sha256"] = artifact_set_sha256
    return receipt


def write_startup_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    """Publish a canonical no-overwrite receipt and return its byte digest."""

    payload = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return hashlib.sha256(payload).hexdigest()


def validate_generation_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before model loading when startup sampling controls are invalid."""

    unknown = sorted(set(values) - {"num_beams", "temperature", "top_k", "top_p"})
    if unknown:
        raise ValueError(f"unsupported generation setting(s): {', '.join(unknown)}")
    result = dict(values)
    for name in ("top_k", "num_beams"):
        value = result.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    top_p = result.get("top_p")
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError("top_p must be finite and in the interval (0, 1]")
    temperature = result.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    return result


class IndexTTS2SpeechEngine:
    """Load one operator-selected checkpoint and speaker prompt exactly once."""

    def __init__(
        self,
        *,
        config_path: Path,
        model_dir: Path,
        gpt_checkpoint: Path,
        speaker: Path,
        tokenizer: Path | None,
        device: str | None,
        fp16: bool,
        seed: int,
        max_text_tokens: int,
        interval_silence: int,
        generation_kwargs: Mapping[str, Any],
    ) -> None:
        self.config_path = _resolve_file(config_path, "config")
        self.model_dir = _resolve_directory(model_dir, "model directory")
        self.gpt_checkpoint = _resolve_file(gpt_checkpoint, "GPT checkpoint")
        self.speaker = _resolve_file(speaker, "speaker prompt")
        self.tokenizer = _resolve_file(tokenizer, "tokenizer") if tokenizer else None
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(max_text_tokens, bool) or not isinstance(max_text_tokens, int) or max_text_tokens <= 0:
            raise ValueError("max text tokens must be positive")
        if isinstance(interval_silence, bool) or not isinstance(interval_silence, int) or interval_silence < 0:
            raise ValueError("interval silence must be nonnegative")
        self.device = device
        self.fp16 = fp16
        self.seed = seed
        self.max_text_tokens = max_text_tokens
        self.interval_silence = interval_silence
        self.generation_kwargs = validate_generation_kwargs(generation_kwargs)
        self._engine = self._load_engine()

    def _load_engine(self) -> Any:
        omega_conf = importlib.import_module("omegaconf").OmegaConf
        index_tts_2 = importlib.import_module("indextts.infer_v2").IndexTTS2
        config = omega_conf.load(self.config_path)
        config.gpt_checkpoint = str(self.gpt_checkpoint)
        if self.tokenizer is not None:
            if "dataset" not in config or "bpe_model" not in config["dataset"]:
                raise KeyError("config does not contain dataset.bpe_model to override")
            config.dataset["bpe_model"] = str(self.tokenizer)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            omega_conf.save(config, str(temporary_path))
            return index_tts_2(
                cfg_path=str(temporary_path),
                model_dir=str(self.model_dir),
                device=self.device,
                use_fp16=self.fp16,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def generate(self, text: str, output_path: Path) -> Mapping[str, int | float]:
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        self._engine.infer(
            spk_audio_prompt=str(self.speaker),
            text=text,
            output_path=str(output_path),
            interval_silence=self.interval_silence,
            max_text_tokens_per_segment=self.max_text_tokens,
            **self.generation_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "generation_seconds": time.perf_counter() - started,
            **(
                {"peak_memory_bytes": int(torch.cuda.max_memory_allocated())}
                if torch.cuda.is_available()
                else {}
            ),
        }


class SpeechHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class IPv6SpeechHTTPServer(SpeechHTTPServer):
    address_family = socket.AF_INET6


def create_http_server(
    host: str, port: int, handler: type[BaseHTTPRequestHandler]
) -> SpeechHTTPServer:
    try:
        is_ipv6 = ipaddress.ip_address(host).version == 6
    except ValueError:
        is_ipv6 = False
    server_type = IPv6SpeechHTTPServer if is_ipv6 else SpeechHTTPServer
    return server_type((host, port), handler)


def build_handler(service: SpeechService) -> type[BaseHTTPRequestHandler]:
    """Build a handler bound to one immutable service configuration."""

    class SpeechRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "IndexTTS2Reference/0.1"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(service.config.request_timeout_seconds)

        def _request_id(self) -> str:
            return f"req_{uuid.uuid4().hex}"

        def _send_headers(
            self,
            status: HTTPStatus,
            content_type: str,
            length: int,
            request_id: str,
            extra_headers: Mapping[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Request-ID", request_id)
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any], request_id: str) -> None:
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self._send_headers(status, "application/json; charset=utf-8", len(body), request_id)
            self.wfile.write(body)

        def _send_error(self, error: ApiError, request_id: str) -> None:
            self._send_json(
                error.status,
                {"error": {"code": error.code, "message": error.message, "type": "invalid_request_error"}},
                request_id,
            )

        def _authorize(self) -> None:
            expected = service.config.api_key
            if expected is None:
                return
            supplied = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not supplied.startswith(prefix) or not hmac.compare_digest(supplied[len(prefix) :], expected):
                raise ApiError(HTTPStatus.UNAUTHORIZED, "invalid_api_key", "invalid or missing API key")

        def do_GET(self) -> None:
            request_id = self._request_id()
            try:
                self._authorize()
                if self.path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"status": "ok"}, request_id)
                elif self.path == "/readyz":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "model": service.config.model_id,
                            "status": "ready",
                            "voice": service.config.voice_id,
                            **(
                                {"startup_receipt_sha256": service.config.startup_receipt_sha256}
                                if service.config.startup_receipt_sha256
                                else {}
                            ),
                        },
                        request_id,
                    )
                else:
                    raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "route not found")
            except ApiError as error:
                self._send_error(error, request_id)

        def do_POST(self) -> None:
            request_id = self._request_id()
            try:
                self._authorize()
                if self.path != "/v1/audio/speech":
                    raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "route not found")
                media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
                if media_type != "application/json":
                    raise ApiError(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        "unsupported_media_type",
                        "Content-Type must be application/json",
                    )
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise ApiError(HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required")
                try:
                    length = int(raw_length)
                except ValueError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_length", "Content-Length is invalid") from error
                if length <= 0:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_length", "Content-Length must be positive")
                if length > service.config.max_body_bytes:
                    raise ApiError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "body_too_large",
                        "request body exceeds the configured limit",
                    )
                payload = decode_json_body(self.rfile.read(length))
                request = parse_speech_request(payload, service.config)
                synthesis = service.synthesize(request)
                metric_headers = {
                    "X-Generation-Seconds": format(synthesis.generation_seconds, ".9f"),
                    **(
                        {"X-Peak-Memory-Bytes": str(synthesis.peak_memory_bytes)}
                        if synthesis.peak_memory_bytes is not None
                        else {}
                    ),
                }
                self._send_headers(
                    HTTPStatus.OK,
                    "audio/wav",
                    len(synthesis.audio),
                    request_id,
                    metric_headers,
                )
                self.wfile.write(synthesis.audio)
            except ApiError as error:
                self._send_error(error, request_id)
            except TimeoutError:
                self._send_error(
                    ApiError(HTTPStatus.REQUEST_TIMEOUT, "request_timeout", "request body timed out"),
                    request_id,
                )
            except Exception:
                LOG.exception("synthesis failed for request %s", request_id)
                self._send_error(
                    ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "synthesis_failed", "speech synthesis failed"),
                    request_id,
                )

        def log_message(self, message: str, *args: Any) -> None:
            LOG.info("client=%s %s", self.client_address[0], message % args)

    return SpeechRequestHandler


def _is_loopback_bind(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def read_api_key(host: str, env_name: str | None) -> str | None:
    """Require environment-delivered authentication for any non-loopback bind."""

    if env_name is not None and not ENV_NAME_RE.fullmatch(env_name):
        raise ValueError("API key environment variable name is invalid")
    api_key = os.environ.get(env_name, "") if env_name else ""
    if env_name and not api_key:
        raise ValueError(f"API key environment variable {env_name!r} is unset or empty")
    if not _is_loopback_bind(host) and not api_key:
        raise ValueError("non-loopback binding requires --api-key-env with a nonempty environment value")
    return api_key or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpt-checkpoint", type=Path, required=True)
    parser.add_argument("--speaker", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model-id", default="indextts2-finetuned")
    parser.add_argument("--voice-id", default="instavar-reference")
    parser.add_argument("--device")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--interval-silence", type=int, default=200)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--max-input-chars", type=int, default=4_000)
    parser.add_argument("--max-body-bytes", type=int, default=16_384)
    parser.add_argument("--max-audio-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-receipt", type=Path)
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--artifact-set-sha256")
    parser.add_argument(
        "--api-key-env",
        help="Environment variable holding the bearer token. Never pass the token as an argument.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    api_key = read_api_key(args.host, args.api_key_env)
    generation_kwargs = {
        key: value
        for key, value in {
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
        }.items()
        if value is not None
    }
    engine = IndexTTS2SpeechEngine(
        config_path=args.config,
        model_dir=args.model_dir,
        gpt_checkpoint=args.gpt_checkpoint,
        speaker=args.speaker,
        tokenizer=args.tokenizer,
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
        max_text_tokens=args.max_text_tokens,
        interval_silence=args.interval_silence,
        generation_kwargs=generation_kwargs,
    )
    receipt_sha256 = None
    if args.startup_receipt is not None:
        receipt = build_startup_receipt(
            config_path=args.config,
            gpt_checkpoint=args.gpt_checkpoint,
            speaker=args.speaker,
            tokenizer=args.tokenizer,
            model_id=args.model_id,
            voice_id=args.voice_id,
            device=args.device,
            fp16=args.fp16,
            seed=args.seed,
            max_text_tokens=args.max_text_tokens,
            interval_silence=args.interval_silence,
            generation_kwargs=generation_kwargs,
            artifact_set_id=args.artifact_set_id,
            artifact_set_sha256=args.artifact_set_sha256,
        )
        receipt_sha256 = write_startup_receipt(args.startup_receipt, receipt)
    elif args.artifact_set_id or args.artifact_set_sha256:
        raise ValueError("artifact set binding requires --startup-receipt")
    config = SpeechServerConfig(
        model_id=args.model_id,
        voice_id=args.voice_id,
        max_input_chars=args.max_input_chars,
        max_body_bytes=args.max_body_bytes,
        max_audio_bytes=args.max_audio_bytes,
        request_timeout_seconds=args.request_timeout_seconds,
        api_key=api_key,
        startup_receipt_sha256=receipt_sha256,
    )
    config.validate()
    server = create_http_server(args.host, args.port, build_handler(SpeechService(engine, config)))
    LOG.info(
        "ready host=%s port=%d model=%s voice=%s auth=%s",
        args.host,
        args.port,
        config.model_id,
        config.voice_id,
        "required" if config.api_key else "disabled-loopback-only",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutdown requested")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
