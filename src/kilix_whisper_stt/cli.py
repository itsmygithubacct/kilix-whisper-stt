"""Command-line interface for the Kilix Whisper provider.

``transcribe`` runs one WAV file and prints its transcript. ``serve`` keeps
one model loaded for a dictation session and answers length-prefixed PCM
requests on standard input, so a sentence costs only its decode, not a model
load. Neither ever touches the network: the model is a local directory, and
a path that is not one is refused rather than taken for a model name to
download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave

from . import __version__

RATE = 16000
# Ten minutes of 16 kHz 16-bit mono: the longest turn a caller may send.
MAX_PCM_BYTES = RATE * 2 * 600
MAX_HEADER_BYTES = 256
MAX_THREADS = 64
MAX_BEAM = 10
# The files faster-whisper needs from a CTranslate2 Whisper conversion.
REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
# Whisper writes words even for silence or a noise burst ("You", "Thank
# you."), but it also rates each segment's chance of holding no speech. On
# dictated speech that rating was 0.01-0.05; on a microphone pop or silence,
# 0.85-0.90. Segments at or above this are dropped, so a pop reads as nothing
# and the caller keeps listening.
NO_SPEECH_THRESHOLD = 0.6


class ProviderError(Exception):
    """A failure reported to the caller as one line, never a traceback."""


def _check_model_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise ProviderError(f"model directory not found: {path}")
    missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise ProviderError(f"model directory {path} lacks {', '.join(missing)}")
    return path


def load_model(path: str, threads: int):
    """Load a local CTranslate2 Whisper model for int8 CPU inference."""
    _check_model_dir(path)
    # Belt and braces: the path is a checked local directory, so nothing is
    # looked up, but the hub client must not reach out even to check.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise ProviderError(f"faster-whisper is not installed: {error}") from error
    try:
        return WhisperModel(path, device="cpu", compute_type="int8",
                            cpu_threads=threads, local_files_only=True)
    except Exception as error:  # the library raises many types; report one line
        raise ProviderError(f"cannot load the model in {path}: {error}") from error


def samples_of(pcm: bytes):
    """16-bit little-endian mono PCM as the float32 array Whisper expects."""
    import numpy

    return numpy.frombuffer(pcm, dtype="<i2").astype(numpy.float32) / 32768.0


def transcribe_pcm(model, pcm: bytes, beam: int) -> str:
    if not pcm:
        return ""
    try:
        segments, _info = model.transcribe(
            samples_of(pcm), language="en", beam_size=beam, vad_filter=False,
            condition_on_previous_text=False)
        return " ".join(text for text in (segment.text.strip() for segment in segments
                                          if segment.no_speech_prob < NO_SPEECH_THRESHOLD)
                        if text)
    except Exception as error:
        raise ProviderError(f"transcription failed: {error}") from error


def read_wav(path: str) -> bytes:
    try:
        with wave.open(path, "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                raise ProviderError(f"{path} must be 16 kHz, mono, 16-bit PCM")
            if wav.getnframes() * 2 > MAX_PCM_BYTES:
                raise ProviderError(f"{path} is longer than ten minutes")
            return wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise ProviderError(f"cannot read {path}: {error}") from error


def _protocol_stdout():
    """Keep the protocol stream private: fd 1 now points at stderr.

    Anything the native libraries print lands in the diagnostics, never in
    the middle of a reply the caller is parsing.
    """
    sys.stdout.flush()
    private = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(private, "wb", buffering=0)


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise ProviderError("standard input ended inside a request")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _reply(out, value: dict) -> None:
    out.write((json.dumps(value, ensure_ascii=True) + "\n").encode("ascii"))


def serve(args, stdin=None, out=None) -> int:
    stdin = stdin if stdin is not None else sys.stdin.buffer
    out = out if out is not None else _protocol_stdout()
    try:
        model = load_model(args.model, args.threads)
    except ProviderError as error:
        _reply(out, {"error": str(error)})
        return 1
    _reply(out, {"ready": True, "version": __version__})
    while True:
        header = stdin.readline(MAX_HEADER_BYTES + 1)
        if not header:
            return 0
        try:
            if len(header) > MAX_HEADER_BYTES or not header.endswith(b"\n"):
                raise ProviderError("request header is not one short line")
            try:
                request = json.loads(header)
            except ValueError as error:
                raise ProviderError("request header is not JSON") from error
            size = request.get("pcm_bytes") if isinstance(request, dict) else None
            if (not isinstance(request, dict) or set(request) != {"pcm_bytes"}
                    or type(size) is not int or not 0 <= size <= MAX_PCM_BYTES or size % 2):
                raise ProviderError(
                    f"expected {{\"pcm_bytes\": N}} with even N up to {MAX_PCM_BYTES}")
            pcm = _read_exact(stdin, size)
        except ProviderError as error:
            # The stream is out of step: say why and stop, never guess.
            _reply(out, {"error": str(error)})
            return 2
        try:
            _reply(out, {"text": transcribe_pcm(model, pcm, args.beam)})
        except ProviderError as error:
            _reply(out, {"error": str(error)})


def _bounded(low: int, high: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be {low}..{high}")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kilix-whisper-stt",
        description="Local Whisper speech recognition for Kilix dictation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("transcribe", "print the transcript of one 16 kHz mono WAV file"),
                       ("serve", "keep the model loaded and answer PCM requests on stdin")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--model", required=True, help="local CTranslate2 Whisper model directory")
        command.add_argument("--threads", type=_bounded(1, MAX_THREADS), default=4)
        command.add_argument("--beam", type=_bounded(1, MAX_BEAM), default=5)
        if name == "transcribe":
            command.add_argument("--audio", required=True, help="16 kHz mono 16-bit WAV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args)
    try:
        pcm = read_wav(args.audio)
        model = load_model(args.model, args.threads)
        text = transcribe_pcm(model, pcm, args.beam)
    except ProviderError as error:
        print(f"kilix-whisper-stt: {error}", file=sys.stderr)
        return 1
    print(text)
    return 0
