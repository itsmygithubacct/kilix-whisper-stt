"""The provider CLI against a stand-in faster-whisper; no model, no network."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

from kilix_whisper_stt import cli


class FakeModel:
    loads: list = []

    def __init__(self, path, **options):
        FakeModel.loads.append((path, options))
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append((samples, options))
        if samples == b"\xff\xff":
            raise RuntimeError("decoder exploded")
        words = [" hello ", "", " world"] if samples else []
        return iter(SimpleNamespace(text=word) for word in words), None


def model_dir(root: str, skip: str | None = None) -> str:
    path = tempfile.mkdtemp(dir=root)
    for name in cli.REQUIRED_FILES:
        if name != skip:
            with open(os.path.join(path, name), "w") as handle:
                handle.write("x")
    return path


class ProviderTests(unittest.TestCase):
    def setUp(self):
        FakeModel.loads = []
        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeModel
        self.enterContext(mock.patch.dict(sys.modules, {"faster_whisper": fake}))
        self.enterContext(mock.patch.dict(os.environ, {}, clear=False))
        # Identity instead of numpy, so the stand-in sees the raw PCM.
        self.enterContext(mock.patch.object(cli, "samples_of", lambda pcm: pcm))
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())

    def serve(self, requests: bytes, model: str | None = None):
        args = cli.build_parser().parse_args(
            ["serve", "--model", model or model_dir(self.tmp), "--threads", "3"])
        out = io.BytesIO()
        code = cli.serve(args, stdin=io.BytesIO(requests), out=out)
        return code, [json.loads(line) for line in out.getvalue().splitlines()]

    def test_serve_loads_once_and_answers_each_request(self):
        requests = (b'{"pcm_bytes": 4}\n\x01\x00\x02\x00' + b'{"pcm_bytes": 0}\n'
                    + b'{"pcm_bytes": 2}\n\xff\xff' + b'{"pcm_bytes": 2}\n\x01\x00')
        code, replies = self.serve(requests)
        self.assertEqual(code, 0)
        self.assertEqual(replies[0], {"ready": True, "version": cli.__version__})
        self.assertEqual(replies[1:], [{"text": "hello world"}, {"text": ""},
                                       {"error": "transcription failed: decoder exploded"},
                                       {"text": "hello world"}])
        self.assertEqual(len(FakeModel.loads), 1)
        path, options = FakeModel.loads[0]
        self.assertEqual(options, {"device": "cpu", "compute_type": "int8", "cpu_threads": 3,
                                   "local_files_only": True})
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_decode_options_are_fixed(self):
        model = FakeModel("m")
        cli.transcribe_pcm(model, b"\x01\x00", 5)
        self.assertEqual(model.calls[0][1], {"language": "en", "beam_size": 5, "vad_filter": False,
                                             "condition_on_previous_text": False})

    def test_protocol_errors_stop_the_server(self):
        for request in (b"not json\n", b'{"pcm_bytes": 3}\n\x00\x00\x00',
                        b'{"pcm_bytes": -2}\n', b'{"pcm_bytes": true}\n',
                        b'{"pcm_bytes": 2, "x": 1}\n\x00\x00', b"[2]\n",
                        b'{"pcm_bytes": %d}\n' % (cli.MAX_PCM_BYTES + 2),
                        b'{"pcm_bytes": 2}',
                        b"{" + b" " * cli.MAX_HEADER_BYTES + b"}\n"):
            with self.subTest(request=request[:40]):
                code, replies = self.serve(request + b'{"pcm_bytes": 2}\n\x01\x00')
                self.assertEqual(code, 2)
                self.assertEqual(len(replies), 2)
                self.assertIn("error", replies[1])

    def test_a_truncated_body_stops_the_server(self):
        code, replies = self.serve(b'{"pcm_bytes": 4}\n\x00\x00')
        self.assertEqual((code, replies[1:]), (2, [{"error": "standard input ended inside a request"}]))

    def test_missing_or_incomplete_model_is_refused_without_loading(self):
        code, replies = self.serve(b"", model=os.path.join(self.tmp, "small.en"))
        self.assertEqual((code, replies), (1, [{"error": "model directory not found: "
                                                + os.path.join(self.tmp, "small.en")}]))
        incomplete = model_dir(self.tmp, skip="tokenizer.json")
        code, replies = self.serve(b"", model=incomplete)
        self.assertEqual(code, 1)
        self.assertIn("lacks tokenizer.json", replies[0]["error"])
        self.assertEqual(FakeModel.loads, [])

    def test_transcribe_prints_one_line(self):
        audio = os.path.join(self.tmp, "a.wav")
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x01\x00" * 160)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["transcribe", "--model", model_dir(self.tmp), "--audio", audio])
        self.assertEqual((code, out.getvalue(), err.getvalue()), (0, "hello world\n", ""))

    def test_transcribe_rejects_other_formats(self):
        audio = os.path.join(self.tmp, "a.wav")
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00" * 64)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["transcribe", "--model", model_dir(self.tmp), "--audio", audio])
        self.assertEqual((code, out.getvalue()), (1, ""))
        self.assertIn("16 kHz, mono, 16-bit", err.getvalue())
        self.assertEqual(FakeModel.loads, [])

    def test_bounds_on_threads_and_beam(self):
        for flag, value in (("--threads", "0"), ("--threads", "65"), ("--beam", "11"),
                            ("--beam", "x")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.build_parser().parse_args(["serve", "--model", "m", flag, value])
                self.assertEqual(raised.exception.code, 2)


class ProtocolStdoutTests(unittest.TestCase):
    def test_native_prints_go_to_stderr(self):
        # In a child, so redirecting fd 1 cannot disturb the test runner.
        import subprocess
        code = ("import os\n"
                "from kilix_whisper_stt import cli\n"
                "out = cli._protocol_stdout()\n"
                "os.write(1, b'noise\\n')\n"
                "out.write(b'reply\\n')\n")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"reply\n", b"noise\n"))


if __name__ == "__main__":
    unittest.main()
