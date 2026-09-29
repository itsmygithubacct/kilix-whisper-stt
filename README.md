# kilix-whisper-stt

`kilix-whisper-stt` is the Whisper speech-recognition provider for Kilix
dictation. It runs a CTranslate2 conversion of OpenAI's Whisper through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.0, int8 on the
CPU. faster-whisper and its dependencies are pinned in `uv.lock`, and Kilix
installs them into a private environment, never into the system Python.

The provider holds no model. Kilix Content downloads the model the user
chooses, after its licence screen, and Kilix Voice hands this provider the
directory. The default is `faster-whisper-small-en`: Systran's conversion of
Whisper small.en, pinned to one revision and exact SHA-256 digests. Nothing
here uses the network. A `--model` that is not a local directory is refused,
never taken for a model name to download.

## Commands

```sh
uv sync --locked
uv run kilix-whisper-stt transcribe --model DIR --audio speech.wav
uv run kilix-whisper-stt serve --model DIR [--threads 4] [--beam 5]
```

`transcribe` prints the transcript of one 16 kHz, mono, 16-bit WAV file as a
single line. It exits 1 with a one-line reason on stderr for a model or audio
error, and 2 for a usage error.

`serve` loads the model once for a dictation session. It answers on standard
output:

1. At start it writes `{"ready": true, "version": "0.1.0"}` once the model has
   loaded, or `{"error": "..."}` and exits 1.
2. Each request is a header line `{"pcm_bytes": N}`, followed by exactly N bytes
   of 16 kHz, mono, 16-bit little-endian PCM. N is even and at most ten minutes
   of audio.
3. It answers each request with `{"text": "..."}`. If recognition fails it
   answers `{"error": "..."}` and keeps serving.
4. A malformed header or a truncated body gets one `{"error": "..."}` reply and
   exit status 2, because the stream can no longer be trusted.
5. End of input exits 0.

File descriptor 1 is redirected to stderr, so anything a native library prints
cannot corrupt a reply.

Decoding is fixed for dictation: English, beam 5 by default, no voice-activity
filter, and no conditioning on previous text. Kilix detects the end of speech
itself.

## Measured

On an i7-9850H with 4 threads, over 25 dictated prompts (126 s of audio),
Whisper small.en had:

- 2.3% word errors
- about 1.3 s per sentence with the model loaded
- a model load of about 0.6 s
- a peak of about 1.6 GiB of RAM

## Development

`make test` runs the tests against a stand-in faster-whisper, so no model or
installed dependencies are needed. `make lint` byte-compiles the sources.

## Licence

GPL-3.0-or-later. faster-whisper is MIT-licensed. The Whisper weights are
OpenAI's, under MIT. Kilix shows their licence and the model card's cautions
before the user downloads them.
