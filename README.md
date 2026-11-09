# kilix-whisper-stt

`kilix-whisper-stt` is the Whisper speech-recognition provider for Kilix
dictation. It runs a CTranslate2 conversion of OpenAI's Whisper through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.0, int8 on the
CPU. faster-whisper and its dependencies are pinned in `uv.lock`, and Kilix
installs them into a private environment, never into the system Python.

**Status:** not part of the Plebian-OS image. Kilix installs a pinned commit of
this provider (`15ef23b3` for Plebian-OS 0.2.3) into the user's data directory
on first use, with `kilix voice whisper`. The model is a separate Kilix Content
asset (`faster-whisper-small-en`, or `faster-whisper-medium-en`) that downloads
only after its licence screen, through `kilix stt --install`; no weights ship
in the ISO.

## What it does

- Transcribes one WAV file to a single line of text (`transcribe`).
- Serves a dictation session over standard input and output with the model
  loaded once (`serve`).
- Drops segments Whisper itself rates as probably not speech, so a click or a
  pop gives an empty transcript.
- Never downloads anything: `--model` must be a local model directory.

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

| Command | What it does |
|---|---|
| `kilix-whisper-stt transcribe --model DIR --audio F.wav` | Print the transcript of one 16 kHz mono 16-bit WAV file. |
| `kilix-whisper-stt serve --model DIR` | Load the model once and answer requests on stdout. |
| `--threads N` | CPU threads (default 4). |
| `--beam N` | Beam size (default 5). |
| `make test` / `make lint` | Unit tests against a stand-in faster-whisper; byte-compile. |

`transcribe` prints the transcript of one 16 kHz, mono, 16-bit WAV file as a
single line. It exits 1 with a one-line reason on stderr for a model or audio
error, and 2 for a usage error.

## How it works

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

```
Kilix Voice dictation              kilix-whisper-stt serve --model DIR
  microphone -> end-of-speech  -->  {"pcm_bytes": N} + N bytes PCM
                               <--  {"text": "..."}
```

Whisper writes words even for silence or a microphone pop ("You"), but it
rates each segment's chance of holding no speech. Segments rated 0.6 or higher
are dropped, so a pop returns an empty transcript and Kilix keeps listening. On
25 dictated prompts, no speech was dropped. Silence, noise and a pop returned
nothing.

Decoding is fixed for dictation: English, beam 5 by default, no voice-activity
filter, and no conditioning on previous text. Kilix detects the end of speech
itself.

Kilix's installer (`scripts/install-kilix-whisper-stt.sh` in kilix) fetches the
pinned commit, builds it with `uv sync --locked` into a generation directory
under `$KILIX_DATA_HOME/voice/whisper/`, and publishes
`voice/whisper/current/bin/kilix-whisper-stt`, the path Kilix Voice resolves.

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

## Documentation

This repository has no separate `docs/` directory; this README and
`src/kilix_whisper_stt/cli.py` are its documentation.

## Related repositories

- [kilix-voice](https://github.com/itsmygithubacct/kilix-voice#readme): Kilix
  Voice dictation, which runs this provider.
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme): the
  catalog that records, licence-gates and downloads the Whisper models.
- [kilix](https://github.com/itsmygithubacct/kilix#readme): pins and installs
  this provider.
- [kilix-piper-tts](https://github.com/itsmygithubacct/kilix-piper-tts#readme):
  the matching read-aloud provider.
- [kilix-transcribe](https://github.com/itsmygithubacct/kilix-transcribe#readme):
  the separate long-form transcription provider design.

## Part of Plebian-OS

kilix-whisper-stt is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
Kilix installs a pinned commit of it on first use, when a user selects Whisper
dictation. The map of every repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## Licence

GPL-3.0-or-later. faster-whisper is MIT-licensed. The Whisper weights are
OpenAI's, under MIT. Kilix shows their licence and the model card's cautions
before the user downloads them.
