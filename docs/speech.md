# Speech (TTS) Usage

Everything about `/v1/audio/speech` beyond the quickstart in the README: voices,
instructions, streaming knobs, cloning, and the per-checkpoint differences.
Setup and server commands live in the [README](../README.md); runnable scripts
in [`examples/`](../examples/).

## Voices by checkpoint

The loaded checkpoint decides what `voice` and `instructions` mean — mirroring
mlx-audio's own mapping:

| Checkpoint | `voice` | `instructions` |
| --- | --- | --- |
| CustomVoice (0.6B / 1.7B) | preset speaker name — `GET /v1/audio/voices` lists them (9 presets: vivian, ryan, aiden, …) | optional emotion/style prompt (1.7B only — rejected with 400 on 0.6B) |
| Base (1.7B) | **cloning object** — `{"ref_audio": <base64>, "ref_text": "…"}` | — |
| VoiceDesign (1.7B) | rejected (no presets) | **required** — the voice description itself, e.g. "A cheerful young female voice with high pitch and energetic tone" |
| VoxCPM2 (2.5B-class, 48 kHz) | `"default"` (zero-shot) **or** a cloning object `{"ref_audio": <base64>}` — `ref_text` accepted but unused (VoxCPM2 conditions on the clip alone) | **voice design** — a description of the speaker, e.g. "A young woman with a warm and gentle voice" |

`language` forces a language (default auto; not supported on VoxCPM2, which is
multilingual natively); `speed` must be 1.0 for now. VoxCPM2 output is 48 kHz
mono (Qwen3-TTS is 24 kHz); the streaming response headers carry the rate.

## Streaming

`"stream": true` switches from a buffered WAV to chunked raw PCM (24 kHz
16-bit mono, `X-Audio-*` response headers):

- first audio typically lands in **~0.1 s** (single-codec-frame first chunk,
  warm voice — per-voice prefix cache);
- `streaming_interval` (default 0.5 s) sets the steady chunk size;
- `streaming_initial_interval` (default 0.08 s) sets the first-chunk size —
  raise to 0.2 s for a chunkier first beat.

```sh
curl -N -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d '{"input": "Hello.", "voice": "vivian", "stream": true}' \
    http://127.0.0.1:8000/v1/audio/speech -o speech.pcm
```

## Voice cloning (Base checkpoints)

`voice` carries a short reference clip and its transcript:

```sh
vllm-omni-mlx serve mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit --omni --api-key demo
REF=$(base64 -i reference.wav)
curl -H 'Authorization: Bearer demo' -H 'Content-Type: application/json' \
    -d "{\"input\": \"Any text in the cloned voice.\", \"voice\": {\"ref_audio\": \"$REF\", \"ref_text\": \"transcript of the reference clip\"}}" \
    http://127.0.0.1:8000/v1/audio/speech -o cloned.wav
```

The clip is decoded and resampled to 24 kHz mono server-side (any format
miniaudio/ffmpeg reads); keep it 0.5–30 s of clean speech. Cloning streams too
(`stream: true` + the voice object) on the same fast path as presets — the
loop is token-exact against mlx-audio's ICL path; chunked audio differs from
the buffered WAV at waveform level by nature (the vocoder is stateful).
Cloning pays per-request setup (reference re-encode + ICL prefill), so its
time-to-first-audio is ~0.35–0.7 s vs ~0.1 s for presets — a known
optimization target.

## VoxCPM2

```sh
vllm-omni-mlx serve mlx-community/VoxCPM2-4bit --omni        # needs [tts]
vllm-omni-mlx tts --model mlx-community/VoxCPM2-4bit \
    --text "Hello from VoxCPM2." --out out.wav                # or --instruct / --ref-audio clip.wav
```

A tokenizer-free AR + diffusion model (MiniCPM4 backbone → CFM solver →
48 kHz AudioVAE), 30+ languages, no speaker presets: `voice: "default"`
speaks zero-shot, `instructions` designs a voice, and cloning rides the same
`voice` object as Base (`ref_audio` required, `ref_text` unused). `language`
is rejected — the model is multilingual natively.

Generation runs through a vendored compiled loop (the whole CFM solver in one
fixed-shape trace, KV-as-arrays LM steps — bitwise-reproducible against the
library under a fixed seed); `VLLM_OMNI_VOXCPM2_EAGER=1` serves the plain
mlx-audio path. mlx-audio's generate is single-yield, so `stream: true`
delivers interval-sized chunks of the **finished** buffer — first audio lands
when synthesis completes; incremental per-patch decode is follow-up work
(#88). Steady RTF on M4 4-bit ≈ 2 eager, improved by the compiled loop;
further levers (timestep knee, DiT quantization) are tracked on #88.

## Transcription (ASR)

```sh
pip install 'vllm-omni-mlx[asr]'
vllm-omni-mlx serve --asr-model mlx-community/Qwen3-ASR-1.7B-4bit      # combine with --tts-model for one process
curl -F file=@clip.wav http://127.0.0.1:8000/v1/audio/transcriptions
```

`POST /v1/audio/transcriptions` takes a multipart upload (25 MiB cap; any
container mlx-audio's decoder reads, resampled to 16 kHz mono) and the OpenAI
fields it can honor:

| field | behavior |
| --- | --- |
| `file` | required audio upload |
| `language` | ISO-639-1/3 code (`en`, `zh`, `yue`) or language name, checked against the checkpoint's list; omitted → auto-detect |
| `prompt` | context text, becomes the decoder's system prompt (the static prompt head) |
| `hotwords` | extension: comma/newline-separated vocabulary, folded into the same head |
| `response_format` | `json` (default), `text`, `verbose_json` (language, duration, segments, usage); `srt`/`vtt` are a 400 — the decoder family has no word timestamps yet |
| `temperature` | `[0, 2]`, default 0 (greedy) |
| `stream` | `true` → Server-Sent Events, below; `verbose_json` is a 400 with it |

Text fields sent as file parts, an oversized `Content-Length`, and non-audio
payloads are 400/413s before any generation starts.

**Streaming.** `stream=true` returns `text/event-stream` in the OpenAI shape:
one `transcript.text.delta` per decoded token, then a closing
`transcript.text.done` with the full `text`, `language`, `duration` and
`usage`. A bad request is still a plain JSON 400 (validation and decode happen
before the first byte). The service lock is held for the whole stream; a
client disconnect closes the generator and releases it at the next token.
Deltas are decoded token by token, so a character whose UTF-8 bytes span two
tokens (rare CJK) arrives as replacement characters — incremental
detokenization belongs to the vendored loop (#68 task 5).

**Latency probe.** `scripts/latency_probe.py --asr --asr-file clip.wav`
reports, per turn, TTFT (upload + decode + audio-encoder prefill + first
token), ITL p50/p95/max between deltas, TPOT, and RTF. A first reading on an
M1 Pro / 16 GB (Qwen3-ASR-1.7B-4bit, one 4.5 s clip, warm, not a cooled A/B):
TTFT 0.24–0.37 s, TPOT ≈ 8–10 ms/token, RTF ≈ 0.1; adding a ~400-token static
`prompt` raised steady TTFT to ≈ 0.72–0.76 s — that is the cost the
prompt-prefix cache (#68 task 5) is meant to remove.

**Round-trip gate.** `scripts/asr_roundtrip.py` synthesizes sentences with the
TTS checkpoint, transcribes them with the ASR checkpoint and fails above a mean
word-error-rate (default 0.15) — a catastrophic-breakage detector on either
side, not a WER benchmark.

## One-shot, no server

```sh
vllm-omni-mlx tts --voice ryan --text "Hello from the CLI." --out out.wav
```

## Performance & verification

Measured numbers (first-audio, sustained RTF, per-checkpoint tables) and the
public reproduction protocol live in
[issue #84](https://github.com/ThinkFlowLab/vllm-omni-mlx/issues/84); the
benchmark scripts are `scripts/bench_all_checkpoints.py`,
`scripts/bench_prefix_cache.py`, `scripts/bench_stream_rtf.py`. Correctness
gates: HNR floors (harmonics-to-noise, the catastrophic-decode detector) plus
token-exactness harnesses in `tests/`.
