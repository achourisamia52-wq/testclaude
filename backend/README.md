# Zatoona backend

Audio in, text out. Two endpoints share one pipeline: speech-to-text, then Claude.

| Endpoint | Input | Output |
|---|---|---|
| `POST /v1/gist` | `audio` (a voice note someone sent the user) | `transcript`, `language`, `gist`, `action_items[{text, due}]`, `replies[3]` |
| `POST /v1/compose` | `audio` **or** `transcript`, plus `tone` (`formal`/`casual`) and `language` (`auto` or a 2-letter code) | `transcript`, `message` |
| `GET /healthz` | – | `{"status": "ok"}` |

Requests are multipart forms with `Authorization: Bearer <token>`, where the
token is one of `APP_API_KEYS`. To switch tone or language in the keyboard, send
the `transcript` from the first `/v1/compose` response back instead of the audio.

Audio is held in memory for the request and never written to disk.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
cp .env.example .env
```

Fill in `.env`, then:

```bash
.venv/bin/uvicorn app.main:app --reload
```

```bash
curl -H "Authorization: Bearer change-me" -F audio=@note.ogg localhost:8000/v1/gist
```

## Test

```bash
.venv/bin/pytest
```

Tests use fakes and a mocked HTTP transport; they need no keys and no network.

## Speech-to-text provider

`app/stt.py` talks to any provider with an OpenAI-compatible
`/audio/transcriptions` endpoint, set through `STT_BASE_URL`, `STT_API_KEY` and
`STT_MODEL`. No provider is chosen by default: pick one by measuring accuracy on
real voice notes (dialects, Arabic-English mixing, noise). A provider with a
different API only needs a new class with the same `transcribe` method.

## Not built yet

- User accounts and per-user usage limits (auth is a shared static token).
- Rate limiting.
- An evaluation set and script for comparing speech-to-text providers.
