"""Text-to-speech observers for Forge.

``SpeechObserver`` auto-selects the best available TTS backend.

Backend priority:

1. **pyttsx3** — offline, system voices, cross-platform.
   - macOS: uses NSSpeechSynthesizer (built-in voices, no extra install)
   - Windows: uses SAPI5 (install extra language packs via Windows Settings)
   - Linux: uses espeak as driver — requires ``sudo apt install espeak-ng``
   Install: ``pip install pyttsx3``  (already in requirements.txt)

2. **espeak / espeak-ng** (Linux only) — direct subprocess fallback.
   Install: ``sudo apt install espeak-ng``

3. **macOS ``say``** (macOS only) — built-in, zero Python dependencies.

4. **edge-tts** — Microsoft Edge TTS (online, free, best quality, all platforms).
   Install: ``pip install edge-tts``

Usage (standalone agent)::

    from forge.plugins.speech import SpeechObserver
    factory = ForgeAgentFactory(observers=[SpeechObserver(language="Spanish")])

Usage (playbook — speak only the final summary step)::

    from forge.plugins.speech import SpeechObserver
    playbook_factory = ForgePlaybookFactory(
        agent_factory=agent_factory,
        observers=[SpeechObserver(language="Spanish", step="summary")],
    )
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from forge.events.observer import ForgeObserver

# ---------------------------------------------------------------------------
# Language → locale / voice mappings
# ---------------------------------------------------------------------------

# Language name → BCP-47 locale prefix for pyttsx3 voice search
_LANG_LOCALE: dict[str, str] = {
    "Spanish": "es_",
    "English": "en_",
    "French": "fr_",
    "German": "de_",
    "Italian": "it_",
    "Portuguese": "pt_",
    "Japanese": "ja_",
    "Korean": "ko_",
    "Chinese": "zh_",
    "Dutch": "nl_",
    "Russian": "ru_",
    "Polish": "pl_",
}

# espeak / espeak-ng language codes (Linux direct subprocess fallback)
_ESPEAK_LANG: dict[str, str] = {
    "Spanish": "es",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Dutch": "nl",
    "Russian": "ru",
    "Polish": "pl",
}

# macOS ``say`` voices (macOS-only fallback)
_SAY_VOICES: dict[str, str] = {
    "Spanish": "Monica",     # es_ES compact — pre-installed
    "English": "Samantha",   # en_US compact
    "French": "Thomas",      # fr_FR compact
    "German": "Anna",        # de_DE compact
    "Italian": "Alice",      # it_IT compact
    "Portuguese": "Luciana", # pt_BR compact
}

# Microsoft Edge TTS neural voices (edge-tts, all platforms)
_EDGE_VOICES: dict[str, str] = {
    "Spanish": "es-ES-AlvaroNeural",
    "English": "en-US-AriaNeural",
    "French": "fr-FR-HenriNeural",
    "German": "de-DE-ConradNeural",
    "Italian": "it-IT-DiegoNeural",
    "Portuguese": "pt-BR-AntonioNeural",
}
_DEFAULT_EDGE_VOICE = "en-US-AriaNeural"


# ---------------------------------------------------------------------------
# pyttsx3
# ---------------------------------------------------------------------------

def _pyttsx3_voice_id(language: str) -> str | None:
    """Return the best pyttsx3 voice ID for *language*, or None if not found.

    Works across macOS (nsss), Windows (sapi5) and Linux (espeak driver).
    Searches by BCP-47 locale prefix; falls back to matching the voice ID
    string itself (useful for espeak driver on Linux where ``languages`` may
    be empty or use plain language codes like ``['en']``).
    """
    try:
        import pyttsx3  # noqa: PLC0415
    except ImportError:
        return None

    locale = _LANG_LOCALE.get(language)
    if not locale:
        return None

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    # Pass 1: match by the languages attribute (all platforms)
    for v in voices:
        langs = [str(l) for l in getattr(v, "languages", [])]
        if any(l.lower().startswith(locale.lower()) for l in langs):
            return v.id

    # Pass 2: match voice ID or name containing the language code
    # (espeak driver on Linux uses IDs like "english", "spanish", or lang codes)
    lang_code = locale.rstrip("_").lower()  # "es_" → "es"
    for v in voices:
        vid = (v.id or "").lower()
        vname = (getattr(v, "name", "") or "").lower()
        if lang_code in vid or lang_code in vname:
            return v.id

    return None


def _speak_pyttsx3(text: str, language: str) -> bool:
    """Speak via pyttsx3. Returns True on success."""
    try:
        import pyttsx3  # noqa: PLC0415
    except ImportError:
        return False
    try:
        engine = pyttsx3.init()
        voice_id = _pyttsx3_voice_id(language)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as exc:
        print(f"[speech] pyttsx3 error: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# espeak (Linux direct subprocess — no Python dependency)
# ---------------------------------------------------------------------------

def _speak_espeak(text: str, language: str) -> bool:
    """Speak via espeak-ng subprocess (Linux). Returns True on success."""
    lang_code = _ESPEAK_LANG.get(language, "en")
    for binary in ("espeak-ng", "espeak"):
        result = subprocess.run(
            [binary, "-v", lang_code, text],
            capture_output=True,
        )
        if result.returncode == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# macOS say (macOS only)
# ---------------------------------------------------------------------------

def _speak_macos_say(text: str, language: str) -> bool:
    """Speak via macOS built-in ``say`` command. Returns True on success."""
    voice = _SAY_VOICES.get(language)
    cmd = ["say"] + (["-v", voice] if voice else []) + [text]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# edge-tts (all platforms, requires internet + pip install edge-tts)
# ---------------------------------------------------------------------------

def _speak_edge_tts(text: str, language: str) -> bool:
    """Speak via Microsoft Edge TTS. Returns True on success."""
    try:
        import edge_tts  # noqa: PLC0415
    except ImportError:
        return False

    voice = _EDGE_VOICES.get(language, _DEFAULT_EDGE_VOICE)

    async def _synthesize() -> None:
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        _play_mp3(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

    try:
        asyncio.run(_synthesize())
        return True
    except Exception as exc:
        print(f"[speech] edge-tts error: {exc}", file=sys.stderr)
        return False


def _play_mp3(path: str) -> None:
    """Play an mp3 file with the platform audio player."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
    elif sys.platform == "win32":
        ps = f"(New-Object Media.SoundPlayer '{path}').PlaySync()"
        subprocess.run(["powershell", "-c", ps], check=False)
    else:
        for player in ("mpg123", "ffplay", "mplayer"):
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                subprocess.run([player, "-q", path], check=False)
                return


# ---------------------------------------------------------------------------
# wav playback (used by TransformersTTSObserver)
# ---------------------------------------------------------------------------

def _play_wav(path: str) -> None:
    """Play a wav file with the platform audio player."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
    elif sys.platform == "win32":
        ps = f"(New-Object Media.SoundPlayer '{path}').PlaySync()"
        subprocess.run(["powershell", "-c", ps], check=False)
    else:
        for player in ("aplay", "paplay", "ffplay", "mplayer"):
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                subprocess.run([player, "-q", path], check=False)
                return


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_text(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


# ---------------------------------------------------------------------------
# Unified SpeechObserver
# ---------------------------------------------------------------------------

class SpeechObserver(ForgeObserver):
    """Offline-first, cross-platform TTS observer.

    Backend priority:
      macOS:   say (compact voice) → pyttsx3 → edge-tts
      Windows: pyttsx3 (SAPI5)    → edge-tts
      Linux:   pyttsx3 (espeak)   → espeak direct → edge-tts

    ``say`` is preferred over pyttsx3 on macOS because the compact system
    voices sound significantly more natural than the NSSpeechSynthesizer
    voices exposed through pyttsx3.

    Args:
        language: Language name (e.g. ``"Spanish"``, ``"English"``).
                  Use ``"auto"`` to detect from the playbook context.
        step:     Playbook step name whose output should be spoken.
                  If ``None``, speaks every agent final response (standalone mode).
        debug:    Print backend selection and errors to stderr.
    """

    def __init__(
        self,
        language: str = "English",
        step: str | None = None,
        debug: bool = True,
    ) -> None:
        self._language = language
        self._target_step = step
        self._debug = debug

    def _log(self, msg: str) -> None:
        if self._debug:
            print(f"[speech] {msg}", file=sys.stderr)

    def _speak(self, text: str) -> None:
        if not text:
            self._log("nothing to speak (empty text)")
            return

        lang = self._language
        preview = text[:60].replace("\n", " ")
        self._log(f"speaking ({lang}): {preview}…")

        if sys.platform == "darwin":
            # macOS: prefer say (compact voices sound more natural than pyttsx3/nsss)
            voice_say = _SAY_VOICES.get(lang, "")
            self._log(f"backend: macOS say -v {voice_say or '(default)'}")
            if _speak_macos_say(text, lang):
                return
            self._log("macOS say failed, trying pyttsx3")

        if not sys.platform.startswith("linux"):
            # macOS fallback / Windows primary: pyttsx3
            voice_id = _pyttsx3_voice_id(lang)
            self._log(f"backend: pyttsx3 (voice: {voice_id or 'default'})")
            if _speak_pyttsx3(text, lang):
                return
            self._log("pyttsx3 failed or unavailable")
        else:
            # Linux: pyttsx3 first (uses espeak driver), then espeak direct
            voice_id = _pyttsx3_voice_id(lang)
            self._log(f"backend: pyttsx3 (voice: {voice_id or 'default'})")
            if _speak_pyttsx3(text, lang):
                return
            self._log("pyttsx3 failed, trying espeak directly")
            lang_code = _ESPEAK_LANG.get(lang, "en")
            self._log(f"backend: espeak -v {lang_code}")
            if _speak_espeak(text, lang):
                return
            self._log("espeak unavailable — install: sudo apt install espeak-ng")

        # Last resort: edge-tts (requires internet + pip install edge-tts)
        self._log(f"backend: edge-tts ({_EDGE_VOICES.get(lang, _DEFAULT_EDGE_VOICE)})")
        if _speak_edge_tts(text, lang):
            return

        self._log(
            "no TTS backend available.\n"
            "  macOS/Windows: pip install pyttsx3\n"
            "  Linux:         sudo apt install espeak-ng && pip install pyttsx3\n"
            "  Any platform:  pip install edge-tts  (requires internet)"
        )

    # ── Observer hooks ────────────────────────────────────────────────────────

    def on_agent_end(self, agent_name: str, response: Any) -> None:
        if self._target_step is not None:
            return  # playbook mode — wait for the configured step
        self._speak(_extract_text(response))

    def on_step_end(self, step_name: str, result: str) -> None:
        if self._target_step and step_name == self._target_step:
            self._speak(result)

    def on_playbook_end(self, context: dict[str, str]) -> None:
        if self._language == "auto":
            lang = context.get("language", "")
            if lang:
                self._language = lang
                self._log(f"language auto-detected from context: {lang}")


# ---------------------------------------------------------------------------
# OpenAITTSObserver — OpenAI /v1/audio/speech API (cloud or self-hosted)
# ---------------------------------------------------------------------------

class OpenAITTSObserver(ForgeObserver):
    """TTS observer using the OpenAI Audio Speech API.

    Works with OpenAI's cloud API **and** any self-hosted server that
    implements the same endpoint (e.g. a future Ollama TTS, LM Studio, etc.)
    by setting ``base_url``.

    Note: **Ollama does not currently support TTS** — it serves LLMs only.
    This observer is ready for when a compatible local TTS API becomes
    available, and works today with OpenAI's cloud.

    Language is **auto-detected** from the input text — no language config
    needed. The model speaks in whatever language you send it.

    Models:
        - ``tts-1``           — optimised for speed (default)
        - ``tts-1-hd``        — optimised for quality
        - ``gpt-4o-mini-tts`` — latest, most expressive

    Voices (OpenAI): ``alloy``, ``ash``, ``coral``, ``echo``, ``fable``,
    ``nova``, ``onyx``, ``sage``, ``shimmer``

    Args:
        api_key:   OpenAI API key (default: ``OPENAI_API_KEY`` env var).
        base_url:  Override for self-hosted OpenAI-compatible TTS servers.
                   Example: ``"http://localhost:8880/v1"`` for Kokoro-FastAPI.
        model:     TTS model ID (default: ``"tts-1"``).
        voice:     Voice name (default: ``"alloy"``).
        step:      Playbook step name to speak. ``None`` → standalone agent mode.
        debug:     Print progress and errors to stderr.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "tts-1",
        voice: str = "alloy",
        step: str | None = None,
        debug: bool = True,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._voice = voice
        self._target_step = step
        self._debug = debug

    def _log(self, msg: str) -> None:
        if self._debug:
            print(f"[openai-tts] {msg}", file=sys.stderr)

    def _speak(self, text: str) -> None:
        if not text:
            self._log("nothing to speak (empty text)")
            return

        preview = text[:60].replace("\n", " ")
        endpoint = self._base_url or "api.openai.com"
        self._log(f"speaking via {endpoint} ({self._model}, voice={self._voice}): {preview}…")

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            self._log("openai package not installed: pip install openai")
            return

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        try:
            client = OpenAI(**kwargs)
            response = client.audio.speech.create(
                model=self._model,
                voice=self._voice,  # type: ignore[arg-type]
                input=text,
            )
        except Exception as exc:
            self._log(f"API error: {exc}")
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            response.stream_to_file(tmp_path)
            _play_mp3(tmp_path)
        except Exception as exc:
            self._log(f"playback error: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Observer hooks ────────────────────────────────────────────────────────

    def on_agent_end(self, agent_name: str, response: Any) -> None:
        if self._target_step is not None:
            return
        self._speak(_extract_text(response))

    def on_step_end(self, step_name: str, result: str) -> None:
        if self._target_step and step_name == self._target_step:
            self._speak(result)


# ---------------------------------------------------------------------------
# TransformersTTSObserver — HuggingFace neural TTS (offline after first download)
# ---------------------------------------------------------------------------

# ISO 639-3 codes for facebook/mms-tts-{lang} models
# Full list: https://huggingface.co/facebook/mms-tts
_MMS_LANG: dict[str, str] = {
    "Spanish": "spa",
    "English": "eng",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Portuguese": "por",
    "Japanese": "jpn",
    "Korean": "kor",
    "Chinese": "zho",
    "Russian": "rus",
    "Arabic": "ara",
    "Dutch": "nld",
    "Polish": "pol",
    "Turkish": "tur",
    "Hindi": "hin",
}


class TransformersTTSObserver(ForgeObserver):
    """Neural TTS observer using Facebook MMS-TTS transformer models.

    - 100% offline after the first model download (~100 MB per language).
    - 1100+ languages supported via ``facebook/mms-tts-{iso639_3}``.
    - Based on VITS architecture — natural-sounding, fast inference on CPU.
    - No extra pip installs needed (``transformers`` and ``torch`` already
      in requirements.txt).

    The model is loaded lazily on the first ``_speak()`` call and cached
    in memory for the lifetime of the observer.

    Args:
        language:   Language name (e.g. ``"Spanish"``). Use ``"auto"`` to
                    detect from the playbook context.
        model_id:   Override the HuggingFace model ID (default:
                    ``"facebook/mms-tts-{iso_code}"``).
        step:       Playbook step name to speak. ``None`` → speak every
                    agent final response (standalone agent mode).
        debug:      Print progress and errors to stderr.
    """

    def __init__(
        self,
        language: str = "English",
        model_id: str | None = None,
        step: str | None = None,
        debug: bool = True,
    ) -> None:
        self._language = language
        self._model_id_override = model_id
        self._target_step = step
        self._debug = debug
        self._model: Any = None
        self._tokenizer: Any = None

    def _log(self, msg: str) -> None:
        if self._debug:
            print(f"[tts] {msg}", file=sys.stderr)

    def _model_id(self) -> str:
        if self._model_id_override:
            return self._model_id_override
        iso = _MMS_LANG.get(self._language, "eng")
        return f"facebook/mms-tts-{iso}"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, VitsModel  # noqa: PLC0415
        except ImportError:
            raise RuntimeError("transformers is required: pip install transformers torch")

        mid = self._model_id()
        self._log(f"loading {mid} (cached to ~/.cache/huggingface after first run)…")
        self._tokenizer = AutoTokenizer.from_pretrained(mid)
        self._model = VitsModel.from_pretrained(mid)
        self._model.eval()
        self._log("model ready")

    def _synthesize(self, text: str) -> tuple[Any, int]:
        """Return (waveform_numpy_1d, sample_rate). Splits long text by sentence."""
        import re  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        samplerate: int = self._model.config.sampling_rate

        # Split into sentences — VITS performs best on sentence-length inputs.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        # 120 ms silence between sentences: smooths the join without noticeable gaps.
        silence = np.zeros(int(samplerate * 0.12), dtype=np.float32)

        chunks: list[Any] = []
        for i, sentence in enumerate(sentences):
            inputs = self._tokenizer(sentence, return_tensors="pt")
            with torch.no_grad():
                output = self._model(**inputs).waveform
            chunks.append(output.squeeze().numpy())
            if i < len(sentences) - 1:
                chunks.append(silence)

        waveform = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return waveform, samplerate

    def _speak(self, text: str) -> None:
        import wave  # noqa: PLC0415  (stdlib, always available)

        import numpy as np  # noqa: PLC0415

        if not text:
            self._log("nothing to speak (empty text)")
            return

        preview = text[:60].replace("\n", " ")
        self._log(f"speaking ({self._language}): {preview}…")

        try:
            self._load()
        except Exception as exc:
            self._log(f"model load failed: {exc}")
            return

        try:
            waveform, samplerate = self._synthesize(text)
        except Exception as exc:
            self._log(f"synthesis error: {exc}")
            return

        # Write wav using stdlib wave module — no scipy needed.
        data_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(data_int16.tobytes())

        _play_wav(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

    # ── Observer hooks ────────────────────────────────────────────────────────

    def on_agent_end(self, agent_name: str, response: Any) -> None:
        if self._target_step is not None:
            return
        self._speak(_extract_text(response))

    def on_step_end(self, step_name: str, result: str) -> None:
        if self._target_step and step_name == self._target_step:
            self._speak(result)

    def on_playbook_end(self, context: dict[str, str]) -> None:
        if self._language == "auto":
            lang = context.get("language", "")
            if lang:
                self._language = lang
                self._log(f"language auto-detected from context: {lang}")
