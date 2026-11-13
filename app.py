"""Transcribe tu clase — transcripción local con identificación de hablantes.

Se ejecuta con `python app.py` y abre http://localhost:8000
Todo el procesamiento ocurre en este equipo. Internet solo se usa para
descargar los modelos la primera vez.

Flujo de un trabajo (data/trabajos/<id>/):
  1. ffmpeg extrae el audio UNA vez a audio.wav (16 kHz, mono, PCM 16 bits).
  2. mlx-whisper transcribe ese WAV por bloques -> transcripcion.json
     (el modelo se libera antes de seguir).
  3. pyannote identifica hablantes sobre el mismo WAV -> diarizacion.json
  4. Se unen ambos resultados -> data/historial/<id>.json
Si una etapa falla, lo guardado por las anteriores se conserva.
"""
from __future__ import annotations

import contextlib
import gc
import importlib
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

for _stream in (sys.stdout, sys.stderr):  # consola de Windows (cp1252) no soporta →, ✔, …
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).parent.resolve()
DATA = ROOT / "data"
HISTORY_DIR = DATA / "historial"
PROFILES_DIR = DATA / "perfiles"
JOBS_DIR = DATA / "trabajos"
for d in (HISTORY_DIR, PROFILES_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IS_MAC_ARM = sys.platform == "darwin" and platform.machine() == "arm64"
# macOS Apple Silicon: mlx-whisper. Windows/Linux: faster-whisper (CUDA o CPU).
WHISPER_BACKEND = "mlx" if IS_MAC_ARM else "faster"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo" if IS_MAC_ARM else "large-v3-turbo"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
ALLOWED_EXT = {".mp4", ".mkv", ".mp3", ".wav"}
SAMPLE_RATE = 16000
# Similitud coseno mínima para sugerir un perfil de voz.
MATCH_THRESHOLD = 0.50
ID_RE = re.compile(r"[0-9a-f]{12}")


# --------------------------------------------------------------------------
# Configuración (.env)
# --------------------------------------------------------------------------
def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

# Bloques de Whisper (ajustables en .env; los valores por defecto son los recomendados).
CHUNK_SECONDS = int(os.environ.get("WHISPER_CHUNK_SECONDS", 600))
OVERLAP_SECONDS = int(os.environ.get("WHISPER_OVERLAP_SECONDS", 8))


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None


# --------------------------------------------------------------------------
# Diagnóstico por terminal
# --------------------------------------------------------------------------
def mem_info() -> str:
    """RAM residente del proceso (y memoria MPS/CUDA si torch ya está cargado)."""
    parts = []
    try:
        import psutil
        parts.append(f"RAM {psutil.Process().memory_info().rss / 1048576:.0f} MB")
    except Exception:  # noqa: BLE001
        pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.backends.mps.is_available():
            parts.append(f"MPS {torch.mps.current_allocated_memory() / 1048576:.0f} MB")
        elif torch is not None and torch.cuda.is_available():
            parts.append(f"CUDA {torch.cuda.memory_allocated() / 1048576:.0f} MB")
    except Exception:  # noqa: BLE001
        pass
    return " · ".join(parts)


# --------------------------------------------------------------------------
# Trabajos en segundo plano
# --------------------------------------------------------------------------
jobs: dict[str, dict[str, Any]] = {}
job_queue: "queue.Queue[str]" = queue.Queue()
# Solo una etapa pesada a la vez (verificación de arranque o trabajo).
gpu_lock = threading.Lock()


class Cancelled(Exception):
    def __init__(self, segments: list[dict] | None = None):
        super().__init__("cancelado")
        self.segments = segments


STAGE_LABEL = {
    "extrayendo": "extrayendo audio", "transcribiendo": "transcribiendo",
    "hablantes": "identificando hablantes", "uniendo": "uniendo resultados",
}
# (inicio %, fin %) de cada etapa en la barra global
WEIGHTS_DIARIZE = {"extrayendo": (0, 5), "transcribiendo": (5, 65), "hablantes": (65, 95), "uniendo": (95, 100)}
WEIGHTS_PLAIN = {"extrayendo": (0, 8), "transcribiendo": (8, 100)}
WEIGHTS_RETRY = {"hablantes": (0, 95), "uniendo": (95, 100)}


def job_dir(jid: str) -> Path:
    return JOBS_DIR / jid


def check_cancel(job: dict, segments: list[dict] | None = None) -> None:
    if job.get("cancel"):
        raise Cancelled(segments)


def set_stage(job: dict, stage: str, detail: str = "") -> None:
    check_cancel(job)
    job["stage"] = stage
    job["detail"] = detail
    job["stage_started"] = time.time()
    lo, _ = job["weights"].get(stage, (job.get("progress", 0), 0))
    job["progress"] = max(job.get("progress", 0), lo)


def set_progress(job: dict, stage: str, fraction: float, detail: str | None = None) -> None:
    lo, hi = job["weights"][stage]
    job["progress"] = round(lo + (hi - lo) * min(max(fraction, 0.0), 1.0), 1)
    if detail is not None:
        job["detail"] = detail


def public_job(job: dict) -> dict:
    now = time.time()
    end = job.get("finished") or now
    d = {k: job.get(k) for k in (
        "id", "filename", "kind", "status", "stage", "steps", "detail", "progress",
        "error", "warning", "record_id", "diarize", "duration")}
    d["elapsed"] = round(end - job["started"], 1) if job.get("started") else 0
    d["stage_elapsed"] = round(end - job["stage_started"], 1) if job.get("stage_started") else 0
    d["position"] = None
    if job["status"] == "queued":
        waiting = sorted((j for j in jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
        d["position"] = waiting.index(job) + 1 if job in waiting else None
    return d


# ---- Audio ---------------------------------------------------------------
def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def is_target_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnchannels() == 1 and w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2 \
                and w.getcomptype() == "NONE"
    except Exception:  # noqa: BLE001
        return False


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return None


def extract_audio(job: dict, src: Path, dst: Path) -> None:
    """Extrae el audio a WAV 16 kHz mono PCM s16 con ffmpeg, una sola vez."""
    if src.suffix.lower() == ".wav" and is_target_wav(src):
        log(job["id"], "el archivo ya es WAV 16 kHz mono: no se reconvierte")
        src.replace(dst)
        return
    total = probe_duration(src)
    tmp = dst.with_suffix(".part.wav")
    errlog = dst.parent / "ffmpeg.log"
    cmd = ["ffmpeg", "-y", "-nostdin", "-i", str(src), "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
           "-c:a", "pcm_s16le", "-f", "wav", "-progress", "pipe:1", "-nostats", str(tmp)]
    try:
        with errlog.open("w") as ef:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=ef, text=True)
    except FileNotFoundError:
        raise RuntimeError("No se encontró ffmpeg. Instálalo (macOS: `brew install ffmpeg`; Windows: `winget install Gyan.FFmpeg`) y reinicia la app.")
    last_log = 0.0
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            if job.get("cancel"):
                proc.terminate()
                proc.wait()
                raise Cancelled()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    done = int(line.split("=")[1]) / 1e6
                except ValueError:
                    continue
                if total:
                    set_progress(job, "extrayendo", done / total,
                                 f"{fmt_clock(done)} de {fmt_clock(total)} de audio")
                    if time.time() - last_log > 10:
                        last_log = time.time()
                        log(job["id"], f"extrayendo audio {done / total:.0%}")
        proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        tmp.unlink(missing_ok=True)
        raise
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = errlog.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1:] or ["ffmpeg falló"]
        raise RuntimeError(
            f"No se pudo leer el audio de este archivo ({tail[0]}). ¿Está dañado o es un formato no compatible?")
    tmp.replace(dst)
    errlog.unlink(missing_ok=True)


def read_wav_samples(path: Path, start_s: float | None = None, length_s: float | None = None):
    """Lee muestras int16 del WAV (todo o un tramo) como float32 en [-1, 1]."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        if start_s is not None:
            w.setpos(min(int(start_s * sr), w.getnframes()))
        n = w.getnframes() - w.tell() if length_s is None else int(length_s * sr)
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ---- Whisper -------------------------------------------------------------
_FW_MODEL = None


def release_whisper() -> None:
    global _FW_MODEL
    _FW_MODEL = None
    if WHISPER_BACKEND != "mlx":
        gc.collect()
        return
    try:
        mt = importlib.import_module("mlx_whisper.transcribe")
        mt.ModelHolder.model = None
        mt.ModelHolder.model_path = None
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


def whisper_transcribe(audio, language: str | None, prompt: str) -> dict:
    """Transcribe un array float32 16 kHz y devuelve {"segments": [{"start","end","text"}]}."""
    if WHISPER_BACKEND == "mlx":
        import mlx_whisper
        kw: dict[str, Any] = dict(path_or_hf_repo=WHISPER_MODEL, verbose=None)
        if language:
            kw["language"] = language
        if prompt:
            kw["initial_prompt"] = prompt
        return mlx_whisper.transcribe(audio, **kw)
    global _FW_MODEL
    if _FW_MODEL is None:
        from faster_whisper import WhisperModel
        device = "cuda" if resolve_device() == "cuda" else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        log("whisper", f"cargando faster-whisper {WHISPER_MODEL} en {device} ({compute})")
        _FW_MODEL = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    segs, _info = _FW_MODEL.transcribe(audio, language=language, initial_prompt=prompt or None)
    return {"segments": [{"start": x.start, "end": x.end, "text": x.text} for x in segs]}


def _norm_words(text: str) -> list[str]:
    return [re.sub(r"\W+", "", w.lower()) for w in text.split()]


def trim_boundary_repeat(text: str, start: float, prev: list[dict], max_words: int = 10) -> str:
    """Si el bloque nuevo arranca a mitad de frase, puede repetir las últimas palabras del segmento
    anterior que se solapa en el tiempo. Quita ese prefijo repetido (mínimo 2 palabras)."""
    cands = [o for o in prev if o["end"] > start]
    if not cands:
        return text
    old = max(cands, key=lambda o: o["end"])
    ow, nw, raw = _norm_words(old["text"]), _norm_words(text), text.split()
    for k in range(min(max_words, len(ow), len(nw)), 1, -1):
        if ow[-k:] == nw[:k] and any(ow[-k:]):
            return " ".join(raw[k:])
    return text


def transcribe_blocks(job: dict, wav: Path, duration: float, language: str | None, vocabulary: str,
                      out_file: Path) -> list[dict]:
    if WHISPER_BACKEND == "mlx":
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            raise RuntimeError("Falta mlx-whisper. Ejecuta `pip install -r requirements.txt`.")
    else:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            raise RuntimeError("Falta faster-whisper. Ejecuta `pip install -r requirements.txt`.")
    step = CHUNK_SECONDS - OVERLAP_SECONDS
    n = 1 if duration <= CHUNK_SECONDS else math.ceil((duration - OVERLAP_SECONDS) / step)
    prompt = vocabulary.strip()
    kept: list[dict] = []  # segmentos con la marca interna "_k" (bloque de origen)

    def clean() -> list[dict]:
        return [{"start": x["start"], "end": x["end"], "text": x["text"]}
                for x in sorted(kept, key=lambda x: x["start"])]

    def save(complete: bool, done: int) -> None:
        out_file.write_text(json.dumps(
            {"complete": complete, "blocks_done": done, "blocks_total": n, "segments": clean()},
            ensure_ascii=False), encoding="utf-8")

    prev_end = 0.0
    try:
        for k in range(n):
            check_cancel(job, clean())
            start = k * step
            end = min(start + CHUNK_SECONDS, duration)
            if end - start < 0.5:
                continue
            job["detail"] = f"bloque {k + 1} de {n}"
            set_progress(job, "transcribiendo", start / duration)
            log(job["id"], f"transcribiendo bloque {k + 1}/{n} "
                f"[{fmt_clock(start)}–{fmt_clock(end)}] · {job['progress']:.0f}% global · "
                f"{fmt_clock(time.time() - job['started'])} transcurridos")
            audio = read_wav_samples(wav, start, end - start)
            res = whisper_transcribe(audio, language, prompt)
            del audio
            for s in res.get("segments", []):
                text = s["text"].strip()
                if not text:
                    continue
                a, b = float(s["start"]) + start, min(float(s["end"]) + start, duration)
                # Zona de solapamiento: si un segmento coincide (>50 %) con otros del bloque anterior,
                # se queda el que está más lejos del borde de su bloque (tuvo más contexto).
                if k > 0:
                    conflicts = [o for o in kept if o["_k"] == k - 1
                                 and min(b, o["end"]) - max(a, o["start"]) > 0.5 * min(b - a, o["end"] - o["start"])]
                    if conflicts:
                        new_edge = (a + b) / 2 - start
                        old_edge = max(prev_end - (o["start"] + o["end"]) / 2 for o in conflicts)
                        if new_edge <= old_edge:
                            continue
                        kept[:] = [o for o in kept if o not in conflicts]
                if k > 0:
                    text = trim_boundary_repeat(text, a, [o for o in kept if o["_k"] == k - 1])
                    if not text:
                        continue
                kept.append({"start": a, "end": b, "text": text, "_k": k})
            prev_end = end
            save(k == n - 1, k + 1)
            set_progress(job, "transcribiendo", end / duration)
    finally:
        release_whisper()
        log(job["id"], f"modelo de Whisper liberado  ({mem_info()})")
    save(True, n)
    return clean()


# ---- pyannote ------------------------------------------------------------
def friendly_pyannote_error(e: BaseException) -> str:
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    if "torchaudio" in low or "list_audio_backends" in low:
        return ("Versiones incompatibles de torch/torchaudio con pyannote.audio. Instala las versiones fijadas: "
                "`pip install -r requirements.txt` (torch==2.8.0, torchaudio==2.8.0) en un entorno con Python 3.12. "
                f"Detalle: {text[:200]}")
    if any(w in low for w in ("401", "403", "gated", "unauthorized", "access to model", "restricted", "cannot access")):
        return ("Hugging Face rechazó el acceso al modelo de hablantes. Revisa que el token del .env sea válido y "
                "acepta las condiciones de uso de pyannote/speaker-diarization-3.1 y pyannote/segmentation-3.0 "
                "con esa misma cuenta.")
    if any(w in low for w in ("connection", "resolve", "offline", "timed out", "network")):
        return ("No se pudo descargar el modelo de hablantes (sin conexión). La primera vez se necesita internet. "
                f"Detalle: {text[:200]}")
    return text[:400]


def resolve_device() -> str:
    """auto → cuda (NVIDIA) > mps (Apple) > cpu. DIARIZATION_DEVICE=cpu|mps|cuda lo fuerza."""
    pref = os.environ.get("DIARIZATION_DEVICE", "auto").strip().lower()
    has = {"mps": False, "cuda": False}
    try:
        import torch
        has["mps"] = bool(torch.backends.mps.is_available())
        has["cuda"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    if pref == "cpu":
        return "cpu"
    if pref in has:
        if has[pref]:
            return pref
        log("pyannote", f"DIARIZATION_DEVICE={pref} pero no está disponible; se usa cpu")
        return "cpu"
    return "cuda" if has["cuda"] else "mps" if has["mps"] else "cpu"


@contextlib.contextmanager
def trusted_checkpoints():
    """PyTorch >= 2.6 carga con weights_only=True y rechaza los checkpoints de pyannote 3.x (guardan
    clases de lightning/omegaconf/pyannote). Solo mientras se carga el pipeline oficial de pyannote
    se usa weights_only=False; se restaura el comportamiento seguro al terminar."""
    import torch
    original = torch.load

    def load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def load_pipeline(device: str):
    token = hf_token()
    if not token:
        raise RuntimeError("Falta el token de Hugging Face. Crea un archivo .env con HF_TOKEN=hf_… (mira el README).")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(f"Falta pyannote.audio. Ejecuta `pip install -r requirements.txt`. ({e})")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(friendly_pyannote_error(e))
    try:
        with trusted_checkpoints():
            try:
                pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
            except TypeError:
                pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(friendly_pyannote_error(e))
    if pipe is None:
        raise RuntimeError(friendly_pyannote_error(RuntimeError("401 gated repo: no se pudo acceder")))
    try:
        pipe.to(torch.device(device))
    except Exception as e:  # noqa: BLE001
        if device != "cpu":
            log("pyannote", f"no se pudo usar {device} ({type(e).__name__}); se usa cpu")
            pipe.to(torch.device("cpu"))
        else:
            raise
    return pipe


def release_torch() -> None:
    gc.collect()
    try:
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_pipeline(pipe, inputs: dict, num_speakers: int | None, hook):
    kwargs: dict[str, Any] = {"hook": hook, "return_embeddings": True}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    return pipe(inputs, **kwargs)


def diarize(job: dict, wav: Path, num_speakers: int | None) -> tuple[list[dict], dict[str, list[float]]]:
    import numpy as np
    import torch

    device = resolve_device()
    t_stage = time.time()
    pipe = None
    waveform = None
    try:
        with timed(job["id"], f"cargando pyannote en {device}"):
            pipe = load_pipeline(device)
        check_cancel(job)
        # El WAV se carga una sola vez y se pasa en memoria: pyannote no toca torchaudio.load.
        with timed(job["id"], "cargando el WAV como tensor"):
            waveform = torch.from_numpy(read_wav_samples(wav)).unsqueeze(0)
        inputs = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        # Pesos de cada paso dentro de la etapa (segmentación y embeddings dominan el tiempo).
        span = {"segmentation": (0.0, 0.35), "speaker_counting": (0.35, 0.37),
                "embeddings": (0.37, 0.97), "discrete_diarization": (0.97, 1.0)}
        names = {"segmentation": "analizando voz", "speaker_counting": "contando hablantes",
                 "embeddings": "comparando voces", "discrete_diarization": "cerrando turnos"}
        state = {"last_log": 0.0}

        def hook(step_name, artifact=None, file=None, total=None, completed=None):
            check_cancel(job)  # interrumpe pyannote en el siguiente lote
            lo, hi = span.get(step_name, (0.0, 1.0))
            frac = lo + (hi - lo) * ((completed / total) if total else 1.0)
            set_progress(job, "hablantes", frac, f"{names.get(step_name, step_name)}")
            if time.time() - state["last_log"] > 15:
                state["last_log"] = time.time()
                log(job["id"], f"pyannote [{device}] {names.get(step_name, step_name)} "
                    f"{(completed / total) if total else 1:.0%} · global {job['progress']:.0f}%  ({mem_info()})")

        try:
            with timed(job["id"], f"diarización en {device}"):
                out = run_pipeline(pipe, inputs, num_speakers, hook)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            if device == "cpu":
                raise
            log(job["id"], f"falló en {device} ({type(e).__name__}: {str(e)[:120]}); reintentando en cpu. "
                "Si es un error de operaciones no soportadas, prueba PYTORCH_ENABLE_MPS_FALLBACK=1 o "
                "DIARIZATION_DEVICE=cpu en .env")
            job["detail"] = "reintentando en CPU"
            pipe.to(torch.device("cpu"))
            device = "cpu"
            with timed(job["id"], "diarización en cpu"):
                out = run_pipeline(pipe, inputs, num_speakers, hook)
    finally:
        del pipe, waveform
        release_torch()
        log(job["id"], f"pyannote liberado  ({mem_info()})")

    if isinstance(out, tuple):
        diarization, embeddings = out
    else:
        diarization, embeddings = out, None
    turns = [{"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
             for t, _, spk in diarization.itertracks(yield_label=True)]
    emb_by_label: dict[str, list[float]] = {}
    if embeddings is not None:
        for label, vec in zip(diarization.labels(), embeddings):
            vec = np.asarray(vec, dtype=np.float64)
            if np.isfinite(vec).all():
                emb_by_label[str(label)] = vec.tolist()
    log(job["id"], f"diarización: {len(set(t['speaker'] for t in turns))} hablantes, {len(turns)} turnos, "
        f"{fmt_clock(time.time() - t_stage)} en total, dispositivo final {device}")
    return turns, emb_by_label


def merge(segments: list[dict], turns: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Asigna a cada segmento el hablante con mayor solapamiento temporal."""
    order: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t["start"]):
        order.setdefault(t["speaker"], f"S{len(order) + 1}")
    out = []
    for seg in segments:
        overlap: dict[str, float] = {}
        for t in turns:
            o = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if o > 0:
                overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + o
        if overlap:
            spk = max(overlap, key=overlap.get)
        elif turns:  # sin solapamiento: el turno más cercano
            mid = (seg["start"] + seg["end"]) / 2
            spk = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))["speaker"]
        else:
            spk = None
        out.append({**seg, "speaker": order.get(spk) if spk else None})
    return out, order


# ---- Verificación al arrancar ---------------------------------------------
health: dict[str, Any] = {"state": "pending", "message": "Verificación pendiente", "device": None, "seconds": None}


def startup_check() -> None:
    """Carga pyannote y procesa 2 s de audio sintético para detectar fallos en segundos."""
    with gpu_lock:
        health.update(state="running", message="Verificando el modelo de hablantes…")
        t0 = time.time()
        log("arranque", "verificando pyannote con 2 s de audio sintético …")
        pipe = None
        try:
            import numpy as np
            import torch
            device = resolve_device()
            health["device"] = device
            pipe = load_pipeline(device)
            tt = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
            rng = np.random.default_rng(0)
            sig = sum(np.sin(2 * np.pi * f * tt) / (i + 1) for i, f in enumerate((140, 280, 420, 560)))
            sig = sig * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * tt)) * 0.3 + rng.normal(0, 0.01, tt.shape)
            wf = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)
            pipe({"waveform": wf, "sample_rate": SAMPLE_RATE})
            secs = time.time() - t0
            health.update(state="ok", message=f"Listo en {secs:.0f} s ({device})", seconds=round(secs, 1))
            log("arranque", f"✔ pyannote OK en {device} ({secs:.1f} s)  ({mem_info()})")
        except Exception as e:  # noqa: BLE001
            msg = str(e) if isinstance(e, RuntimeError) else friendly_pyannote_error(e)
            health.update(state="error", message=msg, seconds=round(time.time() - t0, 1))
            log("arranque", f"✘ la verificación de pyannote falló: {msg}")
            log("arranque", "  La transcripción sigue disponible; la identificación de hablantes no hasta corregirlo.")
        finally:
            del pipe
            release_torch()
            log("arranque", f"pyannote liberado  ({mem_info()})")


# ---- Orquestación del trabajo -----------------------------------------------
def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def make_record(meta: dict, duration: float, segments: list[dict], *, complete: bool = True,
                diar_status: str = "skipped", diar_error: str | None = None,
                speakers: list[dict] | None = None, embeddings: dict | None = None) -> dict:
    return {
        "id": meta["id"], "title": meta.get("title") or Path(meta["filename"]).stem, "filename": meta["filename"],
        "created": meta.get("created") or time.time(), "duration": duration,
        "language": meta.get("language") or "auto",
        "speakers": speakers or [], "segments": segments, "embeddings": embeddings or {},
        "complete": complete, "diarization": {"status": diar_status, "error": diar_error},
    }


def set_diar_state(jid: str, status: str, error: str | None) -> None:
    try:
        rec = load_record(jid)
    except HTTPException:
        return
    rec["diarization"] = {"status": status, "error": error}
    save_record(rec)


def process_job(job: dict) -> None:
    jid = job["id"]
    jd = job_dir(jid)
    wav = jd / "audio.wav"
    meta = read_json(jd / "meta.json", {})
    t_job = time.time()
    log(jid, f"=== trabajo «{job['filename']}» ({job['kind']}) · dispositivo pyannote: "
        f"{os.environ.get('DIARIZATION_DEVICE', 'auto')} ===")
    try:
        duration = meta.get("duration")
        if job["kind"] == "full":
            if not wav.exists():
                set_stage(job, "extrayendo", "preparando…")
                src = next(jd.glob("source.*"))
                with timed(jid, "extrayendo audio a WAV 16 kHz mono"):
                    extract_audio(job, src, wav)
                src.unlink(missing_ok=True)  # el original ya no hace falta; el WAV se conserva
            duration = wav_duration(wav)
            meta["duration"] = duration
            (jd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            job["duration"] = duration

            set_stage(job, "transcribiendo", "cargando el modelo…")
            with timed(jid, "transcripción"):
                segments = transcribe_blocks(job, wav, duration, meta.get("language"),
                                             meta.get("vocabulary", ""), jd / "transcripcion.json")
            if not segments:
                raise RuntimeError("No se detectó voz en este archivo. Revisa que tenga audio y que el volumen no sea muy bajo.")
            save_record(make_record(meta, duration, segments, diar_status="pending" if job["diarize"] else "skipped"))
            job["record_id"] = jid
        else:
            tr = read_json(jd / "transcripcion.json")
            if not tr or not wav.exists():
                raise RuntimeError("No hay audio o transcripción guardados para este trabajo.")
            segments = tr["segments"]
            duration = duration or wav_duration(wav)
            job["duration"] = duration
            job["record_id"] = jid

        if job["diarize"]:
            try:
                set_stage(job, "hablantes", "cargando el modelo…")
                turns, emb = diarize(job, wav, job.get("num_speakers"))
                (jd / "diarizacion.json").write_text(
                    json.dumps({"turns": turns, "embeddings": emb}), encoding="utf-8")
                set_stage(job, "uniendo", "")
                with timed(jid, "uniendo transcripción y hablantes"):
                    merged, order = merge(segments, turns)
                    speakers, embeddings = [], {}
                    for label, sid in order.items():
                        speakers.append({"id": sid, "name": None})
                        if label in emb:
                            embeddings[sid] = emb[label]
                    rec = make_record(meta, duration, merged, diar_status="ok", speakers=speakers,
                                      embeddings=embeddings)
                    old = read_json(HISTORY_DIR / f"{jid}.json", {})
                    rec["title"] = old.get("title", rec["title"])
                    save_record(rec)
                set_progress(job, "uniendo", 1)
            except Cancelled:
                set_diar_state(jid, "skipped", None)
                raise
            except Exception as e:  # noqa: BLE001
                msg = str(e) if isinstance(e, RuntimeError) else friendly_pyannote_error(e)
                log(jid, f"✘ la identificación de hablantes falló: {msg}")
                set_diar_state(jid, "error", msg)
                job["warning"] = msg
        job["progress"] = 100
        job["status"] = "done"
        job["stage"] = "listo"
    except Cancelled as c:
        # Lo ya procesado queda guardado.
        if c.segments:
            meta_now = read_json(jd / "meta.json", meta)
            save_record(make_record(meta_now, meta_now.get("duration") or c.segments[-1]["end"], c.segments,
                                    complete=False))
            job["record_id"] = jid
            log(jid, f"cancelado: se guardó la transcripción parcial ({len(c.segments)} segmentos)")
        else:
            log(jid, "cancelado")
        job["status"] = "cancelled"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e) or type(e).__name__
        log(jid, f"✘ error: {job['error']}")
        if (jd / "transcripcion.json").exists():
            job["record_id"] = jid if (HISTORY_DIR / f"{jid}.json").exists() else None
    finally:
        release_whisper()
        release_torch()
        for src in jd.glob("source.*"):
            if job["status"] in ("done", "cancelled"):
                src.unlink(missing_ok=True)
        job["finished"] = time.time()
        log(jid, f"=== terminado ({job['status']}) en {fmt_clock(time.time() - t_job)}  ({mem_info()}) ===")


def worker() -> None:
    while True:
        jid = job_queue.get()
        job = jobs.get(jid)
        if not job:
            continue
        if job.get("cancel"):
            job["status"] = "cancelled"
            job["finished"] = time.time()
            continue
        with gpu_lock:
            job["status"] = "running"
            job["started"] = time.time()
            process_job(job)


threading.Thread(target=worker, daemon=True).start()


def new_job(jid: str, kind: str, meta: dict, diarize_on: bool, num_speakers: int | None) -> dict:
    if kind == "retry":
        weights, steps = WEIGHTS_RETRY, ["hablantes", "uniendo"]
    elif diarize_on:
        weights, steps = WEIGHTS_DIARIZE, ["extrayendo", "transcribiendo", "hablantes", "uniendo"]
    else:
        weights, steps = WEIGHTS_PLAIN, ["extrayendo", "transcribiendo"]
    job = {"id": jid, "filename": meta["filename"], "kind": "diarize" if kind == "retry" else "full",
           "status": "queued", "stage": "en cola", "detail": "", "progress": 0, "created": time.time(),
           "error": None, "warning": None, "record_id": None, "diarize": diarize_on,
           "num_speakers": num_speakers, "weights": weights, "steps": steps}
    jobs[jid] = job
    job_queue.put(jid)
    return job


# --------------------------------------------------------------------------
# Historial y perfiles de voz
# --------------------------------------------------------------------------
def save_record(rec: dict) -> None:
    (HISTORY_DIR / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def load_record(rid: str) -> dict:
    if not ID_RE.fullmatch(rid):
        raise HTTPException(404, "Transcripción no encontrada")
    p = HISTORY_DIR / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "Transcripción no encontrada")
    return json.loads(p.read_text(encoding="utf-8"))


def load_profiles() -> list[dict]:
    out = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def suggestions_for(rec: dict) -> dict[str, dict]:
    profiles = load_profiles()
    out = {}
    for sid, emb in (rec.get("embeddings") or {}).items():
        best, best_score = None, 0.0
        for prof in profiles:
            s = cosine(emb, prof["embedding"])
            if s > best_score:
                best, best_score = prof, s
        if best and best_score >= MATCH_THRESHOLD:
            out[sid] = {"profile_id": best["id"], "name": best["name"], "score": round(best_score, 3)}
    return out


def view_record(rec: dict) -> dict:
    v = {k: val for k, val in rec.items() if k != "embeddings"}
    v.setdefault("diarization", {"status": "ok", "error": None})
    v.setdefault("complete", True)
    v["has_audio"] = (job_dir(rec["id"]) / "audio.wav").exists()
    v["has_embeddings"] = sorted((rec.get("embeddings") or {}).keys())
    v["suggestions"] = suggestions_for(rec)
    return v


# --------------------------------------------------------------------------
# Exportación
# --------------------------------------------------------------------------
def fmt_time(t: float, srt: bool = False) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def speaker_names(rec: dict) -> dict[str, str]:
    names = {}
    for i, sp in enumerate(rec["speakers"], 1):
        names[sp["id"]] = sp.get("name") or f"Hablante {i}"
    return names


def turns_of(rec: dict) -> list[dict]:
    """Agrupa segmentos consecutivos del mismo hablante. Sin hablantes, agrupa en párrafos."""
    turns: list[dict] = []
    for seg in rec["segments"]:
        last = turns[-1] if turns else None
        same = last and last["speaker"] == seg["speaker"]
        if same and seg["speaker"] is None:  # sin hablante: párrafo de ~320 caracteres o pausa larga
            same = len(last["text"]) < 320 and seg["start"] - last["end"] < 2.5
        if same:
            last["text"] += " " + seg["text"]
            last["end"] = seg["end"]
        else:
            turns.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"], "text": seg["text"]})
    return turns


def export_text(rec: dict, fmt: str) -> str:
    names = speaker_names(rec)
    has_spk = bool(rec["speakers"])
    name = lambda sid: names.get(sid, "Sin identificar")  # noqa: E731
    if fmt == "srt":
        blocks = []
        for i, seg in enumerate(rec["segments"], 1):
            who = f"{name(seg['speaker'])}: " if has_spk else ""
            blocks.append(f"{i}\n{fmt_time(seg['start'], True)} --> {fmt_time(seg['end'], True)}\n{who}{seg['text']}\n")
        return "\n".join(blocks)
    if fmt == "md":
        info = f"_Duración {fmt_time(rec['duration'])}" + (f" · {len(rec['speakers'])} hablantes_" if has_spk else "_")
        lines = [f"# {rec['title']}", "", info, ""]
        for t in turns_of(rec):
            who = f"**{name(t['speaker'])}** · " if has_spk else ""
            lines += [f"{who}`{fmt_time(t['start'])}`", "", t["text"], ""]
        return "\n".join(lines)
    lines = [rec["title"], "=" * len(rec["title"]), ""]
    for t in turns_of(rec):
        who = f" {name(t['speaker'])}" if has_spk else ""
        lines += [f"[{fmt_time(t['start'])}]{who}", t["text"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=startup_check, daemon=True).start()
    yield


app = FastAPI(title="Transcribe tu clase", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/favicon.svg")
@app.get("/favicon.ico")
def favicon():
    return FileResponse(ROOT / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/status")
def status():
    def has(mod: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001
            return False

    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "token": bool(hf_token()),
        "whisper": has("mlx_whisper" if WHISPER_BACKEND == "mlx" else "faster_whisper"),
        "pyannote": has("pyannote.audio"),
        "health": health,
        "device_pref": os.environ.get("DIARIZATION_DEVICE", "auto"),
    }


def parse_speakers(raw: str | None) -> int | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if not raw.isdigit() or not (1 <= int(raw) <= 20):
        raise HTTPException(400, "El número de hablantes debe estar entre 1 y 20.")
    return int(raw)


@app.post("/api/jobs")
async def create_job(request: Request, filename: str, language: str = "auto", vocabulary: str = "",
                     num_speakers: str = "", diarize: str = "1"):
    """El cuerpo es el archivo en bruto (streaming directo a disco; sin copias intermedias)."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Formato no compatible. Usa mp4, mkv, mp3 o wav.")
    if language not in ("auto", "es", "en"):
        raise HTTPException(400, "Idioma no válido.")
    n = parse_speakers(num_speakers)
    jid = uuid.uuid4().hex[:12]
    jd = job_dir(jid)
    jd.mkdir(parents=True)
    dest = jd / f"source{ext}"
    size = 0
    try:
        with dest.open("wb") as out:
            async for chunk in request.stream():
                out.write(chunk)
                size += len(chunk)
    except Exception:
        shutil.rmtree(jd, ignore_errors=True)
        raise HTTPException(400, "Se interrumpió la subida del archivo.")
    if size == 0:
        shutil.rmtree(jd, ignore_errors=True)
        raise HTTPException(400, "El archivo está vacío.")
    meta = {"id": jid, "filename": filename, "language": None if language == "auto" else language,
            "vocabulary": vocabulary, "num_speakers": n, "diarize": diarize != "0", "created": time.time(),
            "size": size}
    (jd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return public_job(new_job(jid, "full", meta, meta["diarize"], n))


@app.get("/api/jobs")
def list_jobs():
    return [public_job(j) for j in jobs.values() if j["status"] in ("queued", "running")]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Trabajo no encontrado")
    return public_job(job)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Trabajo no encontrado")
    job["cancel"] = True
    return {"ok": True}


class DiarizeIn(BaseModel):
    num_speakers: int | None = None


@app.post("/api/history/{rid}/diarize")
def diarize_existing(rid: str, body: DiarizeIn):
    """Identifica hablantes sobre un trabajo ya transcrito, reutilizando el WAV y la transcripción guardados."""
    load_record(rid)
    jd = job_dir(rid)
    if any(j["id"] == rid and j["status"] in ("queued", "running") for j in jobs.values()):
        raise HTTPException(409, "Este trabajo ya se está procesando.")
    if not (jd / "audio.wav").exists() or not (jd / "transcripcion.json").exists():
        raise HTTPException(400, "El audio de este trabajo ya no está guardado, así que no se puede identificar "
                                 "hablantes. Vuelve a subir el archivo.")
    if body.num_speakers is not None and not (1 <= body.num_speakers <= 20):
        raise HTTPException(400, "El número de hablantes debe estar entre 1 y 20.")
    meta = read_json(jd / "meta.json", {}) or {"id": rid, "filename": load_record(rid)["filename"]}
    return public_job(new_job(rid, "retry", meta, True, body.num_speakers))


@app.get("/api/history")
def history():
    items = []
    for p in HISTORY_DIR.glob("*.json"):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        items.append({"id": r["id"], "title": r["title"], "created": r["created"],
                      "duration": r["duration"], "speakers": len(r["speakers"])})
    return sorted(items, key=lambda i: i["created"], reverse=True)


@app.get("/api/history/{rid}")
def history_item(rid: str):
    return view_record(load_record(rid))


@app.get("/api/history/{rid}/audio")
def history_audio(rid: str):
    load_record(rid)
    p = job_dir(rid) / "audio.wav"
    if not p.exists():
        raise HTTPException(404, "El audio de esta transcripción no está guardado.")
    return FileResponse(p, media_type="audio/wav")


class Patch(BaseModel):
    title: str | None = None
    speaker_names: dict[str, str] | None = None


@app.patch("/api/history/{rid}")
def patch_history(rid: str, body: Patch):
    rec = load_record(rid)
    if body.title is not None and body.title.strip():
        rec["title"] = body.title.strip()[:200]
    if body.speaker_names:
        for sp in rec["speakers"]:
            if sp["id"] in body.speaker_names:
                sp["name"] = body.speaker_names[sp["id"]].strip()[:80] or None
    save_record(rec)
    return view_record(rec)


@app.delete("/api/history/{rid}")
def delete_history(rid: str):
    load_record(rid)
    if any(j["id"] == rid and j["status"] in ("queued", "running") for j in jobs.values()):
        raise HTTPException(409, "Este trabajo se está procesando; cancélalo primero.")
    (HISTORY_DIR / f"{rid}.json").unlink(missing_ok=True)
    shutil.rmtree(job_dir(rid), ignore_errors=True)
    return {"ok": True}


@app.get("/api/history/{rid}/export")
def export(rid: str, format: str = "txt"):
    if format not in ("txt", "srt", "md"):
        raise HTTPException(400, "Formato no válido")
    rec = load_record(rid)
    safe = re.sub(r"[^\w\- ]+", "", rec["title"]).strip() or "transcripcion"
    mime = {"txt": "text/plain", "srt": "application/x-subrip", "md": "text/markdown"}[format]
    return PlainTextResponse(
        export_text(rec, format), media_type=f"{mime}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe}.{format}"; filename*=UTF-8\'\'{safe}.{format}'},
    )


class ProfileIn(BaseModel):
    name: str


@app.get("/api/profiles")
def profiles():
    return [{"id": p["id"], "name": p["name"], "created": p["created"]} for p in load_profiles()]


@app.post("/api/history/{rid}/speakers/{sid}/profile")
def save_profile(rid: str, sid: str, body: ProfileIn):
    rec = load_record(rid)
    emb = (rec.get("embeddings") or {}).get(sid)
    name = body.name.strip()
    if not emb:
        raise HTTPException(400, "Este hablante no tiene huella de voz guardada.")
    if not name:
        raise HTTPException(400, "Escribe un nombre para el perfil.")
    existing = next((p for p in load_profiles() if p["name"].lower() == name.lower()), None)
    pid = existing["id"] if existing else uuid.uuid4().hex[:12]
    prof = {"id": pid, "name": name[:80], "embedding": emb, "created": time.time()}
    (PROFILES_DIR / f"{pid}.json").write_text(json.dumps(prof), encoding="utf-8")
    return {"id": pid, "name": prof["name"], "replaced": bool(existing)}


@app.delete("/api/profiles/{pid}")
def delete_profile(pid: str):
    if not ID_RE.fullmatch(pid):
        raise HTTPException(404, "Perfil no encontrado")
    (PROFILES_DIR / f"{pid}.json").unlink(missing_ok=True)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    print("\n  Transcribe tu clase → http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
