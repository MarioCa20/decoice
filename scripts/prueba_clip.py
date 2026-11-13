#!/usr/bin/env python
"""Prueba por etapas con un clip corto: imprime el tiempo de cada etapa y estima la clase completa.

Uso:
    ffmpeg -i clase.mp4 -t 300 -vn -ar 16000 -ac 1 -c:a pcm_s16le clip.wav
    python scripts/prueba_clip.py clip.wav --full-minutes 110

Opciones:
    --full-minutes N   duración de la clase completa, para estimar el tiempo total
    --no-diarize       solo extracción + transcripción
    --speakers N       número de hablantes (opcional)
    --language es|en   idioma (por defecto es; 'auto' detecta)
Acepta también un mp4/mkv/mp3: la extracción a WAV se mide como primera etapa.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402  (carga .env y reutiliza exactamente el mismo código que la app)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--full-minutes", type=float)
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--speakers", type=int)
    ap.add_argument("--language", default="es", choices=["es", "en", "auto"])
    args = ap.parse_args()

    src = Path(args.entrada).resolve()
    work = app.DATA / "prueba"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio.wav"
    job = {"id": "prueba", "weights": app.WEIGHTS_DIARIZE, "progress": 0, "started": time.time(), "cancel": False,
           "detail": ""}
    times: dict[str, float] = {}
    notes: list[str] = []

    print(f"Entrada: {src.name}\nDispositivo de pyannote: {app.resolve_device()} "
          f"(DIARIZATION_DEVICE={app.os.environ.get('DIARIZATION_DEVICE', 'auto')})\n")

    t = time.time()
    if src.suffix.lower() == ".wav" and app.is_target_wav(src):
        wav.write_bytes(src.read_bytes())
    else:
        app.extract_audio(job, src, wav)
    times["extracción de audio"] = time.time() - t
    dur = app.wav_duration(wav)
    print(f"Audio: {dur / 60:.1f} min\n")

    t = time.time()
    segs = app.transcribe_blocks(job, wav, dur, None if args.language == "auto" else args.language, "",
                                 work / "transcripcion.json")
    times["transcripción (Whisper)"] = time.time() - t
    print(f"  → {len(segs)} segmentos")

    if not args.no_diarize:
        t = time.time()
        try:
            turns, _ = app.diarize(job, wav, args.speakers)
            times["identificación de hablantes"] = time.time() - t
            print(f"  → {len(set(x['speaker'] for x in turns))} hablantes, {len(turns)} turnos")
        except Exception as e:  # noqa: BLE001
            times["identificación de hablantes (FALLÓ)"] = time.time() - t
            notes.append("La diarización falló: " + (str(e) if isinstance(e, RuntimeError) else app.friendly_pyannote_error(e)))

    total = sum(times.values())
    print("\n==== Tiempos ====")
    for k, v in times.items():
        print(f"  {k:<38} {app.fmt_clock(v):>8}   ({v / dur:.2f}× la duración del audio)")
    print(f"  {'TOTAL':<38} {app.fmt_clock(total):>8}")
    if args.full_minutes:
        factor = args.full_minutes * 60 / dur
        print(f"\n==== Estimación para {args.full_minutes:.0f} min ====")
        ok = {k: v for k, v in times.items() if "FALLÓ" not in k}
        for k, v in ok.items():
            print(f"  {k:<38} ~{app.fmt_clock(v * factor):>8}")
        print(f"  {'TOTAL aproximado':<38} ~{app.fmt_clock(sum(ok.values()) * factor):>8}")
        if len(ok) < len(times):
            print("  (sin contar la etapa que falló)")
        print("  (Lineal sobre el clip. Whisper y la extracción escalan casi lineales; pyannote puede crecer algo "
              "más en clases largas. El primer arranque incluye la descarga/carga de modelos y sobreestima.)")
    for n in notes:
        print("\n! " + n)
    print(f"\nMemoria al terminar: {app.mem_info()}")
    return 1 if notes else 0


if __name__ == "__main__":
    sys.exit(main())
