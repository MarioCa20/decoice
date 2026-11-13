# Transcribe tu clase

Aplicación web local para transcribir videos o audios de clases e identificar **quién habla**. Todo el procesamiento ocurre en tu equipo; internet solo se necesita la primera vez para descargar los modelos.

- Transcripción: **mlx-whisper** en Mac con Apple Silicon, **faster-whisper** (CUDA o CPU) en Windows/Linux. Modelo `large-v3-turbo` en ambos.
- Hablantes: [pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.4 (`pyannote/speaker-diarization-3.1`), en GPU (`cuda` o `mps`) si hay, si no en CPU.
- Servidor: FastAPI. Interfaz: un solo `index.html`.

## Requisitos

| Herramienta | Versión | Notas |
|---|---|---|
| Python | **3.12** | No uses 3.13/3.14: torch 2.8 no publica paquetes para ellas. |
| ffmpeg | cualquiera reciente | Ver instalación abajo. |
| Git | cualquiera | Para clonar. |
| Node.js | **no se necesita** | La interfaz es un solo HTML, sin build. |
| GPU NVIDIA (opcional) | driver reciente | Acelera Whisper y pyannote en Windows. Sin ella funciona en CPU (más lento). |

## Instalación en Windows (PowerShell)

```powershell
# 1. Herramientas (una sola vez)
winget install Python.Python.3.12
winget install Gyan.FFmpeg
winget install Git.Git
# Cierra y vuelve a abrir PowerShell para que el PATH se actualice
ffmpeg -version

# 2. Clonar
git clone https://github.com/MarioCa20/decoice.git
cd decoice

# 3. Entorno virtual + dependencias
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Con GPU NVIDIA**, `pip install -r requirements.txt` instala torch solo-CPU. Para usar CUDA, reinstala torch/torchaudio desde el índice de PyTorch (ejemplo CUDA 12.6):

```powershell
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

### Modelos (no están en el repo)

No hay que descargarlos a mano: se bajan solos de Hugging Face **la primera vez** que se usan y quedan en la caché (`%USERPROFILE%\.cache\huggingface`). Whisper large-v3-turbo pesa ~1,6 GB y pyannote unos cientos de MB. Para precargarlos y evitar la espera en la primera transcripción:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"
```

Los de pyannote se descargan al arrancar `python app.py` (necesitan el token del paso siguiente).

## Instalación en macOS (Apple Silicon)

```bash
brew install ffmpeg
python3.12 -m venv .venv          # o: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt   # instala mlx-whisper automáticamente
```

### Versiones fijadas (y por qué)

| Paquete | Versión | Motivo |
|---|---|---|
| `torch` | `2.8.0` | Última serie que conserva `torchaudio.list_audio_backends()`. |
| `torchaudio` | `2.8.0` | Debe coincidir con torch. A partir de 2.9 se eliminó `list_audio_backends()` y pyannote 3.x falla con `module 'torchaudio' has no attribute 'list_audio_backends'`. |
| `pyannote.audio` | `3.4.0` | Versión 3.x probada. |
| `huggingface_hub` | `>=0.30,<1.0` | pyannote 3.x llama a Hugging Face con `use_auth_token`, que `huggingface_hub` 1.x eliminó. |

Además, la app no le pasa a pyannote una ruta de archivo: carga el WAV una vez como tensor y lo entrega en memoria (`{"waveform": tensor, "sample_rate": 16000}`).

### Token de Hugging Face (necesario para identificar hablantes)

Los modelos de pyannote son gratuitos pero exigen **aceptar sus condiciones de uso con tu cuenta**. Sin esto verás un error 403 aunque el token sea válido.

1. Crea una cuenta en <https://huggingface.co>.
2. Abre y acepta las condiciones en estas dos páginas:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
3. Crea un token de lectura en <https://huggingface.co/settings/tokens> (tipo *Read*).
4. Copia `.env.example` como `.env` y pega el token:

   ```powershell
   # Windows (PowerShell)
   Copy-Item .env.example .env
   notepad .env        # HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ```

   ```bash
   # macOS / Linux
   cp .env.example .env
   ```

El token nunca se escribe en el código y `.env` está en `.gitignore`.

## Uso

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
python app.py
```

```bash
# macOS / Linux
source .venv/bin/activate
python app.py
```

Abre <http://localhost:8000>. **Al arrancar**, la app carga pyannote y lo prueba con 2 s de audio sintético; el resultado sale en la terminal (`[arranque] ✔ pyannote OK …`) y, si falla, la interfaz muestra un aviso con la causa probable. Así los errores de versiones, token o condiciones no aceptadas aparecen en segundos y no tras minutos de transcripción. Mientras tanto puedes transcribir sin hablantes.

1. Arrastra el video o audio (mp4, mkv, mp3, wav).
2. Elige idioma, **vocabulario técnico** opcional (se pasa a Whisper como `initial_prompt`), **número de hablantes** opcional y el interruptor **Identificar hablantes** (activado por defecto).
3. Pulsa **Transcribir** y sigue las etapas: subiendo, extrayendo audio, transcribiendo (bloque X de N), identificando hablantes y uniendo resultados, con porcentaje y tiempo transcurrido. La terminal muestra lo mismo.
4. En el resultado, haz clic en “Hablante 1” para ponerle nombre. Haz clic en una marca de tiempo para oír esa parte.
5. Con el icono de micrófono junto a un hablante guardas su **voz**. En transcripciones nuevas, si una voz se parece a un perfil, la app lo **sugiere**; tú decides si lo confirmas.
6. Exporta en `.txt`, `.srt` o `.md` (con los nombres de los hablantes) o copia el texto.

Si desactivaste “Identificar hablantes”, o la identificación falló, la transcripción se muestra igualmente y aparece un botón para **identificar (o reintentar) hablantes** reutilizando el audio y la transcripción ya guardados.

## Cómo procesa (y por qué es más rápido y robusto)

1. **Una sola extracción de audio**: ffmpeg convierte el archivo a `audio.wav` (16 kHz, mono, PCM 16 bits). Ningún modelo decodifica el video original. Si subes un WAV que ya es 16 kHz mono, no se reconvierte. Al terminar se borra el video subido; el WAV se conserva.
2. **Whisper por bloques** de 10 min con 8 s de solapamiento: da progreso real, permite cancelar entre bloques y descarta duplicados en la zona de solapamiento.
3. **Una etapa a la vez**: tras transcribir se libera Whisper y solo entonces se carga pyannote, que también se libera al terminar (limpia caché MPS/CUDA). Nunca hay dos modelos en memoria.
4. **Resultados intermedios en disco** (`data/trabajos/<id>/`): si una etapa falla, no se pierde lo anterior.
5. **Cancelar** interrumpe Whisper entre bloques y pyannote en el siguiente lote; libera memoria y deja guardado lo procesado (una transcripción parcial queda marcada como incompleta).

## Dónde se guardan los datos (`data/`)

| Ruta | Contenido |
|---|---|
| `data/trabajos/<id>/audio.wav` | Audio extraído (se reutiliza para reintentos y para reproducir). Pesa ~2 MB por minuto. |
| `data/trabajos/<id>/transcripcion.json` | Segmentos con tiempos (se actualiza bloque a bloque). |
| `data/trabajos/<id>/diarizacion.json` | Turnos por hablante y huellas de voz. |
| `data/trabajos/<id>/meta.json` | Parámetros del trabajo. |
| `data/historial/<id>.json` | Resultado final unido (lo que ves en la interfaz). |
| `data/perfiles/` | Perfiles de voz (nombre + huella). |

Borrar una transcripción desde la barra lateral elimina también su carpeta de trabajo y el audio.

## Configuración (`.env`)

```bash
HF_TOKEN=hf_...
DIARIZATION_DEVICE=auto        # auto (cuda > mps > cpu) | cuda | mps | cpu
PYTORCH_ENABLE_MPS_FALLBACK=1  # solo macOS, opcional, ver abajo
WHISPER_CHUNK_SECONDS=600      # opcional: tamaño de bloque
WHISPER_OVERLAP_SECONDS=8      # opcional: solapamiento
```

- **`DIARIZATION_DEVICE=cpu`**: úsalo si pyannote en `mps` resulta inestable o más lento. Si falla en `mps`, la app ya reintenta sola en CPU y lo avisa en la terminal.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`**: si aparecen errores de operaciones no soportadas en MPS (`NotImplementedError: ... not currently implemented for the MPS device`), esta variable hace que PyTorch ejecute esas operaciones en CPU. Ponla en `.env` o exporta la variable antes de `python app.py`.

## Diagnóstico por terminal

Cada etapa registra su inicio y fin con tiempo empleado y memoria aproximada (RAM del proceso y memoria MPS), por ejemplo:

```
12:01:03 [ab12cd34ef56] ▶ cargando pyannote en mps …  (RAM 811 MB · MPS 0 MB)
12:01:09 [ab12cd34ef56] pyannote [mps] comparando voces 42% · global 79%  (RAM 2410 MB · MPS 1530 MB)
```

## Probar con un clip de 5 minutos y estimar la clase completa

```powershell
# (en Windows activa antes el entorno: .\.venv\Scripts\Activate.ps1)
ffmpeg -i clase.mp4 -t 300 -vn -ar 16000 -ac 1 -c:a pcm_s16le clip.wav
python scripts/prueba_clip.py clip.wav --full-minutes 110
```

Imprime el tiempo de cada etapa (extracción, Whisper, pyannote) y una estimación para la duración que indiques en `--full-minutes` (escala linealmente desde el clip). Opciones: `--no-diarize`, `--speakers N`, `--language es|en|auto`. La primera ejecución incluye la carga/descarga de modelos y sobreestima: repite la prueba para una cifra realista.

## Problemas frecuentes

| Mensaje | Qué hacer |
|---|---|
| `module 'torchaudio' has no attribute 'list_audio_backends'` | Entorno con versiones nuevas. Recrea el entorno con Python 3.12 y `pip install -r requirements.txt`. |
| Hugging Face rechazó el acceso (403) | Acepta las condiciones de los dos modelos de pyannote con la misma cuenta del token. |
| Falta ffmpeg | macOS: `brew install ffmpeg`. Windows: `winget install Gyan.FFmpeg`; reabre PowerShell. |
| `torch.cuda.is_available()` es `False` en Windows | Tienes torch solo-CPU: reinstálalo con el `--index-url` de CUDA indicado arriba. |
| Faltan hablantes, pero la transcripción está | Usa el botón “Reintentar identificación de hablantes” del resultado. |
| Se detectan hablantes de más o de menos | Indica el número de hablantes. |
| Palabras técnicas mal escritas | Agrégalas al vocabulario técnico. |
| El Mac se calienta | Es normal en clases largas. Transcribe sin hablantes primero e identifica después, o usa `DIARIZATION_DEVICE=cpu`. |

## Notas

- La cola procesa un trabajo a la vez.
- Ajusta el umbral de sugerencia de voz con `MATCH_THRESHOLD` en `app.py`.
