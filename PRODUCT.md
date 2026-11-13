# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack
Backend Python + FastAPI; frontend a single index.html with vanilla CSS/JS (fixed by the user).

## Users
Estudiantes universitarios que transcriben clases grabadas (video/audio) para estudiar y buscar contenido; clases con docente y a veces compañeros hablando.

## Product Purpose
Transcribir localmente (Apple Silicon) videos de clases e identificar quién habla, con marcas de tiempo, exportación y perfiles de voz. Todo el procesamiento es local; internet solo para descargar modelos.

## Positioning
Transcripción con hablantes 100% en el equipo del estudiante, sin subir la clase a ningún servicio; las coincidencias de voz son sugerencias que el usuario confirma.

## Operating Context
macOS Apple Silicon, `python app.py` en localhost:8000, mlx-whisper large-v3-turbo, pyannote.audio en mps, token de Hugging Face en .env. Tareas largas en segundo plano.

## Capabilities and Constraints
Subida mp4/mkv/mp3/wav; idioma es/en/auto; vocabulario técnico (initial_prompt); nº de hablantes opcional; etapas visibles; fusión por mayor solapamiento; renombrar hablantes; perfiles de voz como sugerencia; exportar txt/srt/md; copiar; historial local; modo claro/oscuro. Interfaz en español.

## Brand Commitments
Nombre visible "Transcribe tu clase". Wordmark tipográfico e ícono (onda de sonido que se convierte en texto) usados en título de pestaña y favicon. Libertad visual total, confirmada.

## Evidence on Hand
Sin contenido real; todo ejemplo de demostración debe marcarse como sintético.

## Product Principles
1. Local y privado: nunca sugerir que algo sale del equipo.
2. Las sugerencias de IA se presentan como sugerencias, no hechos.
3. El progreso es honesto: etapas reales, errores con recuperación.
4. El texto de la clase es el protagonista.
