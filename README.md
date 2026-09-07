# Advanced AI Agent v2.5 — Continuous Learning, Reliability & Hardening

Agente de IA **local-first** en Python, pensado como núcleo de un asistente tipo Jarvis. La v2.5 mantiene el modelo **GGUF embebido con llama.cpp**, la memoria neuronal y la memoria episódica, y añade dos capas de aprendizaje real: una **red neuronal de experiencia** que actualiza pesos con el feedback y un sistema **LoRA/PEFT versionado** que puede entrenar adaptadores sobre las mejores experiencias. El modelo base nunca se sobrescribe y cada adaptador puede activarse o revertirse.


## Nuevo en v2.5: estabilidad y diagnóstico

La v2.5 endurece el sistema para uso real y automatizado:

- rutas de `data/`, `models/`, `plugins/` y `.env` resueltas desde la carpeta del proyecto, aunque arranques Python desde otro directorio;
- SQLite con WAL y `busy_timeout` para reducir errores `database is locked` durante chat, feedback e indexación concurrentes;
- bloqueo de inferencias simultáneas sobre una misma instancia GGUF/Transformers;
- límite configurable de rondas de herramientas y turno final obligatorio sin herramientas cuando se alcanza;
- protección adicional del conector HTTP contra destinos privados/loopback por defecto;
- limpieza automática de uploads temporales;
- red neuronal de experiencias con **fallback NumPy**, por lo que el aprendizaje rápido con 👍/👎 no depende obligatoriamente de PyTorch;
- `python doctor.py` y `GET /ready` para distinguir un proceso vivo de una instalación realmente lista;
- GitHub Actions CI para Python 3.11/3.12/3.13, Dependabot y `.gitignore` para evitar subir secretos, datos personales y pesos grandes.

`/health` comprueba que el servicio está vivo. Para revisar también el modelo configurado y dependencias de runtime usa:

```bash
python doctor.py --runtime
```

o consulta `GET /ready`.

---

## Aprendizaje neuronal continuo

La IA ahora tiene dos mecanismos diferentes y complementarios:

1. **Experience Policy Network (rápido)**: una pequeña MLP aprende a predecir el resultado esperado de una situación a partir de experiencias calificadas. Usa PyTorch cuando está disponible y, si no lo está, cambia automáticamente a un backend neuronal NumPy que también actualiza pesos reales. Cada 👍/👎 vuelve a entrenar esta red cuando ya existen suficientes muestras. Sus pesos se guardan en `data/experience_policy/` y su señal entra en el prompt de futuras respuestas. Funciona incluso cuando el LLM principal es un GGUF de solo inferencia.
2. **LoRA/PEFT (profundo, opcional)**: selecciona experiencias positivas, genera ejemplos de entrenamiento y crea un adaptador neuronal LoRA nuevo. Los adaptadores se guardan versionados en `data/adapters/`; nunca se modifica destructivamente el modelo base.

Flujo:

```text
conversación
   ↓
experiencia episódica
   ↓
👍 / 👎
   ├──→ reentreno rápido de la red de experiencia
   │          ↓
   │    señal neuronal para próximas decisiones
   │
   └──→ experiencias positivas suficientes
              ↓
          entrenamiento LoRA
              ↓
       adapter v1 / v2 / v3 ...
              ↓
        activar / rollback
```

### Importante sobre GGUF y LoRA

Un `.gguf` cuantizado está pensado principalmente para inferencia. Para entrenar LoRA necesitas el **modelo base original en formato Hugging Face** (carpeta local o repositorio compatible). Después puedes:

- usar el adaptador directamente con `MODEL_BACKEND=transformers_peft`, o
- convertir el LoRA a formato compatible con llama.cpp y establecer `LOCAL_LORA_PATH=...`.

No se finge que el GGUF se reentrena directamente: el cambio neuronal real ocurre en los pesos del adaptador LoRA.

### Instalar entrenamiento LoRA

Windows:

```bat
install_training_windows.bat
```

Linux:

```bash
./install_training_linux.sh
```

Después configura `.env`:

```env
CONTINUAL_LEARNING_ENABLED=true
LORA_BASE_MODEL=C:/modelos/mi-modelo-hf
LORA_MIN_EXAMPLES=8
LORA_MIN_REWARD=0.65
LORA_EPOCHS=1
```

Entrena desde la interfaz con **🧬 Entrenar LoRA**, desde la API o por consola:

```bash
python train_lora.py --user TU_USER_ID
```

### Backend que aplica el LoRA directamente

```env
MODEL_BACKEND=transformers_peft
HF_MODEL_PATH=C:/modelos/mi-modelo-hf
HF_ADAPTER_PATH=data/adapters/USUARIO/VERSION
```

Este backend carga el modelo Hugging Face y el adaptador PEFT dentro del propio proceso. Es más pesado en RAM/VRAM que GGUF.

### Convertir un LoRA para llama.cpp/GGUF

Si tienes un checkout de `llama.cpp` con `convert_lora_to_gguf.py`:

```bash
python convert_lora_for_gguf.py --adapter data/adapters/USUARIO/VERSION --base C:/modelos/mi-modelo-hf --out models/experience-lora.gguf --llama-cpp-dir C:/llama.cpp
```

Luego:

```env
MODEL_BACKEND=embedded_gguf
LOCAL_MODEL_PATH=models/model.gguf
LOCAL_LORA_PATH=models/experience-lora.gguf
LOCAL_LORA_SCALE=1.0
```

### API de aprendizaje

```text
GET  /v1/learning/status?user_id=...
GET  /v1/learning/policy/status?user_id=...
POST /v1/learning/policy/train
POST /v1/learning/lora/train
GET  /v1/learning/lora/adapters?user_id=...
POST /v1/learning/lora/adapters/{version_id}/activate
POST /v1/learning/lora/rollback
```

El entrenamiento LoRA está **desactivado por defecto** para evitar consumir muchos recursos accidentalmente. La red neuronal de experiencia sí puede permanecer activa y aprende de feedback explícito.

---


## Qué incluye

- **Memoria persistente** por usuario en SQLite, más historial separado por conversación.
- **Memoria neuronal semántica**: un Transformer convierte cada recuerdo en un embedding denso y lo recupera por significado, no solo por palabras exactas.
- **Memoria neuronal de experiencias**: guarda episodios `situación → acción → resultado → recompensa → lección`, los recupera por similitud semántica y usa experiencias negativas como advertencias.
- **Recuperación híbrida**: combina memoria neuronal, búsqueda textual, importancia y recencia.
- **Deduplicación semántica automática**: antes de memorizar, compara el nuevo recuerdo con recuerdos conceptualmente similares.
- **Memoria automática**: el LLM propone recuerdos duraderos y el servidor evita duplicados simples.
- **RAG local** para `.txt`, `.md`, `.json`, `.csv`, `.pdf` y varios archivos de código/configuración.
- **Búsqueda híbrida** en documentos usando SQLite FTS5 + similitud léxica local, sin depender de una base vectorial externa.
- **Herramientas**: calculadora, hora UTC, memoria, documentos, HTTP allowlist, búsqueda SearXNG opcional.
- **Plugins Python** cargados desde `plugins/` sin modificar el núcleo.
- **Visión** usando un endpoint de chat multimodal compatible con OpenAI.
- **STT** (`/audio/transcriptions`) y **TTS** (`/audio/speech`) cuando el proveedor los soporta.
- **Herramientas de PC opcionales**, apagadas por defecto y con allowlist de ejecutables.
- **Telegram** y **Discord** como procesos opcionales.
- **Panel web/PWA** usable desde PC o Android y agregable a la pantalla de inicio.
- **WebSocket** y API REST FastAPI.
- **Token Bearer opcional** para proteger `/v1/*` cuando expones el servidor a la LAN.

## 0. Modelo embebido

El backend predeterminado ahora es:

```env
MODEL_BACKEND=embedded_gguf
LOCAL_MODEL_PATH=models/model.gguf
```

Esto carga las **pesos de la red neuronal GGUF directamente en Python** con `llama-cpp-python`. No hace falta abrir LM Studio, Ollama ni otro servidor. Sí necesitas un archivo GGUF: los pesos son la parte grande y entrenada del LLM; no pueden sustituirse por unas pocas líneas de código.

Instala el runtime una sola vez:

Windows:

```bat
install_local_model_windows.bat
```

Linux:

```bash
./install_local_model_linux.sh
```

Después coloca tu archivo como `models/model.gguf` y ejecuta el inicio normal. También puedes descargar un GGUF de Hugging Face con el ayudante genérico:

```bash
python download_model.py --repo OWNER/REPO-GGUF --file MODELO.Q4_K_M.gguf
```

El endpoint `GET /v1/model/status` muestra si el archivo existe y si ya fue cargado. Para volver a LM Studio/Ollama/API usa `MODEL_BACKEND=openai_compatible`.

## 1. Instalar

Python 3.11+ recomendado.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-neural.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-neural.txt
```

Para Telegram/Discord:

```bash
pip install -r requirements-connectors.txt
```

Copia `.env.example` a `.env`.

## 2. Backend OpenAI-compatible (opcional: LM Studio/Ollama/vLLM)

Si prefieres un servidor externo/local separado, cambia primero `MODEL_BACKEND=openai_compatible` y configura, por ejemplo:

```env
AI_BASE_URL=http://127.0.0.1:1234/v1
AI_API_KEY=local-key
AI_MODEL=nombre-exacto-del-modelo-cargado
```

Luego:

```bash
python run.py
```

Abre:

- Panel: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Salud: `http://127.0.0.1:8000/health`

## 3. Usarlo desde Android

Si el servidor está en tu PC y el teléfono está en la misma red:

```env
APP_HOST=0.0.0.0
API_TOKEN=cambia-esto-por-un-token-largo
```

Obtén la IP LAN del PC (por ejemplo `192.168.1.50`) y abre en Android:

```text
http://192.168.1.50:8000/
```

La interfaz funciona desde la LAN. Para instalarla como PWA completa en Android necesitas un contexto seguro (normalmente HTTPS; `localhost` es la excepción). Si activas `API_TOKEN`, guarda ese token en `localStorage.apiToken` desde la consola del navegador o adapta el panel a tu método de autenticación. Para exposición fuera de la LAN, usa HTTPS y un reverse proxy; no abras el puerto directamente a Internet.


## 4. Memoria neuronal

Por defecto usa un Transformer multilingüe local:

```env
NEURAL_MEMORY_ENABLED=true
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Cada recuerdo se guarda dos veces de forma complementaria:

1. El texto original en SQLite.
2. Su vector neuronal normalizado en `neural_memory_vectors`.

Cuando preguntas algo, el agente calcula el embedding de la consulta y aplica similitud coseno contra los recuerdos. El ranking final mezcla similitud semántica, importancia, recencia y coincidencias textuales. Por ejemplo, el recuerdo `Mi PC usa una RTX 3050` puede recuperarse al preguntar `¿qué tarjeta gráfica tengo?`.

Para usar un servidor de embeddings compatible con OpenAI en vez del Transformer local:

```env
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_BASE_URL=http://127.0.0.1:1234/v1
EMBEDDING_MODEL=nombre-del-modelo-de-embeddings
```

Endpoints:

- `GET /v1/neural-memory/status?user_id=demo`
- `POST /v1/neural-memory/reindex?user_id=demo`
- `GET /v1/neural-memory/search?user_id=demo&q=mi+gpu`

La primera ejecución en modo local puede necesitar descargar el modelo de embeddings. Una vez almacenado en caché puede trabajar localmente.

## 5. Memoria de experiencias y aprendizaje (v2.5)

La IA mantiene una memoria **episódica**, separada de los hechos y preferencias. Cada interacción puede convertirse automáticamente en una experiencia con esta forma:

```text
situación -> acción/respuesta -> herramientas -> resultado -> recompensa -> lección
```

La situación y la lección se convierten en embeddings con el mismo encoder neuronal. Antes de responder, el agente busca experiencias semánticamente parecidas. Las experiencias con recompensa positiva sirven como patrones útiles; las negativas se presentan al modelo como advertencias para evitar repetir un fallo.

Las experiencias nuevas empiezan como `unrated` con recompensa `0`. Una aplicación cliente o el usuario puede calificarlas después entre `-1` y `+1` y añadir el resultado real o una lección:

- `GET /v1/experiences?user_id=demo`
- `GET /v1/experiences/search?user_id=demo&q=problema`
- `GET /v1/experiences/status?user_id=demo`
- `POST /v1/experiences/{id}/feedback`
- `POST /v1/experiences/export`

Ejemplo de feedback:

```json
{
  "user_id": "demo",
  "reward": 1.0,
  "outcome": "La solución funcionó",
  "lesson": "Reutilizar este procedimiento cuando aparezca el mismo error"
}
```

`/v1/experiences/export` sigue disponible para inspección/datasets. En v2.5, el feedback también puede reentrenar inmediatamente la **Experience Policy Network**, cuyos pesos sí cambian y cuya señal se utiliza en futuras respuestas. Si activas `CONTINUAL_LEARNING_ENABLED`, el sistema puede entrenar adaptadores **LoRA/PEFT** directamente desde las experiencias positivas, aplicar una puerta de validación y conservar versiones anteriores para rollback.

Configuración:

```env
EXPERIENCE_MEMORY_ENABLED=true
EXPERIENCE_AUTO_STORE=true
EXPERIENCE_RESULTS=5
EXPERIENCE_MIN_SIMILARITY=0.18
EXPERIENCE_RECENCY_HALF_LIFE_DAYS=90
```

## 6. RAG / documentos

Sube documentos desde el botón **Añadir documento** o con:

```bash
curl -F "user_id=demo" -F "file=@manual.pdf" http://127.0.0.1:8000/v1/knowledge/upload
```

El agente recupera automáticamente fragmentos relevantes antes de responder. Endpoints adicionales:

- `POST /v1/knowledge/text`
- `GET /v1/knowledge/search`
- `GET /v1/knowledge/sources`
- `DELETE /v1/knowledge/source`

## 7. Visión y voz

Puedes reutilizar el endpoint principal o configurar modelos/servidores especializados:

```env
VISION_BASE_URL=http://127.0.0.1:1234/v1
VISION_MODEL=modelo-con-vision
STT_BASE_URL=http://servidor-stt/v1
STT_MODEL=whisper-1
TTS_BASE_URL=http://servidor-tts/v1
TTS_MODEL=gpt-4o-mini-tts
TTS_VOICE=alloy
```

Que una ruta sea compatible depende del servidor/modelo concreto. LM Studio puede manejar chat y visión con modelos compatibles, pero no todos los modelos/servidores ofrecen STT/TTS.

## 8. Web

`HTTP_ALLOWLIST` controla qué dominios puede consultar el agente:

```env
HTTP_ALLOWLIST=api.github.com,example.com
```

Para búsqueda general puedes conectar una instancia SearXNG:

```env
SEARXNG_URL=http://127.0.0.1:8080
```

Los resultados web se introducen al modelo como **datos no confiables**, no como instrucciones.

## 9. Control del PC

Está deshabilitado de fábrica:

```env
ENABLE_PC_TOOLS=false
```

Para activarlo:

```env
ENABLE_PC_TOOLS=true
PC_COMMAND_ALLOWLIST=ipconfig,ping
```

`pc_run_allowed` usa `subprocess` con `shell=False` y solo permite ejecutables incluidos explícitamente. No pongas intérpretes como `python`, `powershell`, `cmd`, `bash` o similares en la allowlist si quieres conservar ese límite de seguridad.

## 10. Plugins

Cada archivo `plugins/*.py` puede registrar una herramienta:

```python
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mi_herramienta",
        "description": "...",
        "parameters": {"type": "object", "properties": {}}
    }
}

def run(args, context):
    return "resultado"
```

Recarga sin reiniciar con `POST /v1/plugins/reload`.

## 11. Telegram

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USER_IDS=123456789
```

```bash
python connectors/telegram_bot.py
```

Si `TELEGRAM_ALLOWED_USER_IDS` está vacío, el bot no restringe por ID; para uso real es mejor configurarlo.

## 12. Discord

```env
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USER_IDS=123456789
```

Activa **Message Content Intent** para el bot y ejecuta:

```bash
python connectors/discord_bot.py
```

En servidores solo responde cuando lo mencionan; en DM responde directamente.

## Arquitectura

```text
Android/PWA ─┐
Discord ─────┤
Telegram ────┼──> FastAPI / Agent
REST/WS ─────┘          │
                        ├── LLM GGUF embebido / OpenAI-compatible
                        ├── Memoria SQLite
                        ├── Memoria neuronal / embeddings
                        ├── RAG SQLite/FTS5
                        ├── Herramientas
                        ├── Plugins
                        ├── Visión / STT / TTS
                        └── PC + Web (opcionales)
```

## Importante

Esto es un **sistema de agente funcional**, no AGI ni una IA que se auto-mejora sin control. Su inteligencia depende de los pesos del modelo GGUF cargado o del modelo conectado. El proyecto le añade persistencia, recuperación de conocimiento, herramientas y canales para convertir ese modelo en un asistente mucho más útil.


## 9. Pruebas y GitHub CI

Ejecuta toda la batería local:

```bash
python -m compileall -q .
python test_smoke.py
python test_neural_memory.py
python test_experience_memory.py
python test_learning_system.py
python test_embedded_model.py
python test_hardening.py
python doctor.py
```

El repositorio incluye `.github/workflows/ci.yml`, que repite estas comprobaciones en Python 3.11, 3.12 y 3.13 sin requerir descargar un LLM grande. Los pesos `*.gguf`, `.env`, `data/` y adaptadores personales están excluidos por `.gitignore`.


## 10. Publicar en GitHub

El proyecto incluye `publish_github.ps1` (Windows) y `publish_github.sh` (Linux/macOS). Ambos crean por defecto un repositorio **privado** mediante GitHub CLI (`gh`), inicializan Git si hace falta y hacen el primer push. Antes de publicarlo, revisa `.env`, `data/` y los pesos del modelo: `.gitignore` ya los excluye, pero nunca conviene subir secretos ni memoria privada.

Windows PowerShell:

```powershell
./publish_github.ps1 -RepoName advanced-ai-agent
```

Linux/macOS:

```bash
./publish_github.sh advanced-ai-agent
```

Para hacerlo público debes pedirlo explícitamente (`-Public` en PowerShell o `public` como segundo argumento del script Bash).
