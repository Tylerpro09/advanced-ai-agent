from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Always load .env from the project directory, not from whichever CWD launched Python.
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    # Main chat / reasoning model.
    # embedded_gguf = load a GGUF directly in this Python process (no LM Studio/Ollama/API required).
    # openai_compatible = use LM Studio, Ollama, vLLM or another /v1 server.
    model_backend: str = "embedded_gguf"
    ai_base_url: str = "http://127.0.0.1:1234/v1"
    ai_api_key: str = "local-key"
    ai_model: str = "local-model"
    ai_temperature: float = 0.35
    ai_timeout_seconds: float = 300.0

    # Embedded GGUF / llama.cpp backend.
    local_model_path: str = "models/model.gguf"
    local_model_name: str = "local-gguf"
    local_model_context: int = 8192
    local_model_gpu_layers: int = 0
    local_model_threads: int = 0
    local_model_chat_format: str = ""
    local_model_verbose: bool = False
    # Optional llama.cpp-compatible LoRA adapter already converted to GGUF.
    local_lora_path: str = ""
    local_lora_scale: float = 1.0

    # Optional specialist endpoints/models. If blank, main endpoint/model is reused.
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = ""
    stt_base_url: str = ""
    stt_api_key: str = ""
    stt_model: str = "whisper-1"
    tts_base_url: str = ""
    tts_api_key: str = ""
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_token: str = ""
    database_path: str = "data/agent.db"
    uploads_dir: str = "data/uploads"
    max_context_messages: int = 20
    memory_results: int = 8
    rag_results: int = 6

    # Neural semantic memory. Local mode uses a real Transformer encoder.
    neural_memory_enabled: bool = True
    embedding_provider: str = "local"  # local | openai-compatible | off
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_batch_size: int = 32
    neural_sync_limit: int = 256
    neural_min_similarity: float = 0.20
    neural_duplicate_similarity: float = 0.88
    neural_semantic_weight: float = 0.78
    neural_importance_weight: float = 0.17
    neural_recency_weight: float = 0.05
    neural_recency_half_life_days: float = 180.0

    # Episodic experience memory: situation -> action -> outcome -> reward -> lesson.
    experience_memory_enabled: bool = True
    experience_results: int = 5
    experience_min_similarity: float = 0.18
    experience_recency_half_life_days: float = 90.0
    experience_auto_store: bool = True

    # Fast neural learning layer. This MLP updates its own weights from rated experiences
    # and feeds an expected success/risk signal back into future prompts.
    experience_policy_enabled: bool = True
    experience_policy_dir: str = "data/experience_policy"
    experience_policy_hidden: int = 64
    experience_policy_dropout: float = 0.10
    experience_policy_lr: float = 0.002
    experience_policy_epochs: int = 20
    experience_policy_min_samples: int = 4
    experience_policy_max_samples: int = 2000
    experience_policy_use_gpu: bool = False

    # Continual LoRA learning. GGUF itself is inference-oriented; LoRA training uses the
    # original Hugging Face model and produces versioned PEFT adapters with rollback.
    continual_learning_enabled: bool = False
    lora_base_model: str = ""
    lora_output_dir: str = "data/adapters"
    lora_min_examples: int = 8
    lora_min_reward: float = 0.65
    lora_max_examples: int = 1000
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    lora_epochs: float = 1.0
    lora_learning_rate: float = 0.0002
    lora_batch_size: int = 1
    lora_gradient_accumulation: int = 8
    lora_max_seq_length: int = 2048
    lora_gradient_checkpointing: bool = True
    lora_local_files_only: bool = False
    lora_validation_fraction: float = 0.15
    lora_min_validation_examples: int = 2
    lora_max_eval_regression: float = 1.05
    lora_filter_sensitive: bool = True
    # 0 = manual only. Example 20 = after each 20 qualifying positive experiences, queue a LoRA training run.
    lora_auto_train_every: int = 0

    # Direct Hugging Face + PEFT inference backend (optional alternative to GGUF).
    hf_model_path: str = ""
    hf_adapter_path: str = ""
    # Optional user whose active adapter from data/adapters/registry.json is restored after restart.
    hf_adapter_user: str = ""
    hf_max_new_tokens: int = 768
    hf_local_files_only: bool = False
    enable_tools: bool = True
    auto_memory: bool = True
    enable_pc_tools: bool = False
    enable_plugins: bool = True
    http_allowlist: str = ""
    searxng_url: str = ""
    http_allow_private_networks: bool = False
    pc_command_allowlist: str = ""
    max_upload_mb: int = 20
    sqlite_busy_timeout_ms: int = 10000
    max_tool_rounds: int = 6
    max_tool_result_chars: int = 40000

    # Optional bot connectors.
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""
    discord_bot_token: str = ""
    discord_allowed_user_ids: str = ""

    @property
    def allowed_http_domains(self) -> set[str]:
        return {x.strip().lower() for x in self.http_allowlist.split(",") if x.strip()}

    @property
    def allowed_pc_commands(self) -> set[str]:
        return {x.strip() for x in self.pc_command_allowlist.split(",") if x.strip()}

    @property
    def telegram_users(self) -> set[int]:
        out: set[int] = set()
        for value in self.telegram_allowed_user_ids.split(","):
            value = value.strip()
            if value:
                try:
                    out.add(int(value))
                except ValueError:
                    pass
        return out

    @property
    def discord_users(self) -> set[int]:
        out: set[int] = set()
        for value in self.discord_allowed_user_ids.split(","):
            value = value.strip()
            if value:
                try:
                    out.add(int(value))
                except ValueError:
                    pass
        return out

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @staticmethod
    def _project_path(value: str) -> str:
        if not value:
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    def normalize_paths(self) -> None:
        # These settings are always filesystem paths. Resolving them makes startup independent
        # from the process working directory (systemd, Task Scheduler, IDEs, etc.).
        self.database_path = self._project_path(self.database_path)
        self.uploads_dir = self._project_path(self.uploads_dir)
        self.experience_policy_dir = self._project_path(self.experience_policy_dir)
        self.lora_output_dir = self._project_path(self.lora_output_dir)
        self.local_model_path = self._project_path(self.local_model_path)
        if self.local_lora_path:
            self.local_lora_path = self._project_path(self.local_lora_path)

    def ensure_dirs(self) -> None:
        self.normalize_paths()
        Path(self.uploads_dir).mkdir(parents=True, exist_ok=True)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.experience_policy_dir).mkdir(parents=True, exist_ok=True)
        Path(self.lora_output_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
