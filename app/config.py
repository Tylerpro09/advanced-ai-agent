from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    # LM Studio/OpenAI-compatible is the zero-config default. Set embedded_gguf
    # explicitly when you want this process to load models/model.gguf itself.
    model_backend: str = "openai_compatible"
    ai_base_url: str = "http://127.0.0.1:1234/v1"
    ai_api_key: str = "lm-studio"
    # "auto" discovers the first model exposed by GET /v1/models.
    ai_model: str = "auto"
    ai_temperature: float = 0.35
    ai_timeout_seconds: float = 300.0

    local_model_path: str = "models/model.gguf"
    local_model_name: str = "local-gguf"
    local_model_context: int = 8192
    local_model_gpu_layers: int = 0
    local_model_threads: int = 0
    local_model_chat_format: str = ""
    local_model_verbose: bool = False
    local_lora_path: str = ""
    local_lora_scale: float = 1.0

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

    neural_memory_enabled: bool = True
    embedding_provider: str = "local"
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

    experience_memory_enabled: bool = True
    experience_results: int = 5
    experience_min_similarity: float = 0.18
    experience_recency_half_life_days: float = 90.0
    experience_auto_store: bool = True

    human_like_learning_enabled: bool = True
    human_learning_results: int = 6
    human_consolidation_similarity: float = 0.84
    human_retrieval_min_similarity: float = 0.18
    human_positive_reward_threshold: float = 0.35
    human_negative_reward_threshold: float = -0.35
    human_reinforcement_rate: float = 0.35
    human_retrieval_reinforcement: float = 0.03
    human_max_memory_strength: float = 8.0
    human_forgetting_half_life_days: float = 45.0
    human_novelty_threshold: float = 0.72
    human_low_confidence_threshold: float = 0.38
    human_reflection_enabled: bool = True

    developmental_learning_enabled: bool = True
    developmental_auto_goals: bool = True
    developmental_max_open_goals: int = 12
    developmental_goal_curiosity_threshold: float = 0.70
    developmental_goal_mastery_successes: int = 4
    developmental_curiosity_novelty_weight: float = 0.50
    developmental_curiosity_uncertainty_weight: float = 0.35
    developmental_curiosity_gap_weight: float = 0.15
    developmental_feedback_xp: float = 2.0
    developmental_failure_learning_xp: float = 0.35
    developmental_practice_xp: float = 1.5
    developmental_practice_success_score: float = 0.72
    developmental_auto_analogies: bool = True
    developmental_analogy_curiosity_threshold: float = 0.78
    developmental_auto_practice: bool = False
    developmental_sandbox_execution_enabled: bool = False
    developmental_sandbox_timeout_seconds: float = 4.0
    developmental_sandbox_memory_mb: int = 256
    developmental_sandbox_max_output_chars: int = 12000
    developmental_sandbox_max_code_chars: int = 12000

    experience_policy_enabled: bool = True
    experience_policy_dir: str = "data/experience_policy"
    experience_policy_hidden: int = 64
    experience_policy_dropout: float = 0.10
    experience_policy_lr: float = 0.002
    experience_policy_epochs: int = 20
    experience_policy_min_samples: int = 4
    experience_policy_max_samples: int = 2000
    experience_policy_use_gpu: bool = False

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
    lora_auto_train_every: int = 0

    hf_model_path: str = ""
    hf_adapter_path: str = ""
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
