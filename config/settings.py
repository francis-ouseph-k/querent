"""
config/settings.py
──────────────────
Central configuration for the NL→SQL system.
All values are read from environment variables or the .env file.
Pydantic-settings validates types at startup — bad config fails fast.

ARCHITECTURE:
    Every nested config class (QdrantSettings, OpenSearchSettings, etc.)
    inherits from BaseSettings, NOT BaseModel.  Only BaseSettings subclasses
    process env_prefix and env_file.  Using BaseModel for nested configs is a
    silent no-op — values fall back to hardcoded defaults regardless of what
    is set in .env.

    Each nested class carries:
        env_prefix  — maps OPENSEARCH_USE_SSL → use_ssl, etc.
        env_file    — same .env file as the parent Settings class
        extra       — "ignore" so unknown env vars don't raise ValidationError

    The ENV_FILE constant resolves to the project root .env at import time
    so all classes point to the same file without duplicating the path.
"""


import re
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = str(ROOT_DIR / ".env")


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str            = "localhost"
    port: int            = 6333
    collection_name: str = "schema_chunks"
    vector_size: int     = 384          # BGE-small-en-v1.5


class OpenSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENSEARCH_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str       = "localhost"
    port: int       = 9200
    index_name: str = "schema_chunks"
    username: str   = "admin"
    # REVIEW FIX (NEW-M1): previously defaulted to a hardcoded plausible-looking
    # password ("pgmagJmL#76L") committed in source. Even though exclude=True
    # keeps it out of log serialisation, the value itself was still readable by
    # anyone with repo access — a credential leak risk if this repo is ever
    # made public, forked, or if the same string happens to match a real
    # password anywhere else. Default is now "" (blank); every environment
    # must set OPENSEARCH_PASSWORD explicitly via .env. The validator below
    # already enforced this for non-localhost hosts — it now also applies to
    # localhost, since local dev OpenSearch instances should set their own
    # password rather than rely on a string baked into the codebase.
    password: str   = Field(default="", exclude=True)
    # SSL settings for this deployment — OpenSearch runs with TLS on port 9200.
    # Override via OPENSEARCH_USE_SSL / OPENSEARCH_VERIFY_CERTS in .env.
    use_ssl:      bool = True
    verify_certs: bool = False

    @model_validator(mode="after")
    def require_password_if_host_set(self) -> "OpenSearchSettings":
        # REVIEW FIX (NEW-M1): previously only required OPENSEARCH_PASSWORD
        # for non-localhost hosts, which made sense when there was a hardcoded
        # default to fall back to on localhost. Now that the default is blank,
        # the localhost exemption would mean local dev silently connects with
        # an empty password rather than failing with a clear error. Requiring
        # it everywhere costs a one-line .env entry and removes the only
        # remaining case where a missing credential fails silently instead of
        # at startup.
        if not self.password:
            raise ValueError(
                "OPENSEARCH_PASSWORD must be set in .env — no default password "
                "is provided. Set it even for local development."
            )
        return self


class PostgreSQLSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PG_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host:                 str  = "localhost"
    port:                 int  = 5432
    database:             str  = "digital_evaluation_db"
    user:                 str  = "postgres"
    password:             str  = ""
    readonly:             bool = True
    statement_timeout_ms: int  = 30_000
    max_rows:             int  = 1_000
    pool_min: int = Field(default=2,  description="Minimum idle connections in pool")
    pool_max: int = Field(default=20, description="Maximum connections in pool")

    # M-7 fix: warn when PG password is empty. Unlike OpenSearch (which hard-fails),
    # PostgreSQL supports peer/ident auth for local dev, so we warn instead of raising.
    @model_validator(mode="after")
    def warn_empty_password(self) -> "PostgreSQLSettings":
        if not self.password and self.host not in ("localhost", "127.0.0.1", "::1"):
            import warnings
            warnings.warn(
                "PG_PASSWORD is empty and PG_HOST is not localhost. "
                "This may indicate a misconfigured remote database connection.",
                stacklevel=2,
            )
        return self


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path:   str   = "./models/qwen/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    context_size: int   = 8_192
    max_tokens:   int   = 512
    temperature:  float = 0.2
    n_gpu_layers: int   = -1
    n_threads:    int   = 8
    grammar_path: str   = "config/sql_select.gbnf"
    base_url:     str   = ""
    frequency_penalty: float = 0.0
    presence_penalty:  float = 0.0


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBED_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    device:     str = "cpu"


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled:      bool = False
    model_name:   str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_input:  int  = 20
    top_k_output: int  = 10


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dense_top_k:               int = 20
    bm25_top_k:                int = 20
    rrf_k:                     int = 60
    # Standard context token budget for the initial retrieval step
    context_budget_tokens:     int = 7_000
    
    # Strict maximum ceiling for retrieval token budget expansion during self-correction retries.
    # Scaled incrementally to prevent query parser/model token overflow while providing sufficient schema context.
    max_context_budget_tokens: int = 12_000


class ValidationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VALIDATION_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_retries:              int = 2
    explain_cost_threshold:   int = 1_000_000
    blocked_statements: list[str] = Field(
        default=["INSERT", "UPDATE", "DELETE", "TRUNCATE", "DROP",
                 "ALTER", "CREATE", "GRANT", "REVOKE", "EXECUTE",
                 "COPY", "VACUUM", "ANALYZE"]
    )

    _blocked_pattern: re.Pattern | None = None

    @model_validator(mode="after")
    def compile_blocked_pattern(self) -> "ValidationSettings":
        joined  = "|".join(re.escape(s) for s in self.blocked_statements)
        pattern = re.compile(rf"\b({joined})\b", re.IGNORECASE)
        object.__setattr__(self, "_blocked_pattern", pattern)
        return self

    @property
    def blocked_pattern(self) -> re.Pattern:
        return self._blocked_pattern
    
class MCPSettings(BaseSettings):
    """
    Bind addresses and ports for the four NL→SQL MCP servers.

    All MCP server config in one place — override via .env.

    .env keys:
        MCP_QDRANT_HOST=127.0.0.1
        MCP_QDRANT_PORT=5010
        MCP_OPENSEARCH_HOST=127.0.0.1
        MCP_OPENSEARCH_PORT=5011
        MCP_POSTGRES_HOST=127.0.0.1
        MCP_POSTGRES_PORT=5012
        MCP_CORPUS_HOST=127.0.0.1
        MCP_CORPUS_PORT=5013
        MCP_CORPUS_BACKEND=local
        MCP_CORPUS_DRIVE_FOLDER_ID=
    """
    model_config = SettingsConfigDict(
        env_prefix        = "MCP_",
        env_file          = ENV_FILE,
        env_file_encoding = "utf-8",
        extra             = "ignore",
    )

    qdrant_host:            str = "127.0.0.1"
    qdrant_port:            int = 5010
    opensearch_host:        str = "127.0.0.1"
    opensearch_port:        int = 5011
    postgres_host:          str = "127.0.0.1"
    postgres_port:          int = 5012
    corpus_host:            str = "127.0.0.1"
    corpus_port:            int = 5013
    corpus_backend:         str = "local"
    corpus_drive_folder_id: str = ""    


class FineTuningSettings(BaseSettings):
    # Phase-2 fine-tuning paths. All overridable via .env with the FT_ prefix:
    #   FT_ADAPTER_DIR, FT_HF_MODEL_DIR, FT_TRAIN_DATA
    # NOTE: defaults match the current hardcoded values in trainer.py, which
    # write INSIDE the project root ("models/..."). The README documents the
    # sibling layout "../models/..." (models kept outside the repo). If you
    # follow the README layout, set in .env:
    #   FT_ADAPTER_DIR=../models/adapters
    #   FT_HF_MODEL_DIR=../models/hf/Qwen2.5-Coder-3B-Instruct
    model_config = SettingsConfigDict(
        env_prefix="FT_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    adapter_dir:  str = "models/adapters"                     # LoRA adapter output root
    hf_model_dir: str = "models/hf/Qwen2.5-Coder-3B-Instruct" # HF base model (training)
    train_data:   str = "data/fine_tuning_train.fit.jsonl"   # trainer input file (FITTED corpus)
    # NOTE: this is the token-fitted artifact, not the raw 554-row formatted file.
    # Both the preprocessor (PreprocessConfig.artifact) and the trainer (TRAIN_DATA)
    # derive from this single setting, so they always agree on the same file.
    eval_data:    str = "data/fine_tuning_eval.jsonl"         # evaluator input (if used)
    baseline_path: str = "data/eval_baseline.json"            # evaluator baseline metrics
    merged_dir:   str = "models/merged"                       # export: merged HF model
    gguf_output_dir: str = "models/qwen"                      # export: final GGUF output
    # SINGLE token-budget knob for Phase 2. Override in .env with FT_MAX_SEQ.
    # This ONE value drives BOTH the preprocessor (fit_rows token budget) and the
    # trainer (SFTConfig.max_seq_length), so the corpus and the training ceiling
    # can never diverge again. If they diverge, rows are silently truncated past
    # the assistant turn and the completion-only collator masks the whole
    # sequence → zero loss. Keep >= the reserve floor (system + question + output
    # JSON + template); measured max reserve on the current corpus is ~1112 tok,
    # so 1024 is too small. 2048 fits the current fit.jsonl with no re-fit.
    max_seq:      int = 2048                                   # FT_MAX_SEQ

    # ── export.py tool paths / merge device ──────────────────────────────────
    # These are read HERE (via settings/.env) rather than directly from
    # os.environ, so export.py behaves exactly like main.py / batch_run.py:
    # one .env, loaded once by pydantic. FT_MERGE_DEVICE uses the FT_ prefix like
    # every other field; the LLAMA_* vars keep their historical un-prefixed names
    # (shared with the llama-server launch command), so each declares an explicit
    # validation_alias to bypass the FT_ prefix and read the exact env name.
    merge_device:       str = "cuda:0"    # FT_MERGE_DEVICE  (cuda:0 | cuda:N | cpu)

    # llama.cpp SOURCE checkout containing convert_hf_to_gguf.py. External/shared,
    # differs per machine → empty default = "unset"; export.py raises a clear
    # prerequisite error (not an import crash) when it's needed but blank.
    llama_cpp_source:   str = Field(default="",                  validation_alias="LLAMA_CPP_SOURCE")
    # Precompiled quantiser/server binaries dir. Empty default = "unset"; export.py
    # then falls back to the repo-relative "llama-precompiled" beside the project.
    llama_precompiled:  str = Field(default="",                  validation_alias="LLAMA_PRECOMPILED")
    # Binary NAMES — override for Linux/WSL (no .exe).
    llama_quantize_bin: str = Field(default="llama-quantize.exe", validation_alias="LLAMA_QUANTIZE_BIN")
    llama_server_bin:   str = Field(default="llama-server.exe",   validation_alias="LLAMA_SERVER_BIN")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Sub-configs ────────────────────────────────────────────────────────
    qdrant:     QdrantSettings     = Field(default_factory=QdrantSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    postgres:   PostgreSQLSettings = Field(default_factory=PostgreSQLSettings)
    llm:        LLMSettings        = Field(default_factory=LLMSettings)
    embedding:  EmbeddingSettings  = Field(default_factory=EmbeddingSettings)
    reranker:   RerankerSettings   = Field(default_factory=RerankerSettings)
    retrieval:  RetrievalSettings  = Field(default_factory=RetrievalSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    mcp:        MCPSettings        = Field(default_factory=MCPSettings)
    fine_tuning: FineTuningSettings = Field(default_factory=FineTuningSettings)

    # ── Paths ──────────────────────────────────────────────────────────────
    ddl_path:              str = "data/docs/digital_evaluation_schema_v10_4_1.sql"
    glossary_path:         str = "data/glossary.json"
    few_shot_examples_path: str = "data/few_shot_examples.json"
    failure_log_dir:       str = "failures"
    log_dir:               str = "logs"
    schema_hash_path:      str = Field(
        default="data/.schema_hash",
        description="Path to the stored DDL hash file. Override via SCHEMA_HASH_PATH.",
    )

    # ── Tenant / security ──────────────────────────────────────────────────
    tenant_column: str = ""
    rls_variable:  str = "app.current_user_id"
    hf_home:       str = Field(default="d:/hugging_face/hf_cache", validation_alias="HF_HOME")

    # ── Feature flags ──────────────────────────────────────────────────────
    dry_run_default:              bool  = True
    show_sql_in_cli:              bool  = True
    show_explanation_in_cli:      bool  = True
    confidence_warn_threshold:    float = 0.60
    use_mcp_servers:              bool  = False
    debug_mode:                   bool  = False
    strict_version_check:         bool  = Field(default=False, validation_alias="STRICT_MODE")


# Module-level singleton — import this everywhere
settings = Settings()