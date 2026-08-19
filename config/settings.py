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
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider names / aliases / defaults live in a leaf module with no imports, so
# that config.settings and generation.llm.factory can both use them without a
# cycle (settings must never import from generation/).
from config import llm_providers as _lp

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
        # FIX (2026-08-19). Every aliased field below (mistral_api_key,
        # gemini_api_key, ...) has a validation_alias, so it can be set from
        # an env var under that alias name. Without populate_by_name=True,
        # whether __init__(field_name=...) ALSO works is undocumented and
        # changed between pydantic-settings 2.14.0 and 2.15.0: on 2.14.0,
        # InitSettingsSource requires the alias and silently drops a kwarg
        # given by field name (falls through to the field default, no error,
        # extra="ignore" swallows it); on 2.15.0 it accepts either. Any code
        # path that constructs LLMSettings(mistral_api_key=...) -- tests,
        # scripts, a future caller -- gets a silently empty key on 2.14.0 and
        # correct behaviour on 2.15.0, with no exception either way to reveal
        # the difference. populate_by_name=True makes field-name construction
        # a guaranteed part of the contract on every version, which is what
        # every caller of this class already assumes.
        populate_by_name=True,
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
    # FIX-A3 (2026-08-14) — proactive client-side throttle. Complements the
    # reactive 429 backoff in generation/llm/langchain_provider.py, which
    # only engages AFTER a request is already rejected. This paces requests
    # BEFORE they are sent, covering every call site that goes through
    # LangChainChatProvider.complete() (top-level generation and any
    # validator-driven retry). 0 disables the limiter entirely (no lock, no
    # per-request overhead) for local/offline providers where it does not
    # apply. MUST be tuned to the actual provider tier in use — this default
    # is deliberately conservative, not a measured value for any specific
    # account.
    requests_per_minute: int = Field(
        default=50, ge=0, validation_alias="LLM_REQUESTS_PER_MINUTE",
    )
    # Burst capacity for the token bucket, in requests. The bucket
    # previously defaulted its capacity to the FULL per-minute rate, which
    # means a cold start could fire `requests_per_minute` calls
    # back-to-back before the limiter engaged at all -- so the very first
    # thing a run does is the thing most likely to trip a 429. A burst of
    # 1 makes the limiter a genuine pacer from the first request. Raise it
    # only for a provider tier that is documented to tolerate bursts.
    rate_limit_burst: int = Field(
        default=1, ge=1, validation_alias="LLM_RATE_LIMIT_BURST",
    )
    # Proactive pacing on TOKENS, independent of requests_per_minute above.
    # 0 (default) disables it, so an existing deployment is unchanged until
    # this is set deliberately.
    #
    # Run 20260818_133351 is the evidence that the two limits are separate
    # constraints and that requests was the wrong one: at 20 req/min the
    # request bucket never engaged (236 inferences / 55 min = 4.3 req/min),
    # yet the run still took 183 rate-limit rejections. Median prompt was
    # 11,665 tokens and the run pushed ~40,000 prompt tokens/min. Set this
    # against the measured tier; 30000 is a reasonable starting point for the
    # Mistral tier in use and can be raised against observed headroom.
    tokens_per_minute: int = Field(
        default=0, ge=0, validation_alias="LLM_TOKENS_PER_MINUTE",
    )
    # Transport-level attempts for a TRANSIENT provider error (429/5xx).
    # This is NOT validation.max_retries: that budget buys SQL-correction
    # round trips, this one buys re-sends of an identical request that the
    # provider shed. Conflating them means a rate-limit storm silently eats
    # the correction budget, or vice versa. Previously a module-level
    # constant in langchain_provider.py with no env knob at all.
    transient_max_attempts: int = Field(
        default=4, ge=1, validation_alias="LLM_TRANSIENT_MAX_ATTEMPTS",
    )
    # FIX-F1 — prompt profile. Which prompt distribution the pipeline serves:
    #   "full" (default) — the rich 10–14k-token serve prompt (base model).
    #   "ft"             — the training-parity prompt: _TRAIN_SYSTEM_PROMPT in a
    #                      real system role + a budgeted SCHEMA+QUESTION user
    #                      turn (PromptBuilder.build_ft). REQUIRED whenever
    #                      llama-server is serving a fine-tuned GGUF — the
    #                      adapter is specialised to this shape; serving it the
    #                      "full" prompt is out-of-distribution and degrades it
    #                      below the base model.
    prompt_profile: str = "full"         # env: LLM_PROMPT_PROFILE = full | ft

    # ── PROVIDER SELECTION (switchable LLM) ───────────────────────────────
    # Which backend generates SQL. Default "local" preserves the pre-existing
    # behaviour exactly: llama.cpp against the Qwen GGUF, llama-server first
    # with the in-process fallback. See config/llm_providers.py.
    #   local            — llama.cpp direct (DEFAULT)
    #   local_langchain  — the same llama-server, via LangChain
    #   mistral          — Mistral La Plateforme, via LangChain
    #   gemini           — Google Gemini, via LangChain
    provider: str = "local"                  # env: LLM_PROVIDER

    # Model id llama-server is serving. Sent in the request body on the
    # LangChain path and shown in the startup banner. NOT the GGUF path —
    # that stays LLM_MODEL_PATH, which the in-process loader still uses.
    primary_model: str = "qwen2.5-coder-3b-instruct"   # env: LLM_PRIMARY_MODEL

    # ── TIMEOUTS ──────────────────────────────────────────────────────────
    # Two separate budgets, deliberately. The local primary can spend its
    # first request loading a 2.4 GB GGUF into VRAM and needs real time on the
    # 10-14k-token serve prompt; a hosted provider taking that long means
    # something is actually wrong, so it gets a shorter leash.
    primary_timeout_seconds: float = 120.0   # env: LLM_PRIMARY_TIMEOUT_SECONDS
    timeout_seconds:         float = 90.0    # env: LLM_TIMEOUT_SECONDS

    # ── HOSTED PROVIDERS ──────────────────────────────────────────────────
    # Both vendors expose an OpenAI-compatible /v1 surface, so all three
    # LangChain providers share one client type and differ only by these
    # three values. Endpoints are overridable for proxies / regional hosts.
    #
    # API keys carry exclude=True so they never appear in model_dump() — the
    # thing any accidental settings-logging call would serialise. They are
    # declared with explicit validation_alias so the conventional bare vendor
    # names (MISTRAL_API_KEY, GOOGLE_API_KEY) work alongside the prefixed
    # LLM_MISTRAL_API_KEY form, the same technique FineTuningSettings uses for
    # the LLAMA_* vars.
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model:    str = "mistral-small-latest"
    mistral_api_key:  str = Field(
        default="", exclude=True,
        validation_alias=AliasChoices("LLM_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
    )

    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model:    str = "gemini-2.0-flash"
    gemini_api_key:  str = Field(
        default="", exclude=True,
        validation_alias=AliasChoices(
            "LLM_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
        ),
    )

    @model_validator(mode="after")
    def normalise_provider(self) -> "LLMSettings":
        """
        Resolve aliases, reject unknown providers at startup, and enforce the
        one cross-field rule: prompt_profile="ft" is meaningless off the local
        GGUF.

        The "ft" profile serves PromptBuilder.build_ft() — the exact training
        distribution of OUR LoRA adapter. A hosted Mistral/Gemini model has
        never seen that distribution; serving it the short training-parity
        prompt strips out the schema context the rich prompt provides and
        degrades it for no reason. Rather than make every reader of
        settings.llm.prompt_profile provider-aware (runner.py x3,
        sql_validator.py, batch_run.py), the profile is corrected ONCE here,
        so every existing read site stays untouched and correct.
        """
        name = _lp.normalise(self.provider)
        if name not in _lp.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER={self.provider!r}. "
                f"Supported: {', '.join(_lp.SUPPORTED_PROVIDERS)}."
            )
        object.__setattr__(self, "provider", name)

        if name not in _lp.LOCAL_PROVIDERS and self.prompt_profile != "full":
            object.__setattr__(self, "prompt_profile", "full")
        return self

    # ── ACTIVE-PROVIDER ACCESSORS ─────────────────────────────────────────
    # The ONLY place provider names are branched on. generation/llm/factory.py
    # reads these and builds one client, so adding a provider never adds an
    # `if` anywhere in generation/.

    @property
    def is_local_provider(self) -> bool:
        """True when the active provider serves the local GGUF."""
        return self.provider in _lp.LOCAL_PROVIDERS

    @property
    def active_base_url(self) -> str:
        if self.provider in _lp.LOCAL_PROVIDERS:
            return self.base_url
        if self.provider == _lp.MISTRAL:
            return self.mistral_base_url
        if self.provider == _lp.GEMINI:
            return self.gemini_base_url
        return ""

    @property
    def active_model(self) -> str:
        if self.provider in _lp.LOCAL_PROVIDERS:
            return self.primary_model
        if self.provider == _lp.MISTRAL:
            return self.mistral_model
        if self.provider == _lp.GEMINI:
            return self.gemini_model
        return ""

    @property
    def active_api_key(self) -> str:
        """Secret for the active provider. Never logged, never in the banner."""
        if self.provider == _lp.MISTRAL:
            return self.mistral_api_key
        if self.provider == _lp.GEMINI:
            return self.gemini_api_key
        # llama-server accepts any bearer token; a placeholder keeps the
        # OpenAI client happy without implying a credential exists.
        return ""

    @property
    def active_timeout(self) -> float:
        return (self.primary_timeout_seconds if self.is_local_provider
                else self.timeout_seconds)


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
        # See LLMSettings above for why this is needed: an aliased field's
        # __init__(field_name=...) construction is version-dependent behaviour
        # in pydantic-settings without it.
        populate_by_name=True,
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
    # SINGLE token-budget knob for Phase 2. Override in .env with MAX_SEQ_LENGTH
    # (canonical name; the historical FT_MAX_SEQ is accepted as a legacy alias —
    # if both are set, MAX_SEQ_LENGTH wins). This ONE value drives BOTH the
    # preprocessor (fit_rows token budget) and the trainer
    # (SFTConfig.max_seq_length), so the corpus and the training ceiling
    # can never diverge again. If they diverge, rows are silently truncated past
    # the assistant turn and the completion-only collator masks the whole
    # sequence → zero loss. Keep >= the reserve floor (system + question + output
    # JSON + template); measured max reserve on the current corpus is ~1112 tok,
    # so 1024 is too small. 2048 fits the current fit.jsonl with no re-fit.
    max_seq:      int = Field(
        default=2048,
        validation_alias=AliasChoices("MAX_SEQ_LENGTH", "FT_MAX_SEQ"),
    )

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
        # See LLMSettings above for why this is needed. batch_query_delay_ms
        # below carries validation_alias="BATCH_QUERY_DELAY_MS"; on
        # pydantic-settings 2.14.0, Settings(batch_query_delay_ms=-1)
        # silently drops the kwarg (falls to the default, extra="ignore"
        # swallows the mismatch) instead of validating it against ge=0 --
        # the negative value never reaches the validator at all.
        populate_by_name=True,
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
    # DDL VERSION: v10.10 is the authoritative schema source. Override with
    # DDL_PATH in .env. Changing this invalidates data/.schema_hash — re-run
    # `python ingest.py` so the vector index and FK graph match the new DDL.
    ddl_path:              str = "data/docs/digital_evaluation_schema_v10_10.sql"
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
    # FIX-S1. Batch run 20260813: 125 of 191 questions touched a tenant-scoped
    # table with no board_id / course_id / user_id in user_context, and every
    # one was allowed through on a WARN. That is the correct behaviour for an
    # admin reporting console and the wrong default for a multi-tenant service.
    # When true, SecurityTransformer REJECTS such a query instead of logging it;
    # a caller that legitimately spans tenants opts out per request by setting
    # user_context["tenant_scope"] = "all".
    # 2026-08-14 (FIX-S1c): the CODE default stays False; the hardening is a
    # DEPLOYMENT opt-in expressed in .env, not a library default.
    #
    # FIX-S1b flipped this to True and broke batch run 20260814_132132: 2 of
    # every 3 questions failed with tenant_filter_rejected, because
    # SQLValidator derives tenant_scoped_tables as "any table carrying board_id
    # OR course_id" — 15 of 61 tables, including question_paper,
    # exam_schedule_cache and revaluation_request — while batch_run.py supplied
    # no user_context at all. Every query reaching one of those 15 tables was
    # rejected before it could be judged for correctness.
    #
    # The lesson: a security posture belongs to a deployment, not to a default
    # that every embedder and test harness silently inherits. Set
    # SECURITY_REQUIRE_TENANT_CONTEXT=true in .env for the served application
    # (it is set there); leave the code default permissive so a caller that has
    # not yet been taught to pass a tenant context fails loudly at its own
    # config boundary rather than mysteriously mid-run.
    require_tenant_context: bool = Field(
        default=False, validation_alias="SECURITY_REQUIRE_TENANT_CONTEXT"
    )
    hf_home:       str = Field(default="d:/hugging_face/hf_cache", validation_alias="HF_HOME")

    # ── Feature flags ──────────────────────────────────────────────────────
    dry_run_default:              bool  = True
    show_sql_in_cli:              bool  = True
    show_explanation_in_cli:      bool  = True
    confidence_warn_threshold:    float = 0.60
    use_mcp_servers:              bool  = False
    debug_mode:                   bool  = False
    strict_version_check:         bool  = Field(default=False, validation_alias="STRICT_MODE")

    # ── Batch runner pacing ──────────────────────────────────────────────────
    # Optional pause between individual questions in batch_run.py, to reduce
    # the risk of hitting a hosted LLM_PROVIDER's rate limit on a long run.
    # Milliseconds, not seconds — consistent with this being a short inter-
    # request gap, not a coarse delay. Default 0 preserves existing behaviour
    # (no delay) for the local provider and anyone who hasn't opted in.
    batch_query_delay_ms:         int   = Field(default=0, ge=0, validation_alias="BATCH_QUERY_DELAY_MS")


# Module-level singleton — import this everywhere
settings = Settings()