"""Typed, environment-driven configuration.

Every setting comes from the environment (``DOCASSIST_`` prefix, ``__`` for nesting) or from
a ``*_FILE`` variable that points at a mounted secret (Docker/Kubernetes secrets). Nothing
sensitive has a default: secrets are required in every environment, and ``docassist init-env``
generates a local ``.env`` with fresh random values.

Security properties:

* ``hide_input_in_errors`` - a bad value never ends up in a traceback or a log line.
* ``SecretStr`` for every credential, so ``repr(settings)`` cannot leak one.
* ``_check_production`` refuses to start a production process with development shortcuts
  (wildcard hosts, API docs exposed, no malware scanner, offline AI components, weak keys).
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from docassist.core.enums import Classification

ENV_PREFIX = "DOCASSIST_"
_PLACEHOLDER_MARKERS = ("change-me", "changeme", "placeholder", "example", "xxxxxxxx")


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class RateRule(_Section):
    """``requests`` allowed per ``per_seconds`` window (GCRA, burst = requests)."""

    requests: Annotated[int, Field(ge=1, le=1_000_000)]
    per_seconds: Annotated[int, Field(ge=1, le=86_400 * 31)]


class ModelPrice(_Section):
    input_per_mtok: Annotated[float, Field(ge=0)]
    output_per_mtok: Annotated[float, Field(ge=0)]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class DatabaseSettings(_Section):
    url: SecretStr = Field(description="API role DSN (postgresql+asyncpg://docassist_app:...)")
    worker_url: SecretStr | None = Field(default=None, description="Worker role DSN")
    migration_url: SecretStr | None = Field(default=None, description="Schema owner DSN")
    app_role: str = "docassist_app"
    worker_role: str = "docassist_worker"
    pool_size: Annotated[int, Field(ge=1, le=200)] = 10
    max_overflow: Annotated[int, Field(ge=0, le=200)] = 10
    pool_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 10.0
    statement_timeout_ms: Annotated[int, Field(ge=100, le=600_000)] = 15_000
    lock_timeout_ms: Annotated[int, Field(ge=100, le=120_000)] = 5_000
    idle_in_transaction_timeout_ms: Annotated[int, Field(ge=1_000, le=600_000)] = 30_000
    ssl: Literal["disable", "prefer", "require", "verify-full"] = "prefer"
    echo: bool = False
    verify_role_privileges: bool = True

    @field_validator("app_role", "worker_role")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or not value[0].isalpha() or len(value) > 63:
            raise ValueError("role names must be simple SQL identifiers")
        return value


class RedisSettings(_Section):
    url: SecretStr | None = None
    key_prefix: str = "docassist"
    socket_timeout_seconds: Annotated[float, Field(gt=0, le=30)] = 1.0
    max_connections: Annotated[int, Field(ge=1, le=1000)] = 50


class SecuritySettings(_Section):
    jwt_signing_key: SecretStr
    jwt_previous_signing_keys: list[SecretStr] = Field(default_factory=list)
    jwt_issuer: str = "docassist"
    jwt_audience: str = "docassist-api"
    access_token_ttl_seconds: Annotated[int, Field(ge=60, le=3600)] = 600
    refresh_token_ttl_seconds: Annotated[int, Field(ge=300, le=86_400 * 30)] = 43_200
    session_absolute_ttl_seconds: Annotated[int, Field(ge=3600, le=86_400 * 90)] = 604_800
    token_pepper: SecretStr
    audit_hmac_key: SecretStr
    encryption_keys: SecretStr = Field(description="Comma-separated kid:base64(32 bytes) entries")
    active_encryption_key_id: str
    password_min_length: Annotated[int, Field(ge=8, le=64)] = 12
    password_max_length: Annotated[int, Field(ge=64, le=1024)] = 256
    lockout_threshold: Annotated[int, Field(ge=3, le=50)] = 5
    lockout_base_seconds: Annotated[int, Field(ge=30, le=86_400)] = 300
    lockout_max_seconds: Annotated[int, Field(ge=60, le=86_400 * 7)] = 3_600
    argon2_time_cost: Annotated[int, Field(ge=1, le=10)] = 3
    argon2_memory_kib: Annotated[int, Field(ge=8_192, le=1_048_576)] = 65_536
    argon2_parallelism: Annotated[int, Field(ge=1, le=16)] = 2
    password_reset_ttl_seconds: Annotated[int, Field(ge=300, le=86_400)] = 1_800
    mfa_issuer: str = "AI Document Assistant"
    mfa_challenge_ttl_seconds: Annotated[int, Field(ge=60, le=900)] = 300
    cookie_secure: bool = True
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "testserver"]
    )
    cors_origins: list[str] = Field(default_factory=list)
    trusted_proxies: list[str] = Field(default_factory=list)
    expose_api_docs: bool = True
    metrics_token: SecretStr | None = None
    max_json_body_bytes: Annotated[int, Field(ge=1_024, le=10_485_760)] = 1_048_576
    csp_report_uri: str | None = None

    @field_validator("trusted_proxies")
    @classmethod
    def _cidrs(cls, value: list[str]) -> list[str]:
        for entry in value:
            ipaddress.ip_network(entry, strict=False)
        return value


class UploadSettings(_Section):
    max_upload_bytes: Annotated[int, Field(ge=1_024, le=1_073_741_824)] = 52_428_800
    allowed_extensions: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "xlsx", "csv", "txt", "md"]
    )
    max_zip_entries: Annotated[int, Field(ge=10, le=100_000)] = 2_000
    max_zip_uncompressed_bytes: Annotated[int, Field(ge=1_048_576, le=4_294_967_296)] = 209_715_200
    max_zip_ratio: Annotated[int, Field(ge=5, le=1_000)] = 100
    max_pdf_pages: Annotated[int, Field(ge=1, le=100_000)] = 2_000
    malware_scanner: Literal["none", "clamav"] = "none"
    allow_unscanned_uploads: bool = False
    clamav_host: str = "clamav"
    clamav_port: Annotated[int, Field(ge=1, le=65_535)] = 3310
    clamav_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 60.0
    reject_active_content: bool = True
    url_import_enabled: bool = False
    url_import_allowed_domains: list[str] = Field(default_factory=list)

    @field_validator("allowed_extensions")
    @classmethod
    def _known_extensions(cls, value: list[str]) -> list[str]:
        supported = {"pdf", "docx", "xlsx", "csv", "txt", "md"}
        normalized = [v.lower().lstrip(".") for v in value]
        unknown = set(normalized) - supported
        if unknown:
            raise ValueError(f"unsupported extensions: {sorted(unknown)}")
        return normalized


class StorageSettings(_Section):
    backend: Literal["local"] = "local"
    root: Path = Path("var/storage")
    temp_dir: Path | None = None


class ParserSettings(_Section):
    timeout_seconds: Annotated[float, Field(gt=0, le=3_600)] = 60.0
    memory_limit_mb: Annotated[int, Field(ge=128, le=16_384)] = 1_024
    cpu_seconds: Annotated[int, Field(ge=1, le=3_600)] = 60
    max_output_bytes: Annotated[int, Field(ge=1_048_576, le=536_870_912)] = 67_108_864
    max_concurrency: Annotated[int, Field(ge=1, le=64)] = 2
    ocr_enabled: bool = False
    ocr_languages: str = "eng"
    tesseract_path: str = "tesseract"


class ChunkingSettings(_Section):
    target_tokens: Annotated[int, Field(ge=64, le=4_000)] = 450
    max_tokens: Annotated[int, Field(ge=128, le=8_000)] = 800
    overlap_tokens: Annotated[int, Field(ge=0, le=1_000)] = 60

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("require overlap_tokens < target_tokens <= max_tokens")
        return self


class EmbeddingSettings(_Section):
    provider: Literal["hashing", "openai_compatible"] = "hashing"
    model: str = "hashing-v1"
    dimensions: Annotated[int, Field(ge=16, le=2_000)] = 1_024
    base_url: str | None = None
    api_key: SecretStr | None = None
    is_external: bool = True
    batch_size: Annotated[int, Field(ge=1, le=2_048)] = 64
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    send_dimensions: bool = True


class LLMSettings(_Section):
    provider: Literal["anthropic", "openai_compatible", "local_extractive"] = "local_extractive"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str | None = None
    main_model: str = "claude-opus-5-5"
    fast_model: str = "claude-haiku-4-5"
    main_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    enable_server_fallbacks: bool = True
    local_provider: Literal["none", "openai_compatible", "local_extractive"] = "local_extractive"
    local_base_url: str | None = None
    local_model: str = "local-model"
    local_api_key: SecretStr | None = None
    request_timeout_seconds: Annotated[float, Field(gt=0, le=900)] = 120.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 2
    max_output_tokens: Annotated[int, Field(ge=256, le=64_000)] = 8_000
    max_context_tokens: Annotated[int, Field(ge=500, le=200_000)] = 12_000
    max_question_chars: Annotated[int, Field(ge=10, le=20_000)] = 2_000
    external_max_classification: Classification = Classification.CONFIDENTIAL
    pseudonymize_pii_for_external: bool = True
    monthly_token_budget_per_org: Annotated[int, Field(ge=0)] = 5_000_000
    circuit_failure_threshold: Annotated[int, Field(ge=1, le=100)] = 5
    circuit_reset_seconds: Annotated[float, Field(gt=0, le=3_600)] = 30.0
    answer_cache_ttl_seconds: Annotated[int, Field(ge=0, le=86_400)] = 900
    agent_max_iterations: Annotated[int, Field(ge=1, le=20)] = 5
    agent_max_tool_calls: Annotated[int, Field(ge=1, le=50)] = 8
    tool_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 15.0
    pricing: dict[str, ModelPrice] = Field(
        default_factory=lambda: {
            "claude-opus-5-5": ModelPrice(input_per_mtok=4.0, output_per_mtok=20.0),
            "claude-sonnet-5-5": ModelPrice(input_per_mtok=2.0, output_per_mtok=10.0),
            "claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
        }
    )


class RetrievalSettings(_Section):
    backend: Literal["pgvector", "qdrant"] = "pgvector"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "docassist_chunks"
    top_k: Annotated[int, Field(ge=1, le=50)] = 8
    candidate_pool: Annotated[int, Field(ge=5, le=500)] = 40
    rrf_k: Annotated[int, Field(ge=1, le=1_000)] = 60
    mmr_lambda: Annotated[float, Field(ge=0, le=1)] = 0.7
    min_relevance: Annotated[float, Field(ge=0, le=1)] = 0.05
    injection_exclude_threshold: Annotated[float, Field(ge=0, le=1)] = 0.8
    injection_warn_threshold: Annotated[float, Field(ge=0, le=1)] = 0.4
    hnsw_ef_search: Annotated[int, Field(ge=10, le=1_000)] = 100
    max_query_chars: Annotated[int, Field(ge=1, le=10_000)] = 1_000


class RateLimitSettings(_Section):
    enabled: bool = True
    login_per_ip: RateRule = RateRule(requests=20, per_seconds=60)
    login_per_account: RateRule = RateRule(requests=5, per_seconds=60)
    refresh_per_session: RateRule = RateRule(requests=30, per_seconds=60)
    password_reset_per_ip: RateRule = RateRule(requests=5, per_seconds=3_600)
    api_per_user: RateRule = RateRule(requests=600, per_seconds=60)
    upload_per_user: RateRule = RateRule(requests=60, per_seconds=3_600)
    search_per_user: RateRule = RateRule(requests=120, per_seconds=60)
    llm_per_user: RateRule = RateRule(requests=20, per_seconds=60)
    llm_per_org: RateRule = RateRule(requests=600, per_seconds=60)
    export_per_user: RateRule = RateRule(requests=10, per_seconds=3_600)
    tool_calls_per_user: RateRule = RateRule(requests=60, per_seconds=60)
    user_create_per_admin: RateRule = RateRule(requests=120, per_seconds=3_600)


class RetentionSettings(_Section):
    conversation_days: Annotated[int, Field(ge=1, le=3_650)] = 90
    audit_days: Annotated[int, Field(ge=30, le=36_500)] = 2_555
    export_ttl_hours: Annotated[int, Field(ge=1, le=720)] = 24
    job_days: Annotated[int, Field(ge=1, le=3_650)] = 30
    llm_usage_days: Annotated[int, Field(ge=30, le=3_650)] = 400
    deleted_document_purge_days: Annotated[int, Field(ge=0, le=3_650)] = 30


class OutboundSettings(_Section):
    """Egress policy for every server-side HTTP request (SSRF defence)."""

    allowed_hosts: list[str] = Field(default_factory=lambda: ["api.anthropic.com"])
    private_network_allowlist: list[str] = Field(default_factory=list)
    max_response_bytes: Annotated[int, Field(ge=1_024, le=1_073_741_824)] = 52_428_800
    timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 30.0
    max_redirects: Annotated[int, Field(ge=0, le=10)] = 3


class WorkerSettings(_Section):
    concurrency: Annotated[int, Field(ge=1, le=64)] = 4
    poll_interval_seconds: Annotated[float, Field(gt=0, le=60)] = 1.0
    lease_seconds: Annotated[int, Field(ge=10, le=3_600)] = 120
    max_attempts: Annotated[int, Field(ge=1, le=50)] = 5
    backoff_base_seconds: Annotated[float, Field(gt=0, le=3_600)] = 5.0
    backoff_max_seconds: Annotated[float, Field(gt=0, le=86_400)] = 900.0
    shutdown_grace_seconds: Annotated[float, Field(gt=0, le=3_600)] = 30.0
    maintenance_interval_seconds: Annotated[float, Field(gt=0, le=86_400)] = 30.0


class ObservabilitySettings(_Section):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    metrics_enabled: bool = True
    audit_query_text: Literal["redacted", "hash", "none"] = "redacted"


def _strip_file_refs(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _strip_file_refs(value) if isinstance(value, dict) else value
        for key, value in values.items()
        if not key.lower().endswith("_file")
    }


class _WithoutFileRefs(PydanticBaseSettingsSource):
    """Hide ``*_FILE`` pointer variables from the env/dotenv sources.

    ``DOCASSIST_DATABASE__URL_FILE`` is a *pointer* resolved by :func:`load_settings`; without
    this wrapper pydantic-settings would also see it as an unknown nested key
    ``database.url_file`` and reject it (sections use ``extra="forbid"``).
    """

    def __init__(self, inner: PydanticBaseSettingsSource) -> None:
        super().__init__(inner.settings_cls)
        self._inner = inner

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._inner.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        return _strip_file_refs(self._inner())


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        case_sensitive=False,
    )

    environment: Environment = Environment.DEVELOPMENT
    service_name: str = "docassist"
    public_base_url: str = "http://localhost:8000"
    allow_offline_ai_in_production: bool = False

    database: DatabaseSettings
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings
    upload: UploadSettings = UploadSettings()
    storage: StorageSettings = StorageSettings()
    parser: ParserSettings = ParserSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    retention: RetentionSettings = RetentionSettings()
    outbound: OutboundSettings = OutboundSettings()
    worker: WorkerSettings = WorkerSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _WithoutFileRefs(env_settings),
            _WithoutFileRefs(dotenv_settings),
            file_secret_settings,
        )

    @property
    def is_production(self) -> bool:
        return self.environment in {Environment.PRODUCTION, Environment.STAGING}

    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _check_secrets(self) -> Self:
        sec = self.security
        _require_strong_secret("security.jwt_signing_key", sec.jwt_signing_key)
        _require_strong_secret("security.token_pepper", sec.token_pepper)
        _require_strong_secret("security.audit_hmac_key", sec.audit_hmac_key)
        keys = parse_encryption_keys(sec.encryption_keys.get_secret_value())
        if sec.active_encryption_key_id not in keys:
            raise ValueError("security.active_encryption_key_id is not in security.encryption_keys")
        distinct = {
            sec.jwt_signing_key.get_secret_value(),
            sec.token_pepper.get_secret_value(),
            sec.audit_hmac_key.get_secret_value(),
        }
        if len(distinct) != 3:
            raise ValueError("jwt_signing_key, token_pepper and audit_hmac_key must all differ")
        return self

    @model_validator(mode="after")
    def _check_production(self) -> Self:
        if not self.is_production:
            return self
        problems: list[str] = []
        sec = self.security
        if not sec.cookie_secure:
            problems.append("security.cookie_secure must be true")
        if "*" in sec.allowed_hosts:
            problems.append("security.allowed_hosts must not contain '*'")
        if "*" in sec.cors_origins:
            problems.append("security.cors_origins must not contain '*'")
        if sec.expose_api_docs:
            problems.append("security.expose_api_docs must be false")
        if self.database.echo:
            problems.append("database.echo must be false (it logs query parameters)")
        if self.database.ssl in {"disable", "prefer"}:
            problems.append("database.ssl must be 'require' or 'verify-full'")
        if self.redis.url is None:
            problems.append("redis.url is required (distributed rate limiting and revocation)")
        if self.upload.malware_scanner == "none" and not self.upload.allow_unscanned_uploads:
            problems.append("upload.malware_scanner must be 'clamav' (or explicitly waive it)")
        if not self.rate_limit.enabled:
            problems.append("rate_limit.enabled must be true")
        if not self.public_base_url.startswith("https://"):
            problems.append("public_base_url must be https")
        offline = self.llm.provider == "local_extractive" or self.embedding.provider == "hashing"
        if offline and not self.allow_offline_ai_in_production:
            problems.append("offline AI components (local_extractive/hashing) are for dev/test")
        if self.observability.log_level == "DEBUG":
            problems.append("observability.log_level DEBUG is not allowed")
        if problems:
            raise ValueError("unsafe production configuration: " + "; ".join(problems))
        return self


def _require_strong_secret(name: str, secret: SecretStr) -> None:
    value = secret.get_secret_value()
    if len(value.encode()) < 32:
        raise ValueError(f"{name} must be at least 32 bytes")
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise ValueError(f"{name} still contains a placeholder value")
    if len(set(value)) < 10:
        raise ValueError(f"{name} has too little entropy")


def parse_encryption_keys(raw: str) -> dict[str, bytes]:
    """Parse ``kid:base64key,kid2:base64key`` into a dict of 32-byte AES keys."""
    keys: dict[str, bytes] = {}
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        kid, sep, encoded = part.partition(":")
        if not sep or not kid or not kid.replace("-", "").replace("_", "").isalnum():
            raise ValueError("encryption key entries must look like 'kid:base64key'")
        try:
            key = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"encryption key {kid!r} is not valid base64") from exc
        if len(key) != 32:
            raise ValueError(f"encryption key {kid!r} must decode to 32 bytes")
        if kid in keys:
            raise ValueError(f"duplicate encryption key id {kid!r}")
        keys[kid] = key
    if not keys:
        raise ValueError("at least one encryption key is required")
    return keys


def _file_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Turn ``DOCASSIST_A__B_FILE=/run/secrets/x`` into ``{"a": {"b": <file contents>}}``."""
    overrides: dict[str, Any] = {}
    for key, path in environ.items():
        upper = key.upper()
        if not upper.startswith(ENV_PREFIX) or not upper.endswith("_FILE"):
            continue
        dotted = upper[len(ENV_PREFIX) : -len("_FILE")].lower().split("__")
        content = Path(path).read_text(encoding="utf-8").strip()
        cursor = overrides
        for part in dotted[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[dotted[-1]] = content
    return overrides


def load_settings(**overrides: Any) -> Settings:
    """Build settings from env, ``.env`` and ``*_FILE`` secrets (explicit overrides win)."""
    merged = _deep_merge(_file_overrides(dict(os.environ)), overrides)
    return Settings(**merged)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
