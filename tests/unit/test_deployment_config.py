"""Static checks of the deployment artefacts (no Docker needed).

The container image, compose stack, Kubernetes manifests and CI workflows are code: these
tests pin their security properties so a later edit cannot silently drop one - non-root
users, read-only root filesystems, dropped capabilities, no published data-store ports,
secrets only as files, SHA-pinned actions, least-privilege tokens - and cross-check them
against the real settings model (every ``*_FILE`` variable names an actual setting, the
production examples pass ``Settings._check_production``, ``.env.example`` documents every
setting and contains no usable secret).
"""

from __future__ import annotations

import base64
import functools
import re
import secrets
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, SecretStr

from docassist.core.config import Settings, _require_strong_secret, load_settings

ROOT = Path(__file__).resolve().parents[2]
APP_SERVICES = ("api", "worker", "migrate")
DATA_SERVICES = ("postgres", "redis", "qdrant", "clamav", "ollama")
APP_UID = "10001:10001"
OPERATOR_SUPPLIED_SECRETS = {"anthropic_api_key", "embedding_api_key"}


# --------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------- #
class _ComposeLoader(yaml.SafeLoader):
    """Safe loader that understands the Compose merge tags used by the override files."""


def _compose_tag(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_ComposeLoader.add_constructor("!reset", _compose_tag)
_ComposeLoader.add_constructor("!override", _compose_tag)


@functools.cache
def _load_yaml(relative: str) -> Any:
    return yaml.load((ROOT / relative).read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506 - SafeLoader subclass


@functools.cache
def _load_all_yaml(path: Path) -> list[dict[str, Any]]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


_INTERPOLATION = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?:(?P<op>:-|:\?)(?P<arg>[^}]*))?\}")


def _interpolate(value: str, variables: dict[str, str]) -> str:
    """The subset of Compose interpolation the files use: ${V}, ${V:-default}, ${V:?error}."""

    def replace(match: re.Match[str]) -> str:
        name, op = match["name"], match["op"]
        if name in variables:
            return variables[name]
        if op == ":-":
            return match["arg"]
        raise KeyError(f"required compose variable {name} is not set")

    return _INTERPOLATION.sub(replace, value).replace("$$", "$")


def _settings_keys() -> dict[str, Any]:
    """Every environment variable name the settings model understands -> its field."""
    keys: dict[str, Any] = {}
    for name, field in Settings.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for sub, sub_field in annotation.model_fields.items():
                keys[f"DOCASSIST_{name}__{sub}".upper()] = sub_field
        else:
            keys[f"DOCASSIST_{name}".upper()] = field
    return keys


def _is_secret(field: Any) -> bool:
    return "SecretStr" in repr(field.annotation)


SETTINGS_KEYS = _settings_keys()
SECRET_KEYS = {key for key, field in SETTINGS_KEYS.items() if _is_secret(field)}


def _secret_overrides() -> dict[str, Any]:
    """Strong, distinct generated secrets - what the *_FILE files would contain."""
    kid = "k20260101"
    return {
        "database": {"url": "postgresql+asyncpg://docassist_app:pw@db.internal:5432/docassist"},
        "redis": {"url": "rediss://docassist:pw@redis.internal:6380/0"},
        "security": {
            "jwt_signing_key": secrets.token_urlsafe(48),
            "token_pepper": secrets.token_urlsafe(48),
            "audit_hmac_key": secrets.token_urlsafe(48),
            "encryption_keys": f"{kid}:{base64.b64encode(secrets.token_bytes(32)).decode()}",
            "active_encryption_key_id": kid,
            "metrics_token": secrets.token_urlsafe(32),
        },
        "llm": {"anthropic_api_key": "sk-test-" + secrets.token_hex(8)},
        "embedding": {"api_key": "emb-test-" + secrets.token_hex(8)},
    }


def _settings_from_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> Settings:
    """Build Settings from ``env`` (minus *_FILE references) plus generated secrets."""
    import os

    for key in list(os.environ):
        if key.upper().startswith("DOCASSIST_"):
            monkeypatch.delenv(key)
    for key, value in env.items():
        if not key.endswith("_FILE"):
            monkeypatch.setenv(key, value)
    return Settings(**_secret_overrides(), _env_file=None)


def _compose() -> dict[str, Any]:
    data = _load_yaml("compose.yaml")
    assert isinstance(data, dict)
    return data


def _service_env(service: dict[str, Any]) -> dict[str, str]:
    env = service.get("environment") or {}
    assert isinstance(env, dict), "environment must be a mapping (no KEY=VALUE lists)"
    return {str(k): str(v) for k, v in env.items()}


def _secret_targets(service: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for entry in service.get("secrets", []):
        if isinstance(entry, str):
            targets.add(f"/run/secrets/{entry}")
        else:
            target = entry.get("target", entry["source"])
            targets.add(target if target.startswith("/") else f"/run/secrets/{target}")
    return targets


# --------------------------------------------------------------------------------------- #
# Dockerfile / .dockerignore
# --------------------------------------------------------------------------------------- #
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_is_multi_stage_on_a_digest_pinned_slim_base() -> None:
    froms = re.findall(r"^FROM\s+(\S+)", DOCKERFILE, flags=re.MULTILINE)
    assert len(froms) >= 2, "builder and runtime stages expected"
    for image in froms:
        assert image.startswith("python:3.12-slim-bookworm@sha256:"), image
        assert re.fullmatch(r"python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}", image), image
    copied = re.findall(r"COPY --from=(\S+)", DOCKERFILE)
    for ref in copied:
        if "/" in ref or ":" in ref:  # an external image, not a stage name
            assert re.search(r"@sha256:[0-9a-f]{64}$", ref), f"unpinned image in COPY --from: {ref}"


def test_dockerfile_runtime_is_non_root_with_healthcheck_and_entrypoint() -> None:
    runtime = DOCKERFILE.split(" AS runtime", 1)[1]
    users = re.findall(r"^USER\s+(\S+)", runtime, flags=re.MULTILINE)
    assert users == [APP_UID], users
    assert re.search(r"^HEALTHCHECK\s", runtime, flags=re.MULTILINE)
    assert "/health/live" in runtime
    assert re.search(r'^ENTRYPOINT \["docassist"\]$', runtime, flags=re.MULTILINE)
    assert re.search(r"^STOPSIGNAL SIGTERM$", runtime, flags=re.MULTILINE)
    assert "PYTHONDONTWRITEBYTECODE=1" in runtime
    assert "PYTHONUNBUFFERED=1" in runtime
    assert 'VOLUME ["/var/lib/docassist"]' in runtime
    assert "DOCASSIST_ENVIRONMENT=production" in runtime, "the image must be secure by default"
    assert "--require-hashes" in DOCKERFILE, "dependencies must be hash-verified"


def test_dockerfile_builder_copies_every_file_the_wheel_force_includes() -> None:
    # `uv build` in the builder stage fails if a force-included file is not in /src yet.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    builder = DOCKERFILE.split(" AS runtime", 1)[0]
    before_build = builder[: builder.index("uv build")]
    copies = re.findall(r"^COPY\s+(?!--)(.+?)\s+\S+$", before_build, flags=re.MULTILINE)
    copied = {source.rstrip("/*") for line in copies for source in line.split()}
    for path in force_include:
        assert path in copied or path.split("/", 1)[0] in copied, (
            f"{path} is force-included in the wheel but not copied before `uv build`"
        )


def test_dockerfile_bakes_in_no_secrets_and_fetches_nothing_remote() -> None:
    assert not re.search(r"^ADD\s", DOCKERFILE, flags=re.MULTILINE), "use COPY, never ADD"
    for line in re.findall(r"^(?:ENV|ARG)\s.*$", DOCKERFILE, flags=re.MULTILINE):
        assert not re.search(r"(PASSWORD|SECRET|TOKEN|API_KEY|PEPPER|HMAC)", line, re.I), line
    assert not re.search(r"curl |wget ", DOCKERFILE)


def test_dockerignore_is_an_allow_list_that_excludes_secrets() -> None:
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*", "the build context must start from 'exclude everything'"
    allowed = {line[1:] for line in lines if line.startswith("!")}
    assert not any(a.startswith((".env", "deployment", "tests", ".venv", "var")) for a in allowed)
    assert {"pyproject.toml", "src/", "migrations/", "alembic.ini"} <= allowed


# --------------------------------------------------------------------------------------- #
# compose.yaml
# --------------------------------------------------------------------------------------- #
def test_compose_app_containers_are_hardened() -> None:
    services = _compose()["services"]
    for name in APP_SERVICES:
        svc = services[name]
        assert svc.get("read_only") is True, name
        assert "ALL" in svc.get("cap_drop", []), name
        assert "no-new-privileges:true" in svc.get("security_opt", []), name
        assert svc.get("user") == APP_UID, name
        assert any(str(t).startswith("/tmp:") for t in svc.get("tmpfs", [])), name
        assert "noexec" in next(t for t in svc["tmpfs"] if str(t).startswith("/tmp:")), name
        for limit in ("pids_limit", "mem_limit", "cpus"):
            assert svc.get(limit), f"{name}: {limit} missing"
        assert not svc.get("privileged"), name
    for name in ("api", "worker"):
        assert services[name]["restart"] == "unless-stopped"
        check = services[name].get("healthcheck", {})
        assert check.get("test") and not check.get("disable"), name


def test_compose_every_container_drops_capabilities_and_privilege_escalation() -> None:
    for name, svc in _compose()["services"].items():
        assert "ALL" in svc.get("cap_drop", []), name
        assert "no-new-privileges:true" in svc.get("security_opt", []), name
        assert not svc.get("privileged"), name
        assert "network_mode" not in svc, name


def test_compose_publishes_only_the_api_and_only_on_loopback() -> None:
    services = _compose()["services"]
    for name, svc in services.items():
        if name != "api":
            assert not svc.get("ports"), f"{name} must not publish ports"
    for port in services["api"]["ports"]:
        assert str(port).startswith("127.0.0.1:"), port


def test_compose_networks_isolate_the_data_stores() -> None:
    data = _compose()
    assert data["networks"]["backend"].get("internal") is True
    services = data["services"]
    for name in ("postgres", "redis", "qdrant", "ollama", "migrate"):
        assert services[name]["networks"] == ["backend"], name
    reaches_out = {n for n, s in services.items() if "egress" in s.get("networks", [])}
    # clamav: signature updates; ollama-pull: one-shot model download (no backend access).
    assert reaches_out == {"api", "worker", "clamav", "ollama-pull"}, reaches_out
    assert "backend" not in services["ollama-pull"]["networks"]
    assert {n for n, s in services.items() if "edge" in s.get("networks", [])} == {"api"}


def test_compose_passes_secrets_only_as_files() -> None:
    data = _compose()
    for name, svc in data["services"].items():
        env = _service_env(svc)
        targets = _secret_targets(svc)
        for key, value in env.items():
            assert key not in SECRET_KEYS, f"{name}: {key} must be passed as {key}_FILE"
            assert not re.search(r"(PASSWORD|SECRET|TOKEN|API_KEY)$", key), f"{name}: {key}"
            if key.endswith("_FILE"):
                assert value.startswith("/run/secrets/"), f"{name}: {key}"
                assert value in targets, f"{name}: {key} points at a secret it does not mount"
        for source in (s if isinstance(s, str) else s["source"] for s in svc.get("secrets", [])):
            assert source in data["secrets"], f"{name}: undefined secret {source}"
    for name, spec in data["secrets"].items():
        assert spec["file"].startswith("./deployment/secrets/"), name


def test_compose_file_variables_name_real_settings() -> None:
    for name, svc in _compose()["services"].items():
        for key in _service_env(svc):
            if key.startswith("DOCASSIST_") and key.endswith("_FILE"):
                target = key.removesuffix("_FILE")
                assert target in SETTINGS_KEYS, f"{name}: {key} matches no setting"
            elif key.startswith("DOCASSIST_") and name in APP_SERVICES:
                assert key in SETTINGS_KEYS, f"{name}: {key} matches no setting"


def test_compose_override_files_only_use_known_settings_and_file_secrets() -> None:
    for path in sorted((ROOT / "deployment" / "compose").glob("*.yaml")):
        data = _load_yaml(str(path.relative_to(ROOT)))
        for name, svc in data.get("services", {}).items():
            for key, value in _service_env(svc).items():
                if not key.startswith("DOCASSIST_"):
                    continue
                base = key.removesuffix("_FILE")
                assert base in SETTINGS_KEYS, f"{path.name}/{name}: {key}"
                assert key.endswith("_FILE") or key not in SECRET_KEYS, f"{path.name}/{name}: {key}"
                if key.endswith("_FILE"):
                    assert value in _secret_targets(svc), f"{path.name}/{name}: {key} not mounted"
            assert not svc.get("ports"), f"{path.name}/{name} publishes a port"
        for name, spec in data.get("secrets", {}).items():
            assert spec["file"].startswith("./deployment/secrets/"), f"{path.name}: {name}"


def test_every_compose_secret_file_is_generated_or_documented() -> None:
    script = (ROOT / "scripts" / "generate-secrets.sh").read_text(encoding="utf-8")
    generated = set(re.findall(r"^\s*ensure\s+([\w.]+)\s", script, flags=re.MULTILINE))
    generated |= set(re.findall(r"write_secret\s+([\w.]+)\s", script))
    files = {Path(spec["file"]).name for spec in _compose()["secrets"].values()}
    for path in (ROOT / "deployment" / "compose").glob("*.yaml"):
        files |= {
            Path(s["file"]).name
            for s in _load_yaml(str(path.relative_to(ROOT))).get("secrets", {}).values()
        }
    missing = files - generated - OPERATOR_SUPPLIED_SECRETS
    assert not missing, f"no generator for {sorted(missing)}"


def test_compose_healthcheck_host_is_allowed_by_trusted_host_middleware() -> None:
    env = _service_env(_compose()["services"]["api"])
    hosts = yaml.safe_load(env["DOCASSIST_SECURITY__ALLOWED_HOSTS"])
    assert "127.0.0.1" in hosts, "the Docker health check calls http://127.0.0.1:8000"


def test_compose_default_stack_is_a_valid_offline_development_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _service_env(_compose()["services"]["worker"])
    settings = _settings_from_env(monkeypatch, env)
    assert settings.environment.value == "development"
    assert settings.llm.provider == "local_extractive"
    assert settings.embedding.provider == "hashing"
    assert str(settings.storage.root).replace("\\", "/") == "/var/lib/docassist/storage"


def test_compose_production_override_passes_the_production_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = {
        "DOCASSIST_PUBLIC_BASE_URL": "https://docs.example.com",
        "DOCASSIST_ALLOWED_HOSTS": '["docs.example.com","127.0.0.1"]',
        "DOCASSIST_OUTBOUND_ALLOWED_HOSTS": '["api.anthropic.com","embeddings.example.com"]',
        "DOCASSIST_EMBEDDING_BASE_URL": "https://embeddings.example.com/v1",
        "DOCASSIST_EMBEDDING_MODEL": "text-embedding-test",
    }
    base = _service_env(_compose()["services"]["api"])
    production = _service_env(_load_yaml("deployment/compose/production.yaml")["services"]["api"])
    merged = {k: _interpolate(v, variables) for k, v in {**base, **production}.items()}
    settings = _settings_from_env(monkeypatch, merged)
    assert settings.is_production
    assert settings.database.ssl == "require"
    assert settings.upload.malware_scanner == "clamav"
    assert not settings.security.expose_api_docs
    with pytest.raises(KeyError, match="DOCASSIST_PUBLIC_BASE_URL"):
        _interpolate(production["DOCASSIST_PUBLIC_BASE_URL"], {})


def test_compose_secret_files_load_through_load_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    values = _secret_overrides()
    flat = {
        "database_url": values["database"]["url"],
        "database_worker_url": values["database"]["url"].replace(
            "docassist_app", "docassist_worker"
        ),
        "redis_url": values["redis"]["url"],
        "encryption_key_id": values["security"]["active_encryption_key_id"],
        **{k: v for k, v in values["security"].items() if k != "active_encryption_key_id"},
    }
    for key in list(os.environ):
        if key.upper().startswith("DOCASSIST_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)  # no stray .env
    for key, value in _service_env(_compose()["services"]["worker"]).items():
        if key.endswith("_FILE"):
            name = value.removeprefix("/run/secrets/")
            path = tmp_path / name
            path.write_text(flat[name] + "\n", encoding="utf-8")
            monkeypatch.setenv(key, str(path))
        else:
            monkeypatch.setenv(key, value)
    settings = load_settings()
    assert settings.database.worker_url is not None
    assert (
        settings.security.active_encryption_key_id == values["security"]["active_encryption_key_id"]
    )


# --------------------------------------------------------------------------------------- #
# supporting configuration (Redis, PostgreSQL init, nginx)
# --------------------------------------------------------------------------------------- #
def test_redis_is_bounded_non_persistent_and_acl_protected() -> None:
    conf = (ROOT / "deployment" / "redis" / "redis.conf").read_text(encoding="utf-8")
    directives = dict(
        line.split(None, 1)
        for line in conf.splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert directives["protected-mode"] == "yes"
    assert directives["maxmemory"] == "256mb"
    assert directives["maxmemory-policy"] == "volatile-lru"
    assert directives["appendonly"] == "no"
    assert directives["aclfile"] == "/run/secrets/redis_users.acl"
    assert "/run/secrets/redis_users.acl" in _secret_targets(_compose()["services"]["redis"])
    script = (ROOT / "scripts" / "generate-secrets.sh").read_text(encoding="utf-8")
    acl_lines = re.findall(r"printf '(user [^']*)'", script)
    assert any("-@dangerous" in line and "~docassist:*" in line for line in acl_lines)
    assert any(
        line.startswith("user default on nopass") and "-@all +ping" in line for line in acl_lines
    )


def test_postgres_init_creates_least_privilege_roles() -> None:
    sql = (ROOT / "deployment" / "postgres" / "initdb" / "10-docassist-init.sh").read_text(
        encoding="utf-8"
    )
    for role in ("docassist_owner", "docassist_app", "docassist_worker"):
        create = re.search(rf"CREATE ROLE {role}\b[^;]*;", sql, flags=re.DOTALL)
        assert create, role
        for flag in ("NOSUPERUSER", "NOCREATEDB", "NOCREATEROLE", "NOBYPASSRLS"):
            assert flag in create.group(0), f"{role}: {flag}"
        assert "PASSWORD :'" in create.group(0), f"{role}: password must be a bound psql variable"
    assert re.search(r"CREATE DATABASE :\"db_name\" OWNER docassist_owner", sql)
    assert "REVOKE ALL ON DATABASE" in sql
    assert "log_min_error_statement = panic" in sql
    services = _compose()["services"]
    assert (
        "--auth-local=scram-sha-256" in services["postgres"]["environment"]["POSTGRES_INITDB_ARGS"]
    )
    assert "POSTGRES_PASSWORD" not in services["postgres"]["environment"]


def test_production_pg_hba_requires_tls_and_scram_everywhere() -> None:
    hba = (ROOT / "deployment" / "postgres" / "pg_hba.production.conf").read_text(encoding="utf-8")
    rules = [line.split() for line in hba.splitlines() if line.strip() and not line.startswith("#")]
    assert rules
    for rule in rules:
        assert rule[0] in {"local", "hostssl"}, rule
        assert rule[-1] == "scram-sha-256", rule


def test_nginx_example_terminates_tls_with_hsts_limits_and_upload_size() -> None:
    conf = (ROOT / "deployment" / "nginx" / "docassist.conf").read_text(encoding="utf-8")
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in conf
    assert re.search(
        r'add_header Strict-Transport-Security "max-age=\d+; includeSubDomains" always;', conf
    )
    assert "server_tokens off;" in conf
    assert "limit_req_zone" in conf
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in conf
    size = re.search(r"client_max_body_size (\d+)m;", conf)
    assert size
    upload_max = Settings.model_fields["upload"].default.max_upload_bytes
    assert int(size.group(1)) * 1024 * 1024 >= upload_max


# --------------------------------------------------------------------------------------- #
# Kubernetes
# --------------------------------------------------------------------------------------- #
K8S_DIR = ROOT / "deployment" / "kubernetes"


def _k8s_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(K8S_DIR.glob("*.yaml")):
        docs.extend(_load_all_yaml(path))
    return docs


def _workloads() -> list[dict[str, Any]]:
    return [
        d for d in _k8s_docs() if d["kind"] in {"Deployment", "Job", "StatefulSet", "DaemonSet"}
    ]


def test_kubernetes_workloads_are_restricted_pod_security_compliant() -> None:
    workloads = _workloads()
    assert {w["kind"] for w in workloads} == {"Deployment", "Job"}
    for workload in workloads:
        pod = workload["spec"]["template"]["spec"]
        name = workload["metadata"].get("name") or workload["metadata"]["generateName"]
        assert pod["automountServiceAccountToken"] is False, name
        assert pod["securityContext"]["runAsNonRoot"] is True, name
        assert pod["securityContext"]["runAsUser"] == 10001, name
        assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault", name
        assert not pod.get("hostNetwork") and not pod.get("hostPID"), name
        for container in pod["containers"]:
            ctx = container["securityContext"]
            assert ctx["allowPrivilegeEscalation"] is False, name
            assert ctx["readOnlyRootFilesystem"] is True, name
            assert ctx["capabilities"]["drop"] == ["ALL"], name
            assert "add" not in ctx["capabilities"], name
            assert container["resources"]["limits"]["memory"], name
            mounts = {m["mountPath"] for m in container.get("volumeMounts", [])}
            assert "/tmp" in mounts, f"{name}: read-only root needs a /tmp volume"


def test_kubernetes_secrets_are_files_naming_real_settings() -> None:
    for workload in _workloads():
        pod = workload["spec"]["template"]["spec"]
        mounted: set[str] = set()
        for volume in pod["volumes"]:
            if "secret" in volume:
                mount = next(
                    m["mountPath"]
                    for c in pod["containers"]
                    for m in c["volumeMounts"]
                    if m["name"] == volume["name"]
                )
                mounted |= {f"{mount}/{item['path']}" for item in volume["secret"]["items"]}
                assert volume["secret"]["defaultMode"] == 0o440
        for container in pod["containers"]:
            for var in container.get("env", []):
                key = var["name"]
                assert key not in SECRET_KEYS, f"{key} must be a *_FILE reference"
                assert "valueFrom" not in var or "secretKeyRef" not in var["valueFrom"], key
                if key.endswith("_FILE"):
                    assert key.removesuffix("_FILE") in SETTINGS_KEYS, key
                    assert var["value"] in mounted, f"{key} -> {var['value']} is not mounted"


def test_kubernetes_network_policies_default_deny() -> None:
    policies = [d for d in _k8s_docs() if d["kind"] == "NetworkPolicy"]
    deny = [
        p
        for p in policies
        if p["spec"]["podSelector"] == {}
        and "ingress" not in p["spec"]
        and "egress" not in p["spec"]
    ]
    assert deny and set(deny[0]["spec"]["policyTypes"]) == {"Ingress", "Egress"}
    namespace = next(d for d in _k8s_docs() if d["kind"] == "Namespace")
    assert namespace["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"


def test_kubernetes_kustomization_ships_no_secret_values() -> None:
    kustomization = yaml.safe_load((K8S_DIR / "kustomization.yaml").read_text(encoding="utf-8"))
    for resource in kustomization["resources"]:
        assert (K8S_DIR / resource).is_file(), resource
        for doc in _load_all_yaml(K8S_DIR / resource):
            assert doc["kind"] != "Secret", f"{resource} contains a Secret"
    example = _load_all_yaml(K8S_DIR / "secret.example.yaml")[0]
    assert all(value == "" for value in example["stringData"].values())
    assert "secret.example.yaml" not in kustomization["resources"]


def test_kubernetes_configmap_is_a_valid_production_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = next(d for d in _k8s_docs() if d["kind"] == "ConfigMap")["data"]
    for key in config:
        assert key in SETTINGS_KEYS, key
        assert key not in SECRET_KEYS, key
    settings = _settings_from_env(monkeypatch, {k: str(v) for k, v in config.items()})
    assert settings.is_production
    probes = [
        header["value"]
        for w in _workloads()
        for c in w["spec"]["template"]["spec"]["containers"]
        for probe in ("startupProbe", "livenessProbe", "readinessProbe")
        if probe in c
        for header in c[probe]["httpGet"].get("httpHeaders", [])
        if header["name"] == "Host"
    ]
    assert probes, "probes must send an allowed Host header"
    assert set(probes) <= set(settings.security.allowed_hosts)


# --------------------------------------------------------------------------------------- #
# GitHub Actions
# --------------------------------------------------------------------------------------- #
WORKFLOW_FILES = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
ACTION_FILES = sorted((ROOT / ".github" / "actions").glob("*/action.yml"))
_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<comment>\s+#.*)?$", re.MULTILINE)


@pytest.mark.parametrize("path", WORKFLOW_FILES + ACTION_FILES, ids=lambda p: p.name)
def test_actions_are_pinned_to_full_commit_shas(path: Path) -> None:
    refs = list(_USES.finditer(path.read_text(encoding="utf-8")))
    assert refs or path in ACTION_FILES
    for match in refs:
        ref = match["ref"]
        if ref.startswith("./"):
            assert (ROOT / ref / "action.yml").is_file(), ref
            continue
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), f"not SHA-pinned: {ref}"
        assert match["comment"] and re.search(r"#\s*v?\d", match["comment"]), (
            f"no version comment: {ref}"
        )


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_workflows_use_least_privilege_tokens(path: Path) -> None:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads "on" as True
    assert "pull_request_target" not in triggers
    assert workflow["permissions"] == {"contents": "read"}
    allowed_writes = {"security-events"}
    for name, job in workflow["jobs"].items():
        assert job.get("timeout-minutes"), f"{name}: timeout-minutes"
        for scope, level in job.get("permissions", {}).items():
            assert level in {"read", "none"} or scope in allowed_writes, f"{name}: {scope}: {level}"
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") is False, name


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_workflows_contain_no_secret_values(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"://[^/\s:@]+:[^@\s]+@", text), "credentials in a URL"
    assert not re.search(r"(?i)(password|api_key|token)\s*:\s*['\"]?[A-Za-z0-9+/_-]{12,}", text)


def test_ci_covers_the_required_gates() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for needle in (
        "ruff check",
        "ruff format --check",
        "mypy",
        "lint-imports",
        "DOCASSIST_TEST_DATABASE_URL",
        "pgvector/pgvector:",
        "--cov=docassist",
        "bandit -c pyproject.toml -r src",
        "pip-audit",
        "gitleaks/gitleaks-action@",
        "aquasecurity/trivy-action@",
        "anchore/sbom-action@",
        "ignore-unfixed: true",
        "severity: HIGH,CRITICAL",
    ):
        assert needle in ci, needle
    matrix = yaml.safe_load(ci)["jobs"]["tests"]["strategy"]["matrix"]["python"]
    assert {"3.12", "3.13"} <= set(matrix)


# --------------------------------------------------------------------------------------- #
# .env.example and .gitignore
# --------------------------------------------------------------------------------------- #
_ENV_LINE = re.compile(r"^(?:#\s*)?(DOCASSIST_[A-Z0-9_]+)=(.*)$")


def _env_example() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = _ENV_LINE.match(line.strip())
        if match:
            entries[match[1]] = match[2].strip().strip("'\"")
    return entries


def test_env_example_documents_every_setting_and_nothing_else() -> None:
    documented = {key for key in _env_example() if not key.endswith("_FILE")}
    assert set(SETTINGS_KEYS) - documented == set(), "undocumented settings"
    assert documented - set(SETTINGS_KEYS) == set(), "documented keys that do not exist"


def test_env_example_contains_no_usable_secret() -> None:
    for key, value in _env_example().items():
        is_url = "://" in value
        if key in SECRET_KEYS and value and not is_url:
            with pytest.raises(ValueError, match=r"at least 32 bytes|placeholder|entropy"):
                _require_strong_secret(key, SecretStr(value))
        # DSNs: the password part must be a placeholder.
        dsn_password = re.search(r"://[^:/@\s]+:([^@\s]+)@", value)
        if dsn_password:
            assert re.fullmatch(r"<[^>]+>", dsn_password.group(1)), f"{key}: real password in DSN"
        assert not re.search(r"[A-Za-z0-9+/]{43}=", value), f"{key}: looks like a real key"


def test_gitignore_keeps_secrets_and_state_out_of_git() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (
        "/.env",
        "/deployment/secrets/",
        "*.pem",
        "*.key",
        "/.venv/",
        "/var/",
        ".demo-credentials*",
        "coverage.xml",
    ):
        assert required in patterns, required
    assert "!/.env.example" in patterns
