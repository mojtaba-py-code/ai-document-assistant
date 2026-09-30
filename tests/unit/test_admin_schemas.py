"""Value rules of the administration schemas (no database)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from docassist.core.enums import Classification
from docassist.identity.admin import _circuit_states, _resource_ids
from docassist.identity.schemas import (
    EXPORT_ROW_CAP,
    CursorError,
    DepartmentCreate,
    DepartmentUpdate,
    OrganizationCreate,
    OrgSettings,
    OrgSettingsPatch,
    UserCreate,
    UserUpdate,
    decode_cursor,
    effective_export_rows,
    effective_external_ceiling,
    effective_retention,
    effective_token_budget,
    encode_cursor,
    escape_like,
    int_cursor,
    normalize_email,
    normalize_slug,
    parse_int_cursor,
    parse_month,
    parse_time_id_cursor,
    slugify,
    time_id_cursor,
    validate_action_filter,
    validate_org_settings,
)
from tests.conftest import make_settings

ZWSP = chr(0x200B)
RLO = chr(0x202E)


# --------------------------------------------------------------------------- #
# Emails, slugs, names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Alice@Example.COM ", "alice@example.com"),
        ("first.last+tag@sub.example.org", "first.last+tag@sub.example.org"),
    ],
)
def test_valid_emails_are_normalised(raw: str, expected: str) -> None:
    assert normalize_email(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "plainaddress",
        "a@b",
        "a@@example.com",
        ".dot@example.com",
        "dot.@example.com",
        "a..b@example.com",
        "a@-example.com",
        "a@example.123",
        "user@exa mple.com",
        "x" * 65 + "@example.com",
        "caf" + chr(0xE9) + "@example.com",  # non-ASCII local part (homoglyph risk)
        "a@ex" + chr(0x0430) + "mple.com",  # Cyrillic 'a' in the domain
    ],
)
def test_invalid_emails_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid email"):
        normalize_email(raw)


def test_slug_rules() -> None:
    assert normalize_slug(" Acme-EU ") == "acme-eu"
    for bad in ("-acme", "acme-", "ac me", "", "a" * 64, "acme_eu"):
        with pytest.raises(ValueError, match="slug"):
            normalize_slug(bad)
    assert slugify("Research & D" + chr(0xE9) + "veloppement") == "research-developpement"
    assert slugify("!!!") == "department"
    assert len(slugify("x" * 200)) == 63


def test_names_are_cleaned_of_invisible_characters() -> None:
    org = OrganizationCreate(
        slug="acme",
        name=f"  Acme{ZWSP}  Corp {RLO}",
        admin_email="A@Example.com",
        admin_name="Ada\nLovelace",
    )
    assert org.name == "Acme Corp"
    assert org.admin_name == "Ada Lovelace"
    assert org.admin_email == "a@example.com"
    with pytest.raises(ValidationError):
        OrganizationCreate(slug="acme", name=ZWSP * 3, admin_email="a@example.com", admin_name="A")


def test_like_escaping() -> None:
    assert escape_like("50%_off\\") == "50\\%\\_off\\\\"


def test_action_filter_validation() -> None:
    assert validate_action_filter("admin.user_created") == "admin.user_created"
    assert validate_action_filter("admin.*") == "admin.*"
    for bad in ("*", "admin.*.x", "Admin.x", "admin.x;drop", "a" * 65, "admin..x"):
        with pytest.raises(ValueError, match="action filter"):
            validate_action_filter(bad)


def test_month_parsing() -> None:
    assert parse_month("2026-09") == (2026, 9)
    for bad in ("2026-13", "2026-00", "1999-01", "26-09", "2026/09", "September"):
        with pytest.raises(ValueError, match="month"):
            parse_month(bad)


# --------------------------------------------------------------------------- #
# Input models
# --------------------------------------------------------------------------- #
def test_user_models_are_strict() -> None:
    dept = uuid.uuid4()
    with pytest.raises(ValidationError):
        UserCreate.model_validate(
            {"email": "a@example.com", "full_name": "A", "role": "employee", "password": "x" * 20}
        )
    with pytest.raises(ValidationError):
        UserCreate.model_validate(
            {
                "email": "a@example.com",
                "full_name": "A",
                "role": "employee",
                "departments": [{"department_id": str(dept)}, {"department_id": str(dept)}],
            }
        )
    with pytest.raises(ValidationError):
        UserUpdate.model_validate({})
    assert UserUpdate.model_validate({"departments": []}).departments == []


def test_department_models() -> None:
    created = DepartmentCreate(name=" Legal ", description=f"Contracts{ZWSP}")
    assert (created.name, created.slug, created.description) == ("Legal", None, "Contracts")
    with pytest.raises(ValidationError):
        DepartmentUpdate.model_validate({})
    with pytest.raises(ValidationError):
        DepartmentUpdate.model_validate({"name": None})
    cleared = DepartmentUpdate.model_validate({"description": None})
    assert cleared.model_fields_set == {"description"} and cleared.description is None


# --------------------------------------------------------------------------- #
# Organisation settings
# --------------------------------------------------------------------------- #
def test_stored_settings_are_parsed_leniently_and_round_trip() -> None:
    stored = {
        "llm": {"external_max_classification": "INTERNAL", "monthly_token_budget": 5, "junk": 1},
        "retention": {"conversation_days": "not-a-number"},
        "feature_flags": {"beta": True},
    }
    parsed = OrgSettings.from_stored(stored)
    assert parsed.llm.external_max_classification is Classification.INTERNAL
    assert parsed.llm.monthly_token_budget == 5
    assert parsed.retention.conversation_days is None  # invalid section -> defaults
    out = parsed.to_stored(stored)
    assert out == {
        "llm": {"external_max_classification": "INTERNAL", "monthly_token_budget": 5},
        "feature_flags": {"beta": True},  # unrelated keys are preserved
    }
    assert OrgSettings.from_stored(None) == OrgSettings()


def test_patch_tracks_explicit_fields_only() -> None:
    patch = OrgSettingsPatch.model_validate({"llm": {"monthly_token_budget": None}})
    assert patch.changed_fields() == {"llm": {"monthly_token_budget": None}}
    with pytest.raises(ValidationError):
        OrgSettingsPatch.model_validate({})
    with pytest.raises(ValidationError):
        OrgSettingsPatch.model_validate({"llm": {"provider": "anthropic"}})


def test_org_settings_may_only_tighten_the_deployment() -> None:
    settings = make_settings(
        llm={"external_max_classification": "CONFIDENTIAL", "monthly_token_budget_per_org": 1000},
        retention={"audit_days": 400, "llm_usage_days": 100},
    )
    assert (
        validate_org_settings(
            {
                "llm": {
                    "external_max_classification": Classification.INTERNAL,
                    "monthly_token_budget": 999,
                }
            },
            settings,
        )
        == []
    )
    problems = validate_org_settings(
        {
            "llm": {
                "external_max_classification": Classification.RESTRICTED,
                "monthly_token_budget": 1001,
            },
            "retention": {"audit_days": 399, "llm_usage_days": 99, "conversation_days": 1},
        },
        settings,
    )
    assert len(problems) == 4
    unlimited = make_settings(llm={"monthly_token_budget_per_org": 0})
    assert validate_org_settings({"llm": {"monthly_token_budget": 10**9}}, unlimited) == []


def test_effective_values_clamp_stale_tenant_settings() -> None:
    settings = make_settings(
        llm={"external_max_classification": "INTERNAL", "monthly_token_budget_per_org": 1000},
        retention={"audit_days": 400},
    )
    # stored before the deployment lowered its limits
    stale = OrgSettings.from_stored(
        {
            "llm": {"external_max_classification": "CONFIDENTIAL", "monthly_token_budget": 5000},
            "retention": {"audit_days": 60, "conversation_days": 7},
        }
    )
    assert effective_external_ceiling(settings, stale) is Classification.INTERNAL
    assert effective_token_budget(settings, stale) == 1000
    retention = effective_retention(settings, stale)
    assert retention["audit_days"] == 400 and retention["conversation_days"] == 7
    stricter = OrgSettings.from_stored({"llm": {"external_max_classification": "PUBLIC"}})
    assert effective_external_ceiling(settings, stricter) is Classification.PUBLIC
    unlimited = make_settings(llm={"monthly_token_budget_per_org": 0})
    assert effective_token_budget(unlimited, OrgSettings()) is None
    assert (
        effective_token_budget(
            unlimited, OrgSettings.from_stored({"llm": {"monthly_token_budget": 7}})
        )
        == 7
    )


# --------------------------------------------------------------------------- #
# Cursors
# --------------------------------------------------------------------------- #
def test_cursor_round_trip_and_tamper_resistance() -> None:
    moment = datetime(2026, 9, 30, 12, 0, 0, 123456, tzinfo=UTC)
    row = uuid.uuid4()
    assert parse_time_id_cursor(time_id_cursor(moment, row)) == (moment, row)
    assert parse_int_cursor(int_cursor(42)) == 42
    for bad in (
        "",
        "!!!",
        "x" * 600,
        encode_cursor({"t": "2026-09-30T12:00:00", "id": str(row)}),  # naive timestamp
        encode_cursor({"t": moment.isoformat(), "id": "not-a-uuid"}),
        encode_cursor({"t": moment.isoformat(), "id": str(row), "extra": 1}),
    ):
        with pytest.raises(CursorError):
            parse_time_id_cursor(bad)
    for bad in (encode_cursor({"id": -1}), encode_cursor({"id": True}), encode_cursor({"id": "7"})):
        with pytest.raises(CursorError):
            parse_int_cursor(bad)
    with pytest.raises(CursorError):
        decode_cursor(encode_cursor({"a": 1}), ("b",))


# --------------------------------------------------------------------------- #
# Output hygiene helpers
# --------------------------------------------------------------------------- #
def test_job_payload_exposes_only_uuid_ids() -> None:
    doc = str(uuid.uuid4())
    payload = {
        "document_id": doc,
        "version_id": "not-a-uuid",
        "note": str(uuid.uuid4()),
        "Evil_Id": str(uuid.uuid4()),
        "export_id": 5,
    }
    assert _resource_ids(payload) == {"document_id": uuid.UUID(doc)}
    assert _resource_ids(None) == {}


async def test_circuit_states_are_read_defensively() -> None:
    class Sync:
        def circuit_states(self) -> dict[str, str]:
            return {"anthropic": "closed", f"local{ZWSP}": "open"}

    class Async:
        async def circuit_states(self) -> dict[str, str]:
            return {"anthropic": "half_open"}

    class Broken:
        def circuit_states(self) -> dict[str, str]:
            raise RuntimeError("boom")

    class Weird:
        def circuit_states(self) -> list[str]:
            return ["closed"]

    assert await _circuit_states(Sync()) == {"anthropic": "closed", "local": "open"}
    assert await _circuit_states(Async()) == {"anthropic": "half_open"}
    assert await _circuit_states(Broken()) is None
    assert await _circuit_states(Weird()) is None
    assert await _circuit_states(object()) is None


def test_export_row_override_can_only_lower_the_platform_cap() -> None:
    assert effective_export_rows(OrgSettings()) == EXPORT_ROW_CAP
    lowered = OrgSettings.from_stored({"exports": {"max_rows": 500}})
    assert effective_export_rows(lowered) == 500
    assert lowered.to_stored() == {"exports": {"max_rows": 500}}
    with pytest.raises(ValidationError):
        OrgSettingsPatch.model_validate({"exports": {"max_rows": EXPORT_ROW_CAP + 1}})
    with pytest.raises(ValidationError):
        OrgSettingsPatch.model_validate({"exports": {"max_rows": 0}})
    # a corrupted stored value falls back to the platform cap
    assert (
        effective_export_rows(OrgSettings.from_stored({"exports": {"max_rows": 10**9}}))
        == EXPORT_ROW_CAP
    )


def test_export_cap_matches_the_export_service() -> None:
    exports = pytest.importorskip("docassist.intelligence.exports")
    assert exports.EXPORT_MAX_ROWS == EXPORT_ROW_CAP
