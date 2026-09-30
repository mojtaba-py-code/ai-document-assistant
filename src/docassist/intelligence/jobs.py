"""Background job handlers for the intelligence area."""

from __future__ import annotations

from typing import Any

from docassist.core.ids import parse_uuid
from docassist.jobs.queue import ClaimedJob, PermanentJobError
from docassist.jobs.registry import JobContext, job_handler


@job_handler("export_generate")
async def export_generate(ctx: JobContext, job: ClaimedJob) -> dict[str, Any] | None:
    """Generate one export (payload ``{"export_id": "<uuid>"}``, IDs only)."""
    raw = job.payload.get("export_id")
    export_id = parse_uuid(raw) if isinstance(raw, str) else None
    if export_id is None or job.organization_id is None:
        raise PermanentJobError("malformed export job payload", code="bad_payload")
    return await ctx.container.intelligence.exports.generate(
        job.organization_id,
        export_id,
        worker_db=ctx.worker_db,
        final_attempt=job.attempts >= job.max_attempts,
    )
