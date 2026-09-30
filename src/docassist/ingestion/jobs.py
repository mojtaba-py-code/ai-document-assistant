"""Background job handlers for the ingestion area.

``ingest_version`` {version_id} is enqueued by the documents service (upload, new version,
URL import) with ``idempotency_key = f"ingest:{version_id}"`` and handled here by the
pipeline. Retryable failures that exhaust the job's attempts are recorded on the version
(``failed`` + error code) so a document never stays "processing" forever.
"""

from __future__ import annotations

from typing import Any

from docassist.core.ids import parse_uuid
from docassist.jobs.queue import ClaimedJob, JobError, PermanentJobError
from docassist.jobs.registry import JobContext, job_handler


@job_handler("ingest_version")
async def ingest_version(ctx: JobContext, job: ClaimedJob) -> dict[str, Any] | None:
    raw = job.payload.get("version_id")
    version_id = parse_uuid(raw) if isinstance(raw, str) else None
    if version_id is None or job.organization_id is None:
        raise PermanentJobError(
            "ingest_version needs an organisation and a version id", code="invalid_payload"
        )
    pipeline = ctx.container.pipeline
    final_attempt = job.attempts >= job.max_attempts
    try:
        return await pipeline.process_version(job.organization_id, version_id)
    except PermanentJobError:
        raise  # already recorded on the version by the pipeline
    except JobError as exc:
        if final_attempt:
            await pipeline.mark_failed(job.organization_id, version_id, exc.code)
        raise
    except Exception:
        if final_attempt:
            await pipeline.mark_failed(job.organization_id, version_id, "internal_error")
        raise
