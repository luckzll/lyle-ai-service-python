import logging
from typing import Any

from app.core.config import settings
from app.core.database import postgres_connection
from app.security.secret_masker import mask_secrets

logger = logging.getLogger(__name__)


def write_execution_audit(
    *,
    user_id: str,
    source_type: str,
    executor_type: str,
    executor_name: str,
    operation: str,
    risk_level: str,
    status: str,
    source_id: str | None = None,
    project_id: int | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    logger.info(
        "Execution audit. user_id=%s executor=%s:%s operation=%s status=%s risk=%s",
        user_id,
        executor_type,
        executor_name,
        operation,
        status,
        risk_level,
    )
    if not settings.postgres_enabled:
        return
    try:
        with postgres_connection() as conn:
            conn.execute(
                """
                INSERT INTO execution_audit
                  (user_id, project_id, source_type, source_id, executor_type, executor_name,
                   operation, risk_level, status, input_summary, output_summary, error_code, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    project_id,
                    source_type,
                    source_id,
                    executor_type,
                    executor_name,
                    operation,
                    risk_level,
                    status,
                    mask_secrets(input_summary)[:2000],
                    mask_secrets(output_summary)[:4000],
                    error_code,
                    mask_secrets(error_message)[:2000],
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("Failed to write execution audit.")


def list_execution_audit(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    if not settings.postgres_enabled:
        return []
    limit = max(1, min(limit, 200))
    with postgres_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, project_id, source_type, source_id, executor_type, executor_name,
                   operation, risk_level, status, input_summary, output_summary,
                   error_code, error_message, created_at
            FROM execution_audit
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "projectId": row[1],
            "sourceType": row[2],
            "sourceId": row[3],
            "executorType": row[4],
            "executorName": row[5],
            "operation": row[6],
            "riskLevel": row[7],
            "status": row[8],
            "inputSummary": row[9],
            "outputSummary": row[10],
            "errorCode": row[11],
            "errorMessage": row[12],
            "createdAt": row[13].isoformat(),
        }
        for row in rows
    ]
