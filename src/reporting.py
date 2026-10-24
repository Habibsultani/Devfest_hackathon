"""Session inspection records: AI observations and human decisions stay separate."""

import csv
import io
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from src.model_config import ModelConfig
from src.model_gateway import InspectionResult

DECISIONS = ("Accept", "Flag for review", "Inconclusive")


def validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
        raise ValueError(f"{label} is required and must be at most 80 characters.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} cannot contain control characters.")
    return value.strip()


@dataclass(frozen=True)
class InspectionRecord:
    inspection_id: str
    timestamp_utc: str
    analyzed_at_utc: str
    batch_id: str
    instrument_id: str
    instrument_class: str
    source_filename: str
    image_sha256: str
    anomaly_score: float
    threshold: float
    requires_review: bool
    model_version: str
    artifact_sha256: str
    config_sha256: str
    inference_ms: float
    operator_decision: str
    operator_note: str


def create_record(
    *,
    inspection_id: str,
    analyzed_at_utc: str,
    batch_id: str,
    instrument_id: str,
    source_filename: str,
    image_sha256: str,
    result: InspectionResult,
    config: ModelConfig,
    inference_ms: float,
    decision: str | None,
    note: str,
) -> InspectionRecord:
    batch_id = validate_identifier(batch_id, "Batch ID")
    instrument_id = validate_identifier(instrument_id, "Instrument ID")
    if decision not in DECISIONS:
        raise ValueError("Choose an operator decision before saving the inspection.")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("The operator note must be at most 2000 characters.")
    if decision == "Inconclusive" and not note.strip():
        raise ValueError("Add a note explaining why this inspection is inconclusive.")
    return InspectionRecord(
        inspection_id=inspection_id,
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        analyzed_at_utc=analyzed_at_utc,
        batch_id=batch_id,
        instrument_id=instrument_id,
        instrument_class=config.instrument_class,
        source_filename=source_filename,
        image_sha256=image_sha256,
        anomaly_score=result.anomaly_score,
        threshold=config.threshold,
        requires_review=result.requires_review,
        model_version=result.model_version,
        artifact_sha256=config.artifact_sha256,
        config_sha256=config.config_sha256,
        inference_ms=round(inference_ms, 3),
        operator_decision=decision,
        operator_note=note.strip(),
    )


def _csv_cell(value: object) -> object:
    # Spreadsheet programs can execute leading formulas even in quoted CSV cells.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def records_to_csv(records: Iterable[InspectionRecord]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(InspectionRecord.__dataclass_fields__))
    writer.writeheader()
    for record in records:
        writer.writerow({key: _csv_cell(value) for key, value in asdict(record).items()})
    return output.getvalue().encode("utf-8-sig")


def filter_records(
    records: Iterable[InspectionRecord], *, query: str = "", decision: str = "All decisions"
) -> list[InspectionRecord]:
    """Search session records without mutating their data or limiting the CSV export."""
    needle = query.strip().casefold()
    return [
        record
        for record in records
        if (decision == "All decisions" or record.operator_decision == decision)
        and (
            not needle
            or needle
            in " ".join(
                (
                    record.batch_id,
                    record.instrument_id,
                    record.instrument_class,
                    record.model_version,
                    record.source_filename,
                    record.operator_note,
                )
            ).casefold()
        )
    ]
