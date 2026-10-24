import csv
import io
from dataclasses import replace

import pytest

from src.reporting import create_record, filter_records, records_to_csv


@pytest.fixture
def record_args(config, result):
    return {
        "inspection_id": "fixture-inspection",
        "analyzed_at_utc": "2026-10-24T12:00:00+00:00",
        "batch_id": "BATCH-001",
        "instrument_id": "INST-014",
        "source_filename": "unseen.png",
        "image_sha256": "b" * 64,
        "result": result,
        "config": config,
        "inference_ms": 123.4567,
        "decision": "Accept",
        "note": "Operator reviewed the image.",
    }


def test_ai_flag_and_operator_decision_are_independent(record_args):
    record = create_record(**record_args)
    assert record.requires_review is True
    assert record.operator_decision == "Accept"
    assert record.timestamp_utc.endswith("+00:00")
    assert record.anomaly_score == 2.5
    assert record.threshold == 2.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_id", " "),
        ("instrument_id", "bad\nID"),
        ("instrument_id", "x" * 81),
        ("decision", None),
        ("decision", "Certified safe"),
        ("note", "x" * 2001),
    ],
)
def test_invalid_review_metadata(record_args, field, value):
    record_args[field] = value
    with pytest.raises(ValueError):
        create_record(**record_args)


def test_inconclusive_needs_explanation(record_args):
    record_args.update(decision="Inconclusive", note=" ")
    with pytest.raises(ValueError, match="explaining"):
        create_record(**record_args)


def test_csv_round_trip_handles_unicode_commas_newlines_and_formulas(record_args):
    record_args.update(
        batch_id="=HYPERLINK(1)",
        instrument_id="İNST-014",
        note='Note, with "quotes"\nand a newline',
    )
    record = create_record(**record_args)
    output = records_to_csv([record])
    assert output.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(output.decode("utf-8-sig"))))
    assert len(rows) == 1
    assert rows[0]["batch_id"] == "'=HYPERLINK(1)"
    assert rows[0]["instrument_id"] == "İNST-014"
    assert rows[0]["operator_note"] == record_args["note"]
    assert rows[0]["model_version"] == "fixture-v1"
    assert rows[0]["requires_review"] == "True"
    assert float(rows[0]["anomaly_score"]) == record.anomaly_score


def test_search_and_decision_filter_do_not_change_full_export(record_args):
    first = create_record(**record_args)
    second = replace(
        first, inspection_id="second", batch_id="BATCH-002", operator_decision="Inconclusive"
    )
    records = [first, second]
    assert filter_records(records, query="batch-002", decision="Inconclusive") == [second]
    assert filter_records(records, query="batch-002", decision="Accept") == []
    assert filter_records(records, query="operator reviewed") == records
    exported = list(csv.DictReader(io.StringIO(records_to_csv(records).decode("utf-8-sig"))))
    assert len(exported) == 2
    assert records == [first, second]
