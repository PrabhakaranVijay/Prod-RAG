import json
import logging
import sys
from pathlib import Path

from app.core.logging import JsonFormatter


REQUIRED_FIELDS = {
    "timestamp",
    "level",
    "message",
    "module",
    "function",
    "latency_ms",
    "input_tokens",
    "output_tokens",
}


def test_json_formatter_emits_required_schema_with_null_optionals():
    record = logging.LogRecord(
        name="app.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
        func="sample_function",
    )

    payload = json.loads(JsonFormatter().format(record))

    assert REQUIRED_FIELDS.issubset(payload)
    assert payload["timestamp"].endswith("Z")
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["module"] == Path(__file__).stem
    assert payload["function"] == "sample_function"
    assert payload["latency_ms"] is None
    assert payload["input_tokens"] is None
    assert payload["output_tokens"] is None


def test_json_formatter_serializes_exception_info():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            name="app.tests",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="operation failed",
            args=(),
            exc_info=sys.exc_info(),
            func="failing_function",
        )
        record.latency_ms = 7

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "ERROR"
    assert payload["latency_ms"] == 7
    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["exception"]["message"] == "boom"
    assert "Traceback" in payload["exception"]["traceback"]
