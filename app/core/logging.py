from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


SAFE_EXTRA_FIELDS = {
    "chunk_count",
    "document_count",
    "error_code",
    "method",
    "path",
    "request_id",
    "result_count",
    "retrieval_mode",
    "status_code",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "latency_ms": getattr(record, "latency_ms", None),
            "input_tokens": getattr(record, "input_tokens", None),
            "output_tokens": getattr(record, "output_tokens", None),
        }

        for key in SAFE_EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(payload, ensure_ascii=False, default=str)


_configured = False
_handler: logging.Handler | None = None


def configure_logging(level: int = logging.INFO) -> None:
    global _configured, _handler

    root_logger = logging.getLogger()

    if not _configured:
        root_logger.handlers.clear()
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(JsonFormatter())
        root_logger.addHandler(_handler)
        _configured = True
    elif _handler is not None:
        _handler.setFormatter(JsonFormatter())
        if _handler not in root_logger.handlers:
            root_logger.addHandler(_handler)

    root_logger.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


configure_logging()
logger = get_logger("prod-rag")
