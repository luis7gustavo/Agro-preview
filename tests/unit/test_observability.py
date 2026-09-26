import json
import logging

from agri_decision.observability import JsonFormatter


def test_json_formatter_preserves_pipeline_fields() -> None:
    record = logging.LogRecord(
        name="agri.pipeline",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="pipeline_finished",
        args=(),
        exc_info=None,
    )
    record.source = "ibge_pam"
    record.rows_received = 12

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "pipeline_finished"
    assert payload["source"] == "ibge_pam"
    assert payload["rows_received"] == 12
