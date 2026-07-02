from app.core.metrics import collect_metrics, collect_prometheus_metrics, record_request_metric


def test_collect_prometheus_metrics_contains_core_gauges():
    text = collect_prometheus_metrics()

    assert "lyle_ai_uptime_seconds" in text
    assert 'lyle_ai_feature_enabled{feature="tool_agent"}' in text
    assert 'lyle_ai_limit{name="memory_recall_limit"}' in text


def test_request_metrics_are_exported_to_json_and_prometheus():
    record_request_metric("GET", "/ai/test/{id}", 200, 0.12)

    metrics = collect_metrics()
    text = collect_prometheus_metrics()

    assert metrics["requests"]["total"] >= 1
    assert 'lyle_ai_http_requests_total{method="GET",path="/ai/test/{id}",status="200"}' in text
    assert 'lyle_ai_http_request_duration_seconds_bucket{method="GET",path="/ai/test/{id}",le="0.25"}' in text
