from fastapi.testclient import TestClient

import server


def test_query_endpoint_uses_standard_single_query_runner(monkeypatch):
    monkeypatch.setattr(server, "validate_runtime_configuration", lambda: (True, ""))

    async def fake_runner(query: str, *, confirmation_handler, benchmark_mode: bool = False) -> str:
        assert query == "What is CRISPR?"
        assert confirmation_handler is None
        assert benchmark_mode is False
        return "standard-response"

    monkeypatch.setattr(server, "run_single_query_native_with_confirmation_async", fake_runner)

    client = TestClient(server.app)
    response = client.post("/query", json={"query": "What is CRISPR?"})

    assert response.status_code == 200
    assert response.json() == {"response": "standard-response"}


def test_benchmark_query_endpoint_uses_benchmark_runner(monkeypatch):
    monkeypatch.setattr(server, "validate_runtime_configuration", lambda: (True, ""))

    async def fake_runner(query: str) -> str:
        assert query == "What is the GTEx value?"
        return "benchmark-response"

    monkeypatch.setattr(server, "run_single_query_native_benchmark_async", fake_runner)

    client = TestClient(server.app)
    response = client.post("/benchmark_query", json={"query": "What is the GTEx value?"})

    assert response.status_code == 200
    assert response.json() == {"response": "benchmark-response"}
