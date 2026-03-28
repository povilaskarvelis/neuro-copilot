"""
LABBench2 external runner for the benchmark-optimized AI Co-Scientist endpoint.

Usage from a LABBench2 checkout:

    export AI_CO_SCIENTIST_BENCHMARK_URL=http://127.0.0.1:8000/benchmark_query
    uv run python -m evals.run_evals \
      --agent external:/absolute/path/to/ai-co-scientist/adk-agent/labbench2_runner.py:LabBench2BenchmarkRunner \
      --tag dbqa2 \
      --mode inject
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from evals.runners import AgentResponse


DEFAULT_BENCHMARK_URL = "http://127.0.0.1:8000/benchmark_query"
DEFAULT_TIMEOUT_SECONDS = 600.0


class LabBench2BenchmarkRunner:
    """HTTP runner for text-only LABBench2 tasks against `/benchmark_query`."""

    def __init__(self) -> None:
        self.endpoint_url = str(
            os.environ.get("AI_CO_SCIENTIST_BENCHMARK_URL", DEFAULT_BENCHMARK_URL)
        ).strip()
        timeout_raw = str(
            os.environ.get("AI_CO_SCIENTIST_BENCHMARK_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        ).strip()
        self.timeout = float(timeout_raw or DEFAULT_TIMEOUT_SECONDS)

    async def upload_files(
        self,
        files: list[Path],
        gcs_prefix: str | None = None,
    ) -> dict[str, str]:
        if files:
            raise RuntimeError(
                "LabBench2BenchmarkRunner does not support file uploads. "
                "Use text-only tasks such as dbqa2 in inject mode."
            )
        return {}

    def _post_query(self, question: str) -> dict:
        payload = json.dumps({"query": question}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Benchmark endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach benchmark endpoint {self.endpoint_url}: {exc}"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Benchmark endpoint returned invalid JSON: {body[:500]}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Benchmark endpoint returned a non-object JSON payload.")
        return parsed

    async def execute(
        self,
        question: str,
        file_refs: dict[str, str] | None = None,
    ) -> AgentResponse:
        if file_refs:
            raise RuntimeError(
                "LabBench2BenchmarkRunner received file references, but the benchmark "
                "endpoint only supports text-only tasks."
            )

        payload = await asyncio.to_thread(self._post_query, question)
        response_text = str(payload.get("response", "") or "").strip()
        if not response_text:
            raise RuntimeError("Benchmark endpoint returned an empty response.")

        return AgentResponse(
            text=response_text,
            raw_output=payload,
            metadata={"endpoint_url": self.endpoint_url},
        )

    def extract_answer(self, response: AgentResponse) -> str:
        return response.text

    async def download_outputs(self, dest_dir: Path) -> Path | None:
        return None

    async def cleanup(self) -> None:
        return None
