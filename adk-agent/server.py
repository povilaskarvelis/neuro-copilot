"""
FastAPI entrypoint for Cloud Run / HTTP execution.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import (
    run_single_query_native_benchmark_async,
    run_single_query_native_with_confirmation_async,
    validate_runtime_configuration,
)


app = FastAPI(
    title="AI Co-Scientist API",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Biomedical research question.",
        min_length=3,
    )


class QueryResponse(BaseModel):
    response: str


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    is_valid, error_message = validate_runtime_configuration()
    return {
        "ok": is_valid,
        "error": error_message or None,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    prompt = request.query.strip()
    if len(prompt) < 3:
        raise HTTPException(status_code=400, detail="query must be at least 3 characters.")

    is_valid, error_message = validate_runtime_configuration()
    if not is_valid:
        raise HTTPException(status_code=500, detail=error_message)

    try:
        response = await run_single_query_native_with_confirmation_async(
            prompt,
            confirmation_handler=None,
        )
    except Exception as exc:  # pragma: no cover - passthrough for runtime diagnostics
        raise HTTPException(status_code=500, detail=f"query execution failed: {exc}") from exc

    return QueryResponse(response=response)


@app.post("/benchmark_query", response_model=QueryResponse)
async def benchmark_query(request: QueryRequest) -> QueryResponse:
    prompt = request.query.strip()
    if len(prompt) < 3:
        raise HTTPException(status_code=400, detail="query must be at least 3 characters.")

    is_valid, error_message = validate_runtime_configuration()
    if not is_valid:
        raise HTTPException(status_code=500, detail=error_message)

    try:
        response = await run_single_query_native_benchmark_async(prompt)
    except Exception as exc:  # pragma: no cover - passthrough for runtime diagnostics
        raise HTTPException(
            status_code=500,
            detail=f"benchmark query execution failed: {exc}",
        ) from exc

    return QueryResponse(response=response)
