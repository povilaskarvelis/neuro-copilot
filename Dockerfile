FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY adk-agent/requirements.txt /app/adk-agent/requirements.txt
RUN pip install --no-cache-dir -r /app/adk-agent/requirements.txt

# Install MCP Node dependencies.
COPY research-mcp/package.json /app/research-mcp/package.json
COPY research-mcp/package-lock.json /app/research-mcp/package-lock.json
RUN cd /app/research-mcp && npm ci --omit=dev

# Copy application code.
COPY adk-agent /app/adk-agent
COPY research-mcp /app/research-mcp

WORKDIR /app/adk-agent
EXPOSE 8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
