# Multi-stage build for MCP Research Agent System
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files needed for installation
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install the package with dependencies
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime image
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ ./src/

# Create directories for data and logs (will be mounted as volumes)
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1
ENV DATABASE_PATH=/app/data/research_agent.db
ENV LOG_DIR=/app/logs

# Default entrypoint runs the CLI
ENTRYPOINT ["research-agent"]