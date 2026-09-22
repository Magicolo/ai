FROM python:3.12-slim

# Self-contained: no host dependencies. ffmpeg for media, pip for install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY voyage/ ./voyage/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e ".[dev]" \
    && voyage doctor

# Tests/gates run via scripts/gates.sh (not at build, to keep iteration fast).
CMD ["voyage", "doctor"]
