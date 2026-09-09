FROM python:3.12-slim

# Emit output immediately and never write bytecode caches: logs stay orderly
# inside docker logs and the image stays free of root-owned __pycache__ noise.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# curl serves the compose healthcheck (same tool as the comfy service check).
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /application

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY pyproject.toml ./
COPY zoomy ./zoomy
COPY tests ./tests

EXPOSE 7861

CMD ["python", "-m", "zoomy"]
