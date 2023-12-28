FROM ollama/ollama

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes curl jq python3 python3-pip python-is-python3
RUN pip install litellm uvicorn fastapi backoff orjson async_generator
HEALTHCHECK --interval=5s --timeout=30s --start-period=5s --retries=10 CMD curl localhost:11434 && curl localhost:8000
ENTRYPOINT ollama serve & litellm --drop_params