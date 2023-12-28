FROM ollama/ollama

RUN apt update
RUN apt upgrade --yes
RUN apt install --yes curl jq python3 python3-pip python-is-python3
RUN pip install litellm uvicorn fastapi backoff orjson
HEALTHCHECK --interval=1s --timeout=30s --start-period=1s --retries=10 CMD curl localhost:8000 && curl localhost:11434
ENTRYPOINT ollama serve & litellm --drop_params