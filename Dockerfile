FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY agentmem/ ./agentmem/

RUN pip install --no-cache-dir ".[all]"

ENV AGENTMEM_DB=/data/memory.db

VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "agentmem"]
