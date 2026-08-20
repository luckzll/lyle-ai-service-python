FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY deploy/docker-entrypoint.sh ./docker-entrypoint.sh
ARG EMBEDDING_PROVIDER=fastembed
ARG EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RUN EMBEDDING_PROVIDER="$EMBEDDING_PROVIDER" EMBEDDING_MODEL="$EMBEDDING_MODEL" python -m app.embedding.preload
RUN chmod +x ./docker-entrypoint.sh
EXPOSE 8000
CMD ["./docker-entrypoint.sh"]
