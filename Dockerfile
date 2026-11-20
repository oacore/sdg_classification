ARG MODEL_IMAGE=corecontainers.azurecr.io/aks-chars-sdg-classification-model:202503311305_88

FROM ${MODEL_IMAGE} AS model

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install -r requirements.txt

RUN chown appuser:appuser /app
USER appuser

RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')"

COPY --from=model --chown=appuser:appuser /202503311305_88 /app/models/202503311305_88

COPY --chown=appuser:appuser . .

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 5007

ENTRYPOINT ["python", "src/client.py"]
CMD ["--dataset", "synthetic"]
