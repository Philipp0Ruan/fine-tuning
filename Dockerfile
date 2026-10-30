FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .
USER app
EXPOSE 8000
CMD ["python", "-m", "retail_intent.api"]
