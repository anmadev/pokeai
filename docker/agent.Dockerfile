FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./

# Install deps without creating a virtualenv inside the container
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY src/ ./src/

CMD ["python", "-m", "src.agents.random_agent"]
