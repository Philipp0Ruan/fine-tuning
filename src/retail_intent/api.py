from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .config import MAX_BATCH_SIZE, MAX_TEXT_CHARS, Settings
from .predict import IntentClassifier

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    label: Literal["suporte", "venda"]
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    needs_review: bool


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        config = settings or Settings.from_env()
        app.state.classifier = IntentClassifier(
            config.model_dir, config.device, config.confidence_threshold, config.allow_smoke_model,
        )
        yield
        del app.state.classifier

    app = FastAPI(title="Retail Intent", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Não devolve nem registra o conteúdo de mensagens rejeitadas.
        return JSONResponse(status_code=422, content={"error": "Entrada inválida: envie somente texts com 1 a 32 mensagens de até 4000 caracteres."})

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_purpose": request.app.state.classifier.manifest["purpose"]}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(payload: ClassifyRequest, request: Request):
        predictions = request.app.state.classifier.predict(payload.texts)
        return {"predictions": predictions}

    return app


app = create_app()


def main():
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.python_host, port=settings.python_port, access_log=False)


if __name__ == "__main__":
    main()
