"""FastAPI server exposing CRAFT retrieval over a preprocessed corpus.

Endpoints:

    GET  /                 the single-page demo UI
    GET  /health           liveness check
    POST /retrieve         {"question": "...", "top_k": 10} -> ranked tables

The engine (index + models) loads once at startup and is shared across requests.
"""

from pathlib import Path

from craft_tabqa.config import CraftConfig
from craft_tabqa.logging_setup import setup_logger

_UI_FILE = Path(__file__).resolve().parent / "ui.html"


def create_app(cfg: CraftConfig, cache_dir: str):
    """Build the FastAPI app with a ready :class:`RetrievalEngine`."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    from craft_tabqa.serve.engine import RetrievalEngine

    logger = setup_logger(name="craft.serve")
    engine = RetrievalEngine(cfg, cache_dir=cache_dir, logger=logger)

    app = FastAPI(title="CRAFT Retrieval", version="1.0.0")

    class SearchRequest(BaseModel):
        question: str
        top_k: int = 10

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _UI_FILE.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "dataset": cfg.data.dataset}

    @app.post("/retrieve")
    def retrieve(request: SearchRequest) -> dict:
        results = engine.search(request.question, top_k=request.top_k)
        return {"question": request.question, "results": results}

    return app


def run_server(cfg: CraftConfig, cache_dir: str, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Create the app and run it with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(cfg, cache_dir), host=host, port=port)
