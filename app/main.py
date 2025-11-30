import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import training, prediction, auth
from app.database import init_db
from app.config import LOGS_DIR


def setup_logging() -> None:
    """
    Configure application-wide logging.
    Logs are written both to console and to a file (server.log).
    """
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / "server.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def create_app() -> FastAPI:
    setup_logging()
    init_db()

    app = FastAPI(
        title="Private Lessons ML API",
        description="Train and use ML models to predict private lesson prices.",
        version="1.0.0",
    )

    @app.get("/health")
    def health_check():
        """
        Simple health-check endpoint to verify that the API is running.
        Does not require authentication.
        """
        return {"status": "ok"}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(training.router)
    app.include_router(prediction.router)

    logger = logging.getLogger("main")
    logger.info("Application startup complete")

    return app


app = create_app()
