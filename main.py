import logging

import uvicorn
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from core.exception_handlers import register_exception_handlers
from core.logging_config import configure_logging
from core.rate_limit import limiter
from routers.v1 import classifier as v1_classifier_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    load_dotenv()
    configure_logging()

    logger.info("Starting Bill Document Classification API")

    app = FastAPI(
        title="Bill Document Classification API",
        description="API for classifying various bill documents using Google Gemini.",
        version="1.0.0",
    )

    # Rate limiting middleware and exception handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    register_exception_handlers(app)
    app.include_router(v1_classifier_router.router, prefix="/v1", tags=["v1"])

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(
            "Incoming request | method=%s path=%s client=%s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        response = await call_next(request)
        logger.info(
            "Completed request | method=%s path=%s status=%s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    @app.on_event("startup")
    async def on_startup():
        logger.info("Application startup complete — ready to serve requests")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Application shutting down")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "message": "Welcome to the Bill Document Classification API. Access /docs for API documentation."
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
