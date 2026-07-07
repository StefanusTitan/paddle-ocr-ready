from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import example, ocr
from app.routers.ocr import get_llm_client
from app.services.ocr_service import OCRService
from app.middlewares.log import LogMiddleware
from app.exceptions.log import LogError
from app.utils.log import logger
from starlette.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi import APIRouter

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    ocr_service = OCRService()
    app.state.ocr_service = ocr_service
    logger.info("OCR pipeline loaded")
    yield
    ocr_service.close()
    client = get_llm_client()
    await client.aclose()
    logger.info("Application shutdown complete")

app = FastAPI(lifespan=lifespan)
log = LogError()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
app.add_middleware(LogMiddleware)

# Exception handlers
app.add_exception_handler(RequestValidationError, log.request_validation_exception_handler)
app.add_exception_handler(HTTPException, log.http_exception_handler)
app.add_exception_handler(Exception, log.unhandled_exception_handler)

# API v1 routes
api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(
    ocr.router,
    tags=["OCR"],
    prefix="/ocr",
)

app.include_router(api_v1)

@app.get("/")
async def root():
    return {"message": "Server is UP ✅"}
