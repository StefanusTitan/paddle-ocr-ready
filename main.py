from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
from app.routers import example, ocr
from app.services.ocr_service import OCRService
from app.middlewares.logging import LogMiddleware
from app.exceptions.handlers import GlobalExceptionHandler
from app.utils.logger import logger
from starlette.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi import APIRouter

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    
    # Initialize OCR Service
    ocr_service = OCRService()
    app.state.ocr_service = ocr_service
    logger.info("OCR pipeline loaded")
    
    # Initialize HTTP Client for LLM / external services
    http_client = httpx.AsyncClient(timeout=60.0)
    app.state.http_client = http_client
    logger.info("HTTP client loaded")
    
    yield
    
    ocr_service.close()
    await http_client.aclose()
    logger.info("Application shutdown complete")

app = FastAPI(lifespan=lifespan)
exception_handlers = GlobalExceptionHandler()

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
app.add_exception_handler(RequestValidationError, exception_handlers.request_validation_exception_handler)
app.add_exception_handler(HTTPException, exception_handlers.http_exception_handler)
app.add_exception_handler(Exception, exception_handlers.unhandled_exception_handler)

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
