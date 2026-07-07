import time
import httpx
from fastapi import APIRouter, UploadFile, File, Request, Form
from app.schemas.ocr import OCRTextLine, OCRResult
from app.utils.response import success_response, error_response
from app.utils.logger import logger
from app.services.llm_service import LLMService
from app.services.document_processing_service import DocumentProcessingService

router = APIRouter()

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), main_claim_type: str = Form(None)):
    """Run OCR on an uploaded image and return detected text lines."""
    req_start_time = time.time()
    ocr_service = request.app.state.ocr_service
    http_client = request.app.state.http_client

    supported_types = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    if file.content_type and not (file.content_type.startswith("image/") or file.content_type in supported_types):
        return error_response(
            message="Uploaded file is not a supported format.",
            errors={"content_type": file.content_type},
            status_code=400,
        )

    file_bytes = await file.read()
    file_read_time = time.time()
    logger.debug("File read: {:.4f}s", file_read_time - req_start_time)

    try:
        content_type = file.content_type or ""
        raw_lines = await DocumentProcessingService.process_document(
            file_bytes=file_bytes, 
            content_type=content_type, 
            ocr_service=ocr_service
        )
    except Exception as e:
        return error_response(
            message="Document processing failed.",
            errors={"detail": str(e)},
            status_code=500,
        )

    ocr_done_time = time.time()
    logger.debug("OCR/doc processing: {:.4f}s", ocr_done_time - file_read_time)

    text_lines = [OCRTextLine(**line) for line in raw_lines]
    ocr_result = OCRResult(
        filename=file.filename or "unknown",
        text_lines=text_lines,
        total_lines=len(text_lines),
    )
    
    if not main_claim_type:
        return success_response(
            message="OCR completed successfully.",
            result=ocr_result.model_dump(),
        )

    llm_start_time = time.time()
    try:
        llm_analysis = await LLMService.analyze_receipt(
            client=http_client,
            main_claim_type=main_claim_type,
            text_lines=text_lines
        )
    except httpx.HTTPStatusError as e:
        return error_response(
            message="LLM analysis failed.",
            errors={"detail": str(e), "status_code": e.response.status_code},
            status_code=502,
        )
    except Exception as e:
        return error_response(
            message="LLM analysis failed.",
            errors={"detail": str(e)},
            status_code=500,
        )

    llm_end_time = time.time()
    logger.debug("LLM API call: {:.4f}s", llm_end_time - llm_start_time)
    logger.debug("Total request: {:.4f}s", llm_end_time - req_start_time)

    return success_response(
        message="OCR and LLM analysis completed successfully.",
        result={
            "ocr": ocr_result.model_dump(),
            "analysis": llm_analysis
        },
    )
