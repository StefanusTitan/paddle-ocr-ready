import json
import os
import time
import httpx
from fastapi import APIRouter, UploadFile, File, Request, Form
from app.schemas.ocr import OCRTextLine, OCRResult
from app.utils.response import success_response, error_response
from app.utils.date import normalize_date
from app.utils.amount import normalize_amount
from app.utils import document_parser
from app.utils.thread import run_in_thread
from app.utils.log import logger
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
LLM_URL_API = os.getenv("LLM_URL_API")
MODEL = os.getenv("MODEL", "qwen2.5:1.5b")

_client: httpx.AsyncClient | None = None


def get_llm_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), main_claim_type: str = Form(None)):
    """Run OCR on an uploaded image and return detected text lines."""
    req_start_time = time.time()
    ocr_service = request.app.state.ocr_service

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
    raw_lines = []

    try:
        content_type = file.content_type or ""
        
        if content_type == "image/gif":
            processed_bytes = await run_in_thread(document_parser.process_gif, file_bytes)
            raw_lines = await run_in_thread(ocr_service.process_image, processed_bytes)
            
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = await run_in_thread(document_parser.process_docx, file_bytes)
            raw_lines = [{"text": line, "confidence": 1.0, "bbox": []} for line in extracted_text.split("\n") if line.strip()]
            
        elif content_type == "application/msword":
            is_scanned, result = await run_in_thread(document_parser.process_doc, file_bytes)
            if not is_scanned:
                extracted_text = result
                raw_lines = [{"text": line, "confidence": 1.0, "bbox": []} for line in extracted_text.split("\n") if line.strip()]
            else:
                for img_bytes in result:
                    page_lines = await run_in_thread(ocr_service.process_image, img_bytes)
                    raw_lines.extend(page_lines)
            
        elif content_type == "application/pdf":
            is_scanned, result = await run_in_thread(document_parser.process_pdf, file_bytes)
            if not is_scanned:
                extracted_text = result
                raw_lines = [{"text": line, "confidence": 1.0, "bbox": []} for line in extracted_text.split("\n") if line.strip()]
            else:
                for img_bytes in result:
                    page_lines = await run_in_thread(ocr_service.process_image, img_bytes)
                    raw_lines.extend(page_lines)
                    
        else:
            raw_lines = await run_in_thread(ocr_service.process_image, file_bytes)
            
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

    if main_claim_type == "advance":
        prompt = (
            "Extract purposes, amount, payment_method, and confidence. "
            "amount: include currency if present. "
            "payment_method: Bank Transfer|Cash|Virtual Account."
            "confidence: 0.0-1.0 based on estimated accuracy."
        )
    elif main_claim_type == "travel":
        prompt = (
            "Extract purpose, description, budget_amount, mode_of_travel, is_roundtrip and confidence. "
            "budget_amount: include currency if present. "
            "mode_of_travel: Plane|Train|Taxi|Bus|Car|Motorcycle|Other."
            "confidence: 0.0-1.0 based on estimated accuracy."
        )
    else:
        prompt = (
            "Extract invoice description, expense_date, amount, and confidence from this receipt. "
            "amount: include currency if present. "
            "confidence: 0.0-1.0 based on estimated accuracy."
        )

    ocr_text = "\n".join(line.text for line in text_lines).lower()
    full_prompt = f"{prompt}\n\n{ocr_text}"
    logger.debug("OCR result:\n{}", ocr_text)

    # Call the Ollama API (POST /api/chat)
    llm_api_url = f"{LLM_URL_API}/chat"
    llm_start_time = time.time()
    try:
        client = get_llm_client()
        response = await client.post(
            llm_api_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Extract receipt data into JSON. No explanations."},
                    {"role": "user", "content": full_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 254,
                },
            },
        )
        response.raise_for_status()
        llm_data = response.json()
        llm_text = llm_data.get("message", {}).get("content", "")

        # Ollama's format: json ensures valid JSON, but parse defensively
        try:
            llm_analysis = json.loads(llm_text)
            # Normalize date in Python (faster than asking the LLM to format)
            if "expense_date" in llm_analysis:
                llm_analysis["expense_date"] = normalize_date(
                    llm_analysis["expense_date"]
                )
            # Normalize amount fields to clean decimal strings
            for amount_key in ("amount", "budget_amount"):
                if amount_key in llm_analysis:
                    llm_analysis[amount_key] = normalize_amount(
                        llm_analysis[amount_key]
                    )
        except json.JSONDecodeError:
            llm_analysis = {"raw_response": llm_text, "full_data": llm_data}

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
