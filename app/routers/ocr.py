import json
import os
import time
import httpx
from fastapi import APIRouter, UploadFile, File, Request, Form
from app.schemas.ocr import OCRTextLine, OCRResult
from app.utils.response import success_response, error_response
from app.utils.date import normalize_date
from app.utils import document_parser
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
LLM_URL_API = os.getenv("LLM_URL_API")


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), main_claim_type: str = Form("expense")):
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
    print(f"[Latency] File read: {file_read_time - req_start_time:.4f}s")
    raw_lines = []

    try:
        content_type = file.content_type or ""
        
        if content_type == "image/gif":
            processed_bytes = document_parser.process_gif(file_bytes)
            raw_lines = ocr_service.process_image(processed_bytes)
            
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = document_parser.process_docx(file_bytes)
            raw_lines = [{"text": line, "confidence": 1.0, "bbox": []} for line in extracted_text.split("\n") if line.strip()]
            
        elif content_type == "application/msword":
            extracted_text = document_parser.process_doc(file_bytes)
            raw_lines = [{"text": line, "confidence": 1.0, "bbox": []} for line in extracted_text.split("\n") if line.strip()]
            
        elif content_type == "application/pdf":
            is_scanned, result = document_parser.process_pdf(file_bytes)
            if not is_scanned:
                # result is a string of extracted text
                extracted_text = result
                raw_lines = [{"text": line, "confidence": 1.0, "bbox": []} for line in extracted_text.split("\n") if line.strip()]
            else:
                # result is a list of image bytes
                for img_bytes in result:
                    page_lines = ocr_service.process_image(img_bytes)
                    raw_lines.extend(page_lines)
                    
        else:
            # Standard image (jpg, png, webp, etc.)
            raw_lines = ocr_service.process_image(file_bytes)
            
    except Exception as e:
        return error_response(
            message="Document processing failed.",
            errors={"detail": str(e)},
            status_code=500,
        )

    ocr_done_time = time.time()
    print(f"[Latency] OCR/Doc processing: {ocr_done_time - file_read_time:.4f}s")

    text_lines = [OCRTextLine(**line) for line in raw_lines]
    ocr_result = OCRResult(
        filename=file.filename or "unknown",
        text_lines=text_lines,
        total_lines=len(text_lines),
    )
    
    # return success_response(
    #     message="OCR completed successfully.",
    #     result=ocr_result.model_dump(),
    # )

    if main_claim_type == "advance":
        prompt = (
            "Extract purpose of advance, total amount, and payment method from this text. "
            "Payment method must be one of: Bank Transfer, Cash, Virtual Account."
        )
        # GBNF grammar for advance claim
        json_grammar = r'''
            root ::= "{" ws "\"purpose\"" ws ":" ws string "," ws "\"total_amount\"" ws ":" ws number "," ws "\"payment_method\"" ws ":" ws string ws "}"
            string ::= "\"" ([^"\\] | "\\" .)* "\""
            number ::= "-"? [0-9]+ ("." [0-9]+)?
            ws ::= [ \t\n]*
        '''.strip()
    elif main_claim_type == "travel":
        prompt = (
            "Extract destination, description, and ID type from this text. "
            "ID type must be one of: KTP, SIM, or Passport."
        )
        # GBNF grammar for travel claim
        json_grammar = r'''
            root ::= "{" ws "\"destination\"" ws ":" ws string "," ws "\"description\"" ws ":" ws string "," ws "\"id_type\"" ws ":" ws string ws "}"
            string ::= "\"" ([^"\\] | "\\" .)* "\""
            number ::= "-"? [0-9]+ ("." [0-9]+)?
            ws ::= [ \t\n]*
        '''.strip()
    else:
        prompt = (
            "Extract claim_type, description, transaction_date, and total_amount from this receipt OCR text. "
            "Return transaction_date exactly as it appears on the receipt. "
            "claim_type must be one of: Makan, Transportasi, Akomodasi, Lain-lain, Office Operational Transport, "
            "Legal & Administration Fee, Office Supplies & Equipment, Software Subscription, Marketing & Promotion, "
            "Business Meal & Entertain."
        )
        # GBNF grammar to constrain output to valid JSON matching our schema
        json_grammar = r'''
            root ::= "{" ws "\"claim_type\"" ws ":" ws string "," ws "\"description\"" ws ":" ws string "," ws "\"transaction_date\"" ws ":" ws string "," ws "\"total_amount\"" ws ":" ws number ws "}"
            string ::= "\"" ([^"\\] | "\\" .)* "\""
            number ::= "-"? [0-9]+ ("." [0-9]+)?
            ws ::= [ \t\n]*
        '''.strip()

    ocr_text = "\n".join([line.text for line in text_lines])
    full_prompt = f"{prompt}\n\n{ocr_text}"
    print("OCR result:", ocr_text)

    # Call the local llama.cpp server API (POST /completion)
    llm_api_url = f"{LLM_URL_API}/completion"
    llm_start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            raw_prompt = (
                "<|im_start|>system\nExtract receipt data into JSON. No explanations.<|im_end|>\n"
                f"<|im_start|>user\n{full_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            response = await client.post(
                llm_api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "prompt": raw_prompt,
                    "stream": False,
                    "cache_prompt": True,
                    "temperature": 0.0,
                    "n_predict": 256,
                    "grammar": json_grammar,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                },
            )
            response.raise_for_status()
            llm_data = response.json()
            llm_text = llm_data.get("content", "")

            # Grammar guarantees valid JSON, but parse defensively
            try:
                llm_analysis = json.loads(llm_text)
                # Normalize date in Python (faster than asking the LLM to format)
                if "transaction_date" in llm_analysis:
                    llm_analysis["transaction_date"] = normalize_date(
                        llm_analysis["transaction_date"]
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
    print(f"[Latency] LLM API Call: {llm_end_time - llm_start_time:.4f}s")
    print(f"[Latency] Total Request: {llm_end_time - req_start_time:.4f}s")

    return success_response(
        message="OCR and LLM analysis completed successfully.",
        result={
            "ocr": ocr_result.model_dump(),
            "analysis": llm_analysis
        },
    )
