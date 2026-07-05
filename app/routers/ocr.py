import json
import os
import httpx
from fastapi import APIRouter, UploadFile, File, Request
from app.schemas.ocr import OCRTextLine, OCRResult
from app.utils.response import success_response, error_response
from app.utils.date import normalize_date
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
LLM_URL_API = os.getenv("LLM_URL_API")


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Run OCR on an uploaded image and return detected text lines."""
    ocr_service = request.app.state.ocr_service

    if file.content_type and not file.content_type.startswith("image/"):
        return error_response(
            message="Uploaded file is not an image.",
            errors={"content_type": file.content_type},
            status_code=400,
        )

    image_bytes = await file.read()

    try:
        raw_lines = ocr_service.process_image(image_bytes)
    except Exception as e:
        return error_response(
            message="OCR processing failed.",
            errors={"detail": str(e)},
            status_code=500,
        )

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

    prompt = (
        "Extract claim_type, description, transaction_date, and total_amount from this receipt OCR text. "
        "Return transaction_date exactly as it appears on the receipt. "
        "claim_type must be one of: Makan, Transportasi, Akomodasi, Lain-lain, Office Operational Transport, "
        "Legal & Administration Fee, Office Supplies & Equipment, Software Subscription, Marketing & Promotion, "
        "Business Meal & Entertain."
    )

    ocr_text = "\n".join([line.text for line in text_lines])
    full_prompt = f"{prompt}\n\n{ocr_text}"
    print("OCR result:", ocr_text)
    # GBNF grammar to constrain output to valid JSON matching our schema
    json_grammar = r'''
        root ::= "{" ws "\"claim_type\"" ws ":" ws string "," ws "\"description\"" ws ":" ws string "," ws "\"transaction_date\"" ws ":" ws string "," ws "\"total_amount\"" ws ":" ws number ws "}"
        string ::= "\"" ([^"\\] | "\\" .)* "\""
        number ::= "-"? [0-9]+ ("." [0-9]+)?
        ws ::= [ \t\n]*
    '''.strip()

    # Call the local llama.cpp server API (POST /completion)
    llm_api_url = f"{LLM_URL_API}/completion"
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

    return success_response(
        message="OCR and LLM analysis completed successfully.",
        result={
            "ocr": ocr_result.model_dump(),
            "analysis": llm_analysis
        },
    )
