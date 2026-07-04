import json
import os
import httpx
from fastapi import APIRouter, UploadFile, File, Request
from app.schemas.ocr import OCRTextLine, OCRResult
from app.utils.response import success_response, error_response
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
        "Tentukan tipe klaim (Makan, Transportasi, Akomodasi, Lain-lain, Office Operational Transport, "
        "Legal & Administration Fee, Office Supplies & Equipment, Software Subscription, Marketing & Promotion, "
        "atau Business Meal & Entertain), deskripsi pengeluaran, tanggal transaksi, dan total jumlah uang yang dikeluarkan, "
        "berdasarkan hasil OCR berikut. Output MUST be in raw JSON format matching this schema: "
        '{"claim_type": "string", "description": "string", "transaction_date": "string", "total_amount": float}'
    )

    ocr_text = "\n".join([line.text for line in text_lines])
    full_prompt = f"{prompt}\n\nOCR Result:\n{ocr_text}"
    print("OCR result:", ocr_text)
    # Call the local Ollama LLM API (POST /api/generate)
    llm_api_url = f"{LLM_URL_API}/generate"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            raw_prompt = (
                "<|im_start|>system\nYou are a helpful assistant that extracts data into JSON. Do not include any explanations or thinking process.<|im_end|>\n"
                f"<|im_start|>user\n{full_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n{\n  \"claim_type\":"
            )
            response = await client.post(
                llm_api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "qwen3.5:0.8b",
                    "prompt": raw_prompt,
                    "raw": True,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 300
                    }
                },
            )
            response.raise_for_status()
            llm_data = response.json()
            llm_text = llm_data.get("response", "")
            
            # Since we pre-filled the assistant response, we need to prepend it back to the output
            if not llm_text.strip().startswith("{"):
                llm_text = "{\n  \"claim_type\":" + llm_text

            # Try to parse a JSON object from the LLM text
            try:
                llm_analysis = json.loads(llm_text)
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
