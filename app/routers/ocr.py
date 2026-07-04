from fastapi import APIRouter, UploadFile, File, Request
from app.schemas.ocr import OCRTextLine, OCRResult
from app.utils.response import success_response, error_response

router = APIRouter()


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
    result = OCRResult(
        filename=file.filename or "unknown",
        text_lines=text_lines,
        total_lines=len(text_lines),
    )

    return success_response(
        message="OCR completed successfully.",
        result=result.model_dump(),
    )
