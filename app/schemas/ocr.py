from pydantic import BaseModel, Field


class OCRTextLine(BaseModel):
    """A single detected and recognised text line."""

    text: str = Field(..., description="The recognised text content.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Recognition confidence score."
    )
    bbox: list[list[float]] = Field(
        default_factory=list,
        description="Four-point polygon bounding box as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].",
    )


class OCRResult(BaseModel):
    """OCR result for a single image."""

    filename: str = Field(..., description="Name of the input file.")
    text_lines: list[OCRTextLine] = Field(
        default_factory=list, description="Detected text lines with positions."
    )
    total_lines: int = Field(
        ..., description="Total number of text lines detected."
    )
