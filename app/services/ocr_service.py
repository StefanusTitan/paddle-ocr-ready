import os
import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class OCRService:
    """OCR service using PaddleOCR pipeline.

    Chains document orientation classification, image unwarping,
    text line orientation classification, text detection, and text
    recognition into a single pipeline call.
    """

    def __init__(self) -> None:
        logger.info("Initializing PaddleOCR pipeline (PP-OCRv6 small)...")
        self.pipeline = PaddleOCR(
            device="cpu",
            engine="paddle_static",
            text_detection_model_name="PP-OCRv6_tiny_det",
            text_recognition_model_name="PP-OCRv6_medium_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False, # Invoice usually only horizontal texts (at least important ones we need to grab)
            text_det_limit_side_len=960,
            text_det_limit_type="min",
            text_rec_score_thresh=0.5,
            text_recognition_batch_size=8,
            # textline_orientation_batch_size=8,
            enable_hpi=False, # Enable when in Linux and has GPU for performance improvement
            enable_mkldnn=True,
        )
        logger.info("PaddleOCR pipeline initialized successfully.")

    def process_image(self, image: str | bytes) -> list[dict]:
        """Run OCR on an image.

        Args:
            image: Either a file path (str) or raw image bytes.

        Returns:
            A list of dicts, each containing:
              - text (str): The recognised text.
              - confidence (float): Recognition confidence score.
              - bbox (list[list[float]]): Four-point polygon bounding box.
        """
        # If bytes are provided, write to a temp file since PaddleOCR
        # expects a file path.
        if isinstance(image, bytes):
            return self._process_bytes(image)
        return self._run_pipeline(image)

    def _process_bytes(self, image_bytes: bytes) -> list[dict]:
        """Handle in-memory image bytes directly, no disk round-trip."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return self._run_pipeline(img)

    def _run_pipeline(self, image_path: str) -> list[dict]:
        """Execute the OCR pipeline on a file path."""
        results = self.pipeline.predict(image_path)

        text_lines: list[dict] = []
        for result in results:
            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])
            dt_polys = result.get("dt_polys", [])

            for i, text in enumerate(rec_texts):
                confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                bbox = (
                    dt_polys[i].tolist()
                    if i < len(dt_polys)
                    else []
                )
                text_lines.append(
                    {
                        "text": str(text),
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    }
                )

        return text_lines

    def close(self) -> None:
        """Release pipeline resources."""
        if hasattr(self, "pipeline"):
            self.pipeline.close()
