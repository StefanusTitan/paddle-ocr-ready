import os
import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

MAX_SIDE_LENGTH = 1920


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
            text_recognition_model_name="PP-OCRv6_tiny_rec",
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
        if isinstance(image, bytes):
            return self._process_bytes(image)
        return self._process_path(image)

    def _process_bytes(self, image_bytes: bytes) -> list[dict]:
        """Handle in-memory image bytes directly, no disk round-trip."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        img = self._preprocess_image(img)
        return self._run_pipeline(img)

    def _process_path(self, image_path: str) -> list[dict]:
        """Load image from a file path, preprocess, and run OCR."""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        img = self._preprocess_image(img)
        return self._run_pipeline(img)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Downscale image if its longest side exceeds MAX_SIDE_LENGTH.

        This caps maximum resolution before OCR to reduce processing time
        on large images (e.g. phone camera photos), while PaddleOCR's own
        ``text_det_limit_type="min"`` setting handles upscaling tiny images.
        """
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > MAX_SIDE_LENGTH:
            scale = MAX_SIDE_LENGTH / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(
                "Downscaling image from %dx%d to %dx%d (longest side capped at %d)",
                w, h, new_w, new_h, MAX_SIDE_LENGTH,
            )
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def _run_pipeline(self, image: np.ndarray) -> list[dict]:
        """Execute the OCR pipeline on a preprocessed image array."""
        results = self.pipeline.predict(image)

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
