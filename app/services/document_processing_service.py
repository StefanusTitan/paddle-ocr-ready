from app.utils import document_parser
from app.utils.thread import run_in_thread
from app.services.ocr_service import OCRService

class DocumentProcessingService:
    @staticmethod
    async def process_document(file_bytes: bytes, content_type: str, ocr_service: OCRService) -> list:
        raw_lines = []
        
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
            
        return raw_lines
