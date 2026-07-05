import io
import os
import tempfile
import subprocess
import fitz  # PyMuPDF
import docx
from PIL import Image

def process_gif(image_bytes: bytes) -> bytes:
    """Extracts the first frame of a GIF and returns it as PNG bytes."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Get the first frame
        img.seek(0)
        # Convert to RGB to ensure compatibility (remove alpha/palette)
        rgb_img = img.convert("RGB")
        
        output = io.BytesIO()
        rgb_img.save(output, format="PNG")
        return output.getvalue()

def process_docx(file_bytes: bytes) -> str:
    """Extracts text from a DOCX file stream."""
    document = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    for para in document.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

def process_doc(file_bytes: bytes) -> str:
    """Converts a DOC file to DOCX using headless LibreOffice and extracts text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_path = os.path.join(temp_dir, "temp.doc")
        with open(doc_path, "wb") as f:
            f.write(file_bytes)
            
        # Run headless LibreOffice to convert doc to docx
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "docx", "--outdir", temp_dir, doc_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert .doc to .docx: {e.stderr.decode()}")
        except FileNotFoundError:
             raise RuntimeError("LibreOffice ('soffice') not found. Ensure it is installed.")
            
        docx_path = os.path.join(temp_dir, "temp.docx")
        if not os.path.exists(docx_path):
            raise RuntimeError("LibreOffice conversion completed but .docx file not found.")
            
        with open(docx_path, "rb") as f:
            docx_bytes = f.read()
            
        return process_docx(docx_bytes)

def process_pdf(file_bytes: bytes) -> tuple[bool, list[bytes] | str]:
    """
    Checks if a PDF is scanned or digital.
    Returns:
        (is_scanned, result)
        If is_scanned is False, result is the extracted text (str).
        If is_scanned is True, result is a list of PNG image bytes (list[bytes]).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    total_text = ""
    for page in doc:
        total_text += page.get_text()
        
    # Heuristic: if substantial text across all pages, it's likely digital
    if len(total_text.strip()) > 50:
        return False, total_text.strip()
        
    # Scanned PDF: render pages to images
    image_bytes_list = []
    for page in doc:
        # Render at a good resolution (approx 144 DPI) for OCR
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image_bytes_list.append(pix.tobytes("png"))
        
    return True, image_bytes_list
