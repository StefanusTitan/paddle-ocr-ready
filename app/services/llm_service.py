import json
import os
import httpx
from app.utils.logger import logger
from app.utils.date import normalize_date
from app.utils.amount import normalize_amount

LLM_URL_API = os.getenv("LLM_URL_API")
MODEL = os.getenv("MODEL", "qwen2.5:1.5b")

class LLMService:
    @staticmethod
    async def analyze_receipt(
        client: httpx.AsyncClient, 
        main_claim_type: str, 
        text_lines: list
    ) -> dict:
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

        llm_api_url = f"{LLM_URL_API}/chat"
        
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

        try:
            llm_analysis = json.loads(llm_text)
            if "expense_date" in llm_analysis:
                llm_analysis["expense_date"] = normalize_date(llm_analysis["expense_date"])
            for amount_key in ("amount", "budget_amount"):
                if amount_key in llm_analysis:
                    llm_analysis[amount_key] = normalize_amount(llm_analysis[amount_key])
        except json.JSONDecodeError:
            llm_analysis = {"raw_response": llm_text, "full_data": llm_data}
            
        return llm_analysis
