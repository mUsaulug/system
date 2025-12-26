from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os

# Initialize FastAPI app
app = FastAPI(title="ComplaintOps AI Service", version="0.1.0")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("complaintops.ai_service")

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

def sanitize_input(text: str) -> dict:
    from pii_masker import masker
    result = masker.mask(text)
    return {
        "masked_text": result["masked_text"],
        "masked_entities": result["masked_entities"],
        "original_text": result["original_text"],
    }

def log_sanitized_request(endpoint: str, masked_text: str, masked_entities: List[str]) -> None:
    logger.info(
        "request_received endpoint=%s masked_text_length=%s masked_entity_types=%s",
        endpoint,
        len(masked_text),
        ",".join(masked_entities),
    )

def store_raw_text_if_needed(raw_text: str) -> None:
    """Store raw text only when required, using encrypted storage with RBAC."""
    if raw_text:
        logger.debug("Raw text storage not configured; skipping secure storage.")

def store_pii_mask_map(mask_map: dict) -> None:
    """Persist PII mask map only with encrypted storage + role-based access."""
    if mask_map:
        logger.debug("PII mask map storage not configured; skipping secure storage.")

# --- Pydantic Models for API Contract ---

class MaskingRequest(BaseModel):
    text: str

class MaskingResponse(BaseModel):
    original_text: Optional[str] = None
    masked_text: str
    masked_entities: List[str]

class TriageRequest(BaseModel):
    text: str

class TriageResponse(BaseModel):
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float

class RAGRequest(BaseModel):
    text: str
    category: Optional[str] = None

class RAGResponse(BaseModel):
    relevant_snippets: List[str]

class GenerateRequest(BaseModel):
    text: str
    category: str
    urgency: str
    relevant_snippets: List[str]

class GenerateResponse(BaseModel):
    action_plan: List[str]
    customer_reply_draft: str
    risk_flags: List[str]

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "ComplaintOps AI Service is running"}

@app.post("/mask", response_model=MaskingResponse)
def mask_pii(request: MaskingRequest):
    result = sanitize_input(request.text)
    log_sanitized_request("/mask", result["masked_text"], result["masked_entities"])
    response_payload = {
        "masked_text": result["masked_text"],
        "masked_entities": result["masked_entities"],
    }
    if DEBUG_MODE:
        response_payload["original_text"] = result["original_text"]
    return MaskingResponse(**response_payload)

@app.post("/predict", response_model=TriageResponse)
def predict_triage(request: TriageRequest):
    from triage_model import triage_engine
    sanitized = sanitize_input(request.text)
    log_sanitized_request("/predict", sanitized["masked_text"], sanitized["masked_entities"])
    result = triage_engine.predict(sanitized["masked_text"])
    return TriageResponse(
        category=result["category"],
        category_confidence=result["category_confidence"],
        urgency=result["urgency"],
        urgency_confidence=result["urgency_confidence"]
    )

@app.post("/retrieve", response_model=RAGResponse)
def retrieve_docs(request: RAGRequest):
    from rag_manager import rag_manager
    sanitized = sanitize_input(request.text)
    log_sanitized_request("/retrieve", sanitized["masked_text"], sanitized["masked_entities"])
    snippets = rag_manager.retrieve(sanitized["masked_text"])
    return RAGResponse(relevant_snippets=snippets)

@app.post("/generate", response_model=GenerateResponse)
def generate_response(request: GenerateRequest):
    from llm_client import llm_client
    sanitized = sanitize_input(request.text)
    log_sanitized_request("/generate", sanitized["masked_text"], sanitized["masked_entities"])
    result = llm_client.generate_response(
        text=sanitized["masked_text"],
        category=request.category,
        urgency=request.urgency,
        snippets=request.relevant_snippets
    )
    return GenerateResponse(
        action_plan=result["action_plan"],
        customer_reply_draft=result["customer_reply_draft"],
        risk_flags=result["risk_flags"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
