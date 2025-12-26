from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import uuid

from schemas import SourceItem

# Initialize FastAPI app
app = FastAPI(title="ComplaintOps AI Service", version="0.1.0")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("complaintops.ai_service")

ALLOW_RAW_PII_RESPONSE = os.getenv("ALLOW_RAW_PII_RESPONSE", "false").lower() == "true"

def sanitize_input(text: str) -> dict:
    from pii_masker import masker
    result = masker.mask(text)
    return {
        "masked_text": result["masked_text"],
        "masked_entities": result["masked_entities"],
        "original_text": result["original_text"],
    }

def log_sanitized_request(
    endpoint: str,
    masked_text: str,
    masked_entities: List[str],
    request_id: str,
) -> None:
    logger.info(
        "request_received endpoint=%s request_id=%s masked_text_length=%s masked_entity_types=%s",
        endpoint,
        request_id,
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
    category: CategoryLiteral
    category_confidence: float
    urgency: str
    urgency_confidence: float
    needs_human_review: bool
    model_loaded: bool
    review_status: str
    review_id: Optional[str] = None

class RAGRequest(BaseModel):
    text: str
    category: Optional[str] = None

class RAGResponse(BaseModel):
    relevant_sources: List[SourceItem]

class GenerateRequest(BaseModel):
    text: str
    category: CategoryLiteral
    urgency: str
    relevant_sources: List[SourceItem] = Field(default_factory=list)

class GenerateResponse(BaseModel):
    action_plan: List[str]
    customer_reply_draft: str
    risk_flags: List[str]
    sources: List[SourceItem]

class ReviewActionRequest(BaseModel):
    review_id: str
    notes: Optional[str] = None

class ReviewActionResponse(BaseModel):
    review_id: str
    status: str
    notes: Optional[str] = None

# --- Endpoints ---

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/")
def read_root():
    return {"message": "ComplaintOps AI Service is running"}

@app.post("/mask", response_model=MaskingResponse)
def mask_pii(payload: MaskingRequest, request: Request):
    result = sanitize_input(payload.text)
    log_sanitized_request(
        "/mask",
        result["masked_text"],
        result["masked_entities"],
        request.state.request_id,
    )
    response_payload = {
        "masked_text": result["masked_text"],
        "masked_entities": result["masked_entities"],
    }
    if ALLOW_RAW_PII_RESPONSE:
        response_payload["original_text"] = result["original_text"]
    return MaskingResponse(**response_payload)

@app.post("/predict", response_model=TriageResponse)
def predict_triage(payload: TriageRequest, request: Request):
    from triage_model import triage_engine
    from review_store import review_store
    sanitized = sanitize_input(payload.text)
    log_sanitized_request(
        "/predict",
        sanitized["masked_text"],
        sanitized["masked_entities"],
        request.state.request_id,
    )
    result = triage_engine.predict(sanitized["masked_text"])
    needs_human_review = (
        result["category_confidence"] < 0.60
        or result["urgency_confidence"] < 0.60
    )
    review_id = None
    review_status = "AUTO_APPROVED"
    if needs_human_review:
        review_id = str(uuid.uuid4())
        review_store.create_review(
            review_id=review_id,
            masked_text=sanitized["masked_text"],
            category=result["category"],
            category_confidence=result["category_confidence"],
            urgency=result["urgency"],
            urgency_confidence=result["urgency_confidence"],
        )
        review_status = "PENDING_REVIEW"
    return TriageResponse(
        category=result["category"],
        category_confidence=result["category_confidence"],
        urgency=result["urgency"],
        urgency_confidence=result["urgency_confidence"],
        needs_human_review=needs_human_review,
        model_loaded=result["model_loaded"],
        review_status=review_status,
        review_id=review_id,
    )

@app.post("/retrieve", response_model=RAGResponse)
def retrieve_docs(payload: RAGRequest, request: Request):
    from rag_manager import rag_manager
    sanitized = sanitize_input(payload.text)
    log_sanitized_request(
        "/retrieve",
        sanitized["masked_text"],
        sanitized["masked_entities"],
        request.state.request_id,
    )
    sources = rag_manager.retrieve(sanitized["masked_text"], category=payload.category)
    return RAGResponse(relevant_sources=sources)

@app.post("/generate", response_model=GenerateResponse)
def generate_response(payload: GenerateRequest, request: Request):
    from llm_client import llm_client
    from rag_manager import rag_manager
    sanitized = sanitize_input(payload.text)
    log_sanitized_request(
        "/generate",
        sanitized["masked_text"],
        sanitized["masked_entities"],
        request.state.request_id,
    )
    risk_flags = []
    sources = payload.relevant_sources
    if not sources:
        try:
            sources = rag_manager.retrieve(
                sanitized["masked_text"],
                category=payload.category,
            )
            if not sources:
                risk_flags.append("RAG_EMPTY_SOURCES")
            else:
                risk_flags.append("RAG_FALLBACK_USED")
        except Exception:
            risk_flags.append("RAG_UNAVAILABLE")
            sources = []
    result = llm_client.generate_response(
        text=sanitized["masked_text"],
        category=payload.category,
        urgency=payload.urgency,
        snippets=[
            source.model_dump() if isinstance(source, SourceItem) else source
            for source in sources
        ]
    )
    return GenerateResponse(
        action_plan=result["action_plan"],
        customer_reply_draft=result["customer_reply_draft"],
        risk_flags=list(dict.fromkeys(result["risk_flags"] + risk_flags)),
        sources=result["sources"]
    )

@app.post("/review/approve", response_model=ReviewActionResponse)
def approve_review(payload: ReviewActionRequest):
    from review_store import review_store
    record = review_store.update_review(payload.review_id, "APPROVED", payload.notes)
    if not record:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewActionResponse(review_id=record.review_id, status=record.status, notes=record.notes)

@app.post("/review/reject", response_model=ReviewActionResponse)
def reject_review(payload: ReviewActionRequest):
    from review_store import review_store
    record = review_store.update_review(payload.review_id, "REJECTED", payload.notes)
    if not record:
        raise HTTPException(status_code=404, detail="Review not found")
    return ReviewActionResponse(review_id=record.review_id, status=record.status, notes=record.notes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
