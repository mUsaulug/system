from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

# Initialize FastAPI app
app = FastAPI(title="ComplaintOps AI Service", version="0.1.0")

# --- Pydantic Models for API Contract ---

class MaskingRequest(BaseModel):
    text: str

class MaskingResponse(BaseModel):
    original_text: str
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
    from pii_masker import masker
    result = masker.mask(request.text)
    return MaskingResponse(
        original_text=result["original_text"],
        masked_text=result["masked_text"],
        masked_entities=result["masked_entities"]
    )

@app.post("/predict", response_model=TriageResponse)
def predict_triage(request: TriageRequest):
    from triage_model import triage_engine
    result = triage_engine.predict(request.text)
    return TriageResponse(
        category=result["category"],
        category_confidence=result["category_confidence"],
        urgency=result["urgency"],
        urgency_confidence=result["urgency_confidence"]
    )

@app.post("/retrieve", response_model=RAGResponse)
def retrieve_docs(request: RAGRequest):
    from rag_manager import rag_manager
    snippets = rag_manager.retrieve(request.text)
    return RAGResponse(relevant_snippets=snippets)

@app.post("/generate", response_model=GenerateResponse)
def generate_response(request: GenerateRequest):
    from llm_client import llm_client
    result = llm_client.generate_response(
        text=request.text,
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
