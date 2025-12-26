from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional


@dataclass
class ReviewRecord:
    review_id: str
    status: str
    created_at: str
    updated_at: str
    masked_text: str
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    notes: Optional[str] = None


class ReviewStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._reviews: Dict[str, ReviewRecord] = {}

    def create_review(
        self,
        review_id: str,
        masked_text: str,
        category: str,
        category_confidence: float,
        urgency: str,
        urgency_confidence: float,
    ) -> ReviewRecord:
        now = datetime.now(timezone.utc).isoformat()
        record = ReviewRecord(
            review_id=review_id,
            status="PENDING_REVIEW",
            created_at=now,
            updated_at=now,
            masked_text=masked_text,
            category=category,
            category_confidence=category_confidence,
            urgency=urgency,
            urgency_confidence=urgency_confidence,
        )
        with self._lock:
            self._reviews[review_id] = record
        return record

    def update_review(self, review_id: str, status: str, notes: Optional[str] = None) -> Optional[ReviewRecord]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            record = self._reviews.get(review_id)
            if not record:
                return None
            record.status = status
            record.updated_at = now
            record.notes = notes
            return record


review_store = ReviewStore()
