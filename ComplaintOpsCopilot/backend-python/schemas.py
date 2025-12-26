from typing import Literal

from pydantic import BaseModel


CATEGORY_VALUES = (
    "ACCESS_LOGIN_MOBILE",
    "CAMPAIGN_POINTS_REWARDS",
    "CARD_LIMIT_CREDIT",
    "CHARGEBACK_DISPUTE",
    "FRAUD_UNAUTHORIZED_TX",
    "INFORMATION_REQUEST",
    "TECHNICAL_ISSUE",
    "TRANSFER_DELAY",
    "UNKNOWN",
)

CategoryLiteral = Literal[
    "ACCESS_LOGIN_MOBILE",
    "CAMPAIGN_POINTS_REWARDS",
    "CARD_LIMIT_CREDIT",
    "CHARGEBACK_DISPUTE",
    "FRAUD_UNAUTHORIZED_TX",
    "INFORMATION_REQUEST",
    "TECHNICAL_ISSUE",
    "TRANSFER_DELAY",
    "UNKNOWN",
]


class SourceItem(BaseModel):
    snippet: str
    source: str
    doc_name: str
    chunk_id: str
