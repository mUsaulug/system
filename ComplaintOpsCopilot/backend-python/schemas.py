from pydantic import BaseModel


class SourceItem(BaseModel):
    snippet: str
    source: str
    doc_name: str
    chunk_id: str
