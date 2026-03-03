from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class QAQuery(BaseModel):
    """问答查询值对象"""
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {
        "frozen": True
    }
