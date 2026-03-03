from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class QAResponse(BaseModel):
    """问答响应值对象"""
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {
        "frozen": True
    }
