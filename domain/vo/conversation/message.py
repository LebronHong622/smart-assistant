"""
消息值对象
定义不可变的查询消息和响应消息
"""

from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class QueryMessage(BaseModel):
    """查询消息值对象"""
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {
        "frozen": True
    }


class ResponseMessage(BaseModel):
    """响应消息值对象"""
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = {
        "frozen": True
    }
