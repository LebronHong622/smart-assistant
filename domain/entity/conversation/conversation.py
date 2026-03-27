"""
对话实体
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import uuid

from domain.vo.conversation.message import QueryMessage, ResponseMessage


class Conversation(BaseModel):
    """对话实体"""
    session_id: str
    queries: List[QueryMessage] = Field(default_factory=list)
    responses: List[ResponseMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_query(self, query: QueryMessage):
        self.queries.append(query)
        self.updated_at = datetime.now()

    def add_response(self, response: ResponseMessage):
        self.responses.append(response)
        self.updated_at = datetime.now()

    @classmethod
    def create(cls, session_id: Optional[str] = None) -> "Conversation":
        return cls(session_id=session_id or str(uuid.uuid4()))

    model_config = {
        "arbitrary_types_allowed": True
    }
