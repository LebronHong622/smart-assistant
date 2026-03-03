from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import uuid

from domain.qa.value_object.qa_query import QAQuery
from domain.qa.value_object.qa_response import QAResponse

class QAConversation(BaseModel):
    """问答对话实体"""
    session_id: str
    queries: List[QAQuery] = Field(default_factory=list)
    responses: List[QAResponse] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_query(self, query: QAQuery):
        self.queries.append(query)
        self.updated_at = datetime.now()

    def add_response(self, response: QAResponse):
        self.responses.append(response)
        self.updated_at = datetime.now()

    @classmethod
    def create(cls, session_id: Optional[str] = None) -> "QAConversation":
        return cls(session_id=session_id or str(uuid.uuid4()))

    model_config = {
        "arbitrary_types_allowed": True
    }
