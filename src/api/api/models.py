from pydantic import BaseModel, Field
from typing import List, Any, Optional


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: Optional[str] = Field(None, description="The thread ID")

class RAGUSEDIMAGE(BaseModel):
    image_url: str = Field(..., description="The image URL")
    price: Optional[float]= Field(..., description="The price of the image")
    description: str = Field(..., description="The description of the image")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The content of the RAG response")
    used_image_urls: List[RAGUSEDIMAGE]