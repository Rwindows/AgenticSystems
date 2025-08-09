from fastapi import APIRouter, Request
import logging
from api.rag.retrival import rag_pipeline_wrapper

from api.api.models import RAGRequest, RAGResponse, RAGUSEDIMAGE


logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/rag")
async def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    result = rag_pipeline_wrapper(payload.query)
    used_image_urls = [RAGUSEDIMAGE(image_url=image["image_url"], price=image["price"], description=image["description"]) for image in result["retrieved_images"]]
    
    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"].answer,
        used_image_urls=used_image_urls
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])