from qdrant_client import QdrantClient
from dotenv import load_dotenv
from openai import OpenAI
from core.config import config
from langsmith import traceable, get_current_run_tree
import instructor
from pydantic import BaseModel
from typing import List
import json
from api.rag.utils.utils import prompt_template_config

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)

# Initialize Qdrant client
qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

@traceable(name="get_embedding", run_type="llm", metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL})
def get_embedding(text, model = config.EMBEDDING_MODEL):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["using_metadata"] ={
            "input_token": response.usage.prompt_tokens,
            "total_token": response.usage.total_tokens,
        }
    
    return response.data[0].embedding

@traceable(name="retrieve_content", run_type="retriever", metadata={"retriever_type": "qdrant"})
def retrieve_content(query, top_k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.search(
        collection_name=config.Qdrant_Collections_Name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["retrieval_metrics"] = {
            "num_results": len(results),
            "top_k": top_k,
            "collection": config.Qdrant_Collections_Name
        }
        # Mark as completed by setting end_time
        if not current_run.end_time:
            current_run.end_time = current_run.start_time  # This ensures it's not pending
    
    return results

@traceable(name="process_context", run_type="parser")
def process_context(context):
    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {id}: {chunk}\n"
    
    return formatted_context

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question based on the provided context.",
        },
        "retrieved_context_ids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The index of the chunk that was used to answer the question.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the item based on the context together with the id.",
                    },
                },
            },
        },
    },
}
@traceable(name="build_prompt", run_type="prompt")
def build_prompt(context, question):
    formatted_context = process_context(context)
    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "rag_generation")
    
    prompt = prompt_template.render(processed_context=formatted_context, question=question, output_json_schema=json.dumps(OUTPUT_SCHEMA, indent=2))
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["prompt_metrics"] = {
            "prompt_length": len(prompt),
            "question_length": len(question),
            "context_length": len(formatted_context)
        }
        # Mark as completed
        if not current_run.end_time:
            current_run.end_time = current_run.start_time
    
    return prompt

class RAGUsedContext(BaseModel):
    id: int
    description: str

class RAGGenerationResponse(BaseModel):
    answer: str
    retrieved_context_ids: List[RAGUsedContext]
    

@traceable(name="generate_answer", run_type="llm", tags=["model:gpt-4o"], metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL})
def generate_answer(prompt):
    client = instructor.from_openai(OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
    model="gpt-4o",
    response_model=RAGGenerationResponse,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5,
)
    print(f"DEBUG: LLM response - answer length: {len(response.answer)}")
    print(f"DEBUG: LLM response - retrieved_context_ids: {response.retrieved_context_ids}")
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["llm_metrics"] = {
            "prompt_tokens": raw_response.usage.prompt_tokens,
            "completion_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,       
        }
    
    return response  


@traceable(name="rag_pipeline", run_type="chain", metadata={"pipeline_type": "rag"})
def rag_pipeline(question, top_k=5):
    retrieved_content = retrieve_content(question, top_k)
    
    # Create context dictionary with separate arrays for IDs and content
    context = {
        "retrieved_context_ids": [result.id for result in retrieved_content],
        "retrieved_context": [result.payload['text'] for result in retrieved_content]
    }
    
    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)

    # Create a dictionary with the retrieved content directly
    final_result = {
        "answer": answer,
        "question": question,
        "retrieved_content": retrieved_content  # Keep original for image retrieval
    }
    
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["pipeline_metrics"] = {
            "question_length": len(question),
            "answer_length": len(answer.answer),
            "retrieved_items": len(retrieved_content)
        }
        # Mark as completed
        if not current_run.end_time:
            current_run.end_time = current_run.start_time
    
    return final_result

@traceable(name="rag_pipeline_wrapper", run_type="chain", metadata={"pipeline_type": "rag"})
def rag_pipeline_wrapper(question, top_k=5):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    results = rag_pipeline(question, top_k)

    image_url_list = []
    print(f"DEBUG: Number of retrieved_context_ids: {len(results['answer'].retrieved_context_ids)}")
    print(f"DEBUG: Retrieved context IDs: {[ctx.id for ctx in results['answer'].retrieved_context_ids]}")
    
    for context_ref in results["answer"].retrieved_context_ids:
        print(f"DEBUG: Processing context_ref.id: {context_ref.id}")
        # context_ref.id is now the actual Qdrant document ID
        # Retrieve the document directly from Qdrant
        try:
            retrieved_docs = qdrant_client.retrieve(
                collection_name=config.Qdrant_Collections_Name,
                ids=[context_ref.id]
            )
            if retrieved_docs:
                payload = retrieved_docs[0].payload
                print(f"DEBUG: Payload keys: {list(payload.keys())}")
                image_url = payload.get("first_large_image")
                price = payload.get("price")
                print(f"DEBUG: image_url: {image_url}, price: {price}")
                if image_url:
                    image_url_list.append({
                        "image_url": image_url, 
                        "price": price, 
                        "description": context_ref.description
                    })
                    print(f"DEBUG: Added image to list")
                else:
                    print(f"DEBUG: No image_url found in payload")
            else:
                print(f"DEBUG: No document found for ID {context_ref.id}")
        except Exception as e:
            print(f"DEBUG: Error retrieving document {context_ref.id}: {e}")
    
    print(f"DEBUG: Final image_url_list length: {len(image_url_list)}")

    return {
        "answer": results["answer"],
        "retrieved_images": image_url_list
    }
    
