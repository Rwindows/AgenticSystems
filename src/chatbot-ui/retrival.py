from qdrant_client import QdrantClient
from dotenv import load_dotenv
from openai import OpenAI
from core.config import config
from langsmith import traceable, get_current_run_tree

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)

# Initialize Qdrant client
qdrant_client = QdrantClient(url=f"http://{config.QDRANT_URL}:6333")

@traceable(name="get_embedding", run_type="llm",
metadata={"ls_proider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL}
        )
def get_embedding(text, model = config.OPENAI_EMBEDDING_MODEL):
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
def retrive_content(query, top_k=5):
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

@traceable(name="process_context", run_type="transform")
def process_context(context):
    formatted_context = ""
    for chunk in context:
        formatted_context += f"- {chunk.payload['text']}\n"
    
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["transform_metrics"] = {
            "context_chunks": len(context),
            "formatted_length": len(formatted_context)
        }
        # Mark as completed
        if not current_run.end_time:
            current_run.end_time = current_run.start_time
    
    return formatted_context

@traceable(name="build_prompt", run_type="prompt")
def build_prompt(context, question):
    formatted_context = process_context(context)
    prompt = f"""
    
    you are the shopping assistant that can answer questions about the product in stock.

you will be given a question and a list of context.
Instructions:
- you need to answer the question bsed on the provided context only
- Never use word context and refer to it as the avliable products

Context:
{formatted_context}

Question:
{question}

    """
    
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

@traceable(name="generate_answer", run_type="llm", tags=["model:gpt-4o"], metadata={"model": "gpt-4o", "temperature": 0.7})
def generate_answer(prompt):
    # Use the OpenAI client to create a chat completion
    response = client.chat.completions.create(
        model = "gpt-4o",  # Updated to a valid model name
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0.7,
    )
    
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["llm_metrics"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "finish_reason": response.choices[0].finish_reason
        }
        # Mark as completed
        if not current_run.end_time:
            current_run.end_time = current_run.start_time
    
    return response.choices[0].message.content
  
@traceable(name="rag_pipeline", run_type="chain", metadata={"pipeline_type": "rag"})
def rag_pipeline(question, top_k=5):
    retrieved_content = retrive_content(question, top_k)
    prompt = build_prompt(retrieved_content, question)
    answer = generate_answer(prompt)

    # Create a dictionary with the retrieved content directly
    final_result = {
        "answer": answer,
        "question": question,
        "retrieved_content": retrieved_content  # Pass the retrieved content directly
    }
    
    # Update run metadata for LangSmith
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["pipeline_metrics"] = {
            "question_length": len(question),
            "answer_length": len(answer),
            "retrieved_items": len(retrieved_content)
        }
        # Mark as completed
        if not current_run.end_time:
            current_run.end_time = current_run.start_time
    
    return final_result
    
    