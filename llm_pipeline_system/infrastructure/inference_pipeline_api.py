
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.clearml_utils import configure_clearml
from llm_engineering.model.inference import InferenceExecutor
from llm_engineering.model.inference.local_inference import LocalLLMInference

# Initialize ClearML for monitoring
clearml_task = configure_clearml()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


def call_llm_service(query: str, context: str | None) -> str:
    """Call local LLM service for inference."""
    llm = LocalLLMInference(
        model_id=settings.HF_MODEL_ID,
        device=settings.LOCAL_INFERENCE_DEVICE,
        load_in_8bit=True  # Enable memory optimization
    )
    answer = InferenceExecutor(llm, query, context).execute()
    
    # Log to ClearML if available
    if clearml_task:
        clearml_task.get_logger().report_text(
            title="LLM Inference",
            series="Query-Response",
            value=f"Query: {query}\nAnswer: {answer}"
        )
    
    return answer


def rag(query: str) -> str:
    """Execute RAG pipeline with ClearML monitoring."""
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    # Log metrics to ClearML
    if clearml_task:
        logger = clearml_task.get_logger()
        
        # Log scalar metrics
        logger.report_scalar(
            title="Token Counts",
            series="Query Tokens",
            value=misc.compute_num_tokens(query),
            iteration=0
        )
        logger.report_scalar(
            title="Token Counts", 
            series="Context Tokens",
            value=misc.compute_num_tokens(context),
            iteration=0
        )
        logger.report_scalar(
            title="Token Counts",
            series="Answer Tokens", 
            value=misc.compute_num_tokens(answer),
            iteration=0
        )
        
        # Log configuration
        clearml_task.connect({
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "retrieved_documents": len(documents)
        })

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
