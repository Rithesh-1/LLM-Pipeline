import bentoml
import mlflow

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.settings import settings

PROMPT_TEMPLATE = '''
You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
User query: {query}
Context: {context}
            '''

@bentoml.service
class RAGService:
    def __init__(self) -> None:
        # Load model from MLflow Model Registry
        logged_model_uri = "models:/TwinLlama-DPO/latest"
        self.model = mlflow.transformers.load_model(logged_model_uri)
        
        # Initialize the retriever
        self.retriever = ContextRetriever(mock=False)

    @bentoml.api
    def rag_query(self, query: str) -> str:
        # Retrieve context
        documents = self.retriever.search(query, k=3)
        context = EmbeddedChunk.to_context(documents)

        # Format the prompt
        prompt = PROMPT_TEMPLATE.format(query=query, context=context)

        # Generate the answer
        generated = self.model(
            prompt,
            max_new_tokens=settings.MAX_NEW_TOKENS_INFERENCE,
            repetition_penalty=1.1,
            temperature=settings.TEMPERATURE_INFERENCE,
        )
        
        # Extract the generated text
        answer = generated[0]["generated_text"]
        
        return answer
