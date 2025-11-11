import gradio as gr
from langchain.globals import set_verbose
from loguru import logger

from llm_pipeline_system.application.rag.retriever import ContextRetriever
from llm_pipeline_system.infrastructure.clearml_utils import configure_clearml

def chat_interface(message, history):
    history = history or []
    
    set_verbose(True)

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(message, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")

    # This is a simplified response for demonstration purposes.
    # A real implementation would pass the retrieved documents to an LLM.
    response = f"I found {len(documents)} documents related to your query. The most relevant one is: {documents[0]}"
    
    history.append((message, response))
    return "", history

if __name__ == "__main__":
    configure_clearml()
    
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(chat_interface, [msg, chatbot], [msg, chatbot], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()
