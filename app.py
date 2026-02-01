from fastapi import FastAPI
import chromadb
import ollama
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")
logging.info(f"Using model: {MODEL_NAME}")


app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection(name="docs")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: str):
    logging.info(f"/query asked: {q}")

    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    use_mock = os.getenv("USE_MOCK_LLM", "0") == "1"

    if use_mock:
        logging.info("Using mock LLM response")
        # deterministic mock response for testing CI pipeline
        return {"answer": context}
    else:
        logging.info("Using Ollama LLM for response generation")

        # Constrain the model to the retrieved context to reduce hallucinations
        answer = ollama.generate(
            model=MODEL_NAME,
            prompt=(
                "You are a QA system that must answer using only the provided context. "
                "If the context is missing the information, reply with \"I don't know.\"\n\n"
                f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely using only the context."
            )
        )

    return {"answer": answer["response"]}


@app.post("/add")
def add_knowledge(content: str):
    """Add knowledge to the knowledge base dynamically."""
    logging.info(f"/add received content (new id will be generated)")
    try:
        import uuid
        doc_id = str(uuid.uuid4())

        # Add the text to the ChromaDB collection
        collection.add(documents=[content], ids=[doc_id])
        return {"status": "success", 
                "message": "Knowledge added successfully.", 
                "id": doc_id}
    except Exception as e:
        return {"status": "error", 
                "message": str(e)}

