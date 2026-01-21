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

    answer = ollama.generate(
        model=MODEL_NAME,
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely.",
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 120,
        }
    )

    return {"Answer:": answer["response"]}


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

