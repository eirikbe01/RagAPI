import chromadb
import os
import uuid
import ollama

DB_PATH = os.getenv("DB_PATH", "./db")
COLLECTION = os.getenv("COLLECTION", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")



client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="docs")

text = open("k8s.txt").read()

collection.add(documents=[text], ids=["k8s"])

print("Embedding stored in ChromaDB.")
