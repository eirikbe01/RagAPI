import chromadb

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="docs")

text = open("k8s.txt").read()

collection.add(documents=[text], ids=["k8s"])

print("Embedding stored in ChromaDB.")
