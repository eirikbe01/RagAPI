import chromadb
import os

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="docs")

# clear existing docs
existing_ids = collection.get()["ids"]
if existing_ids:
    collection.delete(ids=existing_ids)

for filename in os.listdir("./docs"):
    if filename.endswith(".txt"):
        text = open(f"./docs/{filename}", "r").read()
        collection.add(documents=[text], ids=[filename])

print("Re-embedded all documents in docs/ folder")