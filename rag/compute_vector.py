import json
import numpy as np
import faiss

with open("gita_embeddings.json") as f:
    data = json.load(f)

vectors = np.array([v["embedding"] for v in data]).astype("float32")

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(vectors)

faiss.write_index(index, "gita_index.faiss")

print("Index built with", index.ntotal, "verses")