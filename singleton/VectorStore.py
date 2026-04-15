import faiss
import json
import numpy as np

class VectorDB:
    _instance = None
    
    def __new__(cls, index_name: str = None, data_name: str = None):
        if cls._instance is None:
            cls._instance = super(VectorDB, cls).__new__(cls)
            cls._instance._initialize(index_name, data_name)
        return cls._instance
    
    def _initialize(self, index_name, data_name):
        self.index = faiss.read_index(index_name)
        with open(data_name) as f:
            self.data = json.load(f)
    
    def search(self, query_vector, k=5):
        D, I = self.index.search(np.array([query_vector]), k)
        return [self.data[i] for i in I[0]]