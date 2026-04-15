from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None
    
    def __new__(cls, model_name:str = None):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialize(model_name)
        return cls._instance

    def _initialize(self, model_name):
        if not model_name:
            raise ValueError("Embedding model name is required")
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text):
        return self.model.encode(text)