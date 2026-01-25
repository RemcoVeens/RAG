from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def verify(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("Text cannot be empty")
        return self.model.encode(text)


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
