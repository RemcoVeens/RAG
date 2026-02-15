from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .classes import Movie
from .semantic_chunk_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", docs: dict[int, Movie] = []):
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.titles = [f"{doc.title}: {doc.description}" for (_, doc) in docs.items()]
        import hashlib
        import json

        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        # Create a stable hash of the titles list to use as a cache key
        # json.dumps with sort_keys=True ensures consistent string representation
        titles_json_str = json.dumps(self.titles, sort_keys=True)
        titles_hash = hashlib.md5(titles_json_str.encode("utf-8")).hexdigest()
        cache_file = cache_dir / f"text_embeddings_{titles_hash}.npy"

        if cache_file.is_file():
            print(f"Loading text embeddings from cache: {cache_file}")
            # Assuming numpy is imported as 'np' at the module level due to its use later in the class
            self.text_embeddings = np.load(cache_file)
        else:
            print(f"Computing text embeddings and saving to cache: {cache_file}")
            self.text_embeddings = self.model.encode(self.titles, show_progress_bar=True)
            # Assuming numpy is imported as 'np' at the module level
            np.save(cache_file, self.text_embeddings)

    def embed_image(self, image: Path):
        file = Image.open(image)
        return self.model.encode(file)

    def search_with_image(self, img_path: Path):
        image_embedding = self.embed_image(img_path)
        scores = []
        for embedding in self.text_embeddings:
            scores.append(cosine_similarity(image_embedding, embedding))
        doc_items = list(self.docs.items())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
        results = []
        for idx, score in ranked:
            doc_id, doc = doc_items[idx]
            results.append(
                {
                    "id": doc_id,
                    "title": doc.title,
                    "description": doc.description,
                    "score": score,
                }
            )
        return results


def verify_image_embedding(path: Path):
    embedding = MultimodalSearch().embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
