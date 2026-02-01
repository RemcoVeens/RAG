from . import ChunkedSemanticSearch, InvertedIndex
from .classes import ChunkResult, CombinedResults


class HybridSearch:
    def __init__(self):
        pass

    def normalize(self, values: list[float]):
        if not values:
            return []
        values.sort()
        min_score = min(values)
        max_score = max(values)
        if min_score == max_score:
            return [1.0]
        scores: list[float] = []
        for score in values:
            new_score = (score - min_score) / (max_score - min_score)
            scores.append(new_score)
        return scores

    def weighted_search(self, query: str, alpha: float, limit: int):
        ii = InvertedIndex()
        ii.load()
        bm25_results: dict[int, float] = {i: v for i, v in ii.bm25_search(query, limit * 500)}
        cs = ChunkedSemanticSearch()
        cs.load_movies()
        chunck_results = cs.search_chunks(query, limit * 500)

        combined_results: dict[int, CombinedResults] = {}
        for cr in chunck_results:
            id = int(cr["id"])
            score = self.hybrid_score(bm25_results[id], cr["score"], alpha)
            combined_results[id] = CombinedResults(
                bm25_score=bm25_results[id], semantic_score=cr["score"], data=ChunkResult(**cr), hybrid_score=score
            )

        sorted_combined = sorted(combined_results.items(), key=lambda x: x[1].hybrid_score, reverse=True)
        return sorted_combined[:limit]

    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score
