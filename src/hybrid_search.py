from . import ChunkedSemanticSearch, InvertedIndex
from .classes import ChunkResult, CombinedResults, rrfResult


class HybridSearch:
    def __init__(self):
        pass

    def normalize(self, values: list[float]) -> list[float]:
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
        # normalize bm25 scores
        if bm25_results:
            min_bm25 = min(bm25_results.values())
            max_bm25 = max(bm25_results.values())
            if min_bm25 == max_bm25:
                for k in bm25_results:
                    bm25_results[k] = 1.0
            else:
                for k, v in bm25_results.items():
                    bm25_results[k] = (v - min_bm25) / (max_bm25 - min_bm25)

        # normalize chunked semantic scores
        if chunck_results:
            scores: list[float] = [float(cr["score"]) for cr in chunck_results]
            min_sem = min(scores)
            max_sem = max(scores)
            if min_sem == max_sem:
                for cr in chunck_results:
                    cr["score"] = 1.0
            else:
                for cr in chunck_results:
                    cr["score"] = (float(cr["score"]) - min_sem) / (max_sem - min_sem)

        for cr in chunck_results:
            id = int(cr["id"])
            crs = float(cr["score"])
            score = self.hybrid_score(bm25_results[id], crs, alpha)
            combined_results[id] = CombinedResults(
                bm25_score=bm25_results[id], semantic_score=crs, data=ChunkResult(**cr), hybrid_score=score
            )

        sorted_combined = sorted(combined_results.items(), key=lambda x: x[1].hybrid_score, reverse=True)
        return sorted_combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int):
        ii = InvertedIndex()
        ii.load()
        bm25_results: dict[int, float] = {i: v for i, v in ii.bm25_search(query, limit * 500)}
        cs = ChunkedSemanticSearch()
        cs.load_movies()
        chunck_results = cs.search_chunks(query, limit * 500)

        bm25_results = dict(sorted(bm25_results.items(), key=lambda x: x[1], reverse=True))
        bm25_id2rank = {id: int(rank) for rank, (id, _) in enumerate(bm25_results.items(), start=1)}
        chunck_results = sorted(chunck_results, key=lambda x: float(x["score"]), reverse=True)
        chunk_id2rank = {int(cr["id"]): rank for rank, cr in enumerate(chunck_results, start=1)}
        mapping: dict[int, rrfResult] = {}
        for cr in chunck_results:
            id = int(cr["id"])
            bm25_score = rrf_score(bm25_rank := bm25_id2rank[id], k)
            semantic_score = rrf_score(semantic_rank := chunk_id2rank[id], k)
            mapping[id] = rrfResult(
                bm25_rank=bm25_rank,
                semantic_rank=semantic_rank,
                rrf_score=sum([bm25_score, semantic_score]) if bm25_rank > 0 and semantic_rank > 0 else 0.0,
                data=ChunkResult(**cr),
            )

        sorted_mapping = sorted(mapping.items(), key=lambda x: x[1].rrf_score, reverse=True)
        return sorted_mapping[:limit]

    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: float | int, k: int = 60) -> float:
    return 1 / (k + rank)
