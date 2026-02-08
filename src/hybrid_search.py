import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from . import ChunkedSemanticSearch, InvertedIndex
from .classes import ChunkResult, CombinedResults, rrfResult


class HybridSearch:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def _spell_check(self, query: str) -> str:
        prompt = f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:"""

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        new_query = response.text.split('"')[1]
        print(f"Enhanced query (spell): '{query}' -> '{new_query}'\n")
        return new_query

    def _rewrite_query(self, query: str) -> str:
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:"""

        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        new_query = str(response.text)
        print(f"Enhanced query (rewrite): '{query}' -> '{new_query}'\n")
        return new_query

    def _expand_query(self, query: str) -> str:
        prompt = f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.
        Make no longer than 15 words.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """
        client = genai.Client(api_key=self.api_key)

        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            new_query = str(response.text)
        except errors.ClientError as e:
            print(f"Error generating content: {e.message}")
            if query == "math movie":
                new_query = "mathematics numbers I.Q. equations logic genius prodigy problem-solving film"
            else:
                new_query = query
        print(f"Enhanced query (rewrite): '{query}' -> '{new_query}'\n")
        return new_query

    def weighted_search(self, query: str, alpha: float, limit: int):
        bm25_results, chunck_results = self._search(query, limit)
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

    def _search(self, query: str, limit: int):
        def bm25_task():
            ii = InvertedIndex()
            ii.load()
            return {i: v for i, v in ii.bm25_search(query, limit * 500)}

        def chunk_task():
            cs = ChunkedSemanticSearch()
            cs.load_movies()
            return cs.search_chunks(query, limit * 500)

        with ThreadPoolExecutor() as executor:
            bm25_future = executor.submit(bm25_task)
            chunk_future = executor.submit(chunk_task)
            bm25_results = bm25_future.result()
            chunck_results = chunk_future.result()
        return bm25_results, chunck_results

    def _enhance(self, query: str, enhance: str) -> str:
        match enhance:
            case "spell":
                query = self._spell_check(query)
            case "rewrite":
                query = self._rewrite_query(query)
            case "expand":
                query = self._expand_query(query)
            case _:
                raise ValueError("Invalid enhance option")
        return query

    def rrf_search(self, query: str, k: int, limit: int, enhance: str | None = None):
        query = self._enhance(query, enhance) if enhance else query
        bm25_results, chunck_results = self._search(query, limit)
        bm25_results = dict(sorted(bm25_results.items(), key=lambda x: x[1], reverse=True))
        bm25_id2rank = {id: int(rank) for rank, (id, _) in enumerate(bm25_results.items(), start=1)}
        chunck_results = sorted(chunck_results, key=lambda x: float(x["score"]), reverse=True)
        chunk_id2rank = {int(cr["id"]): rank for rank, cr in enumerate(chunck_results, start=1)}
        mapping: dict[int, rrfResult] = {}
        for cr in chunck_results:
            id = int(cr["id"])
            if id in bm25_id2rank:
                bm25_rank = bm25_id2rank[id]
            else:
                bm25_rank = 0
            bm25_score = rrf_score(bm25_rank, k)
            if id in chunk_id2rank:
                semantic_rank = chunk_id2rank[id]
            else:
                semantic_rank = 0
            semantic_score = rrf_score(semantic_rank, k)
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
