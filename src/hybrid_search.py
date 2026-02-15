import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .classes import ChunkResult, CombinedResults, rrfResult
from .invertedIndex import InvertedIndex
from .semantic_chunk_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            print(f"key: {self.api_key[:6]}... found ")
            self.client = genai.Client(api_key=self.api_key)

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

    def _spell_check(self, query: str) -> str:
        prompt = f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:"""

        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
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

        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
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
        try:
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
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
            print(bm25_results)
            score = self.hybrid_score(bm25_results.get(id, 0.0), crs, alpha)
            combined_results[id] = CombinedResults(
                bm25_score=bm25_results.get(id, 0.0), semantic_score=crs, data=ChunkResult(**cr), hybrid_score=score
            )

        sorted_combined = sorted(combined_results.items(), key=lambda x: x[1].hybrid_score, reverse=True)
        return sorted_combined[:limit]

    def _search(self, query: str, limit: int):
        def bm25_task():
            ii = InvertedIndex()
            ii.load()
            res = {i: v for i, v in ii.bm25_search(query, limit * 500)}
            return res

        def chunk_task():
            cs = ChunkedSemanticSearch()
            cs.load_movies()
            res = cs.search_chunks(query, limit * 500)
            return res

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

    def rrf_search(
        self, query: str, k: int, limit: int, enhance: str | None = None, rerank: str | None = None, evaluate: bool = False
    ):
        query = self._enhance(query, enhance) if enhance else query
        og_limit = limit
        if rerank:
            limit *= 5
        bm25_raw, chunk_raw = self._search(query, limit)
        print("done searching")
        bm25_id2rank = {
            doc_id: rank for rank, (doc_id, _) in enumerate(sorted(bm25_raw.items(), key=itemgetter(1), reverse=True), 1)
        }
        chunk_raw.sort(key=itemgetter("score"), reverse=True)
        chunk_id2rank = {int(cr["id"]): rank for rank, cr in enumerate(chunk_raw, 1)}
        mapping: dict[int, rrfResult] = {}
        for cr in chunk_raw:
            id = int(cr["id"])
            bm25_rank = bm25_id2rank.get(id, 0)
            bm25_score = rrf_score(bm25_rank, k)
            semantic_rank = chunk_id2rank.get(id, 0)
            semantic_score = rrf_score(semantic_rank, k)
            data = ChunkResult(**cr)
            mapping[id] = rrfResult(
                bm25_rank=bm25_rank,
                semantic_rank=semantic_rank,
                rrf_score=sum([bm25_score, semantic_score]) if bm25_rank > 0 and semantic_rank > 0 else 0.0,
                data=data,
            )

        sorted_mapping = sorted(mapping.items(), key=lambda x: x[1].rrf_score, reverse=True)
        sorted_mapping = self._rerank(rerank, sorted_mapping[:limit], query, limit) if rerank else sorted_mapping
        if evaluate:
            self._evaluate(sorted_mapping[:og_limit], query)
        return sorted_mapping[:og_limit]

    def _evaluate(self, mappings: list[tuple[int, rrfResult]], query: str):
        print("evaluating")
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {", ".join([f"{id}: {res.data.title}" for id, res in mappings])}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents.
        Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        parces_response = json.loads(response.text)
        for c, (_, mapping) in enumerate(mappings, start=1):
            print(f"{c}. {mapping.data.title}: {parces_response[c - 1]}/3")

    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def _rerank(self, method: str, mapping: list[tuple[int, rrfResult]], query: str, limit: int):
        match method:
            case "individual":
                for _, sr in tqdm(mapping[:limit], desc="Reranking"):
                    prompt = f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {sr.data.title} - {sr.data.document}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Give me ONLY the number in your response, no other text or explanation.

                    Score:"""

                    response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    score = int(response.text)
                    sr.rerank_score = score
                    time.sleep(5)
                mapping.sort(key=lambda x: x[1].rerank_score, reverse=True)
            case "batch":
                doc_list_str = "\n".join([f"{doc.data.id} - {doc.data.title}" for _, doc in mapping])
                print()
                prompt = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else.
                don't make it a md format, just a list
                For example:

                [75, 12, 34, 2, 1]
                """
                with open("prompt.txt", "w") as f:
                    f.write(prompt)
                response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                json_res = json.loads(response.text)
                id2item = {doc_id: res for doc_id, res in mapping}
                mapping = [(doc_id, id2item[doc_id]) for doc_id in json_res if doc_id in id2item]
            case "cross_encoder":
                pairs: list[list[str]] = []
                for _, doc in mapping:
                    pairs.append([query, f"{doc.data.title} - {doc.data.document}"])
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)
                for i, (_, map) in enumerate(mapping):
                    map.cross_encoder_score = scores[i]
                mapping.sort(key=lambda x: x[1].cross_encoder_score, reverse=True)
            case _:
                raise ValueError("Invalid rerank option")
        return mapping


def rrf_score(rank: float | int, k: int = 60) -> float:
    return 1 / (k + rank)
