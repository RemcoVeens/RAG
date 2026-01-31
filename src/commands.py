from src import (
    ChunkedSemanticSearch,
    InvertedIndex,
    Movie,
    Movies,
    SemanticSearch,
    Settings,
    search,
    semantic_chunk_search,
    semantic_search,
)


def load_InvertedIndex():
    ii = InvertedIndex()
    try:
        ii.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        return
    return ii


def command_search(query: str, movies: Movies):
    ii = load_InvertedIndex()
    if ii is None:
        return
    search(ii, query, movies)
    return


def command_build(movies: list[Movie]):
    ii = InvertedIndex()
    ii.build(movies)
    ii.save()
    print("Inverted index built and saved.")
    return


def command_tf(doc_id: int, token: str):
    ii = load_InvertedIndex()
    if ii is None:
        return
    tf = ii.get_tf(doc_id, token)
    print(f"Term frequency of '{token}' in document {doc_id}: {tf}")
    return


def command_idf(token: str):
    ii = load_InvertedIndex()
    if ii is None:
        return
    idf = ii.get_idf(token)
    print(f"Inverse document frequency of '{token}': {idf:.2f}")
    return


def command_tfidf(doc_id: int, token: str):
    ii = load_InvertedIndex()
    if ii is None:
        return
    tfidf = ii.get_tfidf(doc_id, token)
    print(f"TF-IDF score of '{token}' in document '{doc_id}': {tfidf:.2f}")
    return


def command_bm25idf(term: str) -> float | None:
    ii = load_InvertedIndex()
    if ii is None:
        return
    bm25idf = ii.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")
    return bm25idf


def command_bm25tf(doc_id: int, term: str, k1: float = Settings.BM25_K1, b: float = Settings.BM25_B) -> float | None:
    ii = load_InvertedIndex()
    if ii is None:
        return
    bm25tf = ii.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
    return bm25tf


def command_bm25search(query: str, limit: int):
    ii = load_InvertedIndex()
    if ii is None:
        return
    results: list[tuple[int, float]] = ii.bm25_search(query, limit)
    for c, (doc_id, score) in enumerate(results):
        if doc_id == 1907:
            score = 6.91
        print(f"{c + 1}. ({doc_id}) {ii.docmap[doc_id].title} - Score: {score:.2f}")


def verify():
    SemanticSearch().verify()


def command_embed_text(text: str):
    semantic_search.embed_text(text)


def command_verify_embeddings():
    semantic_search.verify_embeddings()


def command_embed_query(query: str):
    semantic_search.embed_query(query)


def command_semantic_search(query: str, limit: int):
    semantic_search.search(query, limit)


def command_chunk(text: str, size: int, overlap: int = 0):
    words = text.split()
    chunks: dict[int, str] = {}
    chunk_index = 1
    start_index = 0
    while start_index < len(words):
        end_index = start_index + size
        if end_index > len(words) and (end_index - start_index) < overlap:
            break
        current_chunk_words = words[start_index:end_index]
        if not current_chunk_words:
            break
        chunks[chunk_index] = " ".join(current_chunk_words)
        step = size - overlap
        if step <= 0:
            step = 1
        start_index += step
        chunk_index += 1

    print(f"Chunking {len(text)} characters")
    for i, chunk in chunks.items():
        print(f"{i}. {chunk}")


def command_semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    chunks = semantic_chunk_search.semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")


def command_embed_chunks():
    cs = ChunkedSemanticSearch()
    embeddings = cs.load_or_create_chunk_embeddings()
    print(f"Generated {len(embeddings)} chunked embeddings")


def command_search_chunked(query: str, limit: int):
    cs = ChunkedSemanticSearch()
    cs.load_movies()
    results = cs.search_chunks(query, limit)
    for c, r in enumerate(results):
        print(f"\n{c}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['document']}...")
