import argparse

from .settings import Settings


def get_keyword_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser("build", help="builds movies inverted index")

    tf = subparsers.add_parser("tf", help="counting tokens")
    _ = tf.add_argument("doc_id", type=int, help="Document ID to calculate term frequency")
    _ = tf.add_argument("token", type=str, help="Token to calculate term frequency")

    idf = subparsers.add_parser("idf", help="calculating inverse document frequency")
    _ = idf.add_argument("term", type=str, help="Token to calculate inverse document frequency")

    tfidf = subparsers.add_parser("tfidf", help="calculating term frequency inverse document frequency")
    _ = tfidf.add_argument("doc_id", type=int, help="Document ID to calculate term frequency inverse document frequency")
    _ = tfidf.add_argument("token", type=str, help="Token to calculate term frequency inverse document frequency")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    _ = bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    _ = bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=Settings.BM25_K1, help="Tunable BM25 K1 parameter")
    _ = bm25_tf_parser.add_argument("b", type=float, nargs="?", default=Settings.BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    _ = bm25search_parser.add_argument("query", type=str, help="Search query")
    _ = bm25search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Number of results to return")

    return parser


def get_semantic_parcer():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _ = subparsers.add_parser("verify", help="Verify the semantic search model")

    sub = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    _ = sub.add_argument("text", type=str, help="Text to generate embedding for")

    _ = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    eq = subparsers.add_parser("embedquery", help="Generate embedding for a given query")
    _ = eq.add_argument("query", type=str, help="Query to generate embedding for")

    search = subparsers.add_parser("search", help="Search movies using semantic search")
    _ = search.add_argument("query", type=str, help="Search query")
    _ = search.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")

    chunk = subparsers.add_parser("chunk", help="Chunk a given text")
    _ = chunk.add_argument("text", type=str, help="Text to chunk")
    _ = chunk.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Chunk size")
    _ = chunk.add_argument("--overlap", type=int, nargs="?", default=0, help="Chunk overlap")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Chunk a given text using semantic search")
    _ = semantic_chunk.add_argument("text", type=str, help="Text to chunk")
    _ = semantic_chunk.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Max chunk size")
    _ = semantic_chunk.add_argument("--overlap", type=int, nargs="?", default=0, help="Chunk overlap")

    _ = subparsers.add_parser("embed_chunks", help="Generate embeddings for a given text")

    search_chunked = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic search")
    _ = search_chunked.add_argument("query", type=str, help="Search query")
    _ = search_chunked.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")

    return parser


def get_hybrid_parcer():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparcers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparcers.add_parser("normalize", help="Normalize values")
    _ = normalize.add_argument("values", type=float, nargs="+", help="Values to normalize")

    weighted_search = subparcers.add_parser("weighted-search", help="Search movies using weighted semantic search")
    _ = weighted_search.add_argument("query", type=str, help="Search query")
    _ = weighted_search.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Weight for semantic search")
    _ = weighted_search.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")

    rrf_search = subparcers.add_parser("rrf-search", help="Search movies using RRF semantic search")
    _ = rrf_search.add_argument("query", type=str, help="Search query")
    _ = rrf_search.add_argument("--k", type=int, nargs="?", default=60, help="Parameter k for RRF")
    _ = rrf_search.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")
    _ = rrf_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        default=None,
        help="Query enhancement method",
    )

    return parser
