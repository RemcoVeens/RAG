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

    return parser
