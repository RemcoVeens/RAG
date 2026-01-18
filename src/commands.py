from src.classes import Movie, Movies
from src.invertedIndex import InvertedIndex
from src.search import search


def load_InvertedIndex():
    ii = InvertedIndex()
    try:
        ii.load()
    except FileNotFoundError:
        print("Inverted index not found. Please build it first.")
        return
    return ii


def command_search(query:str, movies:Movies):
    ii = load_InvertedIndex()
    if ii is None:
        return
    search(ii, query, movies)
    return

def command_build(movies:list[Movie]):
    ii = InvertedIndex()
    ii.build(movies)
    ii.save()
    print("Inverted index built and saved.")
    return

def command_tf(doc_id:int, token:str):
    ii = load_InvertedIndex()
    if ii is None:
        return
    tf = ii.get_tf(doc_id, token)
    print(f"Term frequency of '{token}' in document {doc_id}: {tf}")
    return

def command_idf(token:str):
    ii = load_InvertedIndex()
    if ii is None:
        return
    idf = ii.get_idf(token)
    print(f"Inverse document frequency of '{token}': {idf:.2f}")
    return

def command_tfidf(doc_id:int, token:str):
    ii = load_InvertedIndex()
    if ii is None:
        return
    tfidf = ii.get_tfidf(doc_id, token)
    print(f"TF-IDF score of '{token}' in document '{doc_id}': {tfidf:.2f}")
    return
