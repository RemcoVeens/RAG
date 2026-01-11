import string

from nltk.stem import PorterStemmer

from src import Movie, Movies, InvertedIndex


def prep_input(input:str)->str:
    temp = input.lower().strip(string.punctuation).replace(",","")
    return temp

def search(ii:InvertedIndex, query:str, movies:Movies):
    print(f"Searching for: {query}")
    matches:list[Movie] = []
    stemmer = PorterStemmer()
    with open("data/stopwords.txt", "r") as f:
        stopwords:list[str] = f.read().splitlines()
    tokens = query.split()
    tokens = [token for token in tokens if token not in stopwords]
    matches_found: set[int] = set()
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        movies_for_stem = ii.get_documents(stemmed_token)
        if movies_for_stem:
            matches_found.update(movies_for_stem)
    matches = [movies.from_id(movie_id) for movie_id in sorted(matches_found)][:10]
    for movie in matches:
        print(f"{movie.id}. {movie.title}")
