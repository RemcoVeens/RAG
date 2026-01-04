import string

from nltk.stem import PorterStemmer

from src.classes import Movie


def prep_input(input:str)->str:
    temp = input.lower().strip(string.punctuation).replace(",","")
    return temp
def search(movies:list[Movie], query:str):
    print(f"Searching for: {query}")
    matches:list[Movie] = []
    stemmer = PorterStemmer()
    with open("data/stopwords.txt", "r") as f:
        stopwords:list[str] = f.read().splitlines()
    tokens = query.split()
    tokens = [token for token in tokens if token not in stopwords]
    for token in tokens:
        for movie in movies:
            if stemmer.stem(token) in prep_input(movie.title):
                if movie not in matches:
                    matches.append(movie)
    for movie in matches:
        print(f"{movie.id}. {movie.title}")
