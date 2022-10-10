import math
import re
import nltk

nltk.data.path.append("/tmp")
nltk.download("stopwords", download_dir="/tmp")
nltk.download("punkt", download_dir="/tmp")

from collections import defaultdict
from functools import reduce
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data import korpus_data


STOPWORDS = set(stopwords.words('indonesian'))



def search_vector_model(corpus, query) :
    size_corpus = len(corpus)
    vocabulary = set()
    postings = defaultdict(dict)
    document_frequency = defaultdict(int)
    length = defaultdict(int)

    for id in corpus :
        document = corpus[id]
        document = remove_special_characters(document)
        document = remove_digits(document)
        terms = tokenize(document)
        unique_terms = set(terms)
        vocabulary = vocabulary.union(unique_terms)
        for term in unique_terms:
            postings[term][id] = terms.count(term)

    for term in vocabulary:
        document_frequency[term] = len(postings[term])

    for id in corpus:
        l = 0
        for term in vocabulary:
            l += term_frequency(postings,term, id) ** 2
        length[id] = math.sqrt(l)   

        
    scores = sorted(
        [(id, similarity(vocabulary, postings, size_corpus, tokenize(query), length, document_frequency, id)) for id in range(size_corpus)],
        key=lambda x: x[1],
        reverse=True,
    )

    return print_scores(scores, corpus);
    

def print_scores(scores, corpus) :

    results = [];
    for (id, score) in scores:
        if score != 0.0:
            results.append({"document_id" : id, "corpus" : corpus[id] ,  "scores" : round(score, 3), });

    return results;



def intersection(sets) :
    return reduce(set.intersection, [s for s in sets])


def similarity(vocabulary, postings, size_corpus, query, length, document_frequency, id) : 
    similarity = 0.0
    for term in query :
        if term in vocabulary :
            similarity += term_frequency(postings, term, id) * inverse_document_frequency(vocabulary, size_corpus, document_frequency, term)
    similarity = similarity/length[id]
    return similarity



def tokenize(document):
    terms = word_tokenize(document)
    terms = [term.lower() for term in terms if term not in STOPWORDS]
    return terms


def inverse_document_frequency(vocabulary, size_corpus, document_frequency, term) :
   
    if term in vocabulary :
        return math.log(size_corpus/ document_frequency[term], 2)
    else :
        return 0.0

def term_frequency(postings, term, id) :

    if id in postings[term] :
        return postings[term][id]
    else : 
        return 0.0
    
def remove_special_characters(text) :
    # Menghapus spesial karakter dengan regex subsitusi
    regex = re.compile(r"[^a-zA-Z0-9\s]")
    return re.sub(regex, "", text)

def remove_digits(text) :
    regex = re.compile(r"\d")
    return re.sub(regex, "", text)

search_vector_model(korpus_data, "komputer")