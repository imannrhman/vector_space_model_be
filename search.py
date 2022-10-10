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
        
        #Menghapus spesial karakter
        document = remove_special_characters(document)
        
        #Menghapus digits 
        document = remove_digits(document)
        
        #Memisahkan dokumen perkata 
        terms = tokenize(document)
        
        #Memfilter kata ganda
        unique_terms = set(terms)

        #Memamsukan kata-kata unique kedalam list kata
        vocabulary = vocabulary.union(unique_terms)
        for term in unique_terms:
            #Menghitung frekuensi kata tiap dokumen
            postings[term][id] = terms.count(term)

    for term in vocabulary:
        #Menghitung frekuensi kata yang muncul pada dokumen
        document_frequency[term] = len(postings[term])

    #Menghitung bobot dokumen yang nantinya digunakan untuk normalisasi
    for id in corpus:
        l = 0
        for term in vocabulary:
            l += term_frequency(postings,term, id) ** 2
        length[id] = math.sqrt(l)   


   
    #Mensortir hasil dari W t,d
    scores = sorted(
        #Menjalankan fungsi simmaliry untuk mengcari W t,d
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




def similarity(vocabulary, postings, size_corpus, query, length, document_frequency, id) : 
    similarity = 0.0

    #Membagi query perkata
    for term in query :

        #Mencari query yang ada kumpulan kata
        if term in vocabulary :

            #Mengkalikan W t,f dan idf
            similarity += term_frequency(postings, term, id) * inverse_document_frequency(vocabulary, size_corpus, document_frequency, term)

    #Melakukan normalisasi W t,d
    similarity = similarity/length[id]

    return similarity



def tokenize(document):

    #memisahkan kalimat menjadi sebuah kata
    terms = word_tokenize(document)

    #memfilter kata yang bukan termasuk stopwords
    terms = [term.lower() for term in terms if term not in STOPWORDS]
    return terms


def inverse_document_frequency(vocabulary, size_corpus, document_frequency, term) :
   
    #Menghitung nilai idf
    if term in vocabulary :
        return math.log(size_corpus/ document_frequency[term], 2)
    else :
        return 0.0

def term_frequency(postings, term, id) :

    #Menghitung bobot tf
    if id in postings[term] :
        return postings[term][id]
    else : 
        return 0.0
    
def remove_special_characters(text) :
    # Menghapus spesial karakter dengan regex subsitusi
    regex = re.compile(r"[^a-zA-Z0-9\s]")
    return re.sub(regex, "", text)

def remove_digits(text) :
    # Menghapus digits dengan regex subsitusi
    regex = re.compile(r"\d")
    return re.sub(regex, "", text)

search_vector_model(korpus_data, "komputer")