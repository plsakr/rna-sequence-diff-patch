from operator import itemgetter

from IRMethods import search_collection, wf_score, cosine, pearson
from pymongo import MongoClient


client = MongoClient()
db = client.rna_db
collection = db.sample_sequences


def get_k_similar_docs(query, k, method):
    scores = search_collection(query, 'tf', collection, method)
    sorted_scores = sorted(scores, key=itemgetter(1), reverse=True)
    return sorted_scores[0:k]


query = 'AAAAAAAAAACUCACCAUGCUGAAAAGC'
# wf =    'AAAAAGGGGCCGGCAUUGUGGCGCAA'
# cos =   'AAAAAUANAGCAUUUUACUCCAUUUC'


wf_results = get_k_similar_docs(query, 6, wf_score)
print(wf_results)
wf_sequences = list(map(lambda x: x[0], wf_results))
# print(wf_sequences)

for k in range(1, 101):
    cos_results = get_k_similar_docs(query, k, cosine)
    cos_sequences = list(map(lambda x: x[0], cos_results))
    # print(cos_sequences)

    count = 0
    for s in cos_sequences:
        if s in wf_sequences:
            count += 1

    print(k, count)




