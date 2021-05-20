# import statements
import math
import pickle
import time
import numpy as np
from multiprocessing import Process, Manager
import pandas as pd
from StringEditDistance import wagnerFisher
from operator import itemgetter
import os

# Nucleotide array: will be referenced throughout the project in the following order, each nucleotide will act like an index
nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']
# Base nucleotides array: will also be referenced throughout the project in the following order, each nucleotide will act like an index
base_nucleotides = ['A', 'G', 'C', 'U']

# ambiguous nucleotides with their associated weights
# For example: Nucleotide Y has a probability of 0.5 of being C and 0.5 of being U
# (using the order of the base_nucleotide array so the weight follows its corresponding value in base_nucleotide at the same index)
ambiguous_nucleotides = {
    'Y': np.array([0, 0, 0.5, 0.5]),
    'R': np.array([0.5, 0.5, 0, 0]),
    'W': np.array([0.5, 0, 0, 0.5]),
    'S': np.array([0, 0.5, 0.5, 0]),
    'K': np.array([0, 0.5, 0, 0.5]),
    'M': np.array([0.5, 0, 0.5, 0]),
    'D': np.array([0.33, 0.33, 0, 0.33]),
    'V': np.array([0.33, 0.33, 0.33, 0]),
    'H': np.array([0.33, 0, 0.33, 0.33]),
    'B': np.array([0, 0.33, 0.33, 0.33]),
    'N': np.array([0.25, 0.25, 0.25, 0.25])
}
# Similarly as above but used for vectors
ambiguity_vectors = {
    'Y': {'A': 0, 'G': 0, 'C': 0.5, 'U': 0.5},
    'R': {'A': 0.5, 'G':  0.5, 'C': 0, 'U': 0},
    'W': {'A': 0.5, 'G': 0, 'C': 0, 'U': 0.5},
    'S': {'A': 0, 'G': 0.5, 'C': 0.5, 'U': 0},
    'K': {'A': 0, 'G': 0.5, 'C': 0, 'U': 0.5},
    'M': {'A': 0.5, 'G': 0, 'C': 0.5, 'U': 0},
    'D': {'A': 0.33, 'G': 0.33, 'C': 0, 'U': 0.33},
    'V': {'A': 0.33, 'G': 0.33, 'C': 0.33, 'U': 0},
    'H': {'A': 0.33, 'G': 0, 'C': 0.33, 'U': 0.33},
    'B': {'A': 0, 'G': 0.33, 'C': 0.33, 'U': 0.33},
    'N': {'A': 0.25, 'G': 0.25, 'C': 0.25, 'U': 0.25}
}

# set-based representation
def convert_to_set(sequence):
    # to convert a sequence of nucleotides to a set, we will simply use python's built-in function
    return set(sequence)

# IR measures specific for the set-based methods:
def intersection(a, b, return_dict=None):
    if return_dict is None:
        # if it is not previously calculated, get the intersection of the 2 sets using the set intersection method of python
        return a.intersection(b)
    else:
        # set it in the return_dict
        return_dict['intersection'] = a.intersection(b)


def set_intersection_similarity(a, b, return_dict=None):
    # this method will return the cardinality of the intersection between set A and set B
    if return_dict is None:
        return len(intersection(a, b))
    else:
        return_dict['set_intersection_sim'] = len(intersection(a, b))


def set_jaccard_similarity(a, b, return_dict=None):
    # if it is not previously calculated, get the intersection of the 2 sets using the set_intersection_similarity
    # (prev defined) then divide it by the cardinality of their union
    if return_dict is None:
        return set_intersection_similarity(a, b) / len(a.union(b))
    else:
        # set it in the return_dict
        return_dict['set_jaccard_sim'] = set_intersection_similarity(a, b) / len(a.union(b))


def set_dice_similarity(a, b, return_dict=None):
    # if it is not previously calculated, get the intersection of the 2 sets using the set_intersection_similarity
    # (prev defined), multiply it by 2 and then divide it by the cardinality of each of the given sets
    denom = len(a) + len(b)
    if return_dict is None:
        return 2 * set_intersection_similarity(a, b) / denom
    else:
        # set it in the return_dict
        return_dict['set_dice_sim'] = 2 * set_intersection_similarity(a, b) / denom

# Multiset representation
def convert_to_multi_set(sequence):
    # transform the sequence into a multiset representation: we will refer to the base nucleotides array (numpy array of 4 elements)
    c = np.zeros(4)

    for l in sequence:
        # if this nucleotide is a base nucleotide, add 1 to its frequency of occurrences at its corresponding index
        if l in base_nucleotides:
            c[base_nucleotides.index(l)] += 1
        else:
            # if it is an ambiguous nucleotide, get its probability np array and add it to the created proba array c
            c = c + ambiguous_nucleotides[l]

    return c


def multi_intersection_similarity(ca, cb, return_dict=None):
    sim = 0

    for k in range(4):
        # getting the min multiplicity of Ai and Bi and adding it to the sim
        # (we are interested in the cardinality not the actual intersection multiset)
        sim += min(ca[k], cb[k])
    if return_dict is None:
        return sim
    else:
        #to store it
        return_dict['multi_intersection_sim'] = sim


def multi_jaccard_similarity(ca, cb, return_dict=None):
    # use the intersection similarity function specific to multisets to get the numerator
    num = multi_intersection_similarity(ca, cb)
    # the denominator corresponds to the cardinality of each multi sets - the numerator
    den = np.sum(ca) + np.sum(cb) - num
    if return_dict is None:
        return num / den
    else:
        # to store it
        return_dict['multi_jaccard_sim'] = num/den


def multi_dice_similarity(ca, cb, return_dict=None):
    # use the intersection similarity function specific to multisets to get the numerator
    num = multi_intersection_similarity(ca, cb)

    den = np.sum(ca) + np.sum(cb)
    # the denominator corresponds to the sum of the multisets' cardinality  and num = 2* intersection_sim
    if return_dict is None:
        return 2 * num / den
    else:
        # to store it
        return_dict['multi_dice_sim'] = 2 * num/den

# Vector representation:
# Term Frequency representation:
def convert_to_tf_vector(seq):
    # In total, we have 15 nucleotides (base+ambiguous)
    # We will be doing a pair wise sim so we need a 15x15 matrix
    vec = np.zeros((15, 15))

    for i in range(len(seq) - 1):
        current = seq[i]
        next_char = seq[i + 1]

        vec[nucleotides.index(current)][nucleotides.index(next_char)] += 1
        # Add 1 to their respective indices in the matrix (occured: 1st pair)
        # initialize a 0 cost of base nucleotides
        cost_current = {'A': 0, 'G': 0, 'C': 0, 'U': 0}

        if current in base_nucleotides:
            # if the current nucleotide is a base nucleotide, add 1 to its index in the cost_current dict
            cost_current[current] = cost_current[current] + 1
        else:
            # If not: it is an ambiguous nucleotide, get the corresponding probability of occurrence dict of
            # that ambiguous nucleotide and add it to the cost_current
            cost_current = {k: cost_current[k] + ambiguity_vectors[current][k] for k in set(cost_current)}


        if next_char in base_nucleotides and current not in base_nucleotides:
            # Considering the case where only the second nucleotide of the pair is a base nucleotide
            for k in cost_current.keys():
                vec[nucleotides.index(k)][nucleotides.index(next_char)] += cost_current[k]

        elif next_char not in base_nucleotides:
            # if the second nucleotide of the pair is an ambiguous nucleotide:

            dest_costs = ambiguity_vectors[next_char]
            # Get the corresponding ambiguity vector and add it to the actual cost
            for k in cost_current.keys():
                for j in cost_current.keys():
                    actual_cost = cost_current[k] * dest_costs[j]
                    vec[nucleotides.index(k)][nucleotides.index(j)] += actual_cost

    return vec

# IDF Representation
def convert_to_idf_vector(seq1, collection=None, list_of_docs=None, doc_count=0):
    vec = np.zeros((15,15))
    # similar to the TF representation in terms of going through the pairs of nucleotides

    set1 = set()
    for i in range(len(seq1) - 1):
        current = seq1[i]
        next_char = seq1[i + 1]
        sequence = current + next_char
        set1.add(sequence)

    # get the number of RNA seq in our collection (from db)
    count = doc_count if doc_count != 0 else len(list_of_docs) if list_of_docs is not None else collection.count_documents({})

    for pair in set1:
        cost = 0
        my_range = list_of_docs if list_of_docs is not None else collection.find({})
        for record in my_range:
            # Get the number of RNA seq in the collection that have this pair
            cost = cost + compare_pair_to_seq(pair, record, is_document=collection is not None)

        idf = 0 if cost == 0 else math.log(count/cost, 10)
        index1 = nucleotides.index(pair[0])
        index2 = nucleotides.index(pair[1])
        vec[index1][index2] = idf

    return vec

# TF-IDF representation:
def create_tf_idf_vector(seq, collection=None, list_of_docs=None, doc_count=0, is_document=False):
    # Transform them into both separately and then perform the dot product (multiplying the vectors)
    tf = convert_to_tf_vector(seq) if not is_document else pickle.loads(seq['tf'])
    idf = convert_to_idf_vector(seq, collection, list_of_docs, doc_count) if not is_document else pickle.loads(seq['idf'])
    return np.dot(tf, idf)


def compare_pair_to_seq(pair, record, is_document=True):
    # Checking if a pair is available in a sequence:
    seq = record['sequence'] if is_document else record
    if pair in seq:
        # if the pair occurs exactly as it does in the first seq ('AA' or 'AN') -> return 1
        return 1
    else:
        # If the pair doesn't occur in the same representation: check for possibilities based on the ambiguity
        # nucleotides in both: pair and target sequence

        # Gather the possibilities of each pair (if it is N: it is 25% A, 25% C, 25% G and 25% U:
        # a pair AN: has 4 possibilities: AA, AC, AG and AU)

        first_possibilities, first_probabilities = possibilities(pair[0])
        second_possibilities, second_probabilities = possibilities(pair[1])
        current_max = 0

        for pos1 in range(len(first_possibilities)):
            for pos2 in range(len(second_possibilities)):
                actual = first_possibilities[pos1] + second_possibilities[pos2]
                # If the possible pair is found in seq: cost is equal to the probability of occurrence of each nucleotide
                # multiplied by the other's probability
                if actual in seq:
                    cost = first_probabilities[pos1] * second_probabilities[pos2]
                    if current_max < cost:
                        # Get the max cost for an index (that's what we're interested in for the Vector based rep)
                        current_max = cost

        return current_max


def possibilities(nucleotide):
    # To get the possible nucleotides:
    if nucleotide in base_nucleotides:
        # if this nucleotide is a base nucleotide, get the bases' possibilities
        return get_base_possibilities(nucleotide)
    else:
        # if this nucleotide is ambiguous:
        output = []
        # get the corresponding vector that represents this ambiguous nucleotide's possible representations
        complete_vector = ambiguity_vectors[nucleotide]
        for k in complete_vector.keys():
            # if the probability of having a certain base nucleotide is not 0, append it to the output array
            if complete_vector[k] != 0:
                output.append(k)
        # the probability will be 1/the total length of the output array (nb of occurrences)
        probability = [1/len(output) for i in output]
        # append 1 to the probability and append the ambiguous nucleotide itself to the output :
        # we might find an N in both the sequence and the pair
        probability.append(1)
        output.append(nucleotide)
        return output, probability


def get_base_possibilities(base):
    # get the probability of occurrence of a base nucleotide
    p, prob = [base], [1]

    for k in ambiguity_vectors.keys():
        # check each ambiguous nucleotide if it might represent the base nucleotide and if so, append it
        if ambiguity_vectors[k][base] != 0:
            p.append(k)
            prob.append(ambiguity_vectors[k][base])
    return p, prob


def cosine(a, b, return_dict = None):
    # Following the cosine measure: numerator is the sum of the multiplication of each vector
    num = np.sum(np.multiply(a, b))
    # get the module of each vector squared: a and b
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))
    # denominator is the sqrt of the prev calculated squared modules
    den = math.sqrt(a_sq * b_sq)


    if return_dict is None:
        # return it
        return num / den
    else:
        # save it
        return_dict['cosine'] = num/den


def pearson(a, b, return_dict = None):
    # get the average value of each vector
    a_bar = np.average(a)
    b_bar = np.average(b)

    # subtract the mean val from each elt of each corresponding vector
    a_sub = np.subtract(a, a_bar)
    b_sub = np.subtract(b, b_bar)

    # numerator is the sum of the multiplication of each of the subtracted vectors: a_sub and b_sub
    num = np.sum(np.multiply(a_sub, b_sub))
    # get the module of each vector squared: a and b
    a_sub_sq = np.sum(np.square(a_sub))
    b_sub_sq = np.sum(np.square(b_sub))

    # denominator is the sqrt of the prev calculated squared modules
    den = math.sqrt(a_sub_sq * b_sub_sq)
    if return_dict is None:
        # return it
        return num / den
    else:
        # save it
        return_dict['pearson'] = num/den


def euclidian_distance(a, b, return_dict = None):
    # get the euclidean distance
    dist = math.sqrt(np.sum(np.square(np.subtract(a, b))))
    if return_dict is None:
        # return the similarity following the 1/(1+dist) formula : sim and dist are inversely proportional
        return 1 / (1 + dist)
    else:
        # save the sim
        return_dict['euclidian_dist'] = 1 / (1 + dist)


def manhattan_distance(a, b, return_dict = None):
    # get the manhattan distance
    dist = math.sqrt(np.sum(np.abs(np.subtract(a, b))))
    if return_dict is None:
        # return the similarity following the 1/(1+dist) formula : sim and dist are inversely proportional
        return 1 / (1 + dist)
    else:
        # save the sim
        return_dict['manhattan_distance'] = 1 / (1 + dist)


def tanimoto_distance(a, b, return_dict = None):
    # numerator of the tanimoto sim is the sum of the multiplied vectors
    num = np.sum(np.multiply(a, b))

    # get the magnitude squared of each vector
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    # denominator is the squared magnitude - numerator
    den = a_sq + b_sq - num
    if return_dict is None:
        # return it
        return num / den
    else:
        # save it
        return_dict['tanimoto_dist'] = num/den


def dice_dist(a, b, return_dict = None):
    # numerator of dice sim is 2* the sum of the multiplied vectors
    num = 2 * np.sum(np.multiply(a, b))

    # get the magnitude squared of each vector
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    # denominator is the squared magnitude
    den = a_sq + b_sq

    if return_dict is None:
        # return it
        return num / den
    else:
        # save it
        return_dict['dice_dist'] = num/den


def time_method(method):
    # to check how long each method takes: for the timing diagram in the GUI
    def wrapper(a, b, return_dict):
        start = time.time()
        method(a, b, return_dict)
        end = time.time()
        return_dict[method.__name__ + '_time'] = (end-start)*1000
    return wrapper


def create_and_start_threads(methods_to_execute, a, b):
    # for multithreading: faster results: each method on a different thread
    result_objects = []
    m = Manager()
    return_dict = m.dict()
    jobs = []

    for m in methods_to_execute:

        p = Process(target=time_method(m), args=(a, b, return_dict))
        jobs.append(p)
        p.start()


    for j in jobs:
        j.join()

    return return_dict

def perform_methods(a, b, do_cosine=False, do_pearson=False, do_euclidian_distance=False, do_manhattan_distance=False,
                    do_tanimoto_distance=False, do_dice_dist=False):

    # based on what the user selects in the GUI, perform/call the appropriate function
    jobs = []
    if do_cosine: jobs.append(cosine)
    if do_pearson: jobs.append(pearson)
    if do_euclidian_distance: jobs.append(euclidian_distance)
    if do_manhattan_distance: jobs.append(manhattan_distance)
    if do_tanimoto_distance: jobs.append(tanimoto_distance)
    if do_dice_dist: jobs.append(dice_dist)
    return create_and_start_threads(jobs, a, b)


def wf_score(seq1, seq2, user_cost=False):
    # get the wagner fischer score from the prev implementation (phase 1: script Edit distance)
    dp = wagnerFisher(seq1, seq2, user_cost)
    # the optimal cost of making 2 sequences homomorphic is the value in the last row, last col
    cost = dp[len(dp)-1][len(dp[0])-1].value
    return 1/(1+cost)


def search_collection(query, vector_type,  collection, method, return_dict=None, callback=None):
    # step 1: convert query to idf/tf/tf-idf/set/multiset
    if method == wf_score:
        vector1 = query
        convert_method = lambda x: x['sequence']
    elif method == set_intersection_similarity or method == set_dice_similarity or method == set_jaccard_similarity:
        vector1 = convert_to_set(query)
        convert_method = lambda x: convert_to_set(x['sequence'])
    elif method == multi_intersection_similarity or method == multi_dice_similarity or method == multi_jaccard_similarity:
        vector1 = convert_to_multi_set(query)
        convert_method = lambda x: convert_to_multi_set(x['sequence'])
    else:
        if vector_type == 'tf':
            vector1 = convert_to_tf_vector(query)
            convert_method = lambda x: pickle.loads(x['tf'])
        elif vector_type == 'idf':
            vector1 = convert_to_idf_vector(query, collection)
            convert_method = lambda x: pickle.loads(x['idf'])
        else:
            vector1 = create_tf_idf_vector(query, collection)
            convert_method = lambda x: create_tf_idf_vector(x, collection, is_document=True)
            pass

    # objectIds = []
    scores = []
    # step 2: calculate similarity with collection vectors
    for doc in collection.find({}):
        scores.append((doc['sequence'], method(vector1, convert_method(doc))))

    if callback is not None:
        callback(scores)
    elif return_dict is not None:
        return_dict[method.__name__] = scores
    else:
        return scores


def create_search_threads(methods_to_execute, query, vector_type,  collection, on_search_done=None, on_wf_done=None):
    m = Manager()
    return_dict = m.dict()
    wagner_dict = m.dict()
    jobs = []


    for m in methods_to_execute:

        p = Process(target=search_collection, args=(query, vector_type, collection, m, return_dict))
        jobs.append(p)
        p.start()

    s = time.time()
    for j in jobs:
        j.join()
    print(str(time.time() - s))

    final_pd = pd.DataFrame()

    for k in return_dict.keys():
        final_pd = final_pd.append([{x[0]: x[1] for x in return_dict[k]}], ignore_index=True)

    avg = final_pd.mean(axis=0)

    final_results = list(zip(avg.index, avg))
    print('score time: ' + str(time.time() - s))

    if on_search_done is not None:
        on_search_done(final_results)

    if wf_score in methods_to_execute:
        p = Process(target=search_collection, args=(query, vector_type, collection, wf_score, wagner_dict))
        p.start()
        p.join()
        on_wf_done(wagner_dict['wf_score'])
