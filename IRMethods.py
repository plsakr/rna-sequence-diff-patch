import math
import numpy as np
from multiprocessing import Process, Manager
import os

nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']
base_nucleotides = ['A', 'G', 'C', 'U']

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


def convert_to_set(sequence):
    return set(sequence)


def intersection(a, b, return_dict=None):
    if return_dict is None:
        return a.intersection(b)
    else:
        return_dict['intersection'] = a.intersection(b)


def set_intersection_similarity(a, b, return_dict=None):
    if return_dict is None:
        return len(intersection(a, b))
    else:
        return_dict['set_intersection_sim'] = len(intersection(a, b))


def set_jaccard_similarity(a, b, return_dict=None):
    if return_dict is None:
        return set_intersection_similarity(a, b) / len(a.union(b))
    else:
        return_dict['set_jaccard_sim'] = set_intersection_similarity(a, b) / len(a.union(b))


def set_dice_similarity(a, b, return_dict=None):
    denom = len(a) + len(b)
    if return_dict is None:
        return 2 * set_intersection_similarity(a, b) / denom
    else:
        return_dict['set_dice_sim'] = 2 * set_intersection_similarity(a, b) / denom


def convert_to_multi_set(sequence):
    c = np.zeros(4)

    for l in sequence:
        if l in base_nucleotides:
            c[base_nucleotides.index(l)] += 1
        else:
            c = c + ambiguous_nucleotides[l]

    return c


def multi_intersection_similarity(ca, cb, return_dict=None):
    sim = 0

    for k in range(4):
        sim += min(ca[k], cb[k])
    if return_dict is None:
        return sim
    else:
        return_dict['multi_intersection_sim'] = sim


def multi_jaccard_similarity(ca, cb, return_dict=None):
    num = multi_intersection_similarity(ca, cb)

    den = np.sum(ca) + np.sum(cb) - num
    if return_dict is None:
        return num / den
    else:
        return_dict['multi_jaccard_sim'] = num/den


def multi_dice_similarity(ca, cb, return_dict=None):
    num = multi_intersection_similarity(ca, cb)

    den = np.sum(ca) + np.sum(cb)
    if return_dict is None:
        return 2 * num / den
    else:
        return_dict['multi_dice_sim'] = 2 * num/den


def convert_to_tf_vector(seq):
    vec = np.zeros((4, 4))

    for i in range(len(seq) - 1):
        current = seq[i]
        cost_current = {'A': 0, 'G': 0, 'C': 0, 'U': 0}

        if current in base_nucleotides:
            cost_current[current] = cost_current[current] + 1
        else:
            cost_current = {k: cost_current[k] + ambiguity_vectors[current][k] for k in set(cost_current)}

        next_char = seq[i + 1]
        if next_char in base_nucleotides:
            for k in cost_current.keys():
                vec[base_nucleotides.index(k)][base_nucleotides.index(next_char)] += cost_current[k]

        else:
            dest_costs = ambiguity_vectors[next_char]
            for k in cost_current.keys():
                for j in cost_current.keys():
                    actual_cost = cost_current[k] * dest_costs[j]
                    vec[base_nucleotides.index(k)][base_nucleotides.index(j)] += actual_cost

    return vec

# TODO: adapt to DB instead of only 2 sequences
def convert_to_idf_vector(seq1, seq2):
    vec = np.zeros((15,15))

    set1 = set()
    for i in range(len(seq1) - 1):
        current = seq1[i]
        next_char = seq1[i + 1]
        sequence = current + next_char
        set1.add(sequence)

    for pair in set1:
        cost = compare_pair_to_seq(pair, seq2)
        idf = 0 if cost == 0 else math.log(2/cost, 10)
        index1 = nucleotides.index(pair[0])
        index2 = nucleotides.index(pair[1])
        vec[index1][index2] = idf

    return vec




def compare_pair_to_seq(pair, seq):
    if pair in seq:
        return 1
    else:
        first_possibilities, first_probabilities = possibilities(pair[0])
        second_possibilities, second_probabilities = possibilities(pair[1])
        current_max = 0

        for pos1 in range(len(first_possibilities)):
            for pos2 in range(len(second_possibilities)):
                actual = first_possibilities[pos1] + second_possibilities[pos2]
                if actual in seq:
                    cost = first_probabilities[pos1] * second_probabilities[pos2]
                    if current_max < cost:
                        current_max = cost

        return current_max


def possibilities(nucleotide):
    if nucleotide in base_nucleotides:
        return get_base_possibilities(nucleotide)
    else:
        output = []
        complete_vector = ambiguity_vectors[nucleotide]
        for k in complete_vector.keys():
            if complete_vector[k] != 0:
                output.append(k)
        probability = [1/len(output) for i in output]
        probability.append(1)
        output.append(nucleotide)
        return output, probability


def get_base_possibilities(base):
    p, prob = [base], [1]

    for k in ambiguity_vectors.keys():
        if ambiguity_vectors[k][base] != 0:
            p.append(k)
            prob.append(ambiguity_vectors[k][base])
    return p, prob


def cosine(a, b, return_dict = None):
    num = np.sum(np.multiply(a, b))
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    den = math.sqrt(a_sq * b_sq)
    # print(num)
    # print(den)

    if return_dict is None:
        return num / den
    else:
        return_dict['cosine'] = num/den


def pearson(a, b, return_dict = None):
    a_bar = np.average(a)
    b_bar = np.average(b)

    a_sub = np.subtract(a, a_bar)
    b_sub = np.subtract(b, b_bar)

    num = np.sum(np.multiply(a_sub, b_sub))

    a_sub_sq = np.sum(np.square(a_sub))
    b_sub_sq = np.sum(np.square(b_sub))

    den = math.sqrt(a_sub_sq * b_sub_sq)
    if return_dict is None:
        return num / den
    else:
        return_dict['pearson'] = num/den


def euclidian_distance(a, b, return_dict = None):
    dist = math.sqrt(np.sum(np.square(np.subtract(a, b))))
    if return_dict is None:
        return 1 / (1 + dist)
    else:
        return_dict['euclidian_dist'] = 1 / (1 + dist)


def manhattan_distance(a, b, return_dict = None):
    dist = math.sqrt(np.sum(np.abs(np.subtract(a, b))))
    if return_dict is None:
        return 1 / (1 + dist)
    else:
        return_dict['manhattan_distance'] = 1 / (1 + dist)


def tanimoto_distance(a, b, return_dict = None):
    num = np.sum(np.multiply(a, b))

    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    den = a_sq + b_sq - num
    if return_dict is None:
        return num / den
    else:
        return_dict['tanimoto_dist'] = num/den


def dice_dist(a, b, return_dict = None):
    num = 2 * np.sum(np.multiply(a, b))
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    den = a_sq + b_sq
    if return_dict is None:
        return num / den
    else:
        return_dict['dice_dist'] = num/den


def create_and_start_threads(methods_to_execute, a, b):
    result_objects = []
    m = Manager()
    return_dict = m.dict()
    jobs = []

    print('ma fetet ba3d')

    for m in methods_to_execute:
        print('ana hon v2')
        p = Process(target=m, args=(a, b, return_dict))
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()

    return return_dict

def perform_methods(a, b, do_cosine=False, do_pearson=False, do_euclidian_distance=False, do_manhattan_distance=False,
                    do_tanimoto_distance=False, do_dice_dist=False):
    jobs = []
    if do_cosine: jobs.append(cosine)
    if do_pearson: jobs.append(pearson)
    if do_euclidian_distance: jobs.append(euclidian_distance)
    if do_manhattan_distance: jobs.append(manhattan_distance)
    if do_tanimoto_distance: jobs.append(tanimoto_distance)
    if do_dice_dist: jobs.append(dice_dist)
    return create_and_start_threads(jobs, a, b)

a = 'AACG'
b = 'NAN'

a_vec = convert_to_idf_vector(a, b)
b_vec = convert_to_idf_vector(b, a)

print('a',a_vec)
print('b',b_vec)

print('starting threads')

if __name__ == '__main__':
    thread_test = create_and_start_threads([cosine, pearson, euclidian_distance, manhattan_distance, tanimoto_distance, dice_dist], a_vec, b_vec)
    print('BONJOUR')
    print(thread_test)