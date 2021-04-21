import math
import numpy as np

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


def intersection(a, b):
    return a.intersection(b)


def set_intersection_similarity(a, b):
    return len(intersection(a, b))


def set_jaccard_similarity(a, b):
    return set_intersection_similarity(a, b) / len(a.union(b))


def set_dice_similarity(a, b):
    denom = len(a) + len(b)
    return 2 * set_intersection_similarity(a, b) / denom


def convert_to_multi_set(sequence):
    c = np.zeros(4)

    for l in sequence:
        if l in base_nucleotides:
            c[base_nucleotides.index(l)] += 1
        else:
            c = c + ambiguous_nucleotides[l]

    return c


def multi_intersection_similarity(ca, cb):
    sim = 0

    for k in (ca.keys() & cb.keys()):
        sim += min(ca[k], cb[k])

    return sim


def multi_jaccard_similarity(ca, cb):
    num = multi_intersection_similarity(ca, cb)

    den = sum(ca.values()) + sum(cb.values()) - num
    return num / den


def multi_dice_similarity(ca, cb):
    num = multi_intersection_similarity(ca, cb)

    den = sum(ca.values()) + sum(cb.values())
    return 2 * num / den


def convert_to_vector(seq):
    vec = np.zeros((4, 4))

    for i in range(len(seq) - 1):
        current = seq[i]
        cost_current = {'A': 0, 'G': 0, 'C': 0, 'U': 0}
        cost_next = {'A': 0, 'G': 0, 'C': 0, 'U': 0}

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


def cosine(a, b):
    num = np.sum(np.multiply(a, b))
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    den = math.sqrt(a_sq * b_sq)
    # print(num)
    # print(den)
    return num / den


def pearson(a, b):
    a_bar = np.average(a)
    b_bar = np.average(b)

    a_sub = np.subtract(a, a_bar)
    b_sub = np.subtract(b, b_bar)

    num = np.sum(np.multiply(a_sub, b_sub))

    a_sub_sq = np.sum(np.square(a_sub))
    b_sub_sq = np.sum(np.square(b_sub))

    den = math.sqrt(a_sub_sq * b_sub_sq)
    return num / den


def euclidian_distance(a, b):
    dist = math.sqrt(np.sum(np.square(np.subtract(a, b))))
    return 1 / (1 + dist)


def manhattan_distance(a, b):
    dist = math.sqrt(np.sum(np.abs(np.subtract(a, b))))
    return 1 / (1 + dist)


def tanimoto_distance(a, b):
    num = np.sum(np.multiply(a, b))

    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    den = a_sq + b_sq - num
    return num / den


def dice_dist(a, b):
    num = 2 * np.sum(np.multiply(a, b))
    a_sq = np.sum(np.square(a))
    b_sq = np.sum(np.square(b))

    den = a_sq + b_sq
    return num / den


# a = 'AACG'
b = 'NAN'

# a_vec = convert_to_vector(a)
b_vec = convert_to_vector(b)

print('\n', b_vec)