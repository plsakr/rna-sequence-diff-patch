from StringEditDistance import wagnerFisher
from matplotlib import pyplot as plt
from IRMethods import *
import random
import time


def random_nucleotide(length: int):

    nucleotides = ['A', 'G', 'C', 'U', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N']

    return "".join(random.choices(nucleotides, k=length))

time_wagner_fischer = []
time_convert_set = []
time_convert_multi_set = []
time_convert_vector = []
time_intersection_set = []
time_jaccard_set = []
time_dice_set = []
time_intersection_multiset = []
time_jaccard_multiset = []
time_dice_multiset = []
time_cosine = []
time_pearson = []
time_euclidian = []
time_manhattan = []
time_tanimoto = []
time_dice = []
# time_paths = []
# time_es = []
# time_patching = []
# time_reversing = []



for i in range(10, 260, 10):
    seq1 = random_nucleotide(i)
    seq2 = random_nucleotide(i)
    print(f'--------------- SEQUENCES OF LENGTH {i} -----------')
    print('Seq1:', seq1)
    print('Seq2:', seq2)

    start_time = time.time()
    dp = wagnerFisher(seq1, seq2)
    end_time = time.time()
    time_wagner_fischer.append((end_time - start_time)*1000)
    print('WF Done')


    # start_time = time.time()
    # set1 = convert_to_set(seq1)
    # set2 = convert_to_set(seq2)
    # end_time = time.time()
    # time_convert_set.append((end_time - start_time)*1000)
    # print('Set Convert Done')

    # start_time = time.time()
    # multi1 = convert_to_multi_set(seq1)
    # multi2 = convert_to_multi_set(seq2)
    # end_time = time.time()
    # time_convert_multi_set.append((end_time - start_time)*1000)
    # print('MultiSet Convert Done')

    #
    start_time = time.time()
    vec1 = convert_to_tf_vector(seq1)
    vec2 = convert_to_tf_vector(seq2)
    end_time = time.time()
    time_convert_vector.append((end_time - start_time)*1000)
    print('Vector Convert Done')


    # start_time = time.time()
    # set_intersection_similarity(set1, set2)
    # end_time = time.time()
    # time_intersection_set.append((end_time - start_time)*1000)
    # print('Intersection Done')
    #
    # start_time = time.time()
    # set_jaccard_similarity(set1, set2)
    # end_time = time.time()
    # time_jaccard_set.append((end_time - start_time)*1000)
    # print('Jaccard Done')
    #
    # start_time = time.time()
    # set_dice_similarity(set1, set2)
    # end_time = time.time()
    # time_dice_set.append((end_time - start_time)*1000)
    # print('Dice Done')

    # start_time = time.time()
    # multi_intersection_similarity(multi1, multi2)
    # end_time = time.time()
    # time_intersection_multiset.append((end_time - start_time)*1000)
    # print('Intersection Done')
    #
    # start_time = time.time()
    # multi_jaccard_similarity(multi1, multi2)
    # end_time = time.time()
    # time_jaccard_multiset.append((end_time - start_time)*1000)
    # print('Jaccard Done')
    #
    # start_time = time.time()
    # multi_dice_similarity(multi1, multi2)
    # end_time = time.time()
    # time_dice_multiset.append((end_time - start_time)*1000)
    # print('Dice Done')

    start_time = time.time()
    cosine(vec1, vec2)
    end_time = time.time()
    time_cosine.append((end_time - start_time)*1000)
    print('Cosine Done')

    start_time = time.time()
    pearson(vec1, vec2)
    end_time = time.time()
    time_pearson.append((end_time - start_time)*1000)
    print('Pearson Done')

    start_time = time.time()
    euclidian_distance(vec1, vec2)
    end_time = time.time()
    time_euclidian.append((end_time - start_time)*1000)
    print('Euclidian Done')

    start_time = time.time()
    manhattan_distance(vec1, vec2)
    end_time = time.time()
    time_manhattan.append((end_time - start_time)*1000)
    print('Manhattan Done')

    start_time = time.time()
    tanimoto_distance(vec1, vec2)
    end_time = time.time()
    time_tanimoto.append((end_time - start_time)*1000)
    print('Tanimoto Done')

    start_time = time.time()
    dice_dist(vec1, vec2)
    end_time = time.time()
    time_dice.append((end_time - start_time)*1000)
    print('Dice Done')

    # start_time = time.time()
    # paths = create_paths(dp)
    # end_time = time.time()
    # time_paths.append((end_time - start_time)*1000)
    # print('Create Paths Done')
    #
    # start_time = time.time()
    # for p in paths:
    #     editScripts = generate_es(p, seq1, seq2)
    # end_time = time.time()
    # time_es.append((end_time - start_time)*1000)
    # print('Generate ES Done')
    #
    # start_time = time.time()
    # patched = patching(editScripts, seq1)
    #
    # if patched != seq2:
    #     print('SOMETHING AWFUL HAPPENED!!!!!!')
    #
    # end_time = time.time()
    # time_patching.append((end_time-start_time)*1000)
    # print('Patching Done')
    #
    # start_time = time.time()
    # reversed = generate_rev_es(editScripts)
    # end_time = time.time()
    # time_reversing.append((end_time-start_time)*1000)
    # print('Reversing Done')
    

# print(time_wagner_fischer)
# print(time_convert_set)
# print(time_convert_multi_set)
# print(time_convert_vector)
x_axis = range(10, 260, 10)
#
# # plt.plot(x_axis, time_wagner_fischer, 'r')
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# ax.set(title='IR Conversion', xlabel='Sequence Length', ylabel='Time (ms)')
# plt.plot(x_axis, time_convert_set, color='g', label='Set Conversion')
# plt.plot(x_axis, time_convert_multi_set, color='b', label='MultiSet Conversion')
# plt.plot(x_axis, time_convert_vector, color='y', label='Vector Conversion')
# plt.legend()
# plt.grid()
# plt.savefig('./ir_conversion_time.png')
# plt.show()
# plt.plot(x_axis, time_wagner_fischer)
# print(time_paths)
# print(time_es)
# print(time_patching)
# print(time_reversing)

# # plt.plot(x_axis, time_wagner_fischer, 'r')
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111)
# ax.set(title='Multiset Methods', xlabel='Sequence Length', ylabel='Time (ms)')
# plt.plot(x_axis, time_intersection_multiset, color='g', label='Intersection')
# plt.plot(x_axis, time_jaccard_multiset, color='b', label='Jaccard')
# plt.plot(x_axis, time_dice_multiset, color='y', label='Dice')
# plt.legend()
# plt.grid()
# plt.savefig('./multiset_methods.png')
# plt.show()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set(title='Vector Methods', xlabel='Sequence Length', ylabel='Time (ms)')
plt.plot(x_axis, time_cosine, color='g', label='Cosine')
plt.plot(x_axis, time_pearson, color='b', label='Pearson')
plt.plot(x_axis, time_manhattan, color='y', label='Manhattan')
plt.plot(x_axis, time_euclidian, color='c', label='Euclidean')
plt.plot(x_axis, time_tanimoto, color='m', label='Tanimoto')
plt.plot(x_axis, time_dice, color='r', label='Dice')
plt.legend()
plt.grid()
plt.savefig('./vector_methods.png')
plt.show()
