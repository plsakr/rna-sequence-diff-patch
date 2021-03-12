import numpy as np


class Node:
    def __init__(self, value=None, operation=None, source=None, destination=None):
        if destination is None:
            destination = [None, None, None]
        if source is None:
            source = [None, None, None]
        if operation is None:
            operation = [None, None, None]
        self.value = value
        self.previous = [None, None, None]
        self.operation = operation
        self.source = source
        self.destination = destination

def cost(char1, char2):
    cost_insert = 1
    cost_delete = 1
    if char1.lower() == char2.lower():
        cost_update = 0
    else:
        cost_update = 1
    return cost_update

def min_cost(dp, i, j, str1, str2):
    cost_insert = dp[i][j-1].value + 1
    cost_delete = dp[i-1][j].value + 1
    cost_update = dp[i-1][j-1].value + cost(str1[i-1], str2[j-1])


    costs = [cost_insert, cost_delete, cost_update]


    index = costs.index(min(costs))
    val = min(costs)
    res_list = [i for i, value in enumerate(costs) if value == val]
    op_list = [None, None, None]
    if 0 in res_list:
        # we insert
        op_list[0] = (val, i, j-1, 'insert')
    if 1 in res_list:
        # we delete
        op_list[1] = (val, i-1, j, 'delete')
    if 2 in res_list:
        # we update
        op_list[2] = (val, i-1, j-1, 'update')

    return op_list


# def get_all_edit_scripts(dp, visited, path, current_n = None, current_i=None, current_j=None):
#     if current_i is None:
#         current_i = len(dp)-1
#     if current_j is None:
#         current_j = len(dp[0])-1
#
#     visited[current_i][current_j] = True
#
#     if current_n is None:
#         current_n = dp[current_i][current_j]
#
#     if current_n == dp[0][0]:
#         return path
#
#     for i in current_n.previous:
#         if i is not None:
#             new_i



def create_rev_es(dp, str1, str2,current_n = None, current_es=None):
    if current_es is None:
        current_es = []
    if current_n is None:
        current_n = dp[len(dp)-1][len(dp[0])-1]

    if current_n.previous == [None, None, None]:
        current_es.reverse()
        return current_es

    optimal_rev_path = current_es

    if current_n.operation[0] == 'insert':
        optimal_rev_path.append((current_n.operation[0], current_n.source[0], current_n.destination[0]))
        return create_rev_es(dp, str1, str2, current_n.previous[0], optimal_rev_path)

    if current_n.operation[1] == 'delete':
        optimal_rev_path.append((current_n.operation[1], current_n.source[1], current_n.destination[1]))
        return create_rev_es(dp, str1, str2, current_n.previous[1], optimal_rev_path)

    if current_n.operation[2] == 'update':
        if str1[current_n.source[2]].lower() != str2[current_n.destination[2]].lower():
            optimal_rev_path.append((current_n.operation[2], current_n.source[2], current_n.destination[2]))
        return create_rev_es(dp, str1, str2, current_n.previous[2], optimal_rev_path)


def wagnerFisher(str1, str2):
    rows = len(str1)+1
    cols = len(str2) + 1

    dp = [[Node(0) for x in range(cols)] for x in range(rows)]

    for j in range(cols):
        if j == 0:
            n = Node(0, None, None, None)
            dp[0][j] = n
        else:
            n = Node(j, ['insert', None, None], None, [j-1, None, None])
            n.previous[0] = dp[0][j-1]
            dp[0][j] = n

    for i in range(rows):
        if i == 0:
            pass
        else:
            n = Node(i, [None, 'delete', None], [None, i-1, None], None)
            n.previous[1] = dp[i-1][0]
            dp[i][0] = n

    for i in range(1, rows):
        for j in range(1, cols):
            operations = min_cost(dp, i, j, str1, str2)

            previouses = [None, None, None]
            ops = [None, None, None]
            sources = [None, None, None]
            dests = [None, None, None]

            v = 0
            for my_tuple in operations:
                if my_tuple is not None:
                    val, prev_i, prev_j, operation = my_tuple
                    v = val
                    if operation == 'delete':
                        previouses[1] = dp[prev_i][prev_j]
                        ops[1] = 'delete'
                        sources[1] = i-1

                    elif operation == 'insert':
                        previouses[0] = dp[prev_i][prev_j]
                        ops[0] = 'insert'
                        dests[0] = j-1
                    else:
                        previouses[2] = dp[prev_i][prev_j]
                        ops[2] = 'update'
                        sources[2] = i-1
                        dests[2] = j-1

            n = Node(v, ops, sources, dests)
            n.previous = previouses
            dp[i][j] = n


    return dp


def patching(es, str1, str2):
    count_index_destination=0
    count_index_source=0
    modified_str1 = str1
    for i in range(len(es)):
        current_sequence = es[i]
        # print("Currently patching: ", current_sequence)
        current_operation = current_sequence[0]
        original_source_index = current_sequence[1]
        original_destination_index = current_sequence[2]

        # print(i)
        # print(original_source_index)

        if not current_operation == 'insert':
            original_source_index += count_index_source + count_index_destination
        else:
            original_source_index = original_destination_index
        # print(original_source_index)

        if current_operation == 'update':
            dest_char = str2[original_destination_index]
            modified_str1 = modified_str1[:original_source_index] + dest_char + modified_str1[original_source_index+1 :]

        if current_operation == 'delete':
            modified_str1 = modified_str1[0 : original_source_index : ] + modified_str1[original_source_index + 1 : :]
            count_index_source -=1
        if current_operation == 'insert':
            current_destination_char = str2[original_destination_index]
            modified_str1 = modified_str1[:original_source_index] + current_destination_char + modified_str1[original_source_index:]
            count_index_destination +=1


    return modified_str1

str1 = 'AGGCT'
str2 = 'GAGATAT'
dp = wagnerFisher(str1, str2)
es = create_rev_es(dp, str1, str2)
print('ED(A,B) = ', dp[len(dp)-1][len(dp[0])-1].value)
print('ES(A,B) = ', es)
print('Patched A: ', patching(es, str1, str2))

