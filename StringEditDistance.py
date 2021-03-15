from queue import Queue
import json

with open('costs.json', 'r') as f:
    default_costs = json.load(f)

try:
    with open('user_costs.json', 'r') as f:
        user_costs = json.load(f)
except (OSError, IOError) as e:
    user_costs = default_costs
    print('Could not find user costs file')


def reload_user_costs():
    global user_costs
    with open('user_costs.json', 'r') as f:
        user_costs = json.load(f)
class Edge:
    def __init__(self, source, destination, operation):
        self.source = source
        self.destination = destination
        self.operation = operation


class Node:
    def __init__(self, i, j, value=0):
        self.i = i
        self.j = j
        self.value = value
        self.edges = []
        self.incoming_edges = []
        self.visited = False

    def add_neighbor(self, dest, operation):
        e = Edge(self, dest, operation)
        self.edges.append(e)
        dest.incoming_edges.append(e)

    def __repr__(self):
        return str(self.value)


def cost(char1, char2, userCosts=False):
    global default_costs, user_costs

    if char1.lower() == char2.lower():
        cost_update = 0
    else:
        cost_update = default_costs['update'][char1][char2] if not userCosts else user_costs['update'][char1][char2]
    return cost_update


def min_cost(dp, i, j, str1, str2, userCosts=False):
    global default_costs, user_costs
    cost_insert = dp[i][j - 1].value + default_costs['insert'] if not userCosts else dp[i][j - 1].value + user_costs['insert']
    cost_delete = dp[i - 1][j].value + default_costs['delete'] if not userCosts else dp[i - 1][j].value + user_costs['delete']
    cost_update = dp[i - 1][j - 1].value + cost(str1[i - 1], str2[j - 1], userCosts)

    costs = [cost_insert, cost_delete, cost_update]

    index = costs.index(min(costs))
    val = min(costs)
    res_list = [i for i, value in enumerate(costs) if value == val]
    op_list = [None, None, None]
    if 0 in res_list:
        # we insert
        op_list[0] = (i, j - 1, 'insert')
    if 1 in res_list:
        # we delete
        op_list[1] = (i - 1, j, 'delete')
    if 2 in res_list:
        # we update
        op_list[2] = (i - 1, j - 1, 'update')

    return val, op_list


def wagnerFisher(str1, str2, userCosts=False):
    global user_costs, default_costs
    rows = len(str1) + 1
    cols = len(str2) + 1

    dp = [[Node(-1, -1, 0) for x in range(cols)] for x in range(rows)]

    for j in range(cols):
        if j == 0:
            n = Node(-1, -1, 0)
            dp[0][j] = n
        else:
            i_cost = default_costs['insert'] if not userCosts else user_costs['insert']
            n = Node(-1, j - 1, j*i_cost)
            n_previous = dp[0][j - 1]
            n_previous.add_neighbor(n, 'insert')
            dp[0][j] = n

    for i in range(rows):
        if i == 0:
            pass
        else:
            d_cost = default_costs['delete'] if not userCosts else user_costs['delete']
            n = Node(i - 1, -1, i*d_cost)
            n_previous = dp[i - 1][0]
            n_previous.add_neighbor(n, 'delete')
            dp[i][0] = n

    for i in range(1, rows):
        for j in range(1, cols):
            val, operations = min_cost(dp, i, j, str1, str2, userCosts)

            current_node = Node(i - 1, j - 1, val)

            for my_tuple in operations:
                if my_tuple is not None:
                    prev_i, prev_j, operation = my_tuple

                    if operation == 'delete':
                        previous_n = dp[prev_i][prev_j]
                        previous_n.add_neighbor(current_node, 'delete')

                    elif operation == 'insert':
                        previous_n = dp[prev_i][prev_j]
                        previous_n.add_neighbor(current_node, 'insert')

                    else:
                        previous_n = dp[prev_i][prev_j]
                        previous_n.add_neighbor(current_node, 'update')

            dp[i][j] = current_node

    return dp


# def create_paths(dp):
#     src = dp[0][0]
#     goal = dp[len(dp) - 1][len(dp[0]) - 1]
#
#     src.visited = True
#     q = Queue(maxsize=len(dp) * len(dp[0]))
#     q.put([src])
#
#     final_paths = []
#     while not q.empty():
#         p = q.get()
#         if p[-1] == goal:
#             final_paths.append(p)
#
#         edges = p[-1].edges
#         for e in edges:
#             next_n = e.destination
#             if next_n not in p:
#                 new_p = p.copy()
#                 new_p.append(next_n)
#                 q.put(new_p)
#
#     return final_paths

def create_paths(dp):
    goal = dp[0][0]
    src = dp[len(dp) - 1][len(dp[0]) - 1]

    src.visited = True
    q = Queue(maxsize=len(dp) * len(dp[0]))
    q.put([src])

    final_paths = []
    while not q.empty():
        p = q.get()
        if p[-1] == goal:
            final_paths.append(p)

        edges = p[-1].incoming_edges
        for e in edges:
            next_n = e.source
            if next_n not in p:
                new_p = p.copy()
                new_p.append(next_n)
                q.put(new_p)

    fixed_paths = []
    for i in final_paths:
        fixed_paths.append(i[::-1])
    return fixed_paths


def generate_es(path, str1, str2):
    path_index = 0
    current = path[path_index]
    next = path[path_index + 1]

    edit_script = []

    while len(next.edges) >= 0:
        path_index += 1
        current_edges = current.edges
        actual_edge = list(filter(lambda x: x.source == current and x.destination == next, current_edges))[0]

        if actual_edge.operation == 'update':
            if str1[next.i].lower() == str2[next.j].lower():
                pass
            else:
                edit_script.append(('update', next.i, next.j))
        elif actual_edge.operation == 'delete':
            edit_script.append((actual_edge.operation, next.i))
        else:
            edit_script.append((actual_edge.operation, next.i, next.j))

        current = path[path_index]
        if path_index == len(path) - 1:
            return edit_script
        next = path[path_index + 1]

    return edit_script


def generate_rev_es(es, str1, str2):
    new_es = []

    for e in es:
        operation = e[0]

        if operation == 'insert':
            new_operation = 'delete'
            new_source = e[2]
            new_dest = None
            pass
        elif operation == 'delete':
            new_operation = 'insert'
            new_source = e[1] - 1
            new_dest = e[1]
        elif operation == 'update':
            new_operation = 'update'
            new_source = e[2]
            new_dest = e[1]

        if new_dest is not None:
            new_es.append((new_operation, new_source, new_dest))
        else:
            new_es.append((new_operation, new_source))

    return new_es


def patching(es, str1, str2):
    count_index_destination = 0
    count_index_source = 0
    modified_str1 = str1
    for i in range(len(es)):
        current_sequence = es[i]
        # print("Currently patching: ", current_sequence)
        current_operation = current_sequence[0]
        original_source_index = current_sequence[1]
        original_destination_index = current_sequence[2] if current_operation != 'delete' else -1

        # print(i)
        # print(original_source_index)

        if not current_operation == 'insert':
            original_source_index += count_index_source + count_index_destination
        else:
            original_source_index = original_destination_index
        # print(original_source_index)

        if current_operation == 'update':
            dest_char = str2[original_destination_index]
            modified_str1 = modified_str1[:original_source_index] + dest_char + modified_str1[
                                                                                original_source_index + 1:]

        if current_operation == 'delete':
            modified_str1 = modified_str1[0: original_source_index:] + modified_str1[original_source_index + 1::]
            count_index_source -= 1
        if current_operation == 'insert':
            current_destination_char = str2[original_destination_index]
            modified_str1 = modified_str1[:original_source_index] + current_destination_char + modified_str1[
                                                                                               original_source_index:]
            count_index_destination += 1

    return modified_str1


str1 = 'AGRGA'
str2 = 'AGGGA'
dp = wagnerFisher(str1, str2, True)
print(dp)
all_paths = create_paths(dp)
for path in all_paths:
    es = generate_es(path, str1, str2)
    print(es)
    print(patching(es, str1, str2))
    # print('REVERSING')
    # rev = generate_rev_es(es, str1, str2)
    # print(rev)
    # print(patching(rev, str2, str1))
