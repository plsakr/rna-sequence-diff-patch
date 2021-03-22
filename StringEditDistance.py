from queue import Queue
import json

# Step 1: load the default cost function from the costs.json file.
# Our cost metric will be thoroughly explained in the demo/presentation.
with open('costs.json', 'r') as f:
    default_costs = json.load(f)

# If the user has selected the option of having his own custom cost function
# load the file and save it in the user_costs variable
# otherwise, user_costs is the default cost function .
try:
    with open('user_costs.json', 'r') as f:
        user_costs = json.load(f)
except (OSError, IOError) as e:
    user_costs = default_costs
    #If user_costs not found, use our default cost function that was inputted during the previous stage.
    print('Could not find user costs file')

# Reloading user_costs/default_costs for the gui:
# when the user leaves and re-selects the custom cost function option he shall be able to view his previously inputted
# cost function instead of starting from scratch.
# He will also be able to edit it again.
def reload_user_costs():
    global user_costs
    with open('user_costs.json', 'r') as f:
        user_costs = json.load(f)

# For this project, we have adopted a graph-based measure where all the nodes that lead to each other are connected via an edge.
# To do so, we must define an Edge class and a Node class.
class Edge:
    # An edge has a source, destination and a specific operation that explains how we got from this source to that
    # destination. An operation could be: Inserting, Deleting or Updating.
    def __init__(self, source, destination, operation):
        # The Edge class has only a constructor which will be used to set the corresponding values
        # and retrieve them later on for the path extraction phase.
        self.source = source
        self.destination = destination
        self.operation = operation


class Node:
    # A node has 6 different identifiers: an i index (along the rows), a j index (along the columns),
    # a value (the cost of reaching the node at Dist[i][j]): Default value is 0,
    # a list of edges: where can this node lead us to
    # a list of incoming edges (the possible nodes that could lead to this cell)
    # a boolean indicating whether this node has been visited or not (to be used to get the optimal path later on).
    def __init__(self, i, j, value=0):
        self.i = i
        self.j = j
        self.value = value
        self.edges = []
        self.incoming_edges = []
        self.visited = False

    def add_neighbor(self, dest, operation):
        # This function links a node to its neighbor along with the corresponding operation that allows us to go from
        # the source which is the current node to reach the destination which is our neighbor.

        # to link a node to another, we must first define an edge: where the source is the current node
        # the destination is the one found: dest and the operation is the current operation that leads us there.
        e = Edge(self, dest, operation)
        # then, this edge should be appended to the list of edges of the source node (current node)
        # and to the incoming_edges list of the destination node since it will be leaving the current source node
        # to feed into the destination node.
        self.edges.append(e)
        dest.incoming_edges.append(e)

    def __repr__(self):
        # This method is used to return the cost at a given node
        return str(self.value)

# This is the cost function used to determine the cost of the update operation by referring to the user cost function
# that was previously processed and loaded into the user_costs variable (and where our default costs are saved
# if the user did not choose a custom cost function)
def cost(char1, char2, userCosts=False):
    global default_costs, user_costs

    if char1.lower() == char2.lower():
        #if characters are identical, cost of updating is 0
        cost_update = 0
    else:
        # otherwise, we must get the cost of the update operation from the correct file
        # if the user has chosen a custom cost function, we will get the cost from user_costs dictionary
        # at the key 'update' for source = char1 and dest = char2.
        # If no custom cost function has been inputted, we will do the same steps by using our default_costs dictionary
        cost_update = default_costs['update'][char1][char2] if not userCosts else user_costs['update'][char1][char2]
        # last, return the cost of updating source character char 1 into destination character char 2
    return cost_update

# This function will be used to generate the operation that will lead to the minimum cost.
def min_cost(dp, i, j, str1, str2, userCosts=False):
    global default_costs, user_costs
    # Calculate the cost of inserting by referring to the user_costs dictionary (if valid) otherwise to the default_costs.
    cost_insert = dp[i][j - 1].value + default_costs['insert'] if not userCosts else dp[i][j - 1].value + user_costs['insert']
    # Calculate the cost of deleting by referring to the user_costs dictionary (if valid) otherwise to the default_costs.
    cost_delete = dp[i - 1][j].value + default_costs['delete'] if not userCosts else dp[i - 1][j].value + user_costs['delete']
    # Calculate the cost of updating by referring to previously created cost function.
    cost_update = dp[i - 1][j - 1].value + cost(str1[i - 1], str2[j - 1], userCosts)

    # Store the 3 generated costs in a costs array such as: cost of inserting will always be at index 0,
    # deleting at index 1 and updating at index 2.
    costs = [cost_insert, cost_delete, cost_update]

    # Get the index which has the minimum cost as well as its value
    index = costs.index(min(costs))
    val = min(costs)
    # If we have more than 1 minimal cost (2 or 3 operations lead to the same cost)
    res_list = [i for i, value in enumerate(costs) if value == val]

    #Create an op_list: the maximum number of optimal operations that could take us from a source A to a destination B
    # is 3. By default, they are all null.
    # For the min. index, find the corresponding index and append it (following the same structure/organization):
    # inserting at 0, deleting at 1 and updating at 2.
    op_list = [None, None, None]
    if 0 in res_list:
        # If the min. cost at index i is 0, we insert.
        op_list[0] = (i, j - 1, 'insert')
    if 1 in res_list:
        # If the min. cost at index i is 1, we delete.
        op_list[1] = (i - 1, j, 'delete')
    if 2 in res_list:
        # If the min. cost at index 2 is at 2, we update.
        op_list[2] = (i - 1, j - 1, 'update')
        # Note that for more than one minimal cost, we will have more than 1 elt that is different than Null in the op_list.

    # return the minimum cost that was stored in val as well as the array of optimal operations: op_list.
    return val, op_list


# This function will be used to determine the wagnerFisher cost by filling out the edit_dist matrix.
# Note that wagnerFisher was used since RNA sequences can be viewed as simple strings.
def wagnerFisher(str1, str2, userCosts=False):
    #For this function, we assume that str1 represents the source RNA sequence and str2 is for the destination RNA seq.
    global user_costs, default_costs
    # Determine the number of rows and columns for the edit distance.
    # Nbr of rows is equal to the number of characters in the source (str1) + 1.
    rows = len(str1) + 1
    # Nbr of columns is equal to the number of characters in the destination (str2) + 1.
    cols = len(str2) + 1

    # Initialize a matrix of nodes having the previously calculated dimensions.
    dp = [[Node(-1, -1, 0) for x in range(cols)] for x in range(rows)]

    # For the first row (at index i = 0):
    for j in range(cols):
        if j == 0:
            # The first node (at source index = -1 and destination index j = -1) will always start with a value of 0
            # since it doesn't represent any character of the given sequences.
            n = Node(-1, -1, 0)
            dp[0][j] = n
        else:
            # Moving horizontally (left->right) corresponds to inserting: get the cost of the insert
            # operation and fill it out.
            # Start by getting the corresponding cost for the insert operation: either from user_costs or default_costs.
            i_cost = default_costs['insert'] if not userCosts else user_costs['insert']
            #Create a node that has a row index i = -1 (1st row), col index is j-1
            # with cost= column index * cost of inserting that was calculated previously
            n = Node(-1, j - 1, j*i_cost)
            # Previous node is the one directly above it: at dp[i][j-1] where i = 0
            n_previous = dp[0][j - 1]
            # Link both nodes by making them neighbors achieved using the insert operation.
            n_previous.add_neighbor(n, 'insert')
            dp[0][j] = n

    # For the first column (at index j = 0):
    for i in range(rows):
        if i == 0:
            # The first node (at source index = -1 and destination index j = -1) has been given a value previously.
            pass
        else:
            # Moving vertically corresponds to deleting: get the cost of the delete operation and fill it out.
            # Start by getting the corresponding cost for the delete operation: either from user_costs or default_costs.
            d_cost = default_costs['delete'] if not userCosts else user_costs['delete']
            #Create a node that has a row index i = i-1, col index is -1 (1st col)
            # with cost= row index * cost of deleting that was calculated previously
            n = Node(i - 1, -1, i*d_cost)
            # Previous node is the one directly to the left: at dp[i-1][j] where j = 0
            n_previous = dp[i - 1][0]
            # Link both nodes by making them neighbors achieved using the delete operation.
            n_previous.add_neighbor(n, 'delete')
            dp[i][0] = n

    # As for the remaining cells, we need to get the min_cost using the previously created min_cost function.
    for i in range(1, rows):
        for j in range(1, cols):
            val, operations = min_cost(dp, i, j, str1, str2, userCosts)
            # Create a current node that has rows index=i-1, cols index=j-1 and cost=val.
            current_node = Node(i - 1, j - 1, val)

            # Looping through the multiple optimal operations that are not None and stored in operations:
            for my_tuple in operations:
                if my_tuple is not None:
                    # If this tuple is not None, then this is an operation that leads to a minimal cost.
                    # Link the previous indices i and j as well as the operation to the ones specified in this tuple.
                    prev_i, prev_j, operation = my_tuple

                    # If the specified tuple corresponds to a delete operation:
                    if operation == 'delete':
                        # The previous node will be at the indices generated from my_tuple.
                        previous_n = dp[prev_i][prev_j]
                        # Link the previous node to the current node by making it a neighbor of the previous
                        # node linked by a delete operation.
                        previous_n.add_neighbor(current_node, 'delete')

                    # If the specified tuple corresponds to an insert operation:
                    elif operation == 'insert':
                        # The previous node will be at the indices generated from my_tuple.
                        previous_n = dp[prev_i][prev_j]
                        # Link the previous node to the current node by making it a neighbor of the previous
                        # node linked by an insert operation.
                        previous_n.add_neighbor(current_node, 'insert')

                    # Otherwise, this will correspond to an update operation
                    else:
                        # The previous node will be at the indices generated from my_tuple.
                        previous_n = dp[prev_i][prev_j]
                        # Link the previous node to the current node by making it a neighbor of the previous
                        # node linked by an update operation.
                        previous_n.add_neighbor(current_node, 'update')

            dp[i][j] = current_node

    return dp


# This method will be used to generate the different paths available that will help us reach the final node.
def create_paths(dp):
    # Define first the target to be the 1st node at location [0][0]: 1st row and 1st col.
    goal = dp[0][0]
    # Going backwards: our source node is the last node in the matrix at loc [len(dp)-1][len(dp[0])-1].
    src = dp[len(dp) - 1][len(dp[0]) - 1]

    # Set the visited attribute of source to True (since we are currently dealing with it).
    src.visited = True
    # Generate a queue of paths having a maximum size of nb of rows of the edit distance matrix * nb of cols = max nb of paths.
    q = Queue(maxsize=len(dp) * len(dp[0]))
    # Add the src (current node) to the queue.
    q.put([src])

    # Define an array to hold the path.
    final_paths = []
    # As long as the queue is not empty:
    while not q.empty():
        # get the generated path (so far: it is not complete yet).
        p = q.get()
        # check if the last node in the path (at index -1) is the target/goal node (the one at dp[0][0]):
        if p[-1] == goal:
            # If we have reached our destination, append it to the final_paths.
            final_paths.append(p)

        # Else, get all the edges that feed into the current node (incoming edges).
        edges = p[-1].incoming_edges
        # Loop through each edge part of the incoming edges.
        for e in edges:
            # define the next node to be the edge's source.
            next_n = e.source
            # If this node (edge's source) is not in the currently-generated path:
            if next_n not in p:
                # create a new path copy of the previous one and append to it this node.
                # We didn't use the same path because we could have multiple operations that could lead to the same cost.
                new_p = p.copy()
                new_p.append(next_n)
                # Add this path to the queue.
                q.put(new_p)

    fixed_paths = []
    for i in final_paths:
        # add all the found optimal paths to the fixed_paths array.
        fixed_paths.append(i[::-1])
    return fixed_paths

# This method will be used to generate the edit script.
def generate_es(path, str1, str2):
    # For a given path, start at index 0 and get the current and next nodes.
    path_index = 0
    current = path[path_index]
    next = path[path_index + 1]

    edit_script = []

    # The last node has 0 edges. So, as long as there are edges, do the following:
    while len(next.edges) >= 0:
        # increment the path index
        path_index += 1
        # get the current node's edges.
        current_edges = current.edges
        # Filter out the unneeded edges such as the edge should have a source equal to the current node,
        # a destination equal to the subsequent node in the generated path.
        actual_edge = list(filter(lambda x: x.source == current and x.destination == next, current_edges))[0]

        # Based on the selected edge, get the operation that links the source and destination together
        # (the operation attribute is associated with the edge).
        # If the operation corresponds to an update operation check the following:
        if actual_edge.operation == 'update':
            # If the characters of the source at the corresponding indices (next_i and next_j) are equal:
            # no need to add a meaningless operation that has no cost to the ES: pass.
            if str1[next.i].lower() == str2[next.j].lower():
                pass
            else:
                # Else: append to the ES array the tuple that contains: this is an update operation:
                # updating srcChar with destChar.
                edit_script.append(('update', (next.i, str1[next.i]), (next.j, str2[next.j])))
        # If the operation corresponds to a delete operation:
        elif actual_edge.operation == 'delete':
            # append to the ES array the tuple that contains: this is a delete operation:
            # deleting srcChar at index i.
            edit_script.append((actual_edge.operation, (next.i, str1[next.i])))
        # Otherwise:the operation corresponds to an insert operation:
        else:
            # append to the ES array the tuple that contains: this is an insert operation:
            # inserting destChar at index j.
            edit_script.append((actual_edge.operation, next.i, (next.j, str2[next.j])))

        # get the next node in the path to loop through the different nodes until reaching the target node.
        current = path[path_index]
        # If we have reached the last node, return the edit script (next node will not be defined).
        if path_index == len(path) - 1:
            return edit_script
        # Otherwise, get the next node and repeat the process
        next = path[path_index + 1]

    return edit_script


# In case we wish to reverse the ES to get str1 from str2, we will use the following function.
def generate_rev_es(es, str1_old, str2_old):
    new_es = []

    # For every possible edit script, start by getting the 1st operation.
    for e in es:
        operation = e[0]

        # If it is an insert operation: it will become a delete operation
        if operation == 'insert':
            new_operation = 'delete'
            new_source = e[2]
            # For a delete operation, destination is None.
            new_dest = None
            pass
        # If it is a delete operation: it will become an insert operation: deleting the char that we were inserting in
        # the prev ES but now we need the index at which we want to delete.
        elif operation == 'delete':
            new_operation = 'insert'
            new_source = e[1][0] - 1
            new_dest = e[1]
        # If it is an update operation: it will remain an update operation but with reversed characters.
        elif operation == 'update':
            new_operation = 'update'
            # source character becomes the prev destination character.
            new_source = e[2]
            # destination character becomes the prev source character.
            new_dest = e[1]

        if new_dest is not None:
            # for insert and update operations: append the following tuple (with this format) to the ES:
            new_es.append((new_operation, new_source, new_dest))
        else:
            # for delete operations: append the following tuple (with this format) to the ES:
            new_es.append((new_operation, new_source))
    # return the generated reversed ES.
    return new_es

# Patch str1 and transform it to become homomorphic to str2 using the following method:
def patching(es, str1):
    # 2 counter indices are needed: one for the destination and one for the source.
    # Destination is needed to know what char we are inserting/ with what we are updating.. at which index
    count_index_destination = 0
    # Source index is needed because we must know how many char. we have added/deleted from the source string
    # which would affect its length.
    count_index_source = 0
    modified_str1 = str1

    # Loop through the different operations available in a given ES:
    for i in range(len(es)):
        # Select the current operation sequence at index i in the ES.
        current_sequence = es[i]
        # print("Currently patching: ", current_sequence)
        # Extract the operation from the previously extracted sequence:
        # according to our ES format, the operation is always at index 0 in the sequence tuple.
        current_operation = current_sequence[0]

        # get the current indices of source and destination
        # (based on the operation: ES has a different format for delete and insert).
        original_source_index = current_sequence[1] if current_operation == 'insert' else current_sequence[1][0]
        original_destination_index = current_sequence[2][0] if current_operation != 'delete' else -1

        # print(i)
        # print(original_source_index)

        # If the current operation is either a delete or an update, change the source index accordingly by adding
        # the counters that would signify whether we added or deleted characters: if i am deleting/updating a character
        # that was originally at index i, in the modified string, it will be at index i + nbr of delete operations done
        # so far + nbr of insert operations done so far.
        if not current_operation == 'insert':
            original_source_index += count_index_source + count_index_destination
        else:
            # If this is an insert operation, i am currently dealing with the same index as the one in the destination
            # sequence. I am making A homomorphic to B: I need to add char x available in B at index i to be in A at index i.
            original_source_index = original_destination_index
        # print(original_source_index)

        # If this is an update operation: modify the characters
        if current_operation == 'update':
            # Start by getting the corresponding character from the destination by referring to the ES
            # (and the way we formatted it).
            dest_char = current_sequence[2][1]
            # Update the character which will be at index original_source_index in the new modified string
            modified_str1 = modified_str1[:original_source_index] + dest_char + modified_str1[
                                                                                original_source_index + 1:]

        # If this is a delete operation: delete the source character at the original source index
        if current_operation == 'delete':
            modified_str1 = modified_str1[0: original_source_index:] + modified_str1[original_source_index + 1::]
            # decrement the count_index_source by 1 since we have deleted a character:
            # decreased the string's length by 1.
            count_index_source -= 1
        # If this is an insert operation: add the destination character at the original source index
        if current_operation == 'insert':
            # Start by getting the corresponding character from the destination by referring to the ES
            # (and the way we formatted it).
            current_destination_char = current_sequence[2][1]
            # Increment the count_index_source by 1 since we have inserted a character:
            # increased the string's length by 1
            modified_str1 = modified_str1[:original_source_index] + current_destination_char + modified_str1[
                                                                                               original_source_index:]
            count_index_destination += 1
    # Return the modified string which should be homomorphic to the destination that was specified
    # while generating the ES.
    return modified_str1

# Used for testing
str1 = 'AGRGA'
str2 = 'AGGGAA'
dp = wagnerFisher(str1, str2, True)
print(dp)
all_paths = create_paths(dp)
for path in all_paths:
    es = generate_es(path, str1, str2)
    print(es)
    print(patching(es, str1))
    print('REVERSING')
    rev = generate_rev_es(es, str1, str2)
    print(rev)
    print(patching(rev, str2))
