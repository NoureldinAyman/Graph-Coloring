import random
from termcolor import colored

target = 0
COLORS = ['blue', 'red', 'green', 'yellow']

def encode_adjacency_matrix(edges, num_nodes):
    """
    Encode the graph as an adjacency matrix
    """
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for (i, j) in edges:
        matrix[i][j] = 1
        matrix[j][i] = 1  # Ensure the matrix is symmetric for undirected graphs
    return matrix

def validate_adjacency_matrix(matrix):
    # Check if the matrix is square
    num_rows = len(matrix)
    for row in matrix:
        if len(row) != num_rows:
            return False
    
    # Check if all entries are either 0 or 1, and if the matrix is symmetric
    for i in range(num_rows):
        for j in range(num_rows):
            if matrix[i][j] not in [0, 1]:
                return False
            if matrix[i][j] != matrix[j][i]:
                return False

    return True

def encode(graph):
    nodes = list(graph.keys())
    for neighbors in graph.values():
        nodes.extend(neighbors)
    nodes = list(set(nodes))
    node_index = {node: i for i, node in enumerate(nodes)}
    edges = [(node_index[node], node_index[neighbor]) for node, neighbors in graph.items() for neighbor in neighbors]
    adj_matrix = encode_adjacency_matrix(edges, len(nodes))
    return adj_matrix, node_index

def decode(adj_matrix, node_index):
    inverse_node_index = {v: k for k, v in node_index.items()}
    graph = {node: [] for node in inverse_node_index.values()}
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                graph[inverse_node_index[i]].append(inverse_node_index[j])
    return graph

class GraphColoring:
    def __init__(self, adjacency_matrix, node_index):
        self.adjacency_matrix = adjacency_matrix
        self.node_index = node_index
        if not validate_adjacency_matrix(self.adjacency_matrix):
            raise ValueError("Invalid adjacency matrix created")
        self.coloring = [random.choice(COLORS) for _ in range(len(self.adjacency_matrix))]
        self.conflicts = 0
        self.color_count = set()

    def print_decoded_graph(self):
        decoded_graph = decode(self.adjacency_matrix, self.node_index)
        print("Decoded Graph with Coloring:")
        for node, neighbors in decoded_graph.items():
            color = self.coloring[self.node_index[node]]
            print(f"{colored(node, color)}: ", end="")
            for neighbor in neighbors:
                neighbor_color = self.coloring[self.node_index[neighbor]]
                print(colored(neighbor, neighbor_color), end=" ")
            print()

    def get_fitness(self):
        self.color_count = set(self.coloring)
        self.conflicts = 0
        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):
                if self.adjacency_matrix[i][j] == 1 and self.coloring[i] == self.coloring[j]:
                    self.conflicts += 1

    def crossover(self, parentA, parentB):
        child1 = GraphColoring(parentA.adjacency_matrix, parentA.node_index)
        child2 = GraphColoring(parentA.adjacency_matrix, parentA.node_index)
        point = len(parentA.coloring) // 2
        child1.coloring = parentA.coloring[:point] + parentB.coloring[point:]
        child2.coloring = parentB.coloring[:point] + parentA.coloring[point:]
        return child1, child2

    def mutate(self):
        for i in range(len(self.coloring)):
            if random.random() < 0.10:
                self.coloring[i] = random.choice(COLORS)

# Helper functions to visualize the graph in the terminal
def colorize(graph):
    for row in range(len(graph.adjacency_matrix)):
        for col in range(len(graph.adjacency_matrix[row])):
            color = graph.coloring[col]
            print(colored(graph.adjacency_matrix[row][col], color), end=" ")
        print()

def summarize(generation, graph, fitness):
    print(f"Generation #{generation}:")
    colorize(graph)
    print(f"Conflicts: {fitness:3}")
    graph.print_decoded_graph()

def colorize_conflicts(graph):
    for row in range(len(graph.adjacency_matrix)):
        for col in range(len(graph.adjacency_matrix)):
            if row != col and graph.adjacency_matrix[row][col] == 1 and graph.coloring[row] == graph.coloring[col]:
                print(f"{colored(graph.adjacency_matrix[row][col], 'red')}" + " ", end="")
            elif row == col:
                print(f"{colored(graph.adjacency_matrix[row][col], 'red')}" + " ", end="")
            else:
                print(f"{colored(graph.adjacency_matrix[row][col], 'white')}" + " ", end="")
        print()

POP_SIZE = 5

best_score = float('inf')
population = []
generation = 1

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['C', 'E', 'F'],
    'C': ['D', 'E', 'G'],
    'D': ['E', 'F', 'H'],
    'E': ['F', 'I'],
    'F': ['I'],
    'G': ['H', 'J'],
    'H': ['I', 'J']
}

# Convert the graph into an adjacency matrix and node indicies mapping each node
adjacency_matrix, node_index = encode(graph)

# Generate initial generation
for i in range(POP_SIZE):
    population.append(GraphColoring(adjacency_matrix, node_index))

while best_score > target:
    # Assess the fitness of each individual
    for i in range(POP_SIZE):
        population[i].get_fitness()

        if population[i].conflicts < best_score:
            best_score = population[i].conflicts
            summarize(generation, population[i], population[i].conflicts)
            colorize_conflicts(population[i])
            print(f"Number of colors: {len(population[i].color_count)}")
            print(f"Coloring: {population[i].coloring}")
            print("Best score:", best_score)

    # Selection phase
    mating_pool = []
    parents = population[:]
    population = []

    # Use tournament selection to select parents to create offspring
    tournament_size = 5
    for i in range(POP_SIZE):
        # Pick 5 random samples from the parents
        tournament = random.sample(parents, tournament_size)

        # Select the parent with the least amount of conflicts
        winner = min(tournament, key=lambda parent: (parent.conflicts, len(parent.color_count)))

        # Add the winner to the mating pool
        mating_pool.append(winner)

    # Generating a new population
    for i in range(POP_SIZE):
        parentA = random.choice(mating_pool)
        parentB = random.choice(mating_pool)

        child1, child2 = parentA.crossover(parentA, parentB)

        child1.mutate()
        child2.mutate()
        
        # Append the children to the new population
        population.append(child1)
        population.append(child2)

    generation += 1
