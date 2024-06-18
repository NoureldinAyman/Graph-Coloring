import customtkinter as ctk
from customtkinter import CTk, CTkButton, CTkFrame, CTkLabel, CTkEntry, CTkTextbox
from tkinter import messagebox
from termcolor import colored
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

target = 0
COLORS = ['blue', 'red', 'green', 'yellow']
POP_SIZE = 5  # Default population size

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

def genetic():
    global POP_SIZE, graph
    try:
        POP_SIZE = int(population_entry.get())
    except ValueError:
        messagebox.showerror("Invalid input", "Population size must be an integer.")
        return
    
    try:
        graph = eval(graph_entry.get("1.0", "end-1c"))
        if not isinstance(graph, dict):
            raise ValueError
    except:
        messagebox.showerror("Invalid input", "Graph must be a valid dictionary.")
        return

    best_score = float('inf')
    population = []
    generation = 1

    # Convert the graph into an adjacency matrix and node indices mapping each node
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
                best_graph = population[i]
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
    generation_label.configure(text=f"Generation: {generation}")


    # Generate the layout positions for the graph
    G = nx.Graph()
    for i in range(len(adjacency_matrix)):
        for j in range(i + 1, len(adjacency_matrix)):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(list(node_index.keys())[i], list(node_index.keys())[j])
    pos = nx.spring_layout(G)

    update_canvas(best_graph, colored_canvas, pos)
    update_canvas(GraphColoring(adjacency_matrix, node_index), uncolored_canvas, pos)

def update_canvas(graph, canvas_type, pos):
    global canvas_colored, canvas_uncolored, fig_colored, fig_uncolored

    if canvas_type == colored_canvas:
        if canvas_colored:
            canvas_colored.get_tk_widget().destroy()
            plt.close(fig_colored)

        fig_colored, ax = plt.subplots()
    else:
        if canvas_uncolored:
            canvas_uncolored.get_tk_widget().destroy()
            plt.close(fig_uncolored)

        fig_uncolored, ax = plt.subplots()

    G = nx.Graph()

    node_colors = []
    for node, index in graph.node_index.items():
        G.add_node(node)
        if canvas_type == colored_canvas:
            node_colors.append(graph.coloring[index])
        else:
            node_colors.append('gray')

    for i in range(len(graph.adjacency_matrix)):
        for j in range(i + 1, len(graph.adjacency_matrix)):
            if graph.adjacency_matrix[i][j] == 1:
                G.add_edge(list(graph.node_index.keys())[i], list(graph.node_index.keys())[j])

    nx.draw(G, pos, with_labels=True, node_color=node_colors, ax=ax, node_size=500, font_color='white')

    if canvas_type == colored_canvas:
        canvas_colored = FigureCanvasTkAgg(fig_colored, master=root)
        canvas_colored.draw()
        canvas_colored.get_tk_widget().grid(row=0, column=2, rowspan=4)
    else:
        canvas_uncolored = FigureCanvasTkAgg(fig_uncolored, master=root)
        canvas_uncolored.draw()
        canvas_uncolored.get_tk_widget().grid(row=0, column=1, rowspan=4)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        plt.close('all')
        root.quit()
        root.destroy()

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

# Initial graph
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

root = CTk()
root.title("Graph Coloring Algorithm")

coloring_frame = CTkFrame(root)
coloring_frame.grid(row=0, column=0, rowspan=4)

graph_label = CTkLabel(coloring_frame, text="Graph Coloring Options", font=("Helvetica", 16))
graph_label.pack(pady=10)

population_label = CTkLabel(coloring_frame, text="Population Size:", font=("Helvetica", 14))
population_label.pack(pady=5)

population_entry = CTkEntry(coloring_frame, font=("Helvetica", 14))
population_entry.pack(pady=5)

graph_entry_label = CTkLabel(coloring_frame, text="Custom Graph:", font=("Helvetica", 14), width=200, height=50)
graph_entry_label.pack(pady=5, padx=5)

graph_entry = CTkTextbox(coloring_frame, font=("Helvetica", 14), height=200, width=200)
graph_entry.pack(pady=5)
graph_entry.insert("1.0", str(graph))

run_button = CTkButton(coloring_frame, text="Run Algorithm", command=genetic, font=("Helvetica", 14))
run_button.pack(pady=20)

generation_label = CTkLabel(coloring_frame, text="Generation: 0", font=("Helvetica", 14))
generation_label.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)

colored_canvas = 2  # Column index for the colored graph canvas
uncolored_canvas = 1  # Column index for the uncolored graph canvas

canvas_colored = None
canvas_uncolored = None
fig_colored = None
fig_uncolored = None

# Generate the layout positions for the initial graph
initial_adjacency_matrix, initial_node_index = encode(graph)
G_initial = nx.Graph()
for i in range(len(initial_adjacency_matrix)):
    for j in range(i + 1, len(initial_adjacency_matrix)):
        if initial_adjacency_matrix[i][j] == 1:
            G_initial.add_edge(list(initial_node_index.keys())[i], list(initial_node_index.keys())[j])
initial_pos = nx.spring_layout(G_initial)

# Draw the initial uncolored graph
update_canvas(GraphColoring(*encode(graph)), uncolored_canvas, initial_pos)

root.mainloop()
