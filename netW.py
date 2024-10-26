import networkx as nx
import matplotlib.pyplot as plt

# Define node positions
positions = {
    1: (0, 4), 2: (3, 4), 
    3: (0, 3), 4: (1, 3), 5: (2, 3),  6: (3, 3), 
    7: (4, 2), 8: (3, 2), 9: (2, 2),
    10: (2, 1), 11: (1, 1),
    12: (0, 1), 13: (0, -3), 14: (1, -1), 15: (2, -1), 16: (3, 1), 17: (3, 0),
    18: (4, 1), 19: (3, -1), 20: (3, -3), 21: (2, -3), 22: (2, -2), 23: (1, -2), 24: (1, -3)
}

# Define edges
edges = [
    (1, 2),     (2, 1),     (1, 3),     (3, 1),     (2, 6),     (6, 2),     (3, 4),     (4, 3), 
    (4, 5),     (5, 4),     (5, 6),     (6, 5),     (6, 7),     (7, 6),     (7, 8),     (8, 7),
    (8, 9),     (9, 8),     (8, 16),    (16, 8),    (9, 5),     (5, 9),     (9, 10),    (10, 9),
    (10, 11),   (11, 10),   (10, 16),   (16, 10),   (10, 17),   (17, 10),
    (11, 4),    (4, 11),    (11, 12),   (12, 11),   (12, 3),    (3, 12),    (7,18),     (18,7),
    (12, 13),   (13, 12),   (14, 11),   (11, 14),   (18,20),    (20,18),    (14,23),    (23,14),
    (14, 15),   (15, 14),   (15, 10),   (10, 15),   (15, 19),   (19, 15),   (22, 20),    (20, 22),
    (16, 18),   (18, 16),   (17, 16),   (16, 17),   (15, 22),   (22, 15),
    (17, 19),   (19, 17),   (19, 20),   (20, 19),   (20, 21),   (21, 20),
    (21, 22),   (22, 21),   (24,21),    (21,24),
    (22, 23),   (23, 22),   (23, 24),   (24, 23),   (24, 13),   (13, 24), (6, 8), (8,6)
]

# Define edges to color differently
special_edges = [(1, 2), (1, 3), (2, 6), (5, 4), (5, 9), (6, 2), (7, 8), (10, 9), (10, 11), (11, 4), (12, 11), (13, 12), (13, 24), (18, 20), (20, 18)]

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
G.add_nodes_from(positions.keys())
G.add_edges_from(edges)

# Draw the network graph
plt.figure(figsize=(10, 8))

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos=positions, node_color='lightgreen', node_size=500)
nx.draw_networkx_labels(G, pos=positions)

# Draw edges with the default color
default_edges = [edge for edge in edges if edge not in special_edges]
nx.draw_networkx_edges(G, pos=positions, edgelist=default_edges, arrowstyle='-|>', arrowsize=15, connectionstyle='arc3,rad=0.1', edge_color='black')

# Draw special edges with a different color (e.g., red)
nx.draw_networkx_edges(G, pos=positions, edgelist=special_edges, arrowstyle='-|>', arrowsize=15, connectionstyle='arc3,rad=0.1', edge_color='red', style='dashed')

#plt.title("Tranportation Network with Potentially Damaged links")
plt.savefig("figure_60.jpg", format="jpg", dpi=1000)
plt.show()
#plt.savefig('figure.jpg', dpi = 1000, format = 'jpg')
