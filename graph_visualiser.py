import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


import XY_to_graph
from XY import XY_model

xy = XY_model(5, 0.1)

x = XY_to_graph.get_xy_spin_node_features(xy.spin_grid, xy.spin_vel_grid)
edge_index = XY_to_graph.get_xy_edge_index(xy.spin_grid)
edge_attr = XY_to_graph.get_xy_edge_attr(xy.spin_grid, edge_index)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

G = to_networkx(data, to_undirected=False)

rows, columns = xy.spin_grid.shape

# Shift nodes to make arrows more visible
shift = 0.1
pos = {}
for i in range(rows):
    for j in range(columns):
        node_index = i * columns + j
        row_shift = shift * (i % 2)
        col_shift = shift * (j % 2)
        pos[node_index] = (j + row_shift, -(i + col_shift))  # Apply shift to every other row and column

plt.figure(figsize=(6, 6))
nx.draw(G, pos=pos, with_labels=True, node_size=500, font_weight='bold')

plt.show()