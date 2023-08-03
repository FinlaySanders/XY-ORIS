import XY_to_graph
from XY import XY_model
from torch_geometric.data import Data
import torch

def generate_spin_dataset(size, n_graphs, graph_depth, dt=1, metropolis_sweeps=0):
    graphs = []
    
    for i in range(n_graphs):
        # one model is used to generate all the data
        xy = XY_model(size, 0.1)

        for _ in range(metropolis_sweeps):
            xy.metropolis_sweep()

        edge_index = XY_to_graph.get_xy_edge_index(xy.spin_grid)
        x = XY_to_graph.get_xy_spin_node_features(xy.spin_grid, xy.spin_vel_grid)

        # generates n graphs in the form of a Data object 
        for _ in range(graph_depth):
            edge_attr = XY_to_graph.get_xy_edge_attr(xy.spin_grid, edge_index)
            xy.numerical_integration(dt)
            y = XY_to_graph.get_xy_spin_node_features(xy.spin_grid, xy.spin_vel_grid)

            graphs.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, u=torch.tensor([[dt]], dtype=torch.float)))

            x = y
        
        print("completed graph: ", i, " / ", graphs[len(graphs)-1])

    return graphs


if __name__ == '__main__':
    torch.save(generate_spin_dataset(30, 50, 50), "0_spin_data_200x500.pt")