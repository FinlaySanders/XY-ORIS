import XY_to_graph
from XY import XY_model
from torch_geometric.data import Data
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader



# --------------- generating t+1 dataset

def generate_spin_dataset(size, n_graphs, graph_depth, dt=1, metropolis_sweeps=0):
    graphs = []

    edge_index = XY_to_graph.get_xy_edge_index((size, size))

    for i in range(n_graphs):
        # one model is used to generate all the data
        xy = XY_model(size, 0.1)

        for _ in range(metropolis_sweeps):
            xy.metropolis_sweep()

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

# --------------- generating t+k dataset


def create_k_step_dataset(amount, depth):
    sequences = create_data_sequences(amount, depth)

    # sub sequences with the same length should be appended in groups of batch_size
    sub_sequences = []

    for sequence in sequences:
        
        for dt in [1,2,3,4,5,6,7,8,9,10]:

            print(depth - dt - (10-dt))
            for i in range(depth - dt - (10-dt)):
                sub_sequence = sequence[i:i+dt+1]
                sub_sequences.append(sub_sequence)
    

    loader = DataLoader(sub_sequences, batch_size=10, shuffle=True, collate_fn=custom_collate)
    
    return loader


def custom_collate(batch_list):
    # batch_list is a list of sequences from sequences[]
    # Transpose to group by timestep
    timesteps = list(zip(*batch_list))

    batched_data = []
    for timestep_data in timesteps:
        batch = Batch.from_data_list(timestep_data)
        batched_data.append(batch)

    return batched_data


def create_data_sequences(amount, length):
    sequences = []

    edge_index = XY_to_graph.get_xy_edge_index((30, 30))

    for _ in range(amount):
        new_sequence = []
        xy = XY_model(30, 0.1)

        for _ in range(length):
            x = XY_to_graph.get_xy_spin_node_features(xy.spin_grid, xy.spin_vel_grid)

            new_sequence.append(Data(x=x, edge_index=edge_index))

            xy.numerical_integration(1)
        
        sequences.append(new_sequence)
    
    return sequences


if __name__ == '__main__':
    create_k_step_dataset(5, 2000)
