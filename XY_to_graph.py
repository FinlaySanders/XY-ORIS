import torch
import numpy as np

def get_xy_edge_index(grid):
    idx_grid = np.arange(0,grid.shape[0]*grid.shape[1]).reshape(grid.shape)

    neighbour_arr = (np.stack([
                            np.roll(idx_grid, -1, axis=0),
                            np.roll(idx_grid, 1, axis=0),
                            np.roll(idx_grid, -1, axis=1),
                            np.roll(idx_grid, 1, axis=1)
                           ], axis = 2)).flatten()

    source_arr = np.repeat(idx_grid.flatten(), 4)
    np_edge_index = np.stack((source_arr, neighbour_arr))

    edge_index = torch.from_numpy(np_edge_index).to(torch.int64)
        
    return edge_index

def get_xy_edge_attr(grid, edge_index):
    flat_grid = grid.flatten()
    edge_attr = torch.tensor(np.expand_dims(np.sin(flat_grid[edge_index[0]] - flat_grid[edge_index[1]]), axis=1), dtype=torch.float)
    return edge_attr

def get_xy_spin_node_features(grid, vel_grid):
    x = grid.flatten()
    v = vel_grid.flatten()
    x = np.stack((np.sin(x), np.cos(x), v), axis=1)
    
    return torch.tensor(x, dtype=torch.float)
    



