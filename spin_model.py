import numpy as np
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from matplotlib import pyplot as plt
from XY import XY_model

# --------------------------------------- Model Init

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(4, 20), ReLU(), Lin(20, 3))
        self.node_mlp_2 = Seq(Lin(6, 20), ReLU(), Lin(20, 3))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, out], dim=1)
        #print("out: ", out)
        #print(out.shape)
        out = self.node_mlp_2(out)
        #print(out)
        
        # normalize output spins
        fac = torch.sqrt(torch.sum(out ** 2, dim=1))
        out = out / fac.view(-1, 1)

        return out

model = MetaLayer(node_model=NodeModel())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()


# --------------------------------------- Data Generation
  
# creates a list of data objects, each representing a graph 
# the data objects a given x, y and edge_index
def generate_dataset(size, temp, n_graphs, graph_depth):
    graphs = []

    # this is constant so only needs to be calculated once
    edge_index = get_xy_edge_index(xy.spin_grid)
    
    for i in range(n_graphs):
        # one model is used to generate all the data
        xy = XY_model(size, temp)

        for _ in range(100):
            xy.metropolis_sweep()

        x = get_xy_node_features(xy.spin_grid, xy.spin_vel_grid)

        # generates n graphs in the form of a Data object 
        for _ in range(graph_depth):
            edge_attr = get_xy_edge_attr(xy.spin_grid, edge_index)
            xy.numerical_integration(1)
            y = get_xy_node_features(xy.spin_grid, xy.spin_vel_grid)

            graphs.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))

            x = y
        
        print("completed graph: ", i, " / ", graphs[len(graphs)-1])

    return graphs

def get_xy_node_features(grid, vel_grid):
    x = grid.flatten()
    v = vel_grid.flatten()
    x = np.stack((np.sin(x), np.cos(x), v), axis=1)

    return torch.tensor(x, dtype=torch.float)
    
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
    return torch.tensor(np.expand_dims(-np.cos(flat_grid[edge_index[0]] - flat_grid[edge_index[1]]), axis=1), dtype=torch.float)

#dataset = generate_dataset(30, 0.1, 200, 200)
#torch.save(dataset, "data_200x200.pt")
dataset = torch.load("data_200x200.pt") 
#print(dataset)
#dataset2 = torch.load("data2_sincos_200x200.pt") 
#torch.save(dataset1+dataset2, "data_sincos_400x200.pt")


# --------------------------------------- Training

loader = DataLoader(dataset, batch_size=5, shuffle=True)

print("Dataset Size: ", len(dataset))
n_steps = len(dataset) / loader.batch_size
print("n steps: ", n_steps)

n_epochs = 1
for epoch in range(n_epochs):
    batch_n = 1
    for batch in loader:
        
        optimizer.zero_grad()
        x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, u=None, batch=batch.batch)

        loss_fn = torch.nn.MSELoss()
        #loss_fn = torch.nn.L1Loss()
        loss = loss_fn(x, batch.y)

        loss.backward()
        optimizer.step()

        if batch_n % (n_steps / 100) == 0:
            print("Epoch: ", epoch, int(batch_n/n_steps*100), "%, Loss: ", loss)
        batch_n += 1

model.eval()

# --------------------------------------- Testing

# Validation
dataset = torch.load("test_data_10x10.pt") 
loader = DataLoader(dataset, batch_size=20, shuffle=True)

print("Test Dataset Size: ", len(dataset))
n_steps = len(dataset) / loader.batch_size
print("n steps: ", n_steps)

losses = torch.tensor([])

n_epochs = 1
for epoch in range(n_epochs):
    batch_n = 1
    for batch in loader:
        with torch.no_grad():
            optimizer.zero_grad()
            x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, u=None, batch=batch.batch)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(x, batch.y)
        torch.cat((losses, loss.unsqueeze(0)))

        if batch_n % (n_steps / 100) == 0:
            print(int(batch_n/n_steps*100), "%, Loss: ", loss)
        batch_n += 1
print(losses)
print("Average Loss: ", torch.mean(losses))



# Visualisation

xy = XY_model(30, 0.1)
for _ in range(150):
    xy.metropolis_sweep()

print("Before: ", np.sin(xy.spin_grid)[0][0], np.cos(xy.spin_grid)[0][0])
print(np.arctan2(np.sin(xy.spin_grid)[0][0], np.cos(xy.spin_grid)[0][0]))

x = get_xy_node_features(xy.spin_grid, xy.spin_vel_grid)
edge_index = get_xy_edge_index(xy.spin_grid)

input = x
dt = 1
for _ in range(dt):
    with torch.no_grad():
        edge_attr = get_xy_edge_attr(xy.spin_grid, edge_index)
        x, _, _ = model(x=input, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=None)
        #print(x)

        input = x

# converts nn output to numpy 30x30 prediction
new_spins_sin, new_spins_cos, _ = np.hsplit(input.numpy(), 3)

# the actual updated spin grid
xy.numerical_integration(dt)

# plots the given spin grids on the same axis
fig, ax = plt.subplots()
ax.set_title("XY Model Comparison  Black=Actual  Red=Predicted")

# plotting spins on grid
x = np.arange(30)
y = np.arange(30)
X, Y = np.meshgrid(x, y)
ax.invert_yaxis()
q2 = ax.quiver(X, Y, new_spins_cos, new_spins_sin, pivot='mid', color='red')#, width = 0.003)
q1 = ax.quiver(X, Y, np.cos(xy.spin_grid), np.sin(xy.spin_grid), pivot='mid')

print("prediction: ", new_spins_sin[0][0], new_spins_cos[0][0])
print(np.arctan2(new_spins_sin[0][0], new_spins_cos[0][0]))
print("Actual: ", np.sin(xy.spin_grid)[0][0], np.cos(xy.spin_grid)[0][0])
print(np.arctan2(np.sin(xy.spin_grid)[0][0], np.cos(xy.spin_grid)[0][0]))



plt.show()

