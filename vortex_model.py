import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
from vortex_dataset import generate_dataset

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(3, 16), ReLU(), Lin(16, 16))
        # input is output of above+5
        self.node_mlp_2 = Seq(Lin(19, 16), ReLU(), Lin(16, 16), ReLU(), Lin(16, 3)) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        #out = torch.cat([x[row], edge_attr], dim=1)
        out = x[row]
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),reduce='mean')
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)

        return out

# creates, saves and validates a spin model 
# models are then passed to XY 
def create_model():
    model = MetaLayer(node_model=NodeModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    model.train()

    dataset = generate_dataset(20, 100, 100)

    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    print("Test Dataset Size: ", len(dataset))
    n_steps = len(dataset) / loader.batch_size
    print("n steps: ", n_steps)

    n_epochs = 2
    for epoch in range(n_epochs):
        batch_n = 1
        for batch in loader:
            
            optimizer.zero_grad()

            x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=None, u=None, batch=batch.batch)

            loss = loss_fn(x, batch.y)

            loss.backward()
            optimizer.step()

            print("Batch ", batch_n, " out of ", n_steps, " Loss:", loss)
            batch_n += 1
    
    torch.save(model.node_model.state_dict(), 'VortexModel.pt')

if __name__ == '__main__':
    create_model()