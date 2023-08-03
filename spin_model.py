import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
from dataset import generate_spin_dataset

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(3, 30), ReLU(), Lin(30, 30), ReLU(), Lin(30, 13))
        # input is output of above+3
        self.node_mlp_2 = Seq(Lin(16, 30), ReLU(), Lin(30, 30), ReLU(), Lin(30, 16), ReLU(), Lin(16, 3)) 

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    model.train()

    print("getting dataset")

    dataset = generate_spin_dataset(30, 200, 200)
    val_dataset = generate_spin_dataset(30, 50, 50, metropolis_sweeps=100)
    #val_dataset = torch.load("val_data.pt") 


    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    n_steps = len(dataset) / loader.batch_size

    n_epochs = 1
    for epoch in range(n_epochs):
        batch_n = 1
        for batch in loader:
            
            optimizer.zero_grad()
            x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, u=batch.u, batch=batch.batch)

            loss = loss_fn(x, batch.y)

            loss.backward()
            optimizer.step()

            if batch_n % (n_steps / 100) == 0:
                print("Epoch: ", epoch, int(batch_n/n_steps*100), "%, Loss: ", loss)
            batch_n += 1
    
    torch.save(model.node_model.state_dict(), 'NodeModel.pt')


    model.eval()

    # Model Validation
    loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

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
                x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, u=batch.u, batch=batch.batch)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(x, batch.y)
            losses = torch.cat((losses, loss.unsqueeze(0)))


            if batch_n % (n_steps / 100) == 0:
                print(int(batch_n/n_steps*100), "%, Loss: ", loss)
            batch_n += 1
    print(losses)
    print("Average Loss: ", torch.mean(losses))

if __name__ == '__main__':
    create_model()