import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
from dataset import generate_spin_dataset, create_k_step_dataset

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(3, 15), ReLU(), Lin(15, 15), ReLU(), Lin(15, 10))
        # input is output of above+3
        self.node_mlp_2 = Seq(Lin(13, 15), ReLU(), Lin(15, 30), ReLU(), Lin(30, 15), ReLU(), Lin(15, 3)) 

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

        magnitude = torch.norm(out[:, :2], dim=1, keepdim=True)
        out = torch.cat([out[:, :2] / magnitude, out[:, 2:]], dim=1)


        return out

# creates, saves and validates a spin model 
# models are then passed to XY 
def create_model():
    model = MetaLayer(node_model=NodeModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    model.train()

    print("getting dataset")
    dataset = create_k_step_dataset(100, 100)
        
    n_steps = len(dataset)

    n_epochs = 1
    for epoch in range(n_epochs):
        batch_n = 1
        for batch in dataset:

            initial_data_batch = batch[0]  # batch at beginning of sequence
            ground_truths_batches = batch[1:] # batches at remaining timesteps in the sequnces

            k = len(ground_truths_batches)

            optimizer.zero_grad()
            
            loss = 0

            prediction = initial_data_batch.x
            for i in range(k):
                prediction, _, _ = model(x=prediction, edge_index=initial_data_batch.edge_index, edge_attr=None, u=None, batch=initial_data_batch.batch)

                loss += loss_fn(prediction, ground_truths_batches[i].x)

            print("Batch ", batch_n, " out of ", n_steps, " Loss:", loss)

            loss.backward()
            optimizer.step()

            batch_n += 1
    
    torch.save(model.node_model.state_dict(), 'NodeModel.pt')

    model.eval()

    # Model Validation
    val_dataset = generate_spin_dataset(30,50,50)
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
