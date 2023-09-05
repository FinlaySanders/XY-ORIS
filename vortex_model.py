import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from vortex_dataset import generate_dataset
import matplotlib.pyplot as plt
import time

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.node_mlp_1 = Seq(Lin(5, 8), ReLU(), Lin(8, 8))
        #self.node_mlp_2 = Seq(Lin(13, 8), ReLU(), Lin(8, 16), ReLU(), Lin(16, 8), ReLU(), Lin(8, 5)) 

        self.node_mlp_2 = Seq(Lin(10, 8), ReLU(), Lin(8, 8), ReLU(), Lin(8, 5)) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        #out = torch.cat([x[row], edge_attr], dim=1)
        #out = x[row]
        #out = self.node_mlp_1(out)
        out = scatter(x[row], col, dim=0, dim_size=x.size(0),reduce='mean')
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)

        return out

def train_one_epoch(model, optimizer, loss_fn, loader, lattice_size):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=None, u=None, batch=batch.batch)
        
        loss = loss_fn(x, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_model(model, val_loader, loss_fn, lattice_size):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            
            x, _, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=None, u=None, batch=batch.batch)

            loss = loss_fn(x, batch.y)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss

def main():
    model = MetaLayer(node_model=NodeModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print("Creating Dataset...")
    lattice_size = 20
    train_loader, val_loader = generate_dataset(lattice_size, 300, 300, train_val_split=0.8, cooling=10)
    print("datset len: ", len(train_loader))
    print("Done!")

    train_losses = []
    val_losses = []

    n_epochs = 10
    for epoch in range(n_epochs):
        start_time = time.time()

        # Training
        avg_train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, lattice_size)
        train_losses.append(avg_train_loss)
        
        # Validation
        avg_val_loss = validate_model(model, val_loader, loss_fn, lattice_size)
        val_losses.append(avg_val_loss)

        torch.save(model.node_model.state_dict(), f'VortexModel_tst_epoch_{epoch + 1}.pt')

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{n_epochs} - Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Time: {epoch_duration:.2f} seconds")

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses over Time')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
