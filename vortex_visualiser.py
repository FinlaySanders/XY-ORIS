import torch

class Vortex_Visualiser:
    def __init__(self, size, vortex_poses, avortex_poses, vortex_model):
        vortex_data = []

        for pos in vortex_poses:
            vortex_data.append(pos + (1,))
        for pos in avortex_poses:
            vortex_data.append(pos + (-1,))

        self.vortex_data = torch.tensor(vortex_data, dtype=torch.float)
        self.vortex_model = vortex_model
        self.size = size
    
    def update_vortices(self):
        n_vortices = len(self.vortex_data)
        rows, cols = torch.combinations(torch.arange(n_vortices), 2).t()
        edge_index = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)
        with torch.no_grad():
            self.vortex_data, _, _ = self.vortex_model(x=self.vortex_data, edge_index=edge_index, batch=torch.tensor([0 for _ in range(n_vortices*n_vortices)]))

        self.vortex_data[:, 0] = self.vortex_data[:, 0] % self.size
        self.vortex_data[:, 1] = self.vortex_data[:, 1] % self.size

        n_values = self.vortex_data[:, 2]
        rounded_values = torch.where(n_values > 0, torch.ones_like(n_values), -torch.ones_like(n_values))
        self.vortex_data[:, 2] = rounded_values

        threshold_distance = 1  # Define what "close enough" means in terms of distance

        distances = self.pairwise_distance(self.vortex_data[:, :2], self.size, self.size)
        mask_different_n = (self.vortex_data[:, 2].unsqueeze(1) != self.vortex_data[:, 2].unsqueeze(0))

        # Identify pairs that are close and have different n values
        mask_to_remove = (distances < threshold_distance) & mask_different_n

        # For each row, see if there's any pair that requires its removal
        rows_to_remove = mask_to_remove.any(dim=1)

        # Filter out rows
        self.vortex_data = self.vortex_data[~rows_to_remove]

        self.ax.clear()
        x = []
        y = []
        colors = []
        # stored as row, col
        for vortex in self.vortex_data:
            x.append(vortex[1])
            y.append(vortex[0])
            if vortex[2] > 0:
                colors.append("red")
            else:
                colors.append("green")
        self.ax.scatter(x, y, color=colors)
        self.ax.set_xlim(0, self.size)  
        self.ax.set_ylim(0, self.size)  
        self.ax.invert_yaxis()


    def plot_scatter(self, ax, title="Predicted Spins"):
        self.ax = ax 
        self.ax.set_title(title)

        x = []
        y = []
        colors = []
        # stored as row, col
        for vortex in self.vortex_data:
            x.append(vortex[1])
            y.append(vortex[0])
            if vortex[2] > 0:
                colors.append("red")
            else:
                colors.append("green")
        self.ax.scatter(x, y, color=colors)

        self.ax.set_xlim(0, self.size)  
        self.ax.set_ylim(0, self.size)  
        self.ax.invert_yaxis()

    def pairwise_distance(self, matrix, Lx, Ly):
        """Compute the pairwise distance matrix for a 2D torch tensor with PBC."""
        diff = matrix.unsqueeze(1) - matrix.unsqueeze(0)
        
        # Adjust for periodic boundary conditions in x dimension
        diff[..., 0] = torch.where(diff[..., 0] > 0.5 * Lx, diff[..., 0] - Lx, diff[..., 0])
        diff[..., 0] = torch.where(diff[..., 0] < -0.5 * Lx, diff[..., 0] + Lx, diff[..., 0])
        
        # Adjust for periodic boundary conditions in y dimension
        diff[..., 1] = torch.where(diff[..., 1] > 0.5 * Ly, diff[..., 1] - Ly, diff[..., 1])
        diff[..., 1] = torch.where(diff[..., 1] < -0.5 * Ly, diff[..., 1] + Ly, diff[..., 1])
        
        dist = torch.sum(diff ** 2, dim=-1).sqrt()
        return dist