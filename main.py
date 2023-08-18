import torch
from torch_geometric.nn import MetaLayer
from spin_model import NodeModel as SpinNodeModel
#from vortex_model import NodeModel as VortexNodeModel
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, animation
from XY import XY_model
#from vortex_visualiser import Vortex_Visualiser
from copy import deepcopy

spin_node_model = SpinNodeModel()
spin_node_model.load_state_dict(torch.load('NodeModel_epoch_8.pt'))
spin_model = MetaLayer(node_model=spin_node_model)

#vortex_node_model = VortexNodeModel()
#vortex_node_model.load_state_dict(torch.load('VortexModel.pt'))
#vortex_model = MetaLayer(node_model=vortex_node_model)

fig, axs = plt.subplots(2, 2)

size = 20
real_xy = XY_model(size, 0.1)
for _ in range(30):
    real_xy.metropolis_sweep()
for _ in range(10):
    real_xy.numerical_integration(1)

model_xy = XY_model(size, 0.1, spin_model=spin_model)
model_xy.spin_grid = deepcopy(real_xy.spin_grid)

#vortex_poses, avortex_poses = real_xy.find_vortices()
#vortex_visualiser = Vortex_Visualiser(size, vortex_poses, avortex_poses, vortex_model)

real_xy.plot_quiver(axs[0,0], title="Real Spins")
model_xy.plot_quiver(axs[0,1], title="Predicted Spins")
#vortex_visualiser.plot_scatter(axs[1,0], title="Predicted Vortices")

# error plotting
start_spins = deepcopy(real_xy.spin_grid)
axs[1,1].set_xlabel("Time")
axs[1,1].set_ylabel("Mean Spin Error")
axs[1,1].set_title("Blue - GNN   Green - Identity")

gnn_errors = []
identity_errors = []

def mean_phase_difference(spins1, spins2):
    x1, y1 = np.cos(spins1), np.sin(spins1)
    x2, y2 = np.cos(spins2), np.sin(spins2)

    delta_x, delta_y = np.abs(x1 - x2), np.abs(y1 - y2)

    mean_x, mean_y = np.mean(delta_x), np.mean(delta_y)

    mean_diff = np.arctan2(mean_y, mean_x)

    return mean_diff

def update_figures(frame):
    print(frame)
    if frame < 3:
        return

    #if frame == 100:
    #    real_xy.spin_grid = deepcopy(model_xy.spin_grid)
    
    # updating real/predicted XY figures
    real_xy.update_spins_numerical_integration()
    real_xy.update_vortices()

    model_xy.update_spins_GNN()
    model_xy.update_vortices()

    #vortex_visualiser.update_vortices()

    # updating the loss graph
    identity_errors.append(mean_phase_difference(start_spins, real_xy.spin_grid))
    gnn_errors.append(mean_phase_difference(model_xy.spin_grid, real_xy.spin_grid))

    x = np.arange(len(gnn_errors))
    axs[1,1].plot(x, gnn_errors, color="Blue")
    axs[1,1].plot(x, identity_errors, color ="Green")


def save_anim(anim, filename):
    matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\2175\\Downloads\\ffmpeg"
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="CodersLegacy"), bitrate=4000)
    anim.save(filename, writer=writer, dpi=300) 

anim = animation.FuncAnimation(fig, update_figures, interval=1, frames=500, repeat=False)
save_anim(anim, "XY.mp4")

plt.show()




