import torch
from torch_geometric.nn import MetaLayer
from spin_model import NodeModel as SpinNodeModel
from vortex_model import NodeModel as VortexNodeModel
import numpy as np

import matplotlib
from matplotlib import pyplot as plt, animation
from XY import XY_model
from vortex_visualiser import Vortex_Visualiser
from copy import deepcopy

spin_node_model = SpinNodeModel()
#spin_node_model.load_state_dict(torch.load('NM4.pt'))
spin_node_model.load_state_dict(torch.load('NodeModel_tst_epoch_9.pt'))
spin_model = MetaLayer(node_model=spin_node_model)

vortex_node_model = VortexNodeModel()
#vortex_node_model.load_state_dict(torch.load('VXModel_SC.pt'))
vortex_node_model.load_state_dict(torch.load('VortexModel_tst_epoch_7.pt'))
vortex_model = MetaLayer(node_model=vortex_node_model)

fig, axs = plt.subplots(2, 2)

size = 20
real_xy = XY_model(size, 0.1)
for _ in range(15):
    real_xy.numerical_integration(1)

model_xy = XY_model(size, 0.1, spin_model=spin_model, 
                    spin_grid=deepcopy(real_xy.spin_grid), 
                    spin_vel_grid=deepcopy(real_xy.spin_vel_grid))

vortex_poses, avortex_poses = real_xy.find_vortices()
vortex_visualiser = Vortex_Visualiser(size, vortex_poses, avortex_poses, vortex_model)

real_xy.plot_quiver(axs[0,0], title="Real Spins")
model_xy.plot_quiver(axs[0,1], title="Predicted Spins")
vortex_visualiser.plot_scatter(axs[1,0], title="Predicted Vortices")

# error plotting
start_spins = deepcopy(real_xy.spin_grid)
axs[1,1].set_xlabel("Time")
axs[1,1].set_ylabel("Mean Spin Error")
axs[1,1].set_title("Error Plot")
axs[1,1].legend()

gnn_errors = []
identity_errors = []

def mean_phase_difference(spins1, spins2):
    spins1 %= 2*np.pi
    spins2 %= 2*np.pi


    diff = np.abs(spins1 - spins2)

    diff = np.minimum(diff, 2*np.pi - diff)
    
    return np.mean(diff)

def update_figures(frame):
    print(frame)
    if frame < 3:
        return
    
    # updating real/predicted XY figures
    real_xy.update_spins_numerical_integration()
    real_xy.update_vortices()

    model_xy.update_spins_GNN()
    model_xy.update_vortices()

    #real_xy.numerical_integration(1)
    vortex_visualiser.update_vortices()
    vortex_visualiser.plot_real_vortices(real_xy.find_vortices())

    # updating the loss graph
    identity_errors.append(mean_phase_difference(start_spins, real_xy.spin_grid))
    gnn_errors.append(mean_phase_difference(model_xy.spin_grid, real_xy.spin_grid))

    x = torch.arange(len(gnn_errors))
    axs[1,1].plot(x, gnn_errors, color="Blue")
    axs[1,1].plot(x, identity_errors, color ="Green")


def save_anim(anim, filename):
    matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\2175\\Downloads\\ffmpeg"
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist="CodersLegacy"), bitrate=4000)
    anim.save(filename, writer=writer, dpi=300) 

anim = animation.FuncAnimation(fig, update_figures, interval=1, frames=150, repeat=False)
#save_anim(anim, "XY7.mp4")

plt.show()




