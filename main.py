from XY import XY_model
from spin_model import MetaLayer, NodeModel
import torch
from matplotlib import pyplot as plt, animation
import matplotlib
from copy import deepcopy

node_model = NodeModel()
node_model.load_state_dict(torch.load('NodeModel.pt'))

model = MetaLayer(node_model=node_model)

fig, axs = plt.subplots(1, 2)

real_xy = XY_model(30, 0.1)
for _ in range(50):
    real_xy.metropolis_sweep()

model_xy = XY_model(30, 0.1, spin_model=model)
model_xy.spin_grid = deepcopy(real_xy.spin_grid)

real_xy.plot_quiver(fig, axs[0], title="Real Spins")
model_xy.plot_quiver(fig, axs[1], title="Predicted Spins")

def update_models(frame):
    if frame < 3:
        return
    real_xy.update_spins_numerical_integration()
    real_xy.update_vortices()

    model_xy.update_spins_GNN()
    model_xy.update_vortices()

def save_anim(anim, filename):
    matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\2175\\Downloads\\ffmpeg"
    writer = animation.FFMpegWriter(fps=1, metadata=dict(artist="CodersLegacy"), bitrate=4000)
    anim.save(filename, writer=writer, dpi=300) 

anim = animation.FuncAnimation(fig, update_models, interval=1, frames=400, repeat=False)
save_anim(anim, "anim.mp4")

plt.show()




