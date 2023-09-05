import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from XY import XY_model
import XY_to_graph

from scipy.optimize import linear_sum_assignment
import numpy as np
import math

def generate_discrete_dataset(size, amount, max_depth, train_val_split=0.8):
    dataset = []
    
    for i in range(amount):
        print(i)
        xy = XY_model(size, 0.1)
        for _ in range(50):
            xy.metropolis_sweep()
        for _ in range(20):
            xy.numerical_integration(1)

        prev_v, prev_av = xy.find_vortices()

        for _ in range(max_depth):
            xy.numerical_integration(1)

            new_v, new_av = xy.find_vortices()

            if new_av == [] or new_v == []:
                continue

            v_pairs = pair_vortices(prev_v, new_v, size)
            av_pairs = pair_vortices(prev_av, new_av, size)

            x = []
            y = []

            for prev, new in v_pairs.items():
                x.append(XY_to_graph.pos_to_angles(prev, size) + (1,))
                y.append(XY_to_graph.pos_to_angles(new, size) + (1,))
            
            for prev, new in av_pairs.items():
                x.append(XY_to_graph.pos_to_angles(prev, size) + (-1,))
                y.append(XY_to_graph.pos_to_angles(new, size) + (-1,))

            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            n_vortices = len(x)
            rows, cols = torch.combinations(torch.arange(n_vortices), 2).t()
            edge_index = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)

            dataset.append(Data(x=x, y=y, edge_index=edge_index))

            prev_v, prev_av = new_v, new_av

    print("DATASET LEN: ", len(dataset))
    split_idx = int(train_val_split * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)

    print("TRAIN LENGTH: ", len(train_loader))
    
    return train_loader, val_loader

def generate_dataset(size, amount, max_depth, train_val_split=0.8, cooling=20):
    dataset = []

    for i in range(amount):
        print(i)
        v_traj, av_traj = generate_xy_trajectories(size, max_depth, cooling=cooling)
        #print("it")
        #print(v_traj)
        v_traj = [smooth_trajectory(traj, size) for traj in v_traj]
        #print(v_traj)
        av_traj = [smooth_trajectory(traj, size) for traj in av_traj]
        #v_traj = smooth_trajectory(v_traj, (size, size))
        #av_traj = smooth_trajectory(av_traj, (size, size))

        for i in range(max_depth):
            x = []
            y = []

            for traj in v_traj:
                pair = traj[i:i+2]
                if len(pair) == 2:
                    x.append(XY_to_graph.pos_to_angles(pair[0], size) + (1,))
                    y.append(XY_to_graph.pos_to_angles(pair[1], size) + (1,))
                    #x.append((pair[0][0]/size, pair[0][1]/size, 1))
                    #y.append((pair[1][0]/size, pair[1][1]/size, 1))
            
            for traj in av_traj:
                pair = traj[i:i+2]
                if len(pair) == 2:
                    x.append(XY_to_graph.pos_to_angles(pair[0], size) + (-1,))
                    y.append(XY_to_graph.pos_to_angles(pair[1], size) + (-1,))
                    #x.append((pair[0][0]/size, pair[0][1]/size, -1))
                    #y.append((pair[1][0]/size, pair[1][1]/size, -1))
            
            if x == [] or y == []:
                continue
        
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            n_vortices = len(x)
            rows, cols = torch.combinations(torch.arange(n_vortices), 2).t()
            edge_index = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)

            dataset.append(Data(x=x, y=y, edge_index=edge_index))
    
    print("DATASET LEN: ", len(dataset))
    split_idx = int(train_val_split * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)

    print("TRAIN LENGTH: ", len(train_loader))
    
    return train_loader, val_loader

def generate_xy_trajectories(size, max_depth, cooling=20):
    xy = XY_model(size, 0.1)
    for _ in range(cooling):
        xy.numerical_integration(1)
    
    v_trajectories = []
    av_trajectories = []

    prev_v, prev_av = xy.find_vortices()
    for pos in prev_v:
        v_trajectories.append([pos])
    for pos in prev_av:
        av_trajectories.append([pos])

    for depth in range(max_depth):
        xy.numerical_integration(1)

        new_v, new_av = xy.find_vortices()

        #if new_v == [] or new_av == []:
        #    print("ran out!")

        v_pairs = pair_vortices(prev_v, new_v, size)
        av_pairs = pair_vortices(prev_av, new_av, size)

        
        annihilated_vortices = list(set(prev_v) - set(new_v))
        annihilated_avortices = list(set(prev_av) - set(new_av))
        annihilated_pairs = pair_vortices(annihilated_avortices, annihilated_vortices, size)
        annihilation_poses = {}
        for av, v in annihilated_pairs.items():
            mid = pbc_average([av, v], size)
            annihilation_poses[v] = mid
            annihilation_poses[av] = mid
        
        
        for traj in v_trajectories:
            if len(traj) -1 != depth:
                continue 

            if traj[-1] in v_pairs:
                traj.append(v_pairs[traj[-1]]) 
            elif traj[-1] in annihilation_poses:
                traj.append(annihilation_poses[traj[-1]]) 
            
        for traj in av_trajectories:
            if len(traj) -1 != depth:
                continue 

            if traj[-1] in av_pairs:
                traj.append(av_pairs[traj[-1]]) 
            elif traj[-1] in annihilation_poses:
                traj.append(annihilation_poses[traj[-1]]) 

        prev_v, prev_av = new_v, new_av
    
    return v_trajectories, av_trajectories


# ------------------------------- Trajectory Smoothing

def bezier_curve(points, t_values):
    n = len(points) - 1
    curve = np.zeros((len(t_values), 2))

    for i, t in enumerate(t_values):
        for j, point in enumerate(points):
            curve[i] += point * (np.math.comb(n, j) * (1 - t) ** (n - j) * t ** j)
    return curve

def unwrap_trajectory_1d(trajectory, world_size):
    threshold = world_size / 2

    unwrapped = [trajectory[0]]
    offset = 0
    for i in range(1, len(trajectory)):
        delta = trajectory[i] - trajectory[i-1]
        if delta > threshold:
            offset -= world_size
        elif delta < -threshold:
            offset += world_size
        unwrapped.append(trajectory[i] + offset)
    return np.array(unwrapped)

def unwrap_trajectory_2d(trajectory, world_size):
    x_unwrapped = unwrap_trajectory_1d(trajectory[:, 0], world_size)
    y_unwrapped = unwrap_trajectory_1d(trajectory[:, 1], world_size)
    return np.vstack((x_unwrapped, y_unwrapped)).T


def smooth_trajectory(positions, world_size):
    positions = np.array(positions)
    positions = unwrap_trajectory_2d(positions, world_size)
    
    t_values = np.linspace(0, 1, len(positions))
    smoothed_positions = bezier_curve(positions, t_values)

    smoothed_positions %= world_size
    
    return smoothed_positions.tolist()

# -------------------------------


def pair_vortices(prev_v, new_v, lattice_size):
    pairs = {}

    dist_matrix = np.zeros((len(prev_v), len(new_v)))

    for i in range(len(prev_v)):
        for j in range(len(new_v)):
            dist_matrix[i][j] = pbc_distance(prev_v[i], new_v[j], lattice_size)

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    pairs = {prev_v[i]: new_v[j] for i, j in zip(row_ind, col_ind)}

    #print(f"pairs: {pairs}")
    return pairs

def pbc_distance(pos1, pos2, lattice_size):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    
    dx = min(dx, lattice_size - dx)
    dy = min(dy, lattice_size - dy)
    
    return (dx**2 + dy**2)**0.5

def pbc_average(points, size):
    #print(f"computing pbc with {points} in box size {size}")

    sum_cos_x = 0
    sum_sin_x = 0
    sum_cos_y = 0
    sum_sin_y = 0

    for point in points:
        sum_cos_x += math.cos(2 * math.pi * point[0] / size)
        sum_sin_x += math.sin(2 * math.pi * point[0] / size)
        sum_cos_y += math.cos(2 * math.pi * point[1] / size)
        sum_sin_y += math.sin(2 * math.pi * point[1] / size)

    avg_cos_x = sum_cos_x / len(points)
    avg_sin_x = sum_sin_x / len(points)
    avg_cos_y = sum_cos_y / len(points)
    avg_sin_y = sum_sin_y / len(points)
    
    avg_angle_x = math.atan2(avg_sin_x, avg_cos_x)
    avg_angle_y = math.atan2(avg_sin_y, avg_cos_y)
    
    avg_x = (avg_angle_x / (2 * math.pi)) * size
    avg_y = (avg_angle_y / (2 * math.pi)) * size

    avg_x = avg_x % size
    avg_y = avg_y % size

    return avg_x, avg_y
"""
def plot_trajectories(v_trajectories, av_trajectories, size):

    x = []
    y = []
    color = []

    for traj in v_trajectories:
        x += [row[1] for row in traj]
        y += [row[0] for row in traj]
        color += ["red" for _ in traj]
    for traj in av_trajectories:
        x += [row[1] for row in traj]
        y += [row[0] for row in traj]
        color += ["green" for _ in traj]

    plt.scatter(x, y, color=color)

    plt.xlim([0, size])
    plt.ylim([0, size])

def plot_trajectories_slice(frame, axs, v_trajectories, av_trajectories):
    print(frame)
    x = []
    y = []
    color = []

    for traj in v_trajectories:
        if len(traj)-1 < frame:
            continue
        x.append(traj[frame][1])
        y.append(traj[frame][0])
        color.append("red")
    for traj in av_trajectories:
        if len(traj)-1 < frame:
            continue
        x.append(traj[frame][1])
        y.append(traj[frame][0])
        color.append("green")

    axs.clear()
    v = axs.scatter(x, y, color=color)

    plt.xlim([0, 30])
    plt.ylim([0, 30])

    return v
"""

if __name__ == '__main__':    
    """from matplotlib import pyplot as plt, animation
    fig, axs = plt.subplots()
    axs.set_xticks([])
    axs.set_yticks([])

    size = 20
    
    v_traj, av_traj = generate_xy_trajectories(size, 200, cooling=10)
    
    plot_trajectories(v_traj, av_traj, size)
    plt.show()

    v_traj = [smooth_trajectory(traj, size) for traj in v_traj]
    av_traj = [smooth_trajectory(traj, size) for traj in av_traj]
    
    plot_trajectories(v_traj, av_traj, size)
    #anim2 = animation.FuncAnimation(fig, plot_trajectories_slice, fargs=(axs, v_traj, av_traj), interval=100, frames=1000, repeat=False)
    plt.show()"""

    generate_dataset(20, 20, 20)