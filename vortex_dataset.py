import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

from XY import XY_model
import XY_to_graph
from copy import deepcopy


def shortest_distance_between(pos1, pos2, lattice_size):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    
    dx = min(dx, lattice_size - dx)
    dy = min(dy, lattice_size - dy)
    
    return (dx**2 + dy**2)**0.5

def get_xy_vortex_trajectories(size, depth):
    vortex_trajectories = []
    avortex_trajectories = []

    xy = XY_model(size, 0.1)
    for _ in range(50):
        xy.metropolis_sweep()
    for _ in range(50):
        xy.numerical_integration(1)

    prev_vs, prev_avs = xy.find_vortices()
    for v in prev_vs:
        vortex_trajectories.append([v])
    for av in prev_avs:
        avortex_trajectories.append([av])
    
    for t in range(depth):
        xy.numerical_integration(1)

        new_vs, new_avs = xy.find_vortices()

        if len(new_vs) + len(new_avs) != len(prev_vs) + len(prev_avs):
            print(f"annihilation found after {t}")
            break
            
        v_pairs, av_pairs = match_vortices(new_vs, prev_vs, new_avs, prev_avs, size)

        if v_pairs == {} or av_pairs == {}:
            print(f" --------- ran out of vortices after {t} iterations ------------")
            break
  

        for traj in vortex_trajectories:
            # if trajectory has already concluded, ignore
            if len(traj) != t+1:
                continue
            
            # if trajectory has a new point, append it
            prev_pos = traj[t]
            if prev_pos in v_pairs:
                traj.append(v_pairs[prev_pos])
        
        for traj in avortex_trajectories:
            if len(traj) != t+1:
                continue
            prev_pos = traj[t]
            if prev_pos in av_pairs:
                traj.append(av_pairs[prev_pos])
        
        vortex_sum = 0
        avortex_sum = 0
        for traj in vortex_trajectories:
            vortex_sum += len(traj)
        for traj in avortex_trajectories:
            avortex_sum += len(traj)
        
        if vortex_sum != avortex_sum:
            print(f"Error after {t} iterations")
            
        prev_vs = new_vs
        prev_avs = new_avs
        
    # smoothing out trajectories

    return vortex_trajectories, avortex_trajectories


def smooth_trajectories(vortex_trajectories, avortex_trajectories):
    for traj in vortex_trajectories:
        root = traj[0]
        n_repeats = 0

        for i in range(len(traj) - 1):
            if traj[i+1] == root:
                n_repeats += 1
            
            else:
                if n_repeats > 0:
                    new_root = traj[i+1]
                    pos_diff = (new_root[0] - root[0], new_root[1] - root[1])
                    fac = (pos_diff[0] / (n_repeats + 1), pos_diff[1] / (n_repeats + 1))

                    for j in range(n_repeats):
                        #traj[i - j -1][0] -= fac[0] * (j+1)
                        #traj[i - j -1][1] -= fac[1] * (j+1)

                        traj[i - j - 1] = (
                            traj[i - j][0] - fac[0] * (j+1), 
                            traj[i - j][1] - fac[1] * (j+1)
                            )

                    root = new_root
                    n_repeats = 0 
    
    return vortex_trajectories, avortex_trajectories
        



def match_vortices(new_v_poses, old_v_poses, new_av_poses, old_av_poses, size):
    av_pairs = {}
    v_pairs = {}

    for np in new_av_poses:
        closest_old = None
        min_distance = 999999
        for op in old_av_poses:
            dist = shortest_distance_between(np, op, size)
            if dist < min_distance:
                min_distance = dist
                closest_old = op
        av_pairs[closest_old] = np
    
    for np in new_v_poses:
        closest_old = None
        min_distance = 999999
        for op in old_v_poses:
            dist = shortest_distance_between(np, op, size)
            if dist < min_distance:
                min_distance = dist
                closest_old = op
        v_pairs[closest_old] = np
    
    return v_pairs, av_pairs

def generate_dataset(size, amount, depth):
    dataset = []
    for i in range(amount):
        print(i)
        vortex_trajectories, avortex_trajectories = get_xy_vortex_trajectories(size, depth)

        if vortex_trajectories == [] or avortex_trajectories == []:
            continue

        n_vortices = len(vortex_trajectories) + len(avortex_trajectories)
        trajectory_length = len(vortex_trajectories[0])

        for i in range(trajectory_length - 1):
            vortex_poses = [row[i] for row in vortex_trajectories]
            avortex_poses = [row[i] for row in vortex_trajectories]
            
            x = []
            for pos in vortex_poses:
                x.append(pos + (1,))
            for pos in avortex_poses:
                x.append(pos + (-1,))
            
            vortex_poses = [row[i+1] for row in vortex_trajectories]
            avortex_poses = [row[i+1] for row in vortex_trajectories]

            y = []
            for pos in vortex_poses:
                y.append(pos + (1,))
            for pos in avortex_poses:
                y.append(pos + (-1,))

            rows, cols = torch.combinations(torch.arange(n_vortices), 2).t()
            edge_index = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)

            dataset.append(Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=edge_index))
    
    return dataset




if __name__ == '__main__':
    #generate_dataset(30, 5, 100)
    vortex_trajectories, avortex_trajectories = get_xy_vortex_trajectories(30, 10)
    print(vortex_trajectories)
    vortex_trajectories, avortex_trajectories = smooth_trajectories(vortex_trajectories, avortex_trajectories)
    print(vortex_trajectories)