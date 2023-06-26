import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.widgets import Slider
import time

class XY_model:
    def __init__(self, size, temp):
        self.J  = 1
        self.temp = temp
        self.size = size
        self.spin_grid = np.random.rand(size, size) * 2 * np.pi
        self.spin_vel_grid = np.zeros((size,size))

        self.idxs = np.array([(y, x) for y in range(self.size) for x in range(self.size)])

    # performs a metropolis step on each spin 
    def metropolis_sweep(self):
        np.random.shuffle(self.idxs)
        
        for idx in self.idxs:
            y,x = idx[0],idx[1]

            neighbours = [(y, (x+1) % self.size),(y, (x-1) % self.size),((y-1) % self.size, x),((y+1) % self.size, x)]
            
            prev_theta = self.spin_grid[y,x]
            prev_energy = -self.J*sum(np.cos(prev_theta - self.spin_grid[n]) for n in neighbours) 
            new_theta = (prev_theta + np.random.uniform(-np.pi, np.pi) * 0.1) % (np.pi * 2)
            new_energy = -self.J*sum(np.cos(new_theta - self.spin_grid[n]) for n in neighbours) 
            
            d_energy = new_energy-prev_energy
            if np.random.uniform(0.0, 1.0) < np.exp(-(d_energy/self.temp)):
                self.spin_grid[y,x] = new_theta

    # performs numerical integration and updates spins accordingly
    def numerical_integration(self, time):
        dt = 0.01
        for _ in range(int(time/dt)):
            self.spin_grid += dt * self.spin_vel_grid
            
            # keep angles in (0, 2pi) so vortices can be detected
            self.spin_grid = self.spin_grid % (2*np.pi)

            # prev spin grid ~= new spin grid 
            # to maximise performance (and so allow for a lower dt) 
            # we dont copy the prev grid and use it below
            self.spin_vel_grid += self.temp * -dt * (
                            + np.sin(self.spin_grid - np.roll(self.spin_grid, 1, axis=0))
                            + np.sin(self.spin_grid - np.roll(self.spin_grid, -1, axis=0))
                            + np.sin(self.spin_grid - np.roll(self.spin_grid, 1, axis=1))
                            + np.sin(self.spin_grid - np.roll(self.spin_grid, -1, axis=1))
                            )

    # returns the indices of vortices and antivortices in the form [y,x]
    # SPINS MUST BE (0, 2pi)
    def find_vortices(self):
        # summing acute angles around each potential vortex positions
        sum_angles = (self.get_signed_angles_between(self.spin_grid, np.roll(self.spin_grid, -1, axis=1)) 
                      + self.get_signed_angles_between(np.roll(self.spin_grid, -1, axis=1), np.roll(self.spin_grid, -1, axis=(1,0))) 
                      + self.get_signed_angles_between(np.roll(self.spin_grid, -1, axis=(1,0)), np.roll(self.spin_grid, -1, axis=0)) 
                      + self.get_signed_angles_between(np.roll(self.spin_grid, -1, axis=0), self.spin_grid))

        row, col = np.where(np.isclose(sum_angles, 2*np.pi))
        vortices = list(zip(row+0.5,col+0.5))
        row, col = np.where(np.isclose(sum_angles, -2*np.pi))
        a_vortices = list(zip(row+0.5,col+0.5))

        return vortices, a_vortices
    
    def get_signed_angles_between(self, arr1, arr2):
        diff = arr1 - arr2

        diff[diff>np.pi] = diff[diff>np.pi] % -np.pi
        diff[diff<-np.pi] = diff[diff<-np.pi] % np.pi

        return diff

    # plots the initial spin grid
    def plot_quiver(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("XY Model   Temperature: " + str(self.temp))

        # plotting spins on grid
        x = np.arange(self.size)
        y = np.arange(self.size)
        X, Y = np.meshgrid(x, y)
        self.ax.invert_yaxis()
        self.q = self.ax.quiver(X, Y, np.cos(self.spin_grid), np.sin(self.spin_grid), pivot='mid')

        # finding and plotting vortices
        vortices, a_vortices = self.find_vortices()
        self.v = self.ax.scatter(
            [n[1] for n in vortices] + [n[1] for n in a_vortices], # x values of vortices, anti vortices
            [n[0] for n in vortices] + [n[0] for n in a_vortices], # y values of vortices, anti vortices
            color=["red" for _ in vortices]+["green" for _ in a_vortices]) # corresponding colours of vortex type 

    # begins all the animations of the quiver plot
    def animate_quiver(self, frames=1):
        self.quiver_anim = animation.FuncAnimation(self.fig, self.animate_spins_numerical_integration, frames=frames, interval=0)
        self.vortex_anim = animation.FuncAnimation(self.fig, self.animate_vortices, frames=frames, interval=0)
 
    # animates the plotted grid's spins using monte carlo and the metropolis algorithm
    def animate_spins_monte_carlo(self, frame):
        self.metropolis_sweep()
        
        U, V = np.cos(self.spin_grid), np.sin(self.spin_grid)
        self.q.set_UVC(U, V)
        return self.q
    
    # animates the plotted grid's spins using numerical integration
    def animate_spins_numerical_integration(self, frame):
        self.numerical_integration(0.4)

        U, V = np.cos(self.spin_grid), np.sin(self.spin_grid)
        self.q.set_UVC(U, V)
        return self.q

    # animates the plotted grid's vortices
    def animate_vortices(self, frame):
        self.v.remove()
        vortices, a_vortices = self.find_vortices()
        self.v = self.ax.scatter(
            [n[1] for n in vortices] + [n[1] for n in a_vortices], # x values of vortices, anti vortices
            [n[0] for n in vortices] + [n[0] for n in a_vortices], # y values of vortices, anti vortices
            color=["red" for _ in vortices]+["green" for _ in a_vortices]) # corresponding colours of vortex type 
        return self.v 

    def show(self, animate=True):
        self.plot_quiver()
        if animate:
            self.animate_quiver()
        plt.show()

    # handles the temperature slider
    def plot_slider(self):
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Temperature', 0.0000001, 1, self.temp)
        
    def slider_update(self, frame):
        self.temp = self.slider.val

    def reset(self):
        self.spin_grid = np.random.rand(self.size, self.size) * 2 * np.pi
        self.spin_vel_grid = np.zeros((self.size,self.size))



#model = XY_model(30, 0.1)
#for i in range(100):
#    print(i)
#    model.metropolis_sweep()

#model.show()













