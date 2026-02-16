import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv

# --- Constants ---
N_tot = 2.76e20      # cm^-3
sigma_e = 2.48e-20   # cm^2
sigma_a = 0.11e-20   # cm^2
length = 1.0         # cm
mesh_z = 10 # only value 10 is working, why? 
dz = length / (mesh_z - 1)

def calculate_total_gain(step):
    # Try loading from VTK first since we know those have the data, beta_cell_.vtk has information about 
    # the population inversion at each point in the mesh, we can use that to calculate the gain coefficient g(z) and then integrate it to get the total gain G.
    fname = f"beta_cell_{step}.vtk"
    if not os.path.exists(fname):
        return None
    
    mesh = pv.read(fname)
    beta_all = mesh.point_data['scalars']
    
    # Reshape and average per layer to get 1D profile
    beta_reshaped = beta_all.reshape((mesh_z, 421), order='F') # order='F' to read in column-major order, which is how VTK stores data
    beta_z = np.mean(beta_reshaped, axis=1) 
    # axis 1 means we average across the 421 points in each layer to get a single beta value for each z layer, 
    # resulting in a 1D array of length mesh_z (10) representing the population inversion along the z-axis.
    
    # 1. Calculate local gain coefficient g(z)
    g_z = N_tot * ((sigma_e + sigma_a) * beta_z - sigma_a)
    
    # 2. Integrate g(z) along the length (Trapezoidal rule)== can use simpson's rule if we want more accuracy, try it next time
    integrated_g = np.trapezoid(g_z, dx=dz)
    
    # 3. Total Gain (Amplification Ratio)
    total_gain = np.exp(integrated_g)
    return total_gain

# --- Generate Time Evolution Plot ---
time_steps = range(0, 150)
gain_history = []
time_axis = []

for t in time_steps:
    G = calculate_total_gain(t)
    if G is not None:
        gain_history.append(G)
        time_axis.append(t)

print(f"Calculated gain for {len(gain_history)} time steps.")
print(f"Final Gain at last time step: {gain_history[-1]:.2e}")

plt.figure(figsize=(8, 5))
plt.plot(time_axis, gain_history, color='red', linewidth=2)
plt.title("Total Small Signal Gain ($I_{out}/I_{in}$) vs Time")
plt.xlabel("Time Step")
plt.ylabel("Gain Ratio ($G$)")
plt.grid(True, linestyle='--')
plt.savefig("gain_v_time_exponential.png")
print("Saved exponential gain plot to gain_v_time_exponential.png")