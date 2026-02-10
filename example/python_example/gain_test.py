import numpy as np
import matplotlib
matplotlib.use('Agg') # Necessary for headless clusters
import matplotlib.pyplot as plt
import glob
import re
import os

# --- Configuration ---
SIGMA_E = 2.1e-20  # Emission cross-section (adjust if needed)
L = 0.7            # Disc thickness in cm
FILE_PATH = "./*.vtk" 

def read_vtk_values(filename):
    """Parses a legacy VTK file manually to find the data after LOOKUP_TABLE"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Use regex to find the data section after LOOKUP_TABLE default
    # This is more robust than line-by-line for different VTK formats
    try:
        data_part = content.split("LOOKUP_TABLE default")[1]
        # Extract all numbers (including scientific notation)
        data = np.fromstring(data_part, sep=' ')
        return data
    except IndexError:
        return np.array([])

def terminal_plot(x, y, width=60, height=15):
    """Draws a basic plot in the terminal using text characters"""
    if len(y) == 0: return
    y_min, y_max = min(y), max(y)
    if y_max == y_min: y_max += 1
    
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for i in range(len(x)):
        xi = int((i / len(x)) * (width - 1))
        yi = int(((y[i] - y_min) / (y_max - y_min)) * (height - 1))
        grid[height - 1 - yi][xi] = "*"
    
    print("\n--- TERMINAL PREVIEW: GAIN VS TIME ---")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("-" * (width + 2))

# --- Process Files ---
files = sorted(glob.glob(FILE_PATH), key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])
times = []
gains = []

print(f"Processing {len(files)} files...")

for i, f in enumerate(files):
    match = re.search(r'(\d+)', f)
    t = float(match.group(1)) if match else i
    
    n_values = read_vtk_values(f)
    
    if n_values.size > 0:
        n_avg = np.mean(n_values)
        total_gain = np.exp(SIGMA_E * n_avg * L)
    else:
        total_gain = 1.0 # Default gain when no data is found
        
    times.append(t)
    gains.append(total_gain)

# --- Save Text Data ---
# This creates a file with 2 columns: Time and Gain
data_to_save = np.column_stack((times, gains))
np.savetxt('gain_data_results.txt', data_to_save, header='Time_us Gain', fmt='%.8e')
print("Successfully saved data to: gain_data_results.txt")

# --- Save Plot Image ---
plt.figure(figsize=(10, 6))
plt.plot(times, gains, 'r-o', markersize=4, label='Small Signal Gain')
plt.xlabel('Time (us)')
plt.ylabel('Gain')
plt.title('Verification of Fig 20: Gain vs Time')
plt.grid(True)
plt.savefig('gain_plot.png')

# --- Final Output ---
terminal_plot(times, gains)
print(f"Done! Image saved as 'gain_plot.png'.")
print(f"Peak Gain observed: {max(gains):.4f}")
