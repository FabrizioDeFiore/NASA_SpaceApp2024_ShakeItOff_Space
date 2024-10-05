import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet
from datasetV2 import EarthquakeDataset
from nets.localizer import FPN1DLocalizer as Localizer
import psutil

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / (1024 ** 3)} GB")  # RSS in GB

# Paths configuration
root = 'data'
planet = 'lunar'
dstype = 'test'
subaux = 'S15_GradeB'

# Define paths
in_folder_path = osp.join(root, planet, dstype, 'downsample_data', subaux)
aux_folder_path = osp.join(root, planet, dstype, 'wavelet_transform', subaux)
out_folder_path = osp.join(root, planet, dstype, 'detection', subaux)

# Create output directory if it doesn't exist
os.makedirs(out_folder_path, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = Localizer().to(device)
model.load_state_dict(torch.load('save/localizer/trial_2/quake_localization_model_00930.pth'))
model.eval()

# Function to plot Morlet TFM and draw a red line
def plot_morlet_tfm(signal, percentage, filename, output_path):
    widths = np.arange(1, 31)
    cwt_matrix = cwt(signal, morlet, widths)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(cwt_matrix), extent=[0, len(signal), 1, 31], cmap='jet', aspect='auto', vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    plt.colorbar(label='Magnitude')
    plt.title(f'Morlet TFM for {filename}')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    
    # Draw red line at the corresponding percentage point
    red_line_position = int(len(signal) * percentage)
    plt.axvline(x=red_line_position, color='red', linestyle='--')
    
    # Save the plot
    output_file = osp.join(output_path, f'{filename}_detection.png')
    plt.savefig(output_file)
    plt.close()

# Process each file
for filename in os.listdir(in_folder_path):
    if filename.endswith('.csv'):
        file_path = osp.join(in_folder_path, filename)
        
        # Load the data (assuming the data is in a CSV file)
        data = np.loadtxt(file_path, delimiter=',')
        
        # Get the signal (assuming the signal is in the first column)
        signal = data[:, 0]
        
        # Get the floating-point number (assuming it's stored in a specific way)
        # For demonstration, let's assume it's the mean of the signal
        floating_point_value = np.mean(signal)
        
        # Convert to percentage
        percentage = floating_point_value
        
        # Plot Morlet TFM and draw red line
        plot_morlet_tfm(signal, percentage, filename, out_folder_path)
        
        # Log memory usage
        log_memory_usage()