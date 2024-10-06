import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import stft, istft
import pandas as pd
from tools import mkdir
from matplotlib.gridspec import GridSpec

root = 'data'
planet = 'mars'
dstype = 'test'

# Define paths
in_folder_path = osp.join(root, planet, dstype, 'downsample_data')
#aux_folder_path = osp.join(root, planet, dstype, 'labels', subaux)
out_folder_path = osp.join(root, planet, dstype, 'waves_analysis')

mkdir(out_folder_path)

# Define window size and step size for STFT
window_size = 512
step_size = 64

# Loop through all files in the folder
for filename in tqdm(os.listdir(in_folder_path)):
    
    if ".csv" not in filename:
        continue

    # Load the data from the CSV file
    in_file_path = osp.join(in_folder_path, filename)
    df = pd.read_csv(in_file_path)

    #aux_file_path = osp.join(aux_folder_path, filename)
    #labeled_df = pd.read_csv(aux_file_path)  # labeled_df has two columns, 'time_rel(sec)' and 'label'

    # Extract the velocity signal
    signal = df['velocity(c/s)'].values

    # Perform STFT on the signal
    f, t, Zxx = stft(signal, nperseg=window_size, noverlap=window_size - step_size)

    # Compute the magnitude of the STFT and take the log
    magnitude_spectrum = np.abs(Zxx) + 1e-60

    # Create a 3x1 subplot layout where the first two subplots are combined
    fig = plt.figure(figsize=(16, 10))

    # Third plot (Original Signal)
    ax3 = fig.add_subplot()
    ax3.plot(df['rel_time(sec)'], df['velocity(c/s)'], color='blue', linewidth=0.5)
    ax3.set_title('Original Signal (Velocity)')
    ax3.set_ylabel('Velocity (c/s)')
    ax3.set_xlabel('Time (sec)')

    # Optionally, you can hide the x-ticks of the first plot (ax1) to avoid clutter

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure as PNG in the output folder
    output_png_path = osp.join(out_folder_path, filename.replace('.csv', '.png'))
    plt.savefig(output_png_path)
    plt.close(fig)  # Close the figure after saving to prevent display
