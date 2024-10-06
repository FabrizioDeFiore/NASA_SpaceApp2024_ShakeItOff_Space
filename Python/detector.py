import os
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pywt
from scipy.signal import cwt, morlet

# Paths configuration
root = 'data'
planet = 'lunar'
dstype = 'test'
subaux = 'S15_GradeB'

# Define paths
in_folder_path = osp.join(root, planet, dstype, 'downsample_data', subaux)
folder_path = osp.join(root, planet, dstype, 'detections', subaux)


# Read the CSV file from the detection folder
csv_file_path = osp.join(folder_path, 'detections.csv')
detections_df = pd.read_csv(csv_file_path)

# Function to plot Morlet TFM and draw a red line
def plot_morlet_tfm(percentage, filename, output_path):
    
    # Load the data from the CSV file
    in_file_path = osp.join(in_folder_path, filename)
    df = pd.read_csv(in_file_path)

    # Perform Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 64)  # Define scale range
    coefficients, frequencies = pywt.cwt(df['velocity(m/s)'], scales, 'morl')  # Using Morlet wavelet

    # Create a 3x1 subplot layout where the first two subplots are combined
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(2, 1, height_ratios=[2, 1])  # First two subplots are larger
    
    # First plot (Morlet Transform, part 1 of 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.log(np.abs(coefficients)), extent=[df['time_rel(sec)'].min(), df['time_rel(sec)'].max(), scales.min(), scales.max()],
            cmap='viridis', vmax=-20, vmin=-25, aspect='auto', origin='lower')
    #ax1.set_ylabel('Scale')
    ax1.set_title(f'Wavelet Transform (Morlet) of Velocity Signal - {filename}')

    # Twin axis to plot the label on top of the first plot
    ax1b = ax1.twinx()
    #ax1b.plot(labeled_df['time_rel(sec)'], labeled_df['label'], color='red', linewidth=0.75, label='Label')
    #ax1b.set_ylabel('Label')
    ax1b.set_ylim(-0.1, 1.1)  # Ensure the y-axis is appropriate for binary labels (0 or 1)
    ax1b.tick_params(axis='y', colors='red')  # Color the ticks to differentiate

    # Third plot (Original Signal)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Share x-axis with ax1
    ax3.plot(df['time_rel(sec)'], df['velocity(m/s)'], color='blue', linewidth=0.5)
    ax3.set_title('Original Signal (Velocity)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_xlabel('Time (sec)')

    # Optionally, you can hide the x-ticks of the first plot (ax1) to avoid clutter
    ax1.tick_params(axis='x', which='both', labelbottom=False)

    # Adjust layout for better spacing
    plt.tight_layout()
    duration = (df['time_rel(sec)'].max() - df['time_rel(sec)'].min()) 
    #print(duration)
    # Draw red line at the corresponding percentage point
    red_line_position = df['time_rel(sec)'].min() + duration * percentage 
    ax1.axvline(x=red_line_position, color='red', linestyle='--', label='Arrival Point')
    ax3.axvline(x=red_line_position, color='red', linestyle='--', label='Arrival Point')    
    #plt.show()

    # Save the plot
    output_file = osp.join(output_path, f'{filename}_detection.png')
    plt.savefig(output_file)
    plt.close()



# Process each row in the CSV file
for index, row in detections_df.iterrows():
    filename = row['Filename']
    percentage = row['Percentage']
    
    # Convert the percentage to a float between 0 and 1
    #percentage = percentage / 100.0
    
    # Plot Morlet TFM and draw red line
    plot_morlet_tfm(percentage, filename, folder_path)
        # Load the data from the CSV file
    in_file_path = osp.join(in_folder_path, filename)
    df = pd.read_csv(in_file_path)

    # Calculate the duration of the signal
    duration = df['time_rel(sec)'].max() - df['time_rel(sec)'].min()

    # Calculate the red line position based on the percentage of the duration
    red_line_position = df['time_rel(sec)'].min() + duration * percentage

    # Get the corresponding relative time and velocity
    closest_index = (df['time_rel(sec)'] - red_line_position).abs().idxmin()
    rel_time = df.loc[closest_index, 'time_rel(sec)']
    velocity = df.loc[closest_index, 'velocity(m/s)']

    # Store the relative time and velocity in the detections_df DataFrame
    detections_df.at[index, 'Relative Time'] = rel_time
    detections_df.at[index, 'Velocity'] = velocity

# Save the updated detections_df DataFrame to the CSV file
detections_df.to_csv(csv_file_path, index=False)

#print(f"Updated detections saved to {csv_file_path}")