import pywt
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

input_folder = '/path/to/input/folder'
output_folder = '/path/to/output/folder'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        
        scales = np.arange(1, 64)
        morl_coefficients, frequencies = pywt.cwt(df['velocity(m/s)'], scales, 'morl')
        
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(morl_coefficients), extent=[0, len(df), 1, 64], cmap='PRGn', aspect='auto',
                   vmax=abs(morl_coefficients).max(), vmin=-abs(morl_coefficients).max())
        plt.colorbar()
        plt.title(f'CWT of {filename}')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_cwt.png')
        plt.savefig(output_file_path)
        plt.close()