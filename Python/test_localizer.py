# --------------------GPU Lib---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from datasetV2 import EarthquakeDataset
import pandas as pd
from nets.localizer import FPN1DLocalizer as Localizer
#from nets.localizer import SimpleCNNLocalizer as Localizer

import psutil
def log_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / (1024 ** 3)} GB")  # RSS in GB

# ------------------------------------------------------------
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
num_epochs = 2000
learning_rate = 0.0001

# Dataset and DataLoader
test_dataset = EarthquakeDataset(data_folder='data/lunar/test/downsample_data/S15_GradeB/',
                                  label_folder=None, is_testing=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
#model = Localizer(num_layers=7, in_channels=1, mid_channels=8, kernel_size=13).to(device)
#model = Localizer(num_layers=3, in_channels=1, mid_channels=64, kernel_size=3).to(device)
#model = Localizer(num_pools=10, in_channels=1, mid_channels=64, kernel_size=31).to(device)
model = Localizer(num_pools=4, in_channels=1, mid_channels=12, kernel_size=13).to(device)
#model = Localizer(num_layers=10, in_channels=1, mid_channels=12, kernel_size=13).to(device)



# Load the saved model parameters
model.load_state_dict(torch.load('save/localizer/trial_2/quake_localization_model_00930.pth'))


batch = next(iter(test_loader))

#pred = model(batch[0].to(device)).flatten()

pred = model(batch[0].to(device)).flatten()
a = 1



# Assuming detections is a dictionary with filenames as keys and percentages as values
detections = dict(zip(batch[3], pred.tolist()))

# Convert detections to a DataFrame for CSV writing
detections_df = pd.DataFrame(list(detections.items()), columns=['Filename', 'Percentage'])

# Define the output folder and CSV file path
output_folder = 'data/lunar/test/detections/S15_GradeB'
os.makedirs(output_folder, exist_ok=True)
csv_file_path = os.path.join(output_folder, 'detections.csv')

# Write the detections to a CSV file using pandas
detections_df.to_csv(csv_file_path, index=False)

print(f'Detections saved to {csv_file_path}')