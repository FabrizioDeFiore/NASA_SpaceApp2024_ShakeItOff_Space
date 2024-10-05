# --------------------GPU Lib---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasetV2 import EarthquakeDataset
#from nets.localizer import FPN1DLocalizer as Localizer
from nets.localizer import SimpleCNNLocalizer as Localizer

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
test_dataset = EarthquakeDataset(data_folder='data/lunar/test/downsample_data/S12_GradeB/',
                                  label_folder=None, is_testing=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
#model = Localizer(num_layers=7, in_channels=1, mid_channels=8, kernel_size=13).to(device)
#model = Localizer(num_layers=3, in_channels=1, mid_channels=64, kernel_size=3).to(device)
#model = Localizer(num_pools=10, in_channels=1, mid_channels=64, kernel_size=31).to(device)
# model = Localizer(num_pools=4, in_channels=1, mid_channels=12, kernel_size=13).to(device)
model = Localizer(num_layers=10, in_channels=1, mid_channels=12, kernel_size=13).to(device)



# Load the saved model parameters
model.load_state_dict(torch.load('save/localizer/trial_2/quake_localization_model_00930.pth'))


batch = next(iter(test_loader))

pred = model(batch[0].to(device)).flatten()

a = 1