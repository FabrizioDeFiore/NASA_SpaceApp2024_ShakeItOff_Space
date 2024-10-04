import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.gru = nn.GRU(512, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.gru(x.unsqueeze(1))  # Adding batch dimension for GRU
        x = torch.relu(self.fc2(x.squeeze(1)))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class RNNa(nn.Module):
    def __init__(self):
        super(RNNa, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=2)
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))  # Adding channel dimension for Conv1d
        x, _ = self.gru(x)
        x = torch.relu(self.fc1(x[:, -1, :]))  # Using the last output of GRU
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=16, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=16, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=16, stride=1, padding=0)
        self.fc1 = nn.Linear(128 * 1 * 1, 80)  # Adjust dimensions based input size
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 400)
        
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = torch.sigmoid(self.fc2(x))
        return x

# Hyperparameters
learning_rate = 0.0001
num_epochs = 20

# Model, loss function, optimizer for FCNN
model_fcnn = FCNN()
criterion = nn.CrossEntropyLoss()
optimizer_fcnn = optim.Adam(model_fcnn.parameters(), lr=learning_rate)

# Dummy data for illustration (replace with actual data)
inputs = torch.randn(64, 512)  # Batch size of 64, input size of 512
labels = torch.randint(0, 2, (64,))  # Batch size of 64, binary classification

# Training loop for FCNN
for epoch in range(num_epochs):
    model_fcnn.train()
    optimizer_fcnn.zero_grad()
    outputs = model_fcnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer_fcnn.step()
    
    print(f'FCNN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Model, loss function, optimizer for RNN
model_rnn = RNN()
optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=learning_rate)

# Training loop for RNN
for epoch in range(num_epochs):
    model_rnn.train()
    optimizer_rnn.zero_grad()
    outputs = model_rnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer_rnn.step()
    
    print(f'RNN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Model, loss function, optimizer for RNNa
model_rnna = RNNa()
optimizer_rnna = optim.Adam(model_rnna.parameters(), lr=learning_rate)

# Training loop for RNNa
for epoch in range(num_epochs):
    model_rnna.train()
    optimizer_rnna.zero_grad()
    outputs = model_rnna(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer_rnna.step()
    
    print(f'RNNa Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Hyperparameters for CNN
learning_rate_cnn = 0.001
num_epochs_cnn = 40

# Model, loss function, optimizer for CNN
model_cnn = CNN()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=learning_rate_cnn)

# Dummy data for illustration (replace with actual data)
inputs_cnn = torch.randn(48, 1, 64, 64)  # Batch size of 48, 1 channel, 64x64 input size
labels_cnn = torch.randint(0, 2, (48,))  # Batch size of 48, binary classification

# Training loop for CNN
for epoch in range(num_epochs_cnn):
    model_cnn.train()
    optimizer_cnn.zero_grad()
    outputs = model_cnn(inputs_cnn)
    loss = criterion(outputs, labels_cnn)
    loss.backward()
    optimizer_cnn.step()
    
    print(f'CNN Epoch [{epoch+1}/{num_epochs_cnn}], Loss: {loss.item():.4f}')

# Hyperparameters for GAN
learning_rate_gen = 0.0001
learning_rate_critic = 0.0002
num_epochs_gan = 100
critic_iterations = 5

# Model, loss function, optimizers for GAN
generator = Generator()
critic = Critic()
optimizer_gen = optim.Adam(generator.parameters(), lr=learning_rate_gen)
optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate_critic)
criterion_gan = nn.BCELoss()

# Dummy data for illustration (replace with actual data)
real_data = torch.randn(64, 1, 28, 28)  # Batch size of 64, 1 channel, 28x28 input size
noise = torch.randn(64, 100)  # Batch size of 64, noise vector of size 100

# Training loop for GAN
for epoch in range(num_epochs_gan):
    for _ in range(critic_iterations):
        # Train Critic
        critic.train()
        optimizer_critic.zero_grad()
        
        # Real data
        real_labels = torch.ones(64, 1)
        outputs_real = critic(real_data)
        loss_real = criterion_gan(outputs_real, real_labels)
        
        # Fake data
        fake_data = generator(noise).view(64, 1, 28, 28)
        fake_labels = torch.zeros(64, 1)
        outputs_fake = critic(fake_data.detach())
        loss_fake = criterion_gan(outputs_fake, fake_labels)
        
        # Total loss and backpropagation
        loss_critic = loss_real + loss_fake
        loss_critic.backward()
        optimizer_critic.step()
    
    # Train Generator
    generator.train()
    optimizer_gen.zero_grad()
    
    fake_data = generator(noise).view(64, 1, 28, 28)
    outputs = critic(fake_data)
    loss_gen = criterion_gan(outputs, real_labels)
    
    loss_gen.backward()
    optimizer_gen.step()
    
    print(f'GAN Epoch [{epoch+1}/{num_epochs_gan}], Loss Critic: {loss_critic.item():.4f}, Loss Generator: {loss_gen.item():.4f}')

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=45, criterion='gini')

# Dummy data to be replaced
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randint(0, 2, (100,))  # 100 samples, binary classification

# Training Random Forest
rf.fit(X_train, y_train)

print("Random Forest training complete.")