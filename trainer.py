import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from extractor import get_data_from_file
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    raise "aaaa"
RNG_NAME = "xorshift128plus"
IMPORT_COUNT = 4000000
TEST_COUNT = 20000
PREV_COUNT = 4
BIT = 0
LOSS_FUNCTION = nn.MSELoss()
METRIC_FUNCTION = 'accuracy'
BATCH_SIZE=1024
PREV_COUNT = 2

X, y = get_data_from_file(RNG_NAME+'.rng', IMPORT_COUNT, PREV_COUNT,bit=0)
X = X.reshape(X.shape[0],-1)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
X_train = X[:-TEST_COUNT]
X_test = X[-TEST_COUNT:]
y_train = y[:-TEST_COUNT]
y_test = y[-TEST_COUNT:]
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)
# Create DataLoader for training data with batch size
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# Create DataLoader for test data with batch size
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_heads, dim_feedforward):
        super(Model, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        self.out = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.out(x)
        return x


model = Model(X.shape[1], 1, num_layers=4, num_heads=4, dim_feedforward=48).to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08)
criterion = LOSS_FUNCTION

for epoch in range(100):
    total_loss = 0.0
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch:', epoch, 'Loss:', total_loss / len(train_loader))
# model evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    print('Test Loss:', loss.item())