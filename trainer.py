import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from extractor import get_data_from_file
import sys
bit = int(sys.argv[1])
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    raise "aaaa"
RNG_NAME = "xorshift128plus"
IMPORT_COUNT = 1000000
TEST_COUNT = 100
LOSS_FUNCTION = nn.BCELoss()
METRIC_FUNCTION = 'accuracy'
BATCH_SIZE= 1024
PREV_COUNT = 2

X, y = get_data_from_file(RNG_NAME+'.rng', IMPORT_COUNT, PREV_COUNT,bit=bit)
X = X.reshape(X.shape[0],-1)
y = y.reshape(-1, 1)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
X_train = X[:-TEST_COUNT]
X_test = X[-TEST_COUNT:].to(device)
y_train = y[:-TEST_COUNT]
y_test = y[-TEST_COUNT:].to(device)
dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)
# Create DataLoader for training data with batch size
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# Create DataLoader for test data with batch size
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
class Model(nn.Module):
    def __init__(self, input_size, output_size, num_heads, dim_feedforward):
        super(Model, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(dropout=0,d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.out = nn.Linear(input_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        x = self.out(x)
        x = self.sig(x)
        return x


model = Model(X.shape[1], 1, num_heads=4, dim_feedforward=2).to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.01, eps=1e-08)
criterion = LOSS_FUNCTION
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',threshold=0.001,patience=2,cooldown=3)
for epoch in range(500):
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
    scheduler.step(total_loss)
    print('Epoch:', epoch, 'Loss:', total_loss / len(train_loader))
# model evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    print('Test Loss:', loss.item())
    print(outputs[0],y_test[0])