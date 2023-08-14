import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from extractor import get_input_and_output_from_file

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    raise "aaaa"
IMPORT_COUNT = 100000
TEST_COUNT = 100
LOSS_FUNCTION = nn.MSELoss()
BATCH_SIZE= 512
PREV_COUNT = 2

X, y = get_input_and_output_from_file("xorshift128_forward_pass.rng", IMPORT_COUNT,bit=0)
X = torch.from_numpy(X).float()
X*=2
X-=1
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
        innerDim = 8
        super(Model, self).__init__()
        self.inn = nn.Linear(input_size, innerDim)
        self.sigin = nn.Sigmoid()
        self.transformer = nn.Transformer(d_model=innerDim, nhead=8, num_encoder_layers=4, num_decoder_layers=2, dim_feedforward=64)
        self.out = nn.Linear(innerDim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inn(x)
        x = self.sigin(x)
        x = self.transformer(x,x)
        x = self.out(x)
        x = self.sig(x)
        return x

print("??")
model = Model(X.shape[1], 1, num_heads=6, dim_feedforward=10).to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.01, eps=1e-08)
criterion = LOSS_FUNCTION
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',threshold=0.001,patience=2,cooldown=3)
for epoch in range(25):
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
torch.save(model, "./model.pth")
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    print('Test Loss:', loss.item())