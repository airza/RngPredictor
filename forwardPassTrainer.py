import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from extractor import get_input_and_output_from_file
from forwardPassModel import Model
import sys

if len(sys.argv) == 1:
    bit = 33
else:
    bit = int(sys.argv[1])
# get import count from sys.argv
if len(sys.argv) >= 2:
    IMPORT_COUNT = int(sys.argv[2])
else:
    IMPORT_COUNT = 2000000
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    raise "aaaa"
TEST_COUNT = 2000
LOSS_FUNCTION = nn.HingeEmbeddingLoss()
BATCH_SIZE = 8192
rng_type = "bad"
X, y = get_input_and_output_from_file(f'{rng_type}.rng', IMPORT_COUNT, bit=bit)
X = torch.from_numpy(X).float()
X *= 2
X -= 1
y = torch.from_numpy(y).float()
y *= 2
y -= 1
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

model = Model().to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.01, eps=1e-20)
criterion = LOSS_FUNCTION
epochs = 3000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
for epoch in range(epochs):
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
    scheduler.step()
    print('Epoch:', epoch, 'Loss:', total_loss / len(train_loader))
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        print('Test Loss:', loss.item())
    if total_loss / len(train_loader) < 0.01:
        break
# model evaluation
bit_str = f'{bit:02}'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f'{rng_type}_{bit_str}.pth')
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    print('Test Loss:', loss.item())
