import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from extractor import get_input_and_output_from_file
from forwardPassModel import Model
import sys
if len(sys.argv) == 1:
    bit = 32
else:
    bit = int(sys.argv[1])
print("running on bit %d"%bit)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    raise "aaaa"
IMPORT_COUNT = 120000
TEST_COUNT = 100
LOSS_FUNCTION = nn.MSELoss()
BATCH_SIZE= 64
rng_type = "xorshift128_forward_pass"
X, y = get_input_and_output_from_file(f'{rng_type}.rng', IMPORT_COUNT,bit=bit)
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


print("??")
model = Model().to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.00003125, eps=1e-08)
criterion = LOSS_FUNCTION
epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,verbose=True)
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
    if total_loss / len(train_loader) < 0.1:
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