import torch
import torch.nn as nn

# f(x) = 2 * x

X = torch.tensor([[1],[2],[3],[11],[14],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[22],[28],[8]], dtype=torch.float32)

X_test = torch.tensor([10], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_features = n_features
output_features = n_features

# model = nn.Linear(input_features, output_features)

class LinearRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_features, output_features)

print(f'Prediction before training: f(10) = {model(X_test).item():.5f}')

# Training
LR = 0.01
EPOCHS = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    # forward and loss
    y_hat = model(X)
    l = loss(Y, y_hat)

    # backward pass and step()
    l.backward()
    optimizer.step()

    # zero_grads
    optimizer.zero_grad()

    if (epoch+1) % (EPOCHS/10) == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(10) = {model(X_test).item():.5f}')
