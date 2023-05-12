import torch
import torch.nn as nn

# f(x) = 2 * x

X = torch.tensor([[1],[2],[3],[11],[14],[40]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[22],[28],[80]], dtype=torch.float32)

X_test = torch.tensor([10], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_features = n_features
output_features = n_features

model = nn.Linear(input_features, output_features)

print(f'Prediction before training: f(10) = {model(X_test).item():.5f}')

# Training
LR = 0.001
EPOCHS = 10000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    # forward
    y_hat = model(X)
    # loss
    l = loss(Y, y_hat)
    # gradients = backward pass
    l.backward()
    # update weights
    optimizer.step()
    # zero_grads
    optimizer.zero_grad()

    if (epoch+1) % (EPOCHS/10) == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(10) = {model(X_test).item():.5f}')
