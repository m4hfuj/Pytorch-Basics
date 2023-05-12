import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
device = torch.device('cuda')

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)


n_samples, n_features = X.shape

# 1) model
in_feat = n_features
out_feat = 1
model = nn.Linear(in_feat, out_feat)

# 2) loss and optimizer
LR = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# 3) training loop
EPOCHS = 500

for epoch in range(EPOCHS):
    # forward and loss
    y_hat = model(X)
    l = criterion(y, y_hat)   # there was a problem with this model because I mistakenly switched the input order in criterion

    # backward pass and step()
    l.backward()
    optimizer.step()

    # zero_grads
    optimizer.zero_grad()

    if (epoch+1) % (EPOCHS/10) == 0:
        [w, b] = model.parameters()
        print(f'epoch [{epoch+1}/{EPOCHS}]: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

# plt.plot(X_numpy, y_numpy, 'ro')
# plt.show()