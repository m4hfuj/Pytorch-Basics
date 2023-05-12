import numpy as np

# f(x) = 2 * x

X = np.array([1,2,3,4,5,11], dtype=np.float32)
Y = np.array([2,4,6,8,10,22], dtype=np.float32)

w = 0.0

# model pred
def forward(x):
    return w * x

# MSE loss
def loss(y, y_hat):
    return ((y_hat-y)**2).mean()

# gradient
# dJ/dw = 1/N 2x (w*x - y)
def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat-y).mean()

print(f'Prediction before training: f(10) = {forward(10):.3f}')

# Training
LR = 0.001
EPOCHS = 10

for epoch in range(EPOCHS):
    # forward
    y_hat = forward(X)

    # loss
    l = loss(Y, y_hat)

    # gradients
    dw = gradient(X, Y, y_hat)

    # update weights
    w -= LR * dw

    if epoch % (EPOCHS/10) == 0:
        print(f'epoch {epoch+1:3.0f}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(10) = {forward(10):.3f}')
