import torch

x = torch.randn(3, requires_grad=True)
print("x =", x)

y = x + 2
print('y =', y)

z = 2*y*y
print('z =', z)

z = z.mean()
print('z.mean =', z)

z.backward() # dz/dx
print('z =', x.grad)