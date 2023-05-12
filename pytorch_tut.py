import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.rand(5,5, device=device)

    print(x)