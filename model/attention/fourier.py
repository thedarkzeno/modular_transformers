import torch

def fourier_transform(x):
    return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real