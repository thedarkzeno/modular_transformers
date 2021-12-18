import torch

def fourier_transform(x):
    attention_output = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return (attention_output,)