import torch
import torch.fft

# def fourier_transform(x):
#     attention_output = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
#     return (attention_output,)
#########

def fourier_transform(x):
    attention_output = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2)
    return (attention_output,)

def inverse_fourier_transform(x):
    attention_output = torch.fft.ifft(torch.fft.ifft(x, dim=-1), dim=-2).real
    return (attention_output,)