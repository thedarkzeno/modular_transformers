from scipy.signal import cont2discrete
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# @partial(jit, static_argnums=[1,2])
# def jrfft(x, n, axis):
#     return jax.jit(jnp.fft.rfft)(x, n=n, axis=axis)
jrfft = jit(lambda x: jnp.fft.rfft(x, n=1024, axis=-1))
jirfft = jit(lambda x: jnp.fft.irfft(x, n=1024, axis=-1))

class FlaxLMUFFT(nn.Module):
    hidden_size: int
    memory_size: int
    seq_len: int
    theta: int
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    def setup(self):
        self.W_u = nn.Dense(1, dtype=self.dtype)
        # self.f_u = nn.ReLU()
        self.W_h = nn.Dense(self.hidden_size, dtype=self.dtype)
        # self.f_h = nn.ReLU()
        self.A_i = jnp.identity(self.memory_size)
        self.A, self.B = self.stateSpaceMatrices()
    
    def stateSpaceMatrices(self):
        """ Returns the discretized state space matrices A and B """

        Q = jnp.arange(self.memory_size, dtype = jnp.float64).reshape(-1)
        R = (2*Q + 1) / self.theta
        i, j = jnp.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * jnp.where(i < j, -1, (-1.0)**(i - j + 1))
        # print(A.shape)
        B = (R * ((-1.0)**Q)).reshape(-1, 1)
        # print(B.shape)
        C = jnp.ones((1, self.memory_size))
        D = jnp.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )

        # To torch.tensor
        # A = torch.from_numpy(A).float() # [memory_size, memory_size]
        # B = torch.from_numpy(B).float() # [memory_size, 1]
        
        return A, B
    
    def impulse(self, seq_len):
        """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """

        H = []
        A_i = self.A_i.clone()
        for t in range(seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i

        H = jnp.concatenate(H, axis = -1) # [memory_size, seq_len]
        # fft_H = fft.rfft(H, n = 2*seq_len, dim = -1) # [memory_size, seq_len + 1]
        fft_H = jrfft(H)
        

        return H, fft_H

    def __call__(
        self,
        x,
    ):
        # print(x.shape)
        batch_size, seq_len, input_size = x.shape

        # Equation 18 of the paper
        u = nn.relu(self.W_u(x)) # [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input = jnp.transpose(u, axes=[0, 2, 1]) #u.permute(0, 2, 1) # [batch_size, 1, seq_len]
        # fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) # [batch_size, seq_len, seq_len+1]
        

        fft_u = jrfft(fft_input)
        # Element-wise multiplication (uses broadcasting)
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        H, fft_H = self.impulse(seq_len=seq_len)
        temp = fft_u * jnp.expand_dims(fft_H, axis=0) # [batch_size, memory_size, seq_len+1]

        # m = fft.irfft(temp, n = 2*seq_len, dim = -1) # [batch_size, memory_size, seq_len+1]
        # m = jax.jit(jnp.fft.irfft)(temp, n = 2*seq_len, axis = -1)
        m = jirfft(temp)
        m = m[:, :, :seq_len] # [batch_size, memory_size, seq_len]
        m = jnp.transpose(m, axes=[0, 2, 1])#m.permute(0, 2, 1) # [batch_size, seq_len, memory_size]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = jnp.concatenate((m, x), axis = -1) # [batch_size, seq_len, memory_size + input_size]
        h = nn.relu(self.W_h(input_h)) # [batch_size, seq_len, hidden_size]

        h_n = h[:, -1, :] # [batch_size, hidden_size]

        return h, h_n
if __name__ == "__main__":
    x = jnp.ones((5, 256, 512))
    model = FlaxLMUFFT(512, 512, 512, 256)
    variables = model.init(jax.random.PRNGKey(0), x)
    # print(variables)
    x = jnp.ones((10, 128, 512))
    output = model.apply(variables, x)
    print(output[0])
    print(output[0].shape)
