from ..file_utils import is_torch_available, is_flax_available
if is_torch_available():
    from .fourier import fourier_transform
    from .self_attention import SelfAttention, AttentionHead
if is_flax_available():
    from .fourier_flax import Fourier_transform_flax
