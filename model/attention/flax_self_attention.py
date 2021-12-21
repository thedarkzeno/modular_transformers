import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.attention import dot_product_attention_weights


class FlaxAttentionHead(nn.Module):
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self, dim_k: int, dim_v: int, dim_out: int = None):
        self.q = nn.Dense(
            dim_k,
            dtype=self.dtype,
        )
        self.k = nn.Dense(
            dim_k,
            dtype=self.dtype,
        )
        self.v = nn.Dense(
            dim_v,
            dtype=self.dtype,
        )
        if dim_out is not None:
            self.out = nn.Dense(dim_out, dtype=self.dtype,)
        else:
            self.out = None

    def __call__(
        self,
        hidden_states,
    ):

        query_states = self.query(hidden_states)
        value_states = self.value(hidden_states)
        key_states = self.key(hidden_states)

        attention_bias = None

        dropout_rng = None

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=True,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        return (attn_output,)
