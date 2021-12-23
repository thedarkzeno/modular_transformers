
from typing import Callable, Optional, Tuple

import numpy as np

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention_weights
from jax import lax

from transformers.models.bert.modeling_flax_bert import *

from .config import Config
from .attention import Fourier_transform_flax
from .attention.flax_self_attention import FlaxAttentionHead

class FlaxModelSelfAttention(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`\
                    : {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

class FlaxAttention(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxModelSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxBertSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask=None,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs

class FlaxModelLayer(nn.Module):
    config: Config
    attention_type: str
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.attention_type == "fourier":
            self.attention = Fourier_transform_flax()  # BertAttention(config)
        else:
            self.attention = FlaxAttention(self.config, dtype=self.dtype)
        # self.attention = FlaxAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBertOutput(self.config, dtype=self.dtype)

        self.use_tiny_attention=self.config.use_tiny_attention
        self.tiny_attention = FlaxAttentionHead(self.config.hidden_size, self.config.tiny_dim, self.config.tiny_dim, self.config.hidden_size) if self.use_tiny_attention and self.attention_type != "self-attention" else None
        self.norm_res = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps) if self.tiny_attention is not None else None


    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        if self.attention_type == "fourier":
            attention_outputs = self.attention(hidden_states)
        else:
            attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                layer_head_mask=layer_head_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
        attention_output = attention_outputs[0]

        if self.tiny_attention is not None:
            att_res = self.tiny_attention(hidden_states)
            attention_output = self.norm_res(attention_output + att_res)

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class FlaxModelLayerCollection(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxModelLayer(self.config, attention_type=attention, name=str(i), dtype=self.dtype) for attention, i in zip(self.config.layers_attention_type, range(len(self.config.layers_attention_type)))
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for \
                        {head_mask.shape[0]}."
                )

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask=head_mask[i] if head_mask is not None else None,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxModelEncoder(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxModelLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Config
    base_model_prefix = "bert"
    module_class: nn.Module = None

    def __init__(
        self, config: Config, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
        )["params"]

    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            jnp.array(head_mask, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

class FlaxModelModule(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxModelEncoder(self.config, dtype=self.dtype)
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        head_mask: Optional[np.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class FlaxModel(FlaxBertPreTrainedModel):
    module_class = FlaxModelModule


class FlaxModelForMaskedLMModule(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxModelModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class FlaxModelForMaskedLM(FlaxBertPreTrainedModel):
    module_class = FlaxModelForMaskedLMModule

