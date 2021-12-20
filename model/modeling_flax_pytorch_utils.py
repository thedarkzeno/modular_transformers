from transformers.modeling_flax_pytorch_utils import *

def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    flax_checkpoint_path = os.path.abspath(flax_checkpoint_path)
    logger.info(f"Loading Flax weights from {flax_checkpoint_path}")

    # import correct flax class
    flax_cls = getattr(modular_transformers.model, "Flax" + model.__class__.__name__)

    # load flax weight dict
    with open(flax_checkpoint_path, "rb") as state_f:
        try:
            flax_state_dict = from_bytes(flax_cls, state_f.read())
        except UnpicklingError:
            raise EnvironmentError(f"Unable to convert {flax_checkpoint_path} to Flax deserializable object. ")

    return load_flax_weights_in_pytorch_model(model, flax_state_dict)


def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a Flax weights in PyTorch, requires both PyTorch and Flax to be installed. Please see "
            "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
        )
        raise

    # check if we have bf16 weights
    is_type_bf16 = flatten_dict(jax.tree_map(lambda x: x.dtype == jnp.bfloat16, flax_state)).values()
    if any(is_type_bf16):
        # convert all weights to fp32 if the are bf16 since torch.from_numpy can-not handle bf16
        # and bf16 is not fully supported in PT yet.
        logger.warning(
            "Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` "
            "before loading those in PyTorch model."
        )
        flax_state = jax.tree_map(
            lambda params: params.astype(np.float32) if params.dtype == jnp.bfloat16 else params, flax_state
        )

    flax_state_dict = flatten_dict(flax_state)
    pt_model_dict = pt_model.state_dict()

    load_model_with_head_into_base_model = (pt_model.base_model_prefix in flax_state) and (
        pt_model.base_model_prefix not in set([k.split(".")[0] for k in pt_model_dict.keys()])
    )
    load_base_model_into_model_with_head = (pt_model.base_model_prefix not in flax_state) and (
        pt_model.base_model_prefix in set([k.split(".")[0] for k in pt_model_dict.keys()])
    )

    # keep track of unexpected & missing keys
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    for flax_key_tuple, flax_tensor in flax_state_dict.items():
        has_base_model_prefix = flax_key_tuple[0] == pt_model.base_model_prefix
        require_base_model_prefix = ".".join((pt_model.base_model_prefix,) + flax_key_tuple) in pt_model_dict

        # adapt flax_key to prepare for loading from/to base model only
        if load_model_with_head_into_base_model and has_base_model_prefix:
            flax_key_tuple = flax_key_tuple[1:]
        elif load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key_tuple = (pt_model.base_model_prefix,) + flax_key_tuple

        # rename flax weights to PyTorch format
        if flax_key_tuple[-1] == "kernel" and flax_tensor.ndim == 4 and ".".join(flax_key_tuple) not in pt_model_dict:
            # conv layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = jnp.transpose(flax_tensor, (3, 2, 0, 1))
        elif flax_key_tuple[-1] == "kernel" and ".".join(flax_key_tuple) not in pt_model_dict:
            # linear layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = flax_tensor.T
        elif flax_key_tuple[-1] in ["scale", "embedding"]:
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

        flax_key = ".".join(flax_key_tuple)

        if flax_key in pt_model_dict:
            if flax_tensor.shape != pt_model_dict[flax_key].shape:
                raise ValueError(
                    f"Flax checkpoint seems to be incorrect. Weight {flax_key_tuple} was expected "
                    f"to be of shape {pt_model_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )
            else:
                # add weight to pytorch dict
                flax_tensor = np.asarray(flax_tensor) if not isinstance(flax_tensor, np.ndarray) else flax_tensor
                pt_model_dict[flax_key] = torch.from_numpy(flax_tensor)
                # remove from missing keys
                missing_keys.remove(flax_key)
        else:
            # weight is not expected by PyTorch model
            unexpected_keys.append(flax_key)

    pt_model.load_state_dict(pt_model_dict)

    # re-transform missing_keys to list
    missing_keys = list(missing_keys)

    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the Flax model were not used when "
            f"initializing the PyTorch model {pt_model.__class__.__name__}: {unexpected_keys}\n"
            f"- This IS expected if you are initializing {pt_model.__class__.__name__} from a Flax model trained on another task "
            "or with another architecture (e.g. initializing a BertForSequenceClassification model from a FlaxBertForPreTraining model).\n"
            f"- This IS NOT expected if you are initializing {pt_model.__class__.__name__} from a Flax model that you expect "
            "to be exactly identical (e.g. initializing a BertForSequenceClassification model from a FlaxBertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All Flax model weights were used when initializing {pt_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model "
            f"and are newly initialized: {missing_keys}\n"
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the Flax model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    return pt_model