# model_src/switch_transformer.py

from .activations import ACT2FN
from .generation import GenerationMixin
from .modeling_outputs import (
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput,
)
from .modeling_utils import PreTrainedModel
from .pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from .utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_switch_transformers import SwitchTransformersConfig
