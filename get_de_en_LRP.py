# layer_lrp_attention_heads.py

import torch
from transformers import AutoTokenizer
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration
from captum.attr import LayerLRP
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

#dummy class
class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the SwitchTransformers style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # SwitchTransformers uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

# Step 1: Load the Model and Tokenizer
model_name = 'google/switch-base-8'  # Replace with 'google/switch_transformer-base-8' when available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
model.eval()

print(model.config)
print(model)

# Step 2: Prepare the Input Text
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Step 3: Define a Custom Forward Function
def custom_forward(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Get decoder attentions from the first layer
    attentions = outputs.decoder_attentions[0]  # Shape: (batch_size, num_heads, seq_len, seq_len)
    # Aggregate attentions over the sequence dimension
    attentions_mean = attentions.mean(dim=-1).mean(dim=-1)  # Shape: (batch_size, num_heads)
    # Sum over heads to get a scalar output for attribution
    return attentions_mean.sum(dim=1)

# Step 4: Set Up LayerLRP
# We focus on the self-attention layer in the first decoder block
def assign_lrp_rules(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            # Assign EpsilonRule to Embedding layer
            setattr(module, 'rule', EpsilonRule())
        elif isinstance(module, SwitchTransformersLayerNorm):
            print("here")
            # Assign IdentityRule to LayerNorm layer
            setattr(module, 'rule', IdentityRule())
        # Add additional conditions for other layer types if necessary

assign_lrp_rules(model)

target_layer = model.decoder.block[0].layer[0].SelfAttention
layer_lrp = LayerLRP(model, layer=target_layer)

# layer_lrp.rule_map[torch.nn.Embedding] = EpsilonRule()
print(layer_lrp)

# Step 5: Compute Attributions
attributions = layer_lrp.attribute(
    input_ids,
    # forward_func=custom_forward,
    additional_forward_args=(attention_mask,),
    attribute_to_layer_input=False
)

# Step 6: Extract and Analyze Attention Head Importance
# Convert attributions to numpy array
attr_np = attributions.detach().numpy()[0]  # Assuming batch_size=1

# Normalize the attributions
attr_np_normalized = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())

# Step 7: Visualize the Attributions
num_heads = attr_np.shape[0]
plt.figure(figsize=(10, 6))
plt.bar(range(num_heads), attr_np_normalized)
plt.xlabel('Attention Head')
plt.ylabel('Normalized Attribution Score')
plt.title('Attention Head Importance')
plt.xticks(range(num_heads))

