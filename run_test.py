# layer_lrp_attention_heads.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SwitchTransformersForConditionalGeneration
from captum.attr import LayerLRP
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Model and Tokenizer
model_name = 'google/switch-base-8'  # Replace with 'google/switch_transformer-base-8' when available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
model.eval()

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
target_layer = model.decoder.block[0].layer[0].SelfAttention
layer_lrp = LayerLRP(custom_forward, target_layer)

# Step 5: Compute Attributions
attributions = layer_lrp.attribute(
    input_ids,
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

