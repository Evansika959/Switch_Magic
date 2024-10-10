# layer_lrp_attention_heads.py

import torch
from transformers import AutoTokenizer
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration
import transformers_cp.src.transformers.models.switch_transformers
from captum.attr._core.layer.layer_lrp import LayerLRP
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule, Alpha1_Beta0_Rule
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# Step 1: Load the Model and Tokenizer
model_name = 'google/switch-base-8'  # Replace with 'google/switch_transformer-base-8' when available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
model.load_state_dict(torch.load('./checkpoints_switch_forLRP/best_switch_transformer.pth'))
# model.load_state_dict(torch.load('./checkpoints_switch/best_switch_transformer.pth'))
model.eval()

# Print the model architecture
def print_model(model):
    if isinstance(model, transformers_cp.src.transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersAttention):
        print("\n")
        print("in attention")
        print(model)
        print(model.children())
        print("\n")
    for layer in model.children():
        if len(list(layer.children())) == 0:
            print(layer)
        else:
            print_model(layer)

print_model(model)

# Step 2: Prepare the Input Text
text = "What is this?"
inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids']
inputs_embeds = model.get_input_embeddings()(input_ids)
print(inputs_embeds)
attention_mask = inputs['attention_mask']

# Prepare decoder_input_ids
target_text = "Was ist"
decoder_inputs = tokenizer(target_text, return_tensors='pt')
decoder_input_ids = decoder_inputs['input_ids']
zero_token = torch.tensor([[0]], dtype=torch.long)  # Shape: (batch_size=1, seq_len=1)
decoder_input_ids = torch.cat([zero_token, decoder_input_ids], dim=1)  # Concatenate along sequence dimension
# Remove the last token
decoder_input_ids = decoder_input_ids[:, :-1]
# decoder_input_ids[:, -1] = next_token_id
decoder_inputs_embeds = model.get_input_embeddings()(decoder_input_ids)
print(decoder_input_ids)
print(decoder_inputs_embeds)
decoder_attention_mask = None

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
        elif isinstance(module, transformers_cp.src.transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersLayerNorm):
            # Assign IdentityRule to LayerNorm layer
            setattr(module, 'rule', IdentityRule())
        # elif isinstance(module, transformers_cp.src.transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersAttention):
        #     setattr(module, 'rule', EpsilonRule())
        #     setattr(module, 'relevance_output', None)  

assign_lrp_rules(model)

target_layer = model.decoder.block[0].layer[0].SelfAttention

outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
# print(outputs)
logits = outputs.logits
print("logits: ", logits.shape)
# get the predicted logits
logits_last_token = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
print("logits_last_token: ", logits_last_token)

# Get the top-1 predicted class index
top1_indices = torch.argmax(logits_last_token, dim=-1)  # Shape: (batch_size,)
print("predicted indices: ", top1_indices)
# get the predicted word
predicted_word = tokenizer.decode(top1_indices)
print("predicted word: ", predicted_word)

layer_lrp = LayerLRP(model, layer=target_layer)

# Step 5: Compute Attributions
attributions = layer_lrp.attribute(
    inputs = inputs_embeds,
    additional_forward_args=(None, attention_mask, None, decoder_inputs_embeds, decoder_attention_mask, True),
    target=top1_indices,
    attribute_to_layer_input=False,
    verbose=True
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

