import torch
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersSparseMLP
import transformers_cp
from transformers import AutoTokenizer
from datasets import load_dataset
from torchsummary import summary
import re
from plot_heat_map import plot_heat_map, plot_confidence_map

def calculate_confidence_encoder(attention_weights):
    """
    Calculate the confidence of each attention head.
    Confidence is defined as the average of the maximum attention weights, excluding the EOS token.

    Args:
        attention_weights (torch.Tensor): The attention weights from the model, shape [batch_size, n_heads, seq_length, key_length].
        eos_token_idx (int): Index of the end-of-sequence (EOS) token.

    Returns:
        list: A list of confidence values for each attention head.
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    confidences = []

    # Iterate over each head
    for head in range(num_heads):
        max_weights = []  # Store max attention weights for each token

        # Iterate over each sequence in the batch
        for batch in range(batch_size):
            # Iterate over each token in the sequence (query tokens)
            for token in range(seq_len):
                if token != seq_len:  # Exclude the EOS token
                    # Extract the attention weights for the current head and token
                    head_weights = attention_weights[batch, head, token][:-1]

                    # Find the maximum attention weight for this token over all key positions
                    max_weight = torch.max(head_weights)
                    max_weights.append(max_weight.item())

        # Compute the average of max weights for the current head
        confidence = sum(max_weights) / len(max_weights) if len(max_weights) > 0 else 0
        confidences.append(confidence)

    return confidences


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-base-8",
    device_map="auto"  # Automatically distribute the model across available devices
)
model.load_state_dict(torch.load('./checkpoints_switch/best_switch_transformer.pth'))

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the WMT dataset
dataset = load_dataset("wmt16", "de-en")

# Randomize the test iteration
import random
random.seed(42)
test_num = 200

conf_matrix = torch.zeros(8, 12, 12)  # 8 heads, 12 layers, 12 heads

decoder_conf_matrix = torch.zeros(8, 12, 12)

cross_conf_matrix = torch.zeros(8, 12, 12)

pattern_attn = r'^encoder\..*\.SelfAttention$'
pattern_attn_de = r'^decoder\..*\.SelfAttention$'
pattern_attn_cross = r'^decoder\..*\.EncDecAttention$'

pattern_en_mlp = r'^encoder\..*\.mlp$'
pattern_de_mlp = r'^decoder\..*\.mlp$'

encoder_router_history = {}
decoder_router_history = {}

for i in range(test_num):
    # randomly select 1 test case
    idx = random.randint(0, len(dataset["test"]))

    # Select a test case from the dataset
    test_case = dataset["test"][idx]
    # print("Original English Sentence:")
    # print(test_case["translation"]["en"])

    # Tokenize the input
    input_text = test_case["translation"]["en"]
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    label_text = test_case["translation"]["de"]
    labels = tokenizer(label_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    label_ids = labels['input_ids'].to(device)
    decoder_mask = labels['attention_mask'].to(device)

    # Generate translation
    model.eval()

    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids, decoder_attention_mask=decoder_mask)
        loss = outputs.loss

    for name, module in model.named_modules():
        if re.match(pattern_en_mlp, name) and isinstance(module, SwitchTransformersSparseMLP):
            # print(name)
            print(module.router_history)
        if re.match(pattern_attn, name) and isinstance(module, transformers_cp.src.transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersAttention):
            match = re.search(r'block\.(\d+)', name)

            if match:
                layer_num = int(match.group(1))
            else:
                print("Layer number not found")
            print("attention_weights: ", module.saved_attention_weights.shape)

            mlp_module = model.encoder.block[layer_num].layer[1].mlp
            print("mlp_module: ", mlp_module)

            confidence = calculate_confidence_encoder(module.saved_attention_weights)
            conf_matrix[layer_num] += torch.tensor(confidence)

        # if re.match(pattern_attn_de, name) and isinstance(module, transformers_cp.src.transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersAttention):
        #     match = re.search(r'block\.(\d+)', name)

        #     if match:
        #         layer_num = int(match.group(1))
        #     else:
        #         print("Layer number not found")

        #     confidence = calculate_confidence_encoder(module.saved_attention_weights)
        #     decoder_conf_matrix[layer_num] += torch.tensor(confidence)

        # if re.match(pattern_attn_cross, name) and isinstance(module, transformers_cp.src.transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersAttention):
        #     match = re.search(r'block\.(\d+)', name)

        #     if match:
        #         layer_num = int(match.group(1))
        #     else:
        #         print("Layer number not found")

        #     confidence = calculate_confidence_encoder(module.saved_attention_weights)
        #     cross_conf_matrix[layer_num] += torch.tensor(confidence)

    print("Test Case:", i, "Loss:", loss.item())

decoder_conf_matrix /= test_num
cross_conf_matrix /= test_num
conf_matrix /= test_num

# Tokenize the input
input_text = "What is this?"
inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

print("Original English Sentence:")
print(input_text)
print(inputs)

# Generate translation
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

# Decode the generated tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated translation
print("Generated German Translation:")
print(generated_text)

# Regex pattern to match all strings starting with "encoder" and ending with ".mlp"



# for name, module in model.named_modules():
#     if re.match(pattern, name) and isinstance(module, SwitchTransformersSparseMLP):
#         # print(name)
#         print(module.router_history.shape)
#         encoder_router_history[re.search(r'encoder\.block\.\d+', name).group()] = torch.cat(module.router_history).flatten()
#         # print("\n")
#     if re.match(pattern2, name) and isinstance(module, SwitchTransformersSparseMLP):
#         # print(name)
#         # print(module.router_history)
#         decoder_router_history[re.search(r'decoder\.block\.\d+', name).group()] = torch.cat(module.router_history).flatten()
#         # print("\n")
        

# plot_heat_map(encoder_router_history, filename="encoder_router_history_cmp", title="Router History of Encoder Blocks")
# plot_heat_map(decoder_router_history, filename="decoder_router_history_cmp", title="Router History of Decoder Blocks")


# plot_confidence_map(conf_matrix, filename="conf_mat", title="Confidence Matrix of Encoder Self-Attention")
# plot_confidence_map(decoder_conf_matrix, filename="decoder_conf_mat", title="Confidence Matrix of Decoder Self-Attention")
# plot_confidence_map(cross_conf_matrix, filename="cross_conf_mat", title="Confidence Matrix of Decoder Cross-Attention")


