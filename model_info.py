from transformers import AutoModel, AutoConfig

# Replace with your model's identifier or local path
model_name = "google/switch-base-8"

# Load the model configuration
config = AutoConfig.from_pretrained(model_name)

# Load the model
model = AutoModel.from_pretrained(model_name, config=config)

# Number of transformer layers
num_layers = config.num_hidden_layers
print(f"Number of Layers: {num_layers}")

# Hidden dimension size
hidden_dim = config.hidden_size
print(f"Hidden Dimension: {hidden_dim}")

# Number of attention heads
num_attention_heads = config.num_attention_heads
print(f"Number of Attention Heads: {num_attention_heads}")

model.summary()

# # Feed-forward network (FFN) dimension
# ffn_dim = config.intermediate_size
# print(f"Feed-Forward Network Dimension: {ffn_dim}")

# # Number of experts (specific to MoE models like Switch Transformer)
# # This assumes that the configuration includes a parameter for number of experts
# num_experts = getattr(config, 'num_experts', None)
# print(f"Number of Experts: {num_experts}")

# # Activation function
# activation_function = config.hidden_act
# print(f"Activation Function: {activation_function}")

# # Dropout rates
# dropout_rate = config.hidden_dropout_prob
# attention_dropout = config.attention_probs_dropout_prob
# print(f"Dropout Rate: {dropout_rate}")
# print(f"Attention Dropout Rate: {attention_dropout}")

# Other possible hyperparameters
print("\nOther Hyperparameters:")
for attr in dir(config):
    if not attr.startswith("__") and not callable(getattr(config, attr)):
        print(f"{attr}: {getattr(config, attr)}")
