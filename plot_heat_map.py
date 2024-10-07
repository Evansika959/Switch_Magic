import numpy as np
import matplotlib.pyplot as plt

# Data representing activated expert indices for each encoder block
data = {
    "encoder.block.1.layer.1.mlp": [7, 4, 6, 2, 1],
    "encoder.block.3.layer.1.mlp": [0, 6, 1, 7, 7],
    "encoder.block.5.layer.1.mlp": [2, 1, 2, 6, 1],
    "encoder.block.7.layer.1.mlp": [0, 4, 4, 5, 4],
    "encoder.block.9.layer.1.mlp": [5, 5, 5, 5, 5],
    "encoder.block.11.layer.1.mlp": [0, 0, 0, 0, 0]
}

# Extract the list of experts activated in each block
blocks = list(data.keys())
num_blocks = len(blocks)
max_experts = max(max(v) for v in data.values()) + 1  # Number of experts + 1 for zero-based indexing

# Create a matrix to store the expert activation counts for each block
activation_matrix = np.zeros((num_blocks, max_experts))

# Populate the activation matrix with the count of each expert activation
for idx, key in enumerate(blocks):
    for expert in data[key]:
        activation_matrix[idx, expert] += 1

# Create a heatmap using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(activation_matrix, cmap="hot", interpolation="nearest")
plt.colorbar(label="Activation Count")
plt.xticks(ticks=np.arange(max_experts), labels=[f"Expert {i}" for i in range(max_experts)])
plt.yticks(ticks=np.arange(num_blocks), labels=blocks)
plt.xlabel("Expert Index")
plt.ylabel("Encoder Block")
plt.title("Heatmap of Activated Experts in Encoder Blocks")
plt.savefig("plots/expert_activation_heatmap.png")
