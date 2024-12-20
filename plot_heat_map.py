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
plt.imshow(activation_matrix, cmap="Blues", interpolation="none")
plt.colorbar(label="Activation Count")
plt.xticks(ticks=np.arange(max_experts), labels=[f"Expert {i}" for i in range(max_experts)])
plt.yticks(ticks=np.arange(num_blocks), labels=blocks)
plt.xlabel("Expert Index")
plt.ylabel("Encoder Block")
plt.title("Heatmap of Activated Experts in Encoder Blocks")
# plt.savefig("plots/{name}.png")

def plot_heat_map(data,filename="heatmap",title="Heatmap of Activated Experts in Encoder Blocks"):
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

    activation_matrix = activation_matrix / activation_matrix.sum(axis=1, keepdims=True)

    # Create a heatmap using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(activation_matrix, cmap="Blues", interpolation="none")
    plt.colorbar(label="Activation Count")
    plt.xticks(ticks=np.arange(max_experts), labels=[f"Expert {i}" for i in range(max_experts)])
    plt.yticks(ticks=np.arange(num_blocks), labels=blocks)
    plt.xlabel("Expert Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    plt.savefig(f"plots/{filename}.png")

def plot_confidence_map(data, filename="confidence_map", title="Heatmap of Confidence in Encoder Blocks"):
    """
    Plot a heatmap of the confidence values of attention heads.

    Args:
        data (list or np.array): The confidence values to be plotted.
        filename (str): The filename to save the plot.
        title (str): The title of the heatmap.
    """
    plt.figure(figsize=(15, 12))  # Set the figure size
    plt.imshow(data, cmap="Blues", interpolation="none")  # Use imshow to plot the heatmap
    cbar = plt.colorbar()
    cbar.set_label("Average Confidence", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.title(title, fontsize=26)  # Set the title of the heatmap
    plt.xlabel("Attention Heads", fontsize=24)
    plt.ylabel("Layers", fontsize=24)
    # Annotate the heatmap with the confidence values
    # for i in range(len(data)):
    #     for j in range(len(data[0])):
    #         plt.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', color='black')

    plt.xticks(ticks=range(len(data[0])), labels=[f'{i}' for i in range(len(data[0]))], fontsize=22)  # Set x-axis ticks
    plt.yticks(ticks=range(len(data)), labels=[f'{i}' for i in range(len(data))], fontsize=22)  # Set y-axis ticks
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")  # Save the heatmap as a PNG file

def plot_pos_heat_map(data,filename="heatmap",title="Heatmap of Activated Experts in Encoder Blocks"):
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

    activation_matrix = activation_matrix / activation_matrix.sum(axis=1, keepdims=True)

    # Create a heatmap using matplotlib
    plt.figure(figsize=(15, 12))
    plt.imshow(activation_matrix, cmap="Blues", interpolation="none")
    cbar = plt.colorbar()
    cbar.set_label("Average Confidence", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(ticks=np.arange(max_experts), labels=[f"{i}" for i in range(max_experts)], fontsize=22)
    plt.yticks(ticks=np.arange(num_blocks), labels=blocks, fontsize=22)
    plt.xlabel("Expert Index", fontsize=24)
    plt.ylabel("POS Tags", fontsize=24)
    plt.title(title, fontsize=26)
    plt.savefig(f"plots/{filename}.png")



