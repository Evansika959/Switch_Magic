import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, DataCollatorWithPadding
from datasets import load_dataset
import os
import matplotlib.pyplot as plt

# Step 1: Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-base-8",
    device_map="auto"  # Automatically distribute the model across available devices
)

# Step 2: Load the WMT Dataset
# Load the WMT dataset
dataset = load_dataset("wmt16", "de-en")

# Print the first few items in the dataset to debug
print("First few items in the dataset:")
print(dataset["train"][0:3])

# Function to tokenize and preprocess data
def preprocess_function(examples):
    en_texts = [ex["en"] for ex in examples["translation"]]
    de_texts = [ex["de"] for ex in examples["translation"]]
    model_inputs = tokenizer(en_texts, text_target=de_texts,
                             max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Print the first few items in the tokenized dataset to debug
print("First few items in the tokenized dataset:")
print(tokenized_dataset["train"][0:3])

# Step 3: Dataset Splitting
# Split the dataset into training and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(tokenized_dataset["train"]))
val_size = len(tokenized_dataset["train"]) - train_size

train_dataset, val_dataset = random_split(tokenized_dataset["train"], [train_size, val_size])

# Initialize DataCollator
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# take only 10% of the dataset for faster training
train_dataset = train_dataset[list(range(0, len(train_dataset), len(train_dataset) // 10))]
val_dataset = val_dataset[list(range(0, len(val_dataset), len(val_dataset) // 10))]

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

# Step 4: Fine-tune the Model
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Training on:", device)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Define loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Directory for checkpoints
checkpoint_dir = './checkpoints_switch'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Initialize lists to track loss
train_losses = []
val_losses = []

# Early Stopping Parameters
patience = 5
best_val_loss = float('inf')
counter = 0

# Training Loop with Validation and Loss Tracking
epochs = 10

for epoch in range(epochs):
    # Training Phase
    model.train()
    epoch_train_loss = 0
    loop = tqdm(train_dataloader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        epoch_train_loss += loss.item()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Compute average training loss for the epoch
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_val_loss += loss.item()

    # Compute average validation loss for the epoch
    avg_val_loss = epoch_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    # Scheduler step based on validation loss
    scheduler.step(avg_val_loss)

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), f'{checkpoint_dir}/best_switch_transformer.pth')
        print(f"Epoch {epoch + 1} improved. Saving best model.")
    else:
        counter += 1
        print(f"Epoch {epoch + 1} did not improve.")
        if counter >= patience:
            print("Early stopping triggered.")
            break

    # Save a checkpoint at the end of each epoch
    torch.save(model.state_dict(), f'{checkpoint_dir}/switch_transformer_epoch_{epoch+1}.pth')
    print(f"Epoch {epoch + 1} completed. Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Plotting the Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()