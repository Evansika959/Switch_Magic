import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from datasets import load_dataset

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

# Select a test case from the dataset
test_case = dataset["test"][0]
print("Original German Sentence:")
print(test_case["translation"]["de"])

# Tokenize the input
input_text = test_case["translation"]["de"]
inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

# Generate translation
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

# Decode the generated tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated translation
print("Generated English Translation:")
print(generated_text)