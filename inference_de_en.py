import torch
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration
from transformers import AutoTokenizer
from datasets import load_dataset
from torchsummary import summary


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
test_num = 5

# for i in range(test_num):
#     # randomly select 1 test case
#     idx = random.randint(0, len(dataset["test"]))

#     # Select a test case from the dataset
#     test_case = dataset["test"][idx]
#     print("Original English Sentence:")
#     print(test_case["translation"]["en"])

#     # Tokenize the input
#     input_text = test_case["translation"]["en"]
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

#     # Generate translation
#     model.eval()
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

#     # Decode the generated tokens
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Print the generated translation
#     print("Generated German Translation:")
#     print(generated_text)

#     print("Reference German Translation:")
#     print(test_case["translation"]["de"])
#     print("\n")



# Tokenize the input
input_text = "What is this?"
inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

summary(model,input_size=inputs.shape)  # Replace input_size with the dimensions of your input


print("Original English Sentence:")
print(input_text)

# Generate translation
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

# Decode the generated tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated translation
print("Generated German Translation:")
print(generated_text)

