import torch
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersSparseMLP
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import spacy

import re

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

# Load Spacy English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Load the WMT dataset
dataset = load_dataset("wmt16", "de-en")

# Randomize the test iteration
random.seed(42)
test_num = 1

for i in range(test_num):
    # randomly select 1 test case
    idx = random.randint(0, len(dataset["test"]))

    # Select a test case from the dataset
    test_case = dataset["test"][idx]
    input_text = test_case["translation"]["en"]

    # Step 1: POS Tagging using Spacy
    doc = nlp(input_text)
    pos_tags = [(token.text, token.pos_) for token in doc]

    print("Words and their POS tags:")
    for token_text, pos in pos_tags:
        print(f"{token_text}: {pos}")

    # Step 2: Tokenize the input using Hugging Face tokenizer
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

    # Get the tokenized input (list of tokens)
    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print("Tokenized input:", input_tokens)

    # Step 3: Align the Spacy tokens (words) with Hugging Face tokenizer tokens
    # This step is necessary because Spacy tokens and the Hugging Face tokenizer tokens may not always align directly.
    print("\nAlignment of tokens and POS tags:")
    word_idx = 0
    for token in input_tokens:
        if token.startswith("▁"):  # If it's a new word, check alignment with Spacy token
            word_text, word_pos = pos_tags[word_idx]
            print(f"Token: {token}, Word: {word_text}, POS: {word_pos}")
            word_idx += 1
        else:
            # Sub-word tokens are parts of previous words
            print(f"Sub-token: {token}, belongs to word: {pos_tags[word_idx-1][0]}, POS: {pos_tags[word_idx-1][1]}")

    # Step 4: Generate translation using the model
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated Translation (German):")
    print(generated_text)

    # Print reference German translation
    print("\nReference Translation (German):")
    print(test_case["translation"]["de"])

module = model.encoder.block[0].layer[1].mlp

print(module.router_history)

# Regex pattern to match all strings starting with "encoder" and ending with ".mlp"
pattern = r'^encoder.block.1\..*\.mlp$' 
pattern2 = r'^decoder\..*\.mlp$'

encoder_router_history = {}
decoder_router_history = {}

for name, module in model.named_modules():
    if re.match(pattern, name) and isinstance(module, SwitchTransformersSparseMLP):
        print(name)
        print(module.router_history)
        encoder_router_history[re.search(r'encoder\.block\.\d+', name).group()] = torch.cat(module.router_history).flatten()
        # print("\n")
    # if re.match(pattern2, name) and isinstance(module, SwitchTransformersSparseMLP):
    #     # print(name)
    #     # print(module.router_history)
    #     decoder_router_history[re.search(r'decoder\.block\.\d+', name).group()] = torch.cat(module.router_history).flatten()
        # print("\n")


# plot_heat_map(encoder_router_history, filename="encoder_router_history_de2en", title="Router History of Encoder Blocks")
# plot_heat_map(decoder_router_history, filename="decoder_router_history_de2en", title="Router History of Decoder Blocks")

#calculate LRP

