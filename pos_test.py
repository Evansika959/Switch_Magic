import torch
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersSparseMLP
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import spacy
from spacy.tokens import Doc

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

# Define a custom tokenizer function for Spacy that uses the Hugging Face tokenizer
def custom_hf_tokenizer(nlp, text):
    # Use Hugging Face tokenizer to tokenize the text
    hf_tokens = tokenizer.tokenize(text)
    # Convert tokens to Spacy's token format
    words = [
        token.lstrip("▁") if token != "▁" and token.startswith("▁") else token
        for token in hf_tokens if token.strip()
    ]
    return Doc(nlp.vocab, words=words)

# Set the custom tokenizer in Spacy
nlp.tokenizer = lambda text: custom_hf_tokenizer(nlp, text)

# Load the WMT dataset
dataset = load_dataset("wmt16", "de-en")

# Randomize the test iteration
random.seed(40)
test_num = 1

rout_dict = {}

for i in range(test_num):
    # randomly select 1 test case
    idx = random.randint(0, len(dataset["test"]))

    # Select a test case from the dataset
    test_case = dataset["test"][idx]
    input_text = test_case["translation"]["en"]

    print("Original English Sentence:")
    print(input_text)

    # Step 1: POS Tagging using Spacy
    doc = nlp(input_text)
    pos_tags = [(token.text, token.pos_) for token in doc]

    
    # for token_text, pos in pos_tags:
    #     print(f"{token_text}: {pos}")

    # Step 2: Tokenize the input using Hugging Face tokenizer
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

    # Get the tokenized input (list of tokens)
    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print("Tokenized input:", input_tokens)
    print(inputs)

    print("input length:", len(input_tokens), len(inputs["input_ids"]))

    # Step 3: Align the Spacy tokens (words) with Hugging Face tokenizer tokens
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

    target_module = model.encoder.block[1].layer[1].mlp

    print(enumerate(target_module.router_history[:-1]))
    routing_rst = list(enumerate(target_module.router_history[:-1]))
    print(routing_rst)

    i = 0
    for token_text, pos in pos_tags:
        print(f"{token_text}: {pos}")
        if pos not in rout_dict:
            rout_dict[pos] = []

        rout_dict[pos].append(routing_rst[i])
        i+=1


print(rout_dict)

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

