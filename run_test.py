import torch
from transformers import AutoTokenizer
from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration


# Step 1: Load the Model and Tokenizer
model_name = 'google/switch-base-8'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
model.load_state_dict(torch.load('./checkpoints_switch_forLRP/best_switch_transformer.pth'))

model.eval()  # Set model to evaluation mode

# Step 2: Prepare the Encoder Input
# Source text to encode
text = "What is this?"
inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Step 3: Initialize the Decoder Input with <bos> token
# Prepare the decoder start token (typically <bos>)
decoder_start_token_id = tokenizer.pad_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]])  # Shape: (batch_size=1, seq_len=1)

# Step 4: Perform Forward Pass to Generate the Next Token
# Loop to generate tokens one at a time
max_generation_steps = 5  # Limit the number of tokens to generate for demonstration
for _ in range(max_generation_steps):
    # Forward pass with encoder and decoder inputs
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    # Get logits for the last token in the sequence
    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    logits_last_token = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

    # Get the most likely next token using argmax
    next_token_id = torch.argmax(logits_last_token, dim=-1)  # Shape: (batch_size,)

    # Append the predicted token to the decoder input ids
    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)

    # Decode the generated token for visualization
    generated_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
    print("Generated Token:", generated_text)

# Step 5: Final Generated Sequence
full_generated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
print("Full Generated Sequence:", full_generated_text)
