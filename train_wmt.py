from transformers_cp.src.transformers.models.switch_transformers import SwitchTransformersForConditionalGeneration
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import torch

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    "google/switch-base-8",
    device_map="auto"
)

# Load the WMT16 English-German dataset
dataset = load_dataset("wmt16", "de-en")

print(dataset)

# Define the number of examples to inspect
num_examples = 3

# Function to print examples from a split
def print_examples(split_name, split):
    print(f"\n--- {split_name.capitalize()} Split ---\n")
    for i in range(num_examples):
        example = split[i]
        source = example['translation']['de']
        target = example['translation']['en']
        print(f"Example {i+1}:")
        print(f"Source (German): {source}")
        print(f"Target (English): {target}\n")

# Print examples from the training set
print_examples("train", dataset["train"])

# Print examples from the validation set
print_examples("validation", dataset["validation"])

# Preprocess the data with reduced max_length
def preprocess_function(examples):
    sources = examples['translation']['de']
    targets = examples['translation']['en']
    
    # Tokenize the source sentences
    model_inputs = tokenizer(
        sources,
        max_length=256,           # Reduced from typical higher values
        truncation=True,
        padding='max_length'     # Optional: adjust padding as needed
    )
    
    # Tokenize the target sentences
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=256,       # Reduced from typical higher values
            truncation=True,
            padding='max_length' # Optional: adjust padding as needed
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing to the dataset
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Define the number of tokenized examples to inspect
num_tokenized_examples = 3

# Function to print tokenized examples from a split
def print_tokenized_examples(split_name, split):
    print(f"\n--- Tokenized {split_name.capitalize()} Split ---\n")
    for i in range(num_tokenized_examples):
        example = split[i]
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        labels = example['labels']
        print(f"Example {i+1}:")
        print(f"Input IDs: {input_ids}")
        print(f"Attention Mask: {attention_mask}")
        print(f"Labels: {labels}\n")

# Print tokenized examples from the training set
print_tokenized_examples("train", tokenized_datasets["train"])

# Print tokenized examples from the validation set
print_tokenized_examples("validation", tokenized_datasets["validation"])

# Initialize the data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True  # Enables dynamic padding
)

# Load BLEU metric
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    decoded_labels = [[label] for label in decoded_labels]
    
    # Compute BLEU score
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract the BLEU score
    result = {"bleu": result["score"]}
    
    return result

# Define training arguments with adjusted batch sizes
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,    # Adjust based on GPU memory
    per_device_eval_batch_size=16,     # Adjust based on GPU memory
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(f"BLEU Score: {results['bleu']}")

# Save the trained model and tokenizer
trainer.save_model("./trained_switch_transformer_wmt_en_de")
tokenizer.save_pretrained("./trained_switch_transformer_wmt_en_de")
