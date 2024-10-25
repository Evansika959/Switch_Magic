from torchtext.datasets import Multi30k

# Load the dataset
train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))

# Iterate over the dataset
for i, (en_sentence, de_sentence) in enumerate(train_data):
    print(f"Example {i+1}")
    print("English: ", en_sentence)
    print("German: ", de_sentence)
    if i == 0:  # Only print the first example
        break