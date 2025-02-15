import json
import re
import glob
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

# Parameters
input_files = glob.glob(r"C:\Users\abias\OpenThoughts-114k\data\*.json")  # Adjust file path pattern
max_vocab_size = 10000  # Adjust as needed
prioritize_alphabet = False  # Set to True if you want a-z at the start

# Step 1: Process files in chunks (avoiding memory overload)
token_freqs = Counter()

for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        try:
            text_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {file}, invalid JSON.")
            continue

    # Ensure dataset is a list of conversation dictionaries
    if isinstance(text_data, list):
        for entry in text_data:
            if isinstance(entry, dict) and "conversations" in entry:
                conversations = entry["conversations"]
                if isinstance(conversations, list):
                    for message in conversations:
                        if isinstance(message, dict) and "from" in message and "value" in message:
                            text = f"{message['from']}: {message['value']}"
                            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())  # Extract words
                            token_freqs.update(words)  # Update word frequencies directly
            else:
                print(f"Skipping malformed entry in {file}: {entry}")

    else:
        print(f"Unexpected structure in {file}, skipping...")

# Step 2: Sort tokens by Zipf’s Law and limit vocabulary size
if prioritize_alphabet:
    sorted_tokens = list("abcdefghijklmnopqrstuvwxyz") + [
        token for token, _ in token_freqs.most_common(max_vocab_size - 26)
    ]
else:
    sorted_tokens = [token for token, _ in token_freqs.most_common(max_vocab_size)]

# Step 3: Save sorted tokens to a new training file
sorted_file = "sorted_text.txt"
with open(sorted_file, "w", encoding="utf-8") as f:
    f.write(" ".join(sorted_tokens))

# Step 4: Initialize BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

# Step 5: Define trainer with optional alphabet prioritization
special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
if prioritize_alphabet:
    special_tokens += list("abcdefghijklmnopqrstuvwxyz")

trainer = BpeTrainer(
    vocab_size=max_vocab_size,
    special_tokens=special_tokens,
    min_frequency=2  # Ensures rare tokens aren't added
)

# Step 6: Train tokenizer on the reordered dataset
tokenizer.train(files=[sorted_file], trainer=trainer)

# Step 7: Set a decoder and save
tokenizer.decoder = ByteLevelDecoder()
tokenizer.save("tokenizer.json")

print(f"Tokenizer training completed using {len(input_files)} files and saved as tokenizer.json (English only)")
