import json
import re
import collections
import numpy as np

class BPEZipfTokenizer:
    def __init__(self, vocab_size=30000, max_word_length=25):
        self.vocab_size = vocab_size
        self.max_word_length = max_word_length
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_frequencies = collections.Counter()
    
    def process_corpus(self, file_paths):
        """Reads multiple JSON files and processes text to build the vocabulary."""
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._extract_text(data)
        
        self._build_vocab()
    
    def _extract_text(self, data):
        """Recursively extracts text from JSON fields."""
        if isinstance(data, dict):
            for value in data.values():
                self._extract_text(value)
        elif isinstance(data, list):
            for item in data:
                self._extract_text(item)
        elif isinstance(data, str):
            words = re.findall(r'\b\w{1,' + str(self.max_word_length) + r'}\b', data.lower())  # Limit word length
            self.word_frequencies.update(words)
    
    def _build_vocab(self):
        """Applies Byte Pair Encoding (BPE) and assigns indices based on Zipf's Law."""
        sorted_words = [word for word, _ in self.word_frequencies.most_common(self.vocab_size - 4)]
        
        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.vocab = {word: idx for idx, word in enumerate(special_tokens + sorted_words)}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def encode(self, text):
        """Encodes a given text into token indices."""
        words = re.findall(r'\b\w{1,' + str(self.max_word_length) + r'}\b', text.lower())
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in words] + [self.vocab["<EOS>"]]
    
    def decode(self, token_indices):
        """Decodes token indices back into text."""
        return ' '.join([self.reverse_vocab.get(idx, '<UNK>') for idx in token_indices if idx not in [self.vocab["<PAD>"], self.vocab["<EOS>"]]])
    
    def save_tokenizer(self, output_file):
        """Saves tokenizer vocabulary and settings to a JSON file."""
        tokenizer_data = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "max_word_length": self.max_word_length
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
    
    def load_tokenizer(self, input_file):
        """Loads tokenizer vocabulary and settings from a JSON file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            self.vocab = tokenizer_data["vocab"]
            self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
            self.vocab_size = tokenizer_data["vocab_size"]
            self.max_word_length = tokenizer_data["max_word_length"]

# Example Usage
if __name__ == "__main__":
    tokenizer = BPEZipfTokenizer()
    json_files = [r"INSERT DATASET PATH HERE", r"INSERT ADDITIONAL DATASETS SEPARATED BY COMMAS, BEGINNING WITH r TO ALLOW FOR PATH"]
    tokenizer.process_corpus(json_files)
    
    # Save tokenizer for future use
    tokenizer.save_tokenizer("tokenizer.json")
    
    # Test encoding and decoding
    sample_text = "Hello world, this is a test."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("Encoded:", encoded)
    print("Decoded:", decoded)

