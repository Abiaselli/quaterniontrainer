import json
import re
import collections

class BPEZipfTokenizer:
    def __init__(self, vocab_size=30000, max_word_length=25):
        self.vocab_size = vocab_size
        self.max_word_length = max_word_length
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_frequencies = collections.Counter()
        self.bpe_merges = []  # List to hold learned merge operations

    def process_corpus(self, file_paths):
        """Reads multiple JSON files and processes text to build the vocabulary."""
        for file_path in file_paths:
            print(f"Processing file: {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._extract_text(data)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        print(f"Finished processing files. Unique words count: {len(self.word_frequencies)}")
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
            # Use regex to capture words up to max_word_length.
            # (Note: this omits punctuation. If you want full punctuation retention,
            #  consider using a different regex or a split that preserves whitespace.)
            words = re.findall(r'\b\w{1,' + str(self.max_word_length) + r'}\b', data.lower())
            self.word_frequencies.update(words)
    
    def _build_vocab(self):
        """Learns BPE merges and builds the subword vocabulary based on subword frequencies."""
        print("Learning BPE merges...")
        self.bpe_merges = self._learn_bpe()
        print(f"Learned {len(self.bpe_merges)} merge operations.")
        
        # Build a frequency dictionary of subword tokens
        subword_freq = collections.Counter()
        for word, freq in self.word_frequencies.items():
            bpe_tokens = self._apply_bpe_to_word(word)
            for token in bpe_tokens:
                subword_freq[token] += freq
        
        # Define special tokens first.
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        num_tokens_to_select = self.vocab_size - len(special_tokens)
        sorted_tokens = [token for token, _ in subword_freq.most_common(num_tokens_to_select)]
        all_tokens = special_tokens + sorted_tokens
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"Vocabulary built with {len(self.vocab)} tokens.")
    
    def _learn_bpe(self):
        """
        Learns BPE merge operations from the corpus.
        Each word is represented as a tuple of characters plus an end-of-word marker.
        """
        vocab = {}
        for word, freq in self.word_frequencies.items():
            word = word[:self.max_word_length]
            tokens = tuple(word) + ("</w>",)
            vocab[tokens] = freq
        
        merges = []
        symbols = set()
        for tokens in vocab.keys():
            symbols.update(tokens)
        # Determine number of merges to perform (accounting for special tokens)
        num_merges = self.vocab_size - len(symbols) - 4
        if num_merges < 0:
            num_merges = 0
        
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            merges.append(best_pair)
            vocab = self._merge_vocab(best_pair, vocab)
            if (i + 1) % 100 == 0 or i == num_merges - 1:
                print(f"Completed merge {i + 1}/{num_merges}")
        return merges
    
    def _get_stats(self, vocab):
        """Count frequency of adjacent symbol pairs."""
        stats = collections.Counter()
        for tokens, freq in vocab.items():
            for i in range(len(tokens) - 1):
                stats[(tokens[i], tokens[i+1])] += freq
        return stats
    
    def _merge_vocab(self, pair, vocab):
        """Merge all occurrences of the given pair in the vocabulary."""
        merged_vocab = {}
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            merged_vocab[tuple(new_tokens)] = freq
        return merged_vocab
    
    def _get_pairs(self, tokens):
        """Return the set of adjacent token pairs."""
        return {(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)}
    
    def _apply_bpe_to_word(self, word, add_marker=True):
        """
        Applies learned BPE merges to a given word and returns a list of subword tokens.
        If add_marker is True, all tokens except the last are marked with '@@' (a common BPE convention).
        """
        tokens = list(word) + ["</w>"]
        merge_rules = {pair: i for i, pair in enumerate(self.bpe_merges)}
        
        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            candidate = None
            candidate_rank = float('inf')
            for pair in pairs:
                if pair in merge_rules and merge_rules[pair] < candidate_rank:
                    candidate = pair
                    candidate_rank = merge_rules[pair]
            if candidate is None:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == candidate:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Remove end-of-word marker.
        if tokens[-1] == "</w>":
            tokens = tokens[:-1]
        elif tokens[-1].endswith("</w>"):
            tokens[-1] = tokens[-1].replace("</w>", "")
        
        if add_marker and len(tokens) > 1:
            # This marker signals that the subword is not final.
            return [token + "@@" for token in tokens[:-1]] + [tokens[-1]]
        else:
            return tokens
    
    def encode(self, text):
        """
        Encodes text into token indices using BPE segmentation.
        
        Note: In this simple implementation we split on word boundaries.  
        For a full ByteLevel approach you might want to preserve punctuation and spaces.
        """
        # Here we use a simple word split. To better align with a ByteLevel paradigm,
        # you could change this to a regex that preserves punctuation and whitespace.
        words = re.findall(r'\w+', text.lower())
        token_indices = []
        for word in words:
            bpe_tokens = self._apply_bpe_to_word(word)
            for token in bpe_tokens:
                token_indices.append(self.vocab.get(token, self.vocab["<UNK>"]))
        token_indices.append(self.vocab["<EOS>"])
        return token_indices
    
    def decode(self, token_indices):
        """
        Decodes token indices back into text.
        
        The method strips out special tokens (<PAD>, <EOS>) and then reassembles words
        by removing the subword marker ("@@") from all tokens except the final subword.
        """
        tokens = [self.reverse_vocab.get(idx, "<UNK>") 
                  for idx in token_indices 
                  if idx not in (self.vocab["<PAD>"], self.vocab["<EOS>"])]
        words = []
        current_word = ""
        for token in tokens:
            if token.endswith("@@"):
                current_word += token[:-2]
            else:
                current_word += token
                words.append(current_word)
                current_word = ""
        if current_word:
            words.append(current_word)
        return " ".join(words)
    
    def save_tokenizer(self, output_file):
        """
        Saves the tokenizer in a production format.
        
        This implementation has been updated to use a ByteLevel pre-tokenizer and decoder,
        and to specify the model type as "BPE" with the learned merges included.
        """
        print(f"Saving tokenizer to {output_file}...")
        tokenizer_data = {
            "version": "1.0",
            "truncation": {
                "direction": "Right",
                "max_length": 1024,
                "strategy": "LongestFirst",
                "stride": 0
            },
            "padding": None,
            "added_tokens": [
                {
                    "id": self.vocab.get("<PAD>"),
                    "content": "<PAD>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": self.vocab.get("<UNK>"),
                    "content": "<UNK>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": self.vocab.get("<BOS>"),
                    "content": "<BOS>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                },
                {
                    "id": self.vocab.get("<EOS>"),
                    "content": "<EOS>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                }
            ],
            "normalizer": None,
            # Use a ByteLevel pre-tokenizer with an added prefix space.
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": True
            },
            "post_processor": None,
            # Set the decoder to ByteLevel so that whitespace is reinserted properly.
            "decoder": {
                "type": "ByteLevel"
            },
            # Set the model type to "BPE" and include both the vocabulary and the learned merges.
            "model": {
                "type": "BPE",
                "vocab": self.vocab,
                "merges": [list(pair) for pair in self.bpe_merges]
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
        print("Tokenizer saved.")
    
    def save_vocab(self, vocab_file):
        """Saves the vocabulary to a separate vocab.json file."""
        print(f"Saving vocabulary to {vocab_file}...")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        print("Vocabulary saved.")
    
    def save_special_tokens_map(self, special_tokens_file):
        """Saves the special tokens map to a JSON file."""
        special_tokens_map = {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>"
        }
        print(f"Saving special tokens map to {special_tokens_file}...")
        with open(special_tokens_file, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)
        print("Special tokens map saved.")

# Example Usage
if __name__ == "__main__":
    tokenizer = BPEZipfTokenizer(vocab_size=30000)
    json_files = [
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk0.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk1.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk2.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk3.json"
    ]
    
    # Process the corpus and build the tokenizer
    tokenizer.process_corpus(json_files)
    
    # Save the tokenizer in production format along with additional files
    tokenizer.save_tokenizer("tokenizer.json")
    tokenizer.save_vocab("vocab.json")
    tokenizer.save_special_tokens_map("special_tokens_map.json")
    
    # Test encoding and decoding
    sample_text = "Hello world, this is a test."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print("Encoded:", encoded)
    print("Decoded:", decoded)
