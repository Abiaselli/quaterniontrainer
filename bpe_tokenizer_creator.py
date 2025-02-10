from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer

# Initialize a BPE tokenizer with an unknown token.
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()  # You can choose a more sophisticated pre-tokenizer if needed.
trainer = BpeTrainer(
    vocab_size=10000, 
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

# Train on one or more text files (each line of your file is treated as a sentence).
tokenizer.train(files=[
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk0.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk1.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk2.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk3.json"
    ], trainer=trainer)

# Set a decoder (optional, for converting token IDs back to text)
tokenizer.decoder = BPEDecoder()

# Save the trained tokenizer in a format that Hugging Face can load.
tokenizer.save("tokenizer.json")
