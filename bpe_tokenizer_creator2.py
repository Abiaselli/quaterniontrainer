from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

# Initialize a BPE tokenizer with an unknown token.
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

# Use the ByteLevel pre-tokenizer, which preserves whitespace information.
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

# Train with your trainer (you might want to adjust special tokens as needed)
trainer = BpeTrainer(
    vocab_size=10000, 
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)
tokenizer.train(files=[
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk0.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk1.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk2.json",
        r"C:\Users\abias\.cursor-tutor\vccdoe\0000waveperceptron\dataset\smoltalk3.json"
    ], trainer=trainer)

# Set a decoder that correctly reassembles text with whitespace.
tokenizer.decoder = ByteLevelDecoder()

# Save the trained tokenizer in a format that Hugging Face can load.
tokenizer.save("tokenizer.json")
