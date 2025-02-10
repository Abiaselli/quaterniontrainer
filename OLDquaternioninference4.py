import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox
import json
import os
import logging
import psutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
import numpy as np
import psutil
import threading 

def log_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory()
    ram_used = virtual_memory.used / (1024 ** 3)  # Convert to GB
    ram_total = virtual_memory.total / (1024 ** 3)  # Convert to GB

    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"RAM Usage: {ram_used:.2f} GB / {ram_total:.2f} GB")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")
#debug for cuda
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
max_length = 1024

def save_checkpoint(model, optimizer, epoch, phase, path):
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['phase']


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    logging.info(f"Tokenizer pad_token set to: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")



# Collate function
def collate_fn(batch):
    input_ids, attention_masks, labels, seq_lengths = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    return input_ids, attention_masks, labels, seq_lengths


class QuaternionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(QuaternionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Each token is represented by a quaternion: a + bi + cj + dk
        self.scalar = nn.Embedding(vocab_size, embedding_dim)  # a
        self.vector_i = nn.Embedding(vocab_size, embedding_dim)  # b
        self.vector_j = nn.Embedding(vocab_size, embedding_dim)  # c
        self.vector_k = nn.Embedding(vocab_size, embedding_dim)  # d

    def forward(self, x):
        # Retrieve scalar and vector components
        a = self.scalar(x)
        b = self.vector_i(x)
        c = self.vector_j(x)
        d = self.vector_k(x)
        
        # Combine into quaternion format: a + bi + cj + dk
        return torch.stack([a, b, c, d], dim=-1)  # Shape: (batch_size, seq_length, embedding_dim, 4)

class QuaternionRotationalEncoding(nn.Module):
    def __init__(self, seq_length, embedding_dim):
        super(QuaternionRotationalEncoding, self).__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        # Generate rotation quaternions
        positions = torch.arange(seq_length).unsqueeze(-1).float()
        self.theta = nn.Parameter(positions * (np.pi / seq_length))  # Scale theta by position

    def forward(self, x):
        # Compute rotation quaternions
        cos_theta = torch.cos(self.theta).unsqueeze(-1)
        sin_theta = torch.sin(self.theta).unsqueeze(-1)
        r = torch.cat([cos_theta, sin_theta, sin_theta, sin_theta], dim=-1)  # (seq_length, 4)

        # Apply rotation: r * x * r^-1
        rotated = self.quaternion_multiply(r, x)
        rotated = self.quaternion_multiply(rotated, self.quaternion_conjugate(r))

        return rotated

    def quaternion_multiply(self, q1, q2):
        """Quaternion multiplication."""
        a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        scalar = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        return torch.stack([scalar, i, j, k], dim=-1)

    def quaternion_conjugate(self, q):
        """Conjugate of a quaternion."""
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


class QuaternionAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(QuaternionAttention, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable weights for query, key, and value
        self.query_weight = nn.Linear(embedding_dim * 4, embedding_dim * 4)
        self.key_weight = nn.Linear(embedding_dim * 4, embedding_dim * 4)
        self.value_weight = nn.Linear(embedding_dim * 4, embedding_dim * 4)

    def quaternion_dot(self, q1, q2):
        """Compute quaternion dot product (element-wise)."""
        a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        scalar_part = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        vector_part_i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        vector_part_j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        vector_part_k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        return torch.stack([scalar_part, vector_part_i, vector_part_j, vector_part_k], dim=-1)

    def forward(self, query, key, value, mask=None):
        query = query.reshape(query.shape[0], query.shape[1], -1)
        key = key.reshape(key.shape[0], key.shape[1], -1)
        value = value.reshape(value.shape[0], value.shape[1], -1)

        Q = self.query_weight(query)
        K = self.key_weight(key)
        V = self.value_weight(value)

        Q = Q.reshape(Q.shape[0], Q.shape[1], -1, 4)
        K = K.reshape(K.shape[0], K.shape[1], -1, 4)
        V = V.reshape(V.shape[0], V.shape[1], -1, 4)

        attention_scores = torch.einsum('bseq,bkeq->bsk', Q, K)  # (batch, seq_len, seq_len)

        # Apply mask before softmax
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = attention_scores / np.sqrt(self.embedding_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.einsum('bsk,bseq->bseq', attention_weights, V)
        return output, attention_weights

class QuaternionFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(QuaternionFeedForward, self).__init__()
        self.hidden_dim = hidden_dim

        # Fully connected layers for quaternion processing
        self.fc1 = nn.Linear(embedding_dim * 4, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, embedding_dim * 4)

    def forward(self, x):
        batch_size, seq_len, embedding_dim, quaternion_dim = x.shape

        # Flatten quaternion dimension
        x = x.reshape(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, embedding_dim * 4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(batch_size, seq_len, embedding_dim, 4)  # Restore quaternion structure
        return x

class QuaternionTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, seq_length):
        super(QuaternionTransformerBlock, self).__init__()
        self.rotation_encoding = QuaternionRotationalEncoding(seq_length, embedding_dim)
        self.attention = QuaternionAttention(embedding_dim)
        self.feed_forward = QuaternionFeedForward(embedding_dim, hidden_dim)
        self.norm1 = nn.LayerNorm([embedding_dim, 4])
        self.norm2 = nn.LayerNorm([embedding_dim, 4])

    def forward(self, x, mask=None):
        # Apply rotational encoding
        x = self.rotation_encoding(x)

        # Attention layer
        attn_output, attention_weights = self.attention(x, x, x, mask)  
        x = self.norm1(x + attn_output)

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x, attention_weights  



class QuaternionTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_length, num_layers):
        super(QuaternionTransformer, self).__init__()
        self.embedding = QuaternionEmbedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([
            QuaternionTransformerBlock(embedding_dim, hidden_dim, seq_length) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(embedding_dim * 4, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        attention_weights_list = []  # Store attention maps

        for layer in self.layers:
            x, attn_weights = layer(x, mask)  # Capture attention weights per layer
            attention_weights_list.append(attn_weights)
        logging.debug(f"Logits shape before projection: {x.shape}")
        logits = self.output_projection(x.reshape(x.shape[0], x.shape[1], -1))
        logging.debug(f"Logits shape qtrans: {logits.shape}")
        return logits, attention_weights_list  


# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Tokenizer Validation and Loading
def validate_tokenizer_folder(tokenizer_path):
    required_files = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(tokenizer_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing files in tokenizer folder: {missing_files}")

def load_tokenizer(tokenizer_path):
    validate_tokenizer_folder(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Special tokens loaded: {tokenizer.special_tokens_map}")
    return tokenizer

def ensure_special_tokens(tokenizer):
    special_tokens = {}
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '<eos>'
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<pad>'
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print(f"Added special tokens: {special_tokens}")
    else:
        print("All special tokens are already present.")
    
    print(f"EOS Token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    return tokenizer

def load_model_parameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def ensure_special_tokens(tokenizer):
    special_tokens = {}
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '<eos>'
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<pad>'
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print(f"Added special tokens: {special_tokens}")
    else:
        print("All special tokens are already present.")
    
    print(f"EOS Token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    return tokenizer

# Model Loading Function
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint and 'model_parameters' in checkpoint:
        # New model format with parameters included
        state_dict = checkpoint['state_dict']
        model_parameters = checkpoint['model_parameters']
    else:
        # Old model format without parameters
        state_dict = checkpoint
        model_parameters = None

    return state_dict, model_parameters

# Top-K and Top-P Filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    batch_size, vocab_size = logits.size()
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), vocab_size)
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, filter_value, logits)

    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

# Text Generation Function
def generate_text_gui(model, tokenizer, input_text, max_length=max_length, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
    model.to(device)
    model.eval()
    logging.info(f"input text length: {len(input_text)}")


    generated_tokens = tokenizer.encode(input_text)
    
    with torch.no_grad():

        logging.info("max_length set")
        for _ in range(max_length):
            input_tokens = []
            input_tokens.append(generated_tokens)
                # Convert lists of token IDs to tensors and calculate original sequence lengths
            input_tensor, seq_lengths = zip(*[
                        (
                            torch.tensor(tokens + [tokenizer.pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length],
                            min(len(tokens), max_length)
                        )
                        for tokens in input_tokens
                    ])
            logging.info("input ids torched to tensor")
            

            attention_mask = [(ids != tokenizer.pad_token_id).long() for ids in input_tensor]
            logging.info("attention masks set for pad tokens")
            input_tensor = torch.stack(input_tensor)
            logging.info(f"input_tensor shape: {input_tensor.shape}")

            attention_mask = torch.stack(attention_mask)

            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

            logging.debug(f"Shape of batch_input_ids before generate_square_subsequent_mask: {input_tensor.shape}")
            batch_size, seq_len = input_tensor.size()

            src_mask = generate_square_subsequent_mask(seq_len).to(device)

                # Log the shapes before combining
            logging.debug(f"Shape of src_mask: {src_mask.shape}")
            logging.debug(f"Shape of attention_mask: {attention_mask.shape}")
                # Expand src_mask to match batch size and number of heads 
            src_mask = src_mask.unsqueeze(0).expand(input_tensor.size(0), -1, -1) 
            logging.debug(f"Shape of src_mask after expansion: {src_mask.shape}")
                # Combine masks without slicing (corrected)
            combined_mask = src_mask.masked_fill(attention_mask[:, None, :].expand(-1, seq_len, seq_len) == 0, float('-inf'))
                
            if isinstance(model, QuaternionTransformer):
                
                outputs, attention_weights = model(input_tensor, mask=combined_mask)
            else:
                outputs = model(input_tensor)
            logging.info(f"outputs shape: {outputs.shape}")

            next_token_logits = outputs[:, -1, :]
            logging.info(f"next_token_logtis shape: {next_token_logits.shape}")
            
            # Rest of the function remains the same

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Repetition Penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    next_token_logits[0, token_id] /= repetition_penalty

            # Filter logits using top-k and/or top-p sampling
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            generated_tokens.append(next_token)

            if next_token == tokenizer.vocab["<EOS>"]:
                break
            
            if len(generated_tokens) == 512:
                break
        

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text

def create_combined_mask(batch_input_ids, pad_token_id):
    """
    Create a combined attention mask that incorporates both the causal (subsequent) mask
    and the padding mask. This function ensures that each row has at least one valid token.
    """
    batch_size, seq_len = batch_input_ids.size()
    device = batch_input_ids.device
    
    # Generate causal (subsequent) mask: shape (seq_len, seq_len)
    causal_mask = generate_square_subsequent_mask(seq_len).to(device)
    # Expand to batch dimension: (batch_size, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Create padding mask: valid tokens are True, padded tokens are False.
    # Shape: (batch_size, seq_len)
    padding_mask = (batch_input_ids != pad_token_id)
    # Expand padding mask to match the shape (batch_size, seq_len, seq_len)
    # Here we broadcast along one dimension so that we mask out positions in each row.
    padding_mask_expanded = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    
    # Combine masks: where padding_mask is False, set to -inf.
    # This keeps the causal structure while ensuring that padded positions are fully masked.
    combined_mask = causal_mask.masked_fill(~padding_mask_expanded, float('-inf'))
    
    # Check each row: if an entire row is -inf, force the first token (or a designated position) to be valid.
    for i in range(batch_size):
        for j in range(seq_len):
            if torch.all(combined_mask[i, j] == float('-inf')):
                combined_mask[i, j, 0] = 0.0  # Force at least one valid position
    
    return combined_mask


class QuaternionInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quaternion Inference Model")

        # Initialize model and tokenizer as None
        self.model = None
        self.tokenizer = None

        # Define Entry widgets for model and tokenizer paths
        Label(root, text="Model Path:").pack(pady=(10, 0))
        self.model_path_entry = Entry(root, width=60)
        self.model_path_entry.pack(pady=(0, 10))

        Label(root, text="Tokenizer Path:").pack(pady=(0, 0))
        self.tokenizer_path_entry = Entry(root, width=60)
        self.tokenizer_path_entry.pack(pady=(0, 10))

        # Select Folder Button
        self.select_button = Button(root, text="Select Model Folder", command=self.select_folder)
        self.select_button.pack(pady=(0, 10))

        # Model Parameters
        Label(root, text="Vocabulary Size:").pack(pady=(10, 0))
        self.vocab_size_entry = Entry(root, width=60)
        self.vocab_size_entry.pack(pady=(0, 10))
        self.vocab_size_entry.insert(0, "30000")  # Default value

        Label(root, text="Embedding Size:").pack(pady=(0, 0))
        self.embed_size_entry = Entry(root, width=60)
        self.embed_size_entry.pack(pady=(0, 10))
        self.embed_size_entry.insert(0, "60")  # Default value

        Label(root, text="Hidden Size:").pack(pady=(0, 0))
        self.hidden_size_entry = Entry(root, width=60)
        self.hidden_size_entry.pack(pady=(0, 10))
        self.hidden_size_entry.insert(0, "60")  # Default value

        # Input Text
        Label(root, text="Input Text:").pack(pady=(10, 0))
        self.input_box = Text(root, height=5, width=60)
        self.input_box.pack(pady=(0, 10))

        # Generation Parameters
        Label(root, text="Max Length:").pack(pady=(10, 0))
        self.max_length_entry = Entry(root, width=60)
        self.max_length_entry.pack(pady=(0, 10))
        self.max_length_entry.insert(0, max_length)

        Label(root, text="Temperature:").pack(pady=(0, 0))
        self.temperature_entry = Entry(root, width=60)
        self.temperature_entry.pack(pady=(0, 10))
        self.temperature_entry.insert(0, "1.0")

        Label(root, text="Top-K:").pack(pady=(0, 0))
        self.top_k_entry = Entry(root, width=60)
        self.top_k_entry.pack(pady=(0, 10))
        self.top_k_entry.insert(0, "0")

        Label(root, text="Top-P:").pack(pady=(0, 0))
        self.top_p_entry = Entry(root, width=60)
        self.top_p_entry.pack(pady=(0, 10))
        self.top_p_entry.insert(0, "0.0")

        Label(root, text="Repetition Penalty:").pack(pady=(0, 0))
        self.repetition_penalty_entry = Entry(root, width=60)
        self.repetition_penalty_entry.pack(pady=(0, 10))
        self.repetition_penalty_entry.insert(0, "1.0")

        # Generate Button
        self.generate_button = Button(root, text="Generate Text", command=self.generate_text_callback)
        self.generate_button.pack(pady=(0, 10))

        # Output Box
        Label(root, text="Generated Output:").pack(pady=(10, 0))
        self.output_box = Text(root, height=10, width=60)
        self.output_box.pack(pady=(0, 10))
        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')


    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")
            
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Set model and tokenizer paths
            model_path = os.path.join(folder_path, ".pth")
            tokenizer_path = folder_path  # Assuming tokenizer files are in the same folder

            # Update Entry widgets
            self.model_path_entry.delete(0, END)
            self.model_path_entry.insert(0, model_path)

            self.tokenizer_path_entry.delete(0, END)
            self.tokenizer_path_entry.insert(0, tokenizer_path)

            # Load model and tokenizer
            try:
                self.load_model_and_tokenizer(model_path, tokenizer_path)
                messagebox.showinfo("Success", "Model and Tokenizer loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model/tokenizer:\n{e}")

    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        # Load tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer = ensure_special_tokens(tokenizer)

        # Load model parameters from model_config.json
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if not os.path.exists(config_path):
            messagebox.showerror("Error", "model_config.json not found.")
            return

        model_parameters = load_model_parameters(config_path)

        # Update Entry widgets with loaded parameters
        self.vocab_size_entry.config(state='normal')
        self.vocab_size_entry.delete(0, END)
        self.vocab_size_entry.insert(0, str(model_parameters['vocab_size']))
        self.vocab_size_entry.config(state='readonly')

        self.embed_size_entry.config(state='normal')
        self.embed_size_entry.delete(0, END)
        self.embed_size_entry.insert(0, str(model_parameters['embed_size']))
        self.embed_size_entry.config(state='readonly')

        self.hidden_size_entry.config(state='normal')
        self.hidden_size_entry.delete(0, END)
        self.hidden_size_entry.insert(0, str(model_parameters['hidden_size']))
        self.hidden_size_entry.config(state='readonly')


        # Create the appropriate model based on the architecture
        architecture = model_parameters.get('architecture', "Quaternion Transformer" or 'Quaternion MatMul-Free LM')

        if architecture == 'Quaternion Transformer':
            model = QuaternionTransformer(
                vocab_size=model_parameters['vocab_size'],
                embedding_dim=model_parameters['embed_size'],
                hidden_dim=model_parameters['hidden_size'],
                num_layers=model_parameters['num_layers'],
                seq_length=1024
            )
            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'quaternion_transformer_model.pth')
        elif architecture =='Quaternion MatMul-Free LM':
            model = QuaternionTransformer(
                vocab_size=model_parameters['vocab_size'],
                embedding_dim=model_parameters['embed_size'],
                hidden_dim=model_parameters['hidden_size'],
                seq_length=1024
            )

            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'quaternion_matmul_free_lm.pth')
        else:
            messagebox.showerror("Error", f"Unsupported architecture: {architecture}")
            return

        print(f"Model Parameters:")
        print(f"  Vocab Size: {model_parameters['vocab_size']}")
        print(f"  Embed Size: {model_parameters['embed_size']}")
        print(f"  Hidden Size: {model_parameters['hidden_size']}")
        print(f"  Num Layers: {model_parameters['num_layers']}")
        # Load state_dict
        state_dict, _ = load_model(model_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Update class attributes
        self.tokenizer = tokenizer
        self.model = model


    def generate_text_callback(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("Warning", "Please load a model and tokenizer first.")
            return

        input_text = self.input_box.get("1.0", END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some input text.")
            return

        # Retrieve generation parameters
        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            top_k = int(self.top_k_entry.get())
            top_p = float(self.top_p_entry.get())
            repetition_penalty = float(self.repetition_penalty_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid generation parameters.")
            return

        # Start generation in a separate thread to keep GUI responsive
        threading.Thread(
            target=self.generate_and_display,
            args=(input_text, max_length, temperature, top_k, top_p, repetition_penalty)
        ).start()

    def generate_and_display(self, input_text, max_length, temperature, top_k, top_p, repetition_penalty):
        try:
            output = generate_text_gui(
                model=self.model,
                tokenizer=self.tokenizer,
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            self.output_box.delete("1.0", END)
            self.output_box.insert(END, output)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text:\n{e}")
            
if __name__ == "__main__":
    root = Tk()
    app = QuaternionInferenceGUI(root)
    root.mainloop()
