import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, normalizers
import psutil
import copy
import random
import math
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")
#debug for cuda
import os
import time
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
seq_len = 128

# Global tokenizer variable for multiprocessing
tokenizer = None

def log_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory()
    ram_used = virtual_memory.used / (1024 ** 3)  # Convert to GB
    ram_total = virtual_memory.total / (1024 ** 3)  # Convert to GB

    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"RAM Usage: {ram_used:.2f} GB / {ram_total:.2f} GB")



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


@staticmethod
def tokenize_chunk(chunk):
    # Tokenizer is now the global variable initialized in each process
    encoded = tokenizer(chunk, return_attention_mask=False, truncation=True, max_length=seq_len)
    return encoded['input_ids']

# Collate function
def collate_fn(batch):
    if len(batch[0]) == 3:
        # Dataset returns: input_ids, labels, seq_lengths
        input_ids, labels, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return input_ids, labels, seq_lengths
    elif len(batch[0]) == 4:
        # Dataset returns: input_ids, attention_masks, labels, seq_lengths
        input_ids, attention_masks, labels, seq_lengths = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return input_ids, attention_masks, labels, seq_lengths
    else:
        raise ValueError("Unexpected number of elements in dataset sample.")

def quaternion_mse_loss(output, target):
    """
    Quaternion Mean Squared Error (QMSE) Loss.
    Computes the squared difference between quaternion components.
    Modify Output projection to use Quaternion linear for this.
    
    Args:
        output: (batch, seq_len, features, 4) - Quaternion predictions
        target: (batch, seq_len, features, 4) - Quaternion targets

    Returns:
        QMSE Loss (scalar)
    """
    return torch.mean((output - target) ** 2)

def quaternion_cross_entropy(output, target):
    """
    Quaternion Cross-Entropy Loss.
    Measures divergence between quaternion softmax outputs and targets.
    Modify Output projection to use Quaternion linear for this.

    Args:
        output: (batch, seq_len, features, 4) - Quaternion predictions
        target: (batch, seq_len, features, 4) - Quaternion targets

    Returns:
        Scalar loss
    """
    # Compute quaternion dot product as a similarity measure
    similarity = torch.sum(output * target, dim=-1)
    loss = -torch.mean(torch.log(torch.exp(similarity) / torch.sum(torch.exp(similarity), dim=-1, keepdim=True)))
    return loss

def quaternion_angular_loss(output, target):
    """
    Quaternion Angular Loss.
    Encourages quaternion outputs to maintain rotational consistency with targets.
    Modify Output projection to use Quaternion linear for this.

    Args:
        output: (batch, seq_len, features, 4) - Quaternion predictions
        target: (batch, seq_len, features, 4) - Quaternion targets

    Returns:
        Scalar loss
    """
    # Normalize quaternions to avoid scale issues
    output = output / torch.norm(output, dim=-1, keepdim=True)
    target = target / torch.norm(target, dim=-1, keepdim=True)
    
    # Compute cosine similarity loss (similar to quaternion dot product)
    angular_distance = torch.acos(torch.clamp(torch.sum(output * target, dim=-1), -1, 1))
    return torch.mean(angular_distance ** 2)

class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data_path, tokenizer, max_length=seq_len):
        self.tokenized_data_path = tokenized_data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get a list of chunk files
        self.chunk_files = [os.path.join(self.tokenized_data_path, f) 
                            for f in os.listdir(self.tokenized_data_path) 
                            if f.startswith('chunk_') and f.endswith('.jsonl')]
        self.chunk_files.sort()  # Ensure the chunks are in order

        # Build an index mapping from global indices to (chunk_idx, sample_idx)
        self.index_mapping = []
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            self.index_mapping.extend([(chunk_idx, i) for i in range(num_lines)])

        # Initialize current chunk data
        self.current_chunk_idx = -1  # Indicates no chunk is currently loaded
        self.current_chunk_data = []  # Will hold the data from the current chunk

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")

        chunk_idx, sample_idx = self.index_mapping[idx]

        # Load the appropriate chunk if not already loaded
        if self.current_chunk_idx != chunk_idx:
            self.load_chunk(chunk_idx)

        record = self.current_chunk_data[sample_idx]
        input_ids = record['input_ids']
        labels = record['labels']

        # Calculate original sequence length before padding
        original_seq_length = min(len(input_ids), self.max_length)
        logging.debug(f"original sequence length = {original_seq_length}")
        # Pad sequences to max_length
        input_ids = input_ids[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(input_ids))
        labels = labels[:self.max_length] + [self.tokenizer.pad_token_id] * max(0, self.max_length - len(labels))

        assert isinstance(input_ids, list), "input_ids should be a list"
        assert isinstance(labels, list), "labels should be a list"
        assert all(isinstance(id, int) for id in input_ids), "All input_ids should be integers"
        assert all(isinstance(id, int) for id in labels), "All labels should be integers"
        assert len(input_ids) == self.max_length, "input_ids should be padded to max_length"
        assert len(labels) == self.max_length, "labels should be padded to max_length"
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        seq_lengths = torch.tensor(original_seq_length, dtype=torch.long)

        # Check for empty sequences
        if len(input_ids) == 0:
            logging.error(f"Empty input_ids at index {idx}.")
            raise ValueError(f"Empty input_ids at index {idx}.")
        if len(labels) == 0:
            logging.error(f"Empty labels at index {idx}.")
            raise ValueError(f"Empty labels at index {idx}.")
    
        return input_ids, attention_mask, labels, seq_lengths

    def load_chunk(self, idx):
        chunk_file = self.chunk_files[idx]
        with open(chunk_file, 'r', encoding='utf-8') as f:
            self.current_chunk_data = [json.loads(line.strip()) for line in f]
        self.current_chunk_idx = idx

def output_projection_hook(grad):
    logging.debug(f"Output projection gradient shape: {grad.shape}")

class QuaternionEmbedding(nn.Module):
    """
        Quaternion Embedding Layer with normalization and RoPE-style scaling.

        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            embedding_dim (int): Number of quaternion features.
        """
    def __init__(self, vocab_size, embedding_dim):
        super(QuaternionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Each token is represented by a quaternion: a + bi + cj + dk
        self.scalar = nn.Embedding(vocab_size, embedding_dim)  # a
        self.vector_i = nn.Embedding(vocab_size, embedding_dim)  # b
        self.vector_j = nn.Embedding(vocab_size, embedding_dim)  # c
        self.vector_k = nn.Embedding(vocab_size, embedding_dim)  # d

        # RoPE-style scaling decay per embedding dimension
        #self.scale_factor = nn.Parameter(1 / (10000 ** (torch.arange(embedding_dim) / embedding_dim)), requires_grad=False) #non-trainable parameter
        # registered as a buffer.
        scale = 1 / (10000 ** (torch.arange(embedding_dim, dtype=torch.float32) / embedding_dim))
        self.register_buffer("scale_factor", scale)

    def forward(self, x):
        """
        Args:
            x: Input token IDs (batch, seq_len)

        Returns:
            Quaternion embeddings (batch, seq_len, embedding_dim, 4)
        """
        # Retrieve scalar and vector components
        # Quaternion Embedding Table: [vocab_size, embed_dim, quaternion_dim]
        logging.debug(f"Shape of x passed to quaternion embedding: {x.shape}")
        r = self.scalar(x)
        logging.debug(f"Shape of r passed to quaternion embedding: {r.shape}")

        i = self.vector_i(x)
        logging.debug(f"Shape of i passed to quaternion embedding: {i.shape}")

        j = self.vector_j(x)
        logging.debug(f"Shape of j passed to quaternion embedding: {j.shape}")

        k = self.vector_k(x)
        logging.debug(f"Shape of k passed to quaternion embedding: {k.shape}")

        # Apply RoPE-style scaling
        i = i * self.scale_factor
        logging.debug(f"Shape of i scaled to quaternion embedding: {i.shape}")

        j = j * self.scale_factor
        logging.debug(f"Shape of j scaled to quaternion embedding: {j.shape}")

        k = k * self.scale_factor
        logging.debug(f"Shape of k scaled to quaternion embedding: {k.shape}")


        # Normalize quaternion embeddings to unit norm
        norm = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-6)
        r, i, j, k = r / norm, i / norm, j / norm, k / norm
        logging.debug(f"Shape of r norm to quaternion embedding: {r.shape}")
        logging.debug(f"Shape of i norm to quaternion embedding: {i.shape}")
        logging.debug(f"Shape of j norm to quaternion embedding: {j.shape}")
        logging.debug(f"Shape of k norm to quaternion embedding: {k.shape}")
    
        x=torch.stack([r, i, j, k], dim=-1)
        logging.debug(f"Shape of quaternion embedding: {x.shape}")
        if torch.isnan(x).any():
            logging.error("NaN detected in quaternion embedding output!")

        # Combine into quaternion format: a + bi + cj + dk
        return x   # Shape: (batch_size, seq_length, embedding_dim, 4)
    
class QuaternionRotationalEncodingLNS(nn.Module):
    """
    Quaternion Rotational Encoding with LNS-based trigonometric approximations.
    """
    def __init__(self, seq_length, embedding_dim):
        super(QuaternionRotationalEncodingLNS, self).__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.lns = LogarithmicNumberSystem()

        # Generate position indices
        positions = torch.arange(seq_length).unsqueeze(-1).float()

        # RoPE-style scaling decay per dimension
        scale_factor = 1 / (10000 ** (torch.arange(embedding_dim, dtype=torch.float32) / embedding_dim))
        theta = positions * (torch.pi / seq_length) * scale_factor
        self.register_buffer("theta", theta)

    def forward(self, x):
        """
        Apply quaternion rotational encoding using LNS.
        """
        # Compute rotation angles in log-space
        log_cos_theta = self.lns.to_log(torch.log1p(torch.cos(self.theta)).unsqueeze(-1))
        log_sin_theta = self.lns.to_log(torch.log1p(torch.sin(self.theta)).unsqueeze(-1))

        # Pairwise quaternion rotations:
        r_ij = torch.cat([log_cos_theta, -log_sin_theta, log_sin_theta, log_cos_theta], dim=-1)
        r_k = torch.cat([log_cos_theta, log_sin_theta, -log_sin_theta, log_cos_theta], dim=-1)

        # ✅ Expand `r_ij` and `r_k` to match `x`
        r_ij = r_ij.expand(x.shape)
        r_k = r_k.expand(x.shape)

        # Apply quaternion multiplication in log-space
        rotated_ij = self.quaternion_multiply(r_ij, self.lns.to_log(x))
        rotated_k = self.quaternion_multiply(r_k, rotated_ij)

        return self.lns.from_log(rotated_k)


    def quaternion_multiply(self, log_q1, log_q2):
        """
        Performs quaternion multiplication using LNS.
        """
        a1, b1, c1, d1 = log_q1[..., 0], log_q1[..., 1], log_q1[..., 2], log_q1[..., 3]
        a2, b2, c2, d2 = log_q2[..., 0], log_q2[..., 1], log_q2[..., 2], log_q2[..., 3]

        # Perform quaternion multiplication in log-space (add instead of multiply)
        scalar = self.lns.log_add(self.lns.log_add(a1, a2), self.lns.log_sub(b1, b2))

        i = self.lns.log_add(self.lns.log_add(self.lns.log_add(a1, b2), self.lns.log_add(b1, a2)),
                            self.lns.log_add(self.lns.log_sub(c1, d2), d1))

        j = self.lns.log_add(self.lns.log_add(self.lns.log_add(a1, c2), self.lns.log_sub(b1, d2)),
                            self.lns.log_add(self.lns.log_add(c1, a2), d1))

        k = self.lns.log_add(self.lns.log_add(self.lns.log_add(a1, d2), self.lns.log_add(b1, c2)),
                            self.lns.log_sub(c1, self.lns.log_add(b2, d1)))

        return torch.stack([scalar, i, j, k], dim=-1)



class QuaternionMultiplication:
    """
    CUDA-optimized quaternion multiplication for efficient computation in log-space.
    """
    def __init__(self, use_cuda=True):
        self.lns = LogarithmicNumberSystem(use_cuda=use_cuda)

    def multiply(self, log_q1, log_q2):
        """
        Performs quaternion multiplication in log-space using LNS.
        """
        a1, b1, c1, d1 = log_q1[..., 0], log_q1[..., 1], log_q1[..., 2], log_q1[..., 3]
        a2, b2, c2, d2 = log_q2[..., 0], log_q2[..., 1], log_q2[..., 2], log_q2[..., 3]
        logging.debug(f"log_q1 shape {log_q1.shape}")
        logging.debug(f"log_q2 shape {log_q2.shape}")

        # Perform quaternion multiplication in log-space (addition instead of multiplication)
        scalar = self.lns.log_add(self.lns.log_add(a1, a2), 
                                self.lns.log_sub(b1, b2))
        logging.debug(f"Scalar shape {scalar.shape}")

        i = self.lns.log_add(self.lns.log_add(self.lns.log_add(a1, b2), self.lns.log_add(b1, a2)), 
                            self.lns.log_add(self.lns.log_sub(c1, d2), d1))
        logging.debug(f"i shape {i.shape}")

        j = self.lns.log_add(self.lns.log_add(self.lns.log_add(a1, c2), self.lns.log_sub(b1, d2)), 
                            self.lns.log_add(self.lns.log_add(c1, a2), d1))
        logging.debug(f"j shape {j.shape}")

        k = self.lns.log_add(self.lns.log_add(self.lns.log_add(a1, d2), self.lns.log_add(b1, c2)), 
                            self.lns.log_sub(c1, self.lns.log_add(b2, d1)))
        logging.debug(f"k shape {k.shape}")

        return torch.stack([scalar, i, j, k], dim=-1)



class QuaternionAttentionLNS(nn.Module):
    """
    Optimized Quaternion Attention using CUDA-accelerated Logarithmic Number System.
    """
    def __init__(self, embedding_dim):
        super(QuaternionAttentionLNS, self).__init__()
        self.lns = LogarithmicNumberSystem()
        self.quat_mult = QuaternionMultiplication()
        self.query_weight = QuaternionLinearLNS(embedding_dim, embedding_dim, seq_len)
        self.key_weight = QuaternionLinearLNS(embedding_dim, embedding_dim, seq_len)
        self.value_weight = QuaternionLinearLNS(embedding_dim, embedding_dim, seq_len)

    def forward(self, query, key, value, mask=None):
        logging.debug(f"Attention: query shape {query.shape}, key shape {key.shape}, value shape {value.shape}")

        # Apply learned weights
        log_query = self.lns.to_log(self.query_weight(query))
        log_key = self.lns.to_log(self.key_weight(key))
        log_value = self.lns.to_log(self.value_weight(value))

        logging.debug(f"Attention: log-space Q shape {log_query.shape}, K shape {log_key.shape}, V shape {log_value.shape}")

        # ✅ Use log-space einsum for attention scores
        attention_scores = self.lns.log_add_einsum('bqfd,bkfd->bqk', log_query, log_key)
        logging.debug(f"Attention: attention_scores shape {attention_scores.shape}")
        # ✅ Fix: Ensure attention_scores retains seq_len x seq_len
        if attention_scores.dim() < 3:
            logging.warning(f"Unexpected attention_scores shape: {attention_scores.shape}. Expanding.")
            attention_scores = attention_scores.unsqueeze(-1).expand(-1, -1, mask.shape[-1])

        logging.debug(f"Attention: attention_scores shape after fix: {attention_scores.shape}")
        # ✅ Fix: Ensure Mask Matches Attention Score Shape
        if mask is not None:
            mask = mask.to(attention_scores.device)
            logging.debug(f"Mask shape before expansion: {mask.shape}")

            # Expand only if necessary
            while mask.dim() < 3:
                mask = mask.unsqueeze(-1)

            # Trim and expand correctly
            expected_dim = attention_scores.shape[2]
            if mask.shape[2] != expected_dim:
                logging.warning(f"Expected mask seq_len {expected_dim}, got {mask.shape[2]}. Fixing.")
                mask = mask[:, :, :expected_dim]

            logging.debug(f"Mask expanded shape: {mask.shape}")

            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        logging.debug(f"Attention: masked attention_scores shape {attention_scores.shape}")

        # ✅ Normalize scores
        attention_scores = attention_scores / math.sqrt(query.size(-2) * 4)
        attention_weights = F.softmax(attention_scores, dim=-1)
        logging.debug(f"Attention: attention_weights shape {attention_weights.shape}")

        # ✅ Use log-space einsum for value aggregation
        output = torch.einsum('bqk,bkfd->bqfd', attention_weights, log_value)
        logging.debug(f"Attention: output shape {output.shape}")

        return output, attention_weights


class QuaternionFeedForwardLNS(nn.Module):
    """
    Optimized Quaternion Feed Forward Network using CUDA-accelerated Logarithmic Number System (LNS).
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(QuaternionFeedForwardLNS, self).__init__()
        self.lns = LogarithmicNumberSystem()
        self.fc1 = QuaternionLinearLNS(embedding_dim, hidden_dim, seq_len)
        self.activation = nn.ReLU()  # Can replace with log-space activation like GELU
        self.fc2 = QuaternionLinearLNS(hidden_dim, embedding_dim, seq_len)

    def forward(self, x):
        logging.debug(f"Feed-forward input shape: {x.shape}")

        log_x = self.lns.to_log(x)
        logging.debug(f"Log-space input shape: {log_x.shape}")
        # ✅ Dynamically determine correct expansion size
        expected_features = self.fc1.out_features if hasattr(self.fc1, "out_features") else x.shape[-1]

        if log_x.shape[-1] != expected_features:
            log_x_expanded = log_x[..., 0].unsqueeze(-1).expand(log_x.shape[:-1] + (expected_features,))
            logging.debug(f"Log-space log_x_expanded shape 1: {log_x_expanded.shape}")

        else:
            log_x_expanded = log_x[..., 0]
            logging.debug(f"Log-space log_x_expanded shape 2: {log_x_expanded.shape}")

        log_hidden = self.lns.log_add(self.fc1(log_x_expanded), self.lns.to_log(self.activation(self.lns.from_log(log_x))))
        logging.debug(f"Hidden layer output shape: {log_hidden.shape}")

        log_output = self.fc2(log_hidden)
        logging.debug(f"Feed-forward output shape: {log_output.shape}")

        return self.lns.from_log(log_output)




class QuaternionTransformerBlockLNS(nn.Module):
    """
    Quaternion Transformer Block using CUDA accelerated Logarithmic Number System (LNS).
    """
    def __init__(self, embedding_dim, hidden_dim, seq_length):
        super(QuaternionTransformerBlockLNS, self).__init__()
        self.rotation_encoding = QuaternionRotationalEncodingLNS(seq_length, embedding_dim)
        self.attention = QuaternionAttentionLNS(embedding_dim)
        self.feed_forward = QuaternionFeedForwardLNS(embedding_dim, hidden_dim)
        self.norm1 = QuaternionLayerNorm(embedding_dim)
        self.norm2 = QuaternionLayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        x = self.rotation_encoding(x)
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attention_weights


class QuaternionLayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(QuaternionLayerNorm, self).__init__()
        # Learnable scale and shift per quaternion feature
        self.gamma = nn.Parameter(torch.ones(embedding_dim, 4))
        self.beta = nn.Parameter(torch.zeros(embedding_dim, 4))
        self.eps = eps

    def forward(self, x):
        device = x.device  # Get device of input tensor
        self.gamma.data = self.gamma.data.to(device)  # Move gamma to same device
        self.beta.data = self.beta.data.to(device)  # Move beta to same device

        norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / norm
        logging.debug(f"x_norm shape {x_norm.shape}")

        return self.gamma.unsqueeze(0).unsqueeze(0) * x_norm + self.beta.unsqueeze(0).unsqueeze(0)


class QuaternionTransformerLNS(nn.Module):
    """
    Optimized Quaternion Transformer with CUDA-accelerated LNS.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_length, num_layers, convert_to_real=True):
        super(QuaternionTransformerLNS, self).__init__()
        self.embedding = QuaternionEmbedding(vocab_size, embedding_dim)  
        self.layers = nn.ModuleList([
            QuaternionTransformerBlockLNS(embedding_dim, hidden_dim, seq_length)
            for _ in range(num_layers)
        ])
        self.convert_to_real = convert_to_real

        self.quat_to_real = QuaternionToReal(embedding_dim, method="norm")
        self.output_projection = QuaternionLinearLNS(embedding_dim, vocab_size, seq_len)

    def forward(self, x, mask=None):
        logging.debug(f"Model input shape: {x.shape}")

        log_x = self.embedding(x)
        logging.debug(f"Embedding output shape: {log_x.shape}")

        # Ensure output matches expected shape `(batch, seq_len, embedding_dim, 4)`
        expected_embedding_dim = self.output_projection.in_features  # Use projection layer's input size
        if log_x.shape[2] != expected_embedding_dim:
            logging.warning(f"Expected embedding dim {expected_embedding_dim}, got {log_x.shape[2]}. Expanding.")
            log_x = log_x[..., :expected_embedding_dim, :]  # Trim to expected size if needed

        for layer_idx, layer in enumerate(self.layers):
            log_x, _ = layer(log_x, mask)
            logging.debug(f"After Transformer Layer {layer_idx}, shape: {log_x.shape}")
        if self.convert_to_real:
            log_x = self.quat_to_real(log_x)

        logits = self.output_projection(log_x)
        logging.debug(f"Output projection shape: {logits.shape}")

        return logits


class GravitationalRelationalEncodingLNS(nn.Module):
    """
    Implements a relational learning mechanism using gravitational interactions.
    Tokens interact based on their frequency and inverse-square relationships.
    """
    def __init__(self, vocab_size, embedding_dim, seq_length, zipf_factor=1.5, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(GravitationalRelationalEncodingLNS, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.zipf_factor = zipf_factor
        self.device = torch.device(device)

        # Base embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(self.device)

        # Frequency-based weighting (Zipf’s Law)
        self.token_frequencies = self._compute_zipf_weights()

    def _compute_zipf_weights(self):
        """
        Generates frequency-based weights for tokens using Zipf’s law.
        """
        ranks = torch.arange(1, self.vocab_size + 1, dtype=torch.float32, device=self.device)
        logging.debug(f"Shape of ranks to zipf: {ranks.shape}")

        zipf_weights = 1.0 / (ranks ** self.zipf_factor)  # f ~ 1 / rank^factor
        logging.debug(f"Shape of zipf_weights to zipf: {zipf_weights.shape}")

        zipf_weights /= zipf_weights.sum()  # Normalize
        logging.debug(f"Shape of zipf_weights to zipf after norm: {zipf_weights.shape}")
        if torch.isnan(zipf_weights).any():
            logging.error("NaN detected in zipf weight!")

        return zipf_weights

    def compute_gravitational_forces(self, token_indices):
        """
        Computes token relational forces using an inverse-square law.
        """
        batch_size, seq_len = token_indices.shape

        # Retrieve embeddings
        token_embeddings = self.embeddings(token_indices)  # (batch, seq_len, embedding_dim)
        logging.debug(f"Shape of token_indices to compute GRE: {token_indices.shape}")
        if torch.isnan(token_embeddings).any():
            logging.error("NaN detected in compute GRE token embeddings!")

        # Compute pairwise gravitational forces
        forces = torch.zeros((batch_size, seq_len, seq_len), device=self.device)
        logging.debug(f"Shape of forces to compute GRE: {forces.shape}")

        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    dist_sq = torch.norm(token_embeddings[:, i] - token_embeddings[:, j], dim=-1, keepdim=True) ** 2
                    dist_sq = torch.clamp(dist_sq, min=1e-6)  # Avoid division by zero

                    if torch.isnan(dist_sq).any():
                        logging.error("NaN detected in gravitational distance!")

                    token_freq_i = self.token_frequencies[token_indices[:, i]]
                    token_freq_j = self.token_frequencies[token_indices[:, j]]
                    if torch.isnan(self.token_frequencies).any():
                        logging.error("NaN detected in token frequencies!")
                    forces[:, i, j] = token_freq_i * token_freq_j / dist_sq.squeeze()

                    forces[:, i, j] = self.token_frequencies[token_indices[:, i]] * self.token_frequencies[token_indices[:, j]] / dist_sq.squeeze()
                    if torch.isnan(forces).any():
                        logging.error("NaN detected in gravitational forces!")

        logging.debug(f"Shape of forces after dist_sq: {forces.shape}")


        # Normalize and reshape
        forces_sum = forces.sum(dim=-1, keepdim=True)
        forces_sum = torch.clamp(forces_sum, min=1e-6)  # Avoid zero sum
        forces = forces / forces_sum
        logging.debug(f"Shape of forces output from compute GRE: {forces.shape}")
        if torch.isnan(forces).any():
            logging.error("NaN detected in gravitational forces!")

        return forces

    def forward(self, token_indices):
        """
        Computes the relationally-attended token embeddings.
        """
        logging.debug(f"Shape of token_indices to GRE: {token_indices.shape}")

        grav_forces = self.compute_gravitational_forces(token_indices)  # (batch, seq_len, seq_len)
        logging.debug(f"Shape of grav_forces to GRE: {grav_forces.shape}")

        token_embeddings = self.embeddings(token_indices)  # (batch, seq_len, embedding_dim)
        logging.debug(f"Shape of token_embeddings to GRE: {token_embeddings.shape}")

        # Weighted sum of embeddings based on gravitational force
        relational_embeddings = torch.einsum("bij,bjd->bid", grav_forces, token_embeddings)
        logging.debug(f"Shape of relational_embeddings to GRE: {relational_embeddings.shape}")
        if torch.isnan(relational_embeddings).any():
            logging.error("NaN detected in GRE relational embeddings!")

        return relational_embeddings

class QuaternionTransformerBlockRelationalLNS(nn.Module):
    """
    Transformer block with relational learning instead of attention.
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(QuaternionTransformerBlockRelationalLNS, self).__init__()
        self.feed_forward = QuaternionFeedForwardLNS(embedding_dim, hidden_dim)
        self.norm1 = QuaternionLayerNorm(embedding_dim)
        self.norm2 = QuaternionLayerNorm(embedding_dim)

    def forward(self, x):
        # Apply relational learning
        logging.debug(f"Shape of x to qtr block: {x.shape}")

        # Apply feed-forward transformation
        ff_output = self.feed_forward(x)
        logging.debug(f"Shape of feed forward outpt to QTR block: {x.shape}")
        if torch.isnan(x).any():
            logging.error("NaN detected in feedforward output!")

        x = self.norm2(x + ff_output)
        logging.debug(f"Shape of x in block after norm: {x.shape}")
        if torch.isnan(x).any():
            logging.error("NaN detected in  feedforward norm !")

        return x
    
class QuaternionTransformerRelationalLNS(nn.Module):
    """
    Transformer model using relational learning instead of standard attention.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_length, num_layers):
        super(QuaternionTransformerRelationalLNS, self).__init__()
        self.embedding = QuaternionEmbedding(vocab_size, embedding_dim)  # Embedding layer
        self.relational_encoding = GravitationalRelationalEncodingLNS(vocab_size, embedding_dim, seq_length) #relational encoding
        self.relational_to_vocab = nn.Linear(embedding_dim, vocab_size)  # Learnable projection
        self.vocab_size=vocab_size
        self.layers = nn.ModuleList([
            QuaternionTransformerBlockRelationalLNS(embedding_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        # Introduce the quaternion-to-real conversion module.
        self.quat_to_real = QuaternionToReal(embedding_dim, method="norm")
        # The output projection now maps from (batch, seq_len, embedding_dim) to vocab size.
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) as token IDs
        logging.debug(f"Shape of x to QTR: {x.shape}")
        # Quaternion Embedding Table: [vocab_size, embed_dim, quaternion_dim]

        x = self.relational_encoding(x)  # Apply once before embeddings
        logging.debug(f"Shape of x after relational encoding: {x.shape}")
        ##relational probs = embedding over (encoded x)
        x = self.relational_to_vocab(x)  # Shape: [batch, seq_len, vocab_size]
        logging.debug(f"Shape of relational logits: {x.shape}")
        #Relational logits = softmax over relational probabilities
        x = F.softmax(x, dim=-1)  # Shape: [batch, seq_len, vocab_size]
        logging.debug(f"Shape of relational softmax: {x.shape}")
        if torch.isnan(x).any():
            logging.error("NaN detected before softmax application!")

        vocab_indices = torch.arange(self.vocab_size, device=x.device)  # Create indices for all vocab words
        embedding_table = self.embedding(vocab_indices)  # Shape: [vocab_size, embed_dim, quaternion_dim]
        logging.debug(f"Shape of embedding_table: {embedding_table.shape}")
        # Compute weighted sum of embeddings using relational distribution
        x = torch.einsum('bsv,veq->bseq', x, embedding_table)
        logging.debug(f"Shape of weighted_embedding: {x.shape}")  # [batch, seq_len, embed_dim, quaternion_dim]
        if torch.isnan(x).any():
            logging.error("NaN detected in model forward pass before layers!")

        for layer in self.layers:
            x = layer(x)  # Each block works in quaternion space.
            if torch.isnan(x).any():
                logging.error("NaN detected in model forward pass layers!")

            logging.debug(f"Shape of x after layer: {x.shape}")
        if torch.isnan(x).any():
            logging.error("NaN detected in model forward pass after layers!")

        logging.debug(f"Shape before real conversion: {x.shape}")  # (batch, seq_len, embedding_dim, 4)
        # Convert quaternion representation to real numbers.
        x_real = self.quat_to_real(x)  # Shape: (batch, seq_len, embedding_dim)
        logging.debug(f"Shape after real conversion: {x_real.shape}")
        if torch.isnan(x).any():
            logging.error("NaN detected in model forward pass real conversion!")

        logits = self.output_projection(x_real)  # Now produce logits for each token.
        logging.debug(f"Logits shape: {logits.shape}")
        if torch.isnan(logits).any():
            logging.error("NaN detected in model forward pass logits!")

        return logits

class QuaternionToReal(nn.Module):
    def __init__(self, embedding_dim, method='norm'):
        """
        Converts a quaternion representation to a real-valued vector.
        
        Args:
            embedding_dim (int): The number of quaternion features.
            method (str): The conversion method: 'norm', 'real', or 'learned'.
        """
        super(QuaternionToReal, self).__init__()
        self.method = method
        if method == 'learned':
            # Learn a linear combination from the four components to a single scalar per quaternion.
            self.linear = nn.Linear(4, 1)
        # For 'norm' and 'real', no additional parameters are needed.

    def forward(self, x):
        # x is expected to have shape (batch, seq_len, embedding_dim, 4)
        logging.debug(f"QuaternionToReal input shape: {x.shape}")
        if self.method == 'norm':
            # Compute the quaternion norm for each quaternion.
            # This results in a tensor of shape (batch, seq_len, embedding_dim)
            return torch.norm(x, dim=-1)
        elif self.method == 'real':
            # Use only the real part (first component)
            return x[..., 0]
        elif self.method == 'learned':
            # Apply a learned linear combination across the quaternion components.
            # x has shape (batch, seq_len, embedding_dim, 4); we apply self.linear to the last dimension.
            out = self.linear(x)  # Shape: (batch, seq_len, embedding_dim, 1)
            return out.squeeze(-1)  # Shape: (batch, seq_len, embedding_dim)
        else:
            logging.debug(f"QuaternionToReal output shape: {out.shape}")
            raise ValueError("Invalid conversion method. Choose from 'norm', 'real', or 'learned'.")

    def forward(self, x):
        logging.debug(f"QuaternionToReal input shape: {x.shape}")

        # ✅ Prevent log(0) or log(negative)
        x = torch.clamp(x, min=1e-6)

        real_x = x[..., 0]  # Extract real component
        logging.debug(f"Converted real component shape: {real_x.shape}")

        if torch.isnan(real_x).any() or torch.isinf(real_x).any():
            logging.error("NaN or Inf detected in QuaternionToReal output!")

        return real_x

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_combined_mask(batch_input_ids, pad_token_id):
    """
    Create a combined attention mask that incorporates both the causal (subsequent) mask
    and the padding mask. This function ensures that each row has at least one valid token.
    """
    batch_size, seq_length = batch_input_ids.size()
    device = batch_input_ids.device
    
    # Generate causal (subsequent) mask: shape (seq_len, seq_len)
    causal_mask = generate_square_subsequent_mask(seq_len).to(device)
    logging.debug(f"Shape of causal_mask before expand: {causal_mask.shape}")

    # Expand to batch dimension: (batch_size, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    logging.debug(f"Shape of causal_mask after expansion: {causal_mask.shape}")
    # Create padding mask: valid tokens are True, padded tokens are False.
    # Shape: (batch_size, seq_len)
    padding_mask = (batch_input_ids != pad_token_id)
    # Expand padding mask to match the shape (batch_size, seq_len, seq_len)
    # Here we broadcast along one dimension so that we mask out positions in each row.
    padding_mask_expanded = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    logging.debug(f"Shape of padding_mask after expansion: {padding_mask_expanded.shape}")

    # Combine masks: where padding_mask is False, set to -inf.
    # This keeps the causal structure while ensuring that padded positions are fully masked.
    combined_mask = causal_mask.masked_fill(~padding_mask_expanded, float('-inf'))
    logging.debug(f"Shape of combined_mask after fill: {combined_mask.shape}")

    # Check each row: if an entire row is -inf, force the first token (or a designated position) to be valid.
    for i in range(batch_size):
        for j in range(seq_len):
            if torch.all(combined_mask[i, j] == float('-inf')):
                combined_mask[i, j, 0] = 0.0  # Force at least one valid position
    
    return combined_mask

class QuaternionLinearLNS(nn.Module):
    def __init__(self, in_features, out_features, seq_len, bias=True):
        super(QuaternionLinearLNS, self).__init__()
        self.lns = LogarithmicNumberSystem()
        self.out_features = out_features  # ✅ Fix: Define out_features
        self.in_features = in_features

        # Fix: Ensure weight tensors match seq_len
        self.r_weight = nn.Parameter(self.lns.to_log(torch.randn(seq_len, out_features)))  # Fix: Use seq_len instead of 256
        self.i_weight = nn.Parameter(self.lns.to_log(torch.randn(seq_len, out_features)))
        self.j_weight = nn.Parameter(self.lns.to_log(torch.randn(seq_len, out_features)))
        self.k_weight = nn.Parameter(self.lns.to_log(torch.randn(seq_len, out_features)))

        if bias:
            self.bias = nn.Parameter(self.lns.to_log(torch.randn(out_features, 4)))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        log_x = self.lns.to_log(x)

        device = log_x.device  # Ensure all tensors match input device

        # Move parameters correctly
        self.r_weight.data = self.r_weight.data.to(device)
        self.i_weight.data = self.i_weight.data.to(device)
        self.j_weight.data = self.j_weight.data.to(device)
        self.k_weight.data = self.k_weight.data.to(device)

        # Ensure log_x has correct shape
        if log_x.dim() == 2:
            log_x = log_x.unsqueeze(-1).expand(-1, -1, 4)  # Expand to (batch, seq_len, 4)

        # Ensure weight tensors match input shape for broadcasting
        r_weight_expanded = self.r_weight.unsqueeze(0).expand(log_x.shape[0], log_x.shape[1], -1)
        i_weight_expanded = self.i_weight.unsqueeze(0).expand(log_x.shape[0], log_x.shape[1], -1)
        j_weight_expanded = self.j_weight.unsqueeze(0).expand(log_x.shape[0], log_x.shape[1], -1)
        k_weight_expanded = self.k_weight.unsqueeze(0).expand(log_x.shape[0], log_x.shape[1], -1)

        # ✅ Fix expand dimension issue with conditional adjustment
        def adjust_dim_for_expand(tensor, target_tensor):
            if tensor.dim() == target_tensor.dim():
                return tensor
            elif tensor.dim() + 1 == target_tensor.dim():
                return tensor.unsqueeze(-1)
            else:
                raise RuntimeError(f"Unexpected dimension mismatch: {tensor.shape} vs {target_tensor.shape}")

        r_out = self.lns.log_add(adjust_dim_for_expand(log_x[..., 0], r_weight_expanded).expand_as(r_weight_expanded), r_weight_expanded)
        i_out = self.lns.log_add(adjust_dim_for_expand(log_x[..., 1], i_weight_expanded).expand_as(i_weight_expanded), i_weight_expanded)
        j_out = self.lns.log_add(adjust_dim_for_expand(log_x[..., 2], j_weight_expanded).expand_as(j_weight_expanded), j_weight_expanded)
        k_out = self.lns.log_add(adjust_dim_for_expand(log_x[..., 3], k_weight_expanded).expand_as(k_weight_expanded), k_weight_expanded)

        out = torch.stack([r_out, i_out, j_out, k_out], dim=-1)

        # ✅ Ensure bias expansion matches out dimensions
        if self.bias is not None:
            self.bias.data = self.bias.data.to(device)
            bias_expanded = self.bias.unsqueeze(0).unsqueeze(0).expand(out.shape)  # ✅ Fix bias expansion
            out = self.lns.log_add(out, bias_expanded)

        return self.lns.from_log(out)


class ModReLU(nn.Module):
    def __init__(self, features):
        super(ModReLU, self).__init__()
        # One bias per quaternion (per output feature)
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        # x shape: (batch, seq_len, features, 4)
        # Compute the norm of each quaternion
        norm = torch.norm(x, dim=-1)  # shape: (batch, seq_len, features)
        # Compute scaling factor: relu(norm + bias) / (norm + epsilon)
        scale = F.gelu(norm + self.bias.unsqueeze(0).unsqueeze(0))
        scale = scale / (norm + 1e-6)
        scale = scale.unsqueeze(-1)  # shape: (batch, seq_len, features, 1)
        return x * scale

class QuaternionGHROptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super(QuaternionGHROptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # Log parameter name, shape, and gradient shape.
                logging.debug(f"Parameter: {param.shape}, Grad: {param.grad.shape}")
                
                # Process only quaternion parameters (which should have last dimension == 4)
                if param.grad.shape[-1] != 4 or param.data.shape[-1] != 4:
                    logging.debug("Skipping quaternion update for non-quaternion parameter.")
                    continue
                
                # Compute quaternion gradient update...
                q0, q1, q2, q3 = param.grad[..., 0], param.grad[..., 1], param.grad[..., 2], param.grad[..., 3]
                q_star = (q0 - q1 * 1j - q2 * 1j - q3 * 1j) / 4  
                
                # For illustration, assume we update by taking the real and imaginary parts in a fixed pattern:
                update = torch.stack([q_star.real, q_star.imag, q_star.real, q_star.imag], dim=-1)
                
                if update.shape != param.data.shape:
                    logging.error(f"Update shape {update.shape} does not match parameter shape {param.data.shape}")
                    continue
                
                param.data -= group['lr'] * update

        return loss

class CliffordBackprop(nn.Module):
    def __init__(self):
        super(CliffordBackprop, self).__init__()

    def forward(self, grad, activation):
        """
        Compute quaternion gradients using Clifford algebra.
        """
        q0, q1, q2, q3 = activation[..., 0], activation[..., 1], activation[..., 2], activation[..., 3]
        
        # Clifford gradient update using geometric product
        grad_q0 = grad[..., 0] * q0 - grad[..., 1] * q1 - grad[..., 2] * q2 - grad[..., 3] * q3
        grad_q1 = grad[..., 0] * q1 + grad[..., 1] * q0 + grad[..., 2] * q3 - grad[..., 3] * q2
        grad_q2 = grad[..., 0] * q2 - grad[..., 1] * q3 + grad[..., 2] * q0 + grad[..., 3] * q1
        grad_q3 = grad[..., 0] * q3 + grad[..., 1] * q2 - grad[..., 2] * q1 + grad[..., 3] * q0
        
        return torch.stack([grad_q0, grad_q1, grad_q2, grad_q3], dim=-1)

class QuaternionGeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=5):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, targets, mask, architecture):
        best_model = None
        best_loss = float('inf')
        n=0
        logging.debug(f"Evolution step inputs shape: {inputs.shape}")

        for model in self.population:

            loss = 0
            logging.debug(f"Evolution step inputs shape passed to model: {inputs.shape}")
            if architecture == "Quaternion LNS Transformer":
                logits= model(inputs, mask)
                # Flatten logits and targets:
                logits_real = self.model.quat_to_real(logits)  # Convert quaternions to real values (2, 128, 10000)
                logging.debug(f"Logits after real conversion: {logits_real.shape}")  

                logits_flat = logits_real.view(-1, logits_real.size(-1))  # (batch * seq_len, vocab_size)
                logging.debug(f"Logits reshaped for loss: {logits_flat.shape}")  
            else:
                logits = model(inputs)
                logits_flat = logits.reshape(-1, logits.size(-1))

            logging.debug(f"Evolution step logits shape: {logits.shape}")
            logging.debug(f"Evolutions logits sample: {logits[:10]}")

            logging.debug(f"Evolution step logits shape passed to loss: {logits_flat.shape}")
            logging.debug(f"Evolution step targets shape passed to loss: {targets.shape}")

            loss += loss_fn(logits_flat, targets).item()
            logging.debug(f"Logits_flat shape: {logits_flat.shape}")
            logging.debug(f"targets_flat shape: {targets.shape}")  

            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss}")
                    best_model = model
            else:
                if architecture == "Quaternion LNS Transformer":
                    logits= model(inputs, mask)
                    # Flatten logits and targets:
                    logits_real = self.model.quat_to_real(logits)  # Convert quaternions to real values (2, 128, 10000)
                    logging.debug(f"Logits after real conversion: {logits_real.shape}")  

                    logits_flat = logits_real.view(-1, logits_real.size(-1))  # (batch * seq_len, vocab_size)
                    logging.debug(f"Logits reshaped for loss: {logits_flat.shape}")  
                else:
                    logits = model(inputs)
                    logits_flat = logits.reshape(-1, logits.size(-1))
                logging.debug(f"Logits reshaped for loss: {logits_flat.shape}")  
                logging.debug(f"Evolution step logits shape passed to loss: {logits_flat.shape}")
                logging.debug(f"Evolution step targets shape passed to loss: {targets.shape}")
                loss += loss_fn(logits_flat, targets).item()
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss}")
                        best_model = model
                logging.debug(f"Logits_flat shape: {logits_flat.shape}")
                logging.debug(f"targets_flat shape: {targets.shape}") 
   
        return best_model

    def evolve(self, loss_fn, inputs, targets, mask, architecture):
        best_model = self.select_best(loss_fn, inputs, targets, mask, architecture)
        self.population = [copy.deepcopy(best_model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, targets, mask, architecture)
    
class QuaternionFireflyOptimizer:
    def __init__(self, model, num_fireflies=5, alpha=0.1, beta=0.5):
        self.population = [copy.deepcopy(model) for _ in range(num_fireflies)]
        self.alpha = alpha
        self.beta = beta

    def move_towards(self, firefly1, firefly2):
        for p1, p2 in zip(firefly1.parameters(), firefly2.parameters()):
            p1.data += self.beta * (p2.data - p1.data) + self.alpha * torch.randn_like(p1)

    def optimize(self, loss_fn, data_loader):
        fitness = [sum(loss_fn(m(batch[0]), batch[1]).item() for batch in data_loader) for m in self.population]
        best_idx = torch.argmin(torch.tensor(fitness))
        best_firefly = self.population[best_idx]
        n=0
        for i in range(len(self.population)):
            if i != best_idx:
                self.move_towards(self.population[i], best_firefly)
                n=n+1
                print(f"Iteration {n}, Loss: {fitness.item()}")
        return best_firefly

class QuaternionNEAT:
    def __init__(self, model, population_size=5):
        self.population = [copy.deepcopy(model) for _ in range(population_size)]

    def mutate_topology(self, model):
        new_model = copy.deepcopy(model)
        if random.random() < 0.5:
            # Add a new quaternion neuron
            new_layer = QuaternionLinearLNS(new_model.layers[0].in_features, new_model.layers[0].out_features, seq_len)
            new_model.layers.insert(random.randint(0, len(new_model.layers)), new_layer)
        return new_model

    def evolve(self, loss_fn, data_loader):
        n=0
        best_model = min(self.population, key=lambda m: sum(loss_fn(m(batch[0]), batch[1]).item() for batch in data_loader))

        self.population = [self.mutate_topology(best_model) for _ in range(len(self.population))]
        n=n+1
        print(f"Iteration {n}, Loss: {self.population.item()}")
        return best_model

class LogarithmicNumberSystem:
    """
    CUDA-Optimized Logarithmic Number System (LNS) for efficient GPU computation.
    """
    def __init__(self, epsilon=1e-6, use_cuda=True):
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    def to_log(self, x):
        logging.debug(f"shape of x to log {x.shape}")

        """ Convert tensor to log-space using CUDA acceleration. """
        return torch.log(torch.clamp(x.to(self.device), min=self.epsilon))

    def from_log(self, log_x):
        logging.debug(f"shape of log_x to to convert back {log_x.shape}")

        """ Convert back from log-space. """
        return torch.exp(log_x)

    def log_add(self, log_x, log_y):
        """ Logarithmic addition using CUDA-accelerated Log-Sum-Exp trick. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")

        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        return max_val + torch.log(torch.exp(log_x - max_val) + torch.exp(log_y - max_val))

    def log_sub(self, log_x, log_y):
        """ Logarithmic subtraction with CUDA support. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")
        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        logging.debug(f"shape of max_val for sub {log_x.shape}")
        sub_result = torch.exp(log_x - max_val) - torch.exp(log_y - max_val)
        logging.debug(f"shape of sub_result for sub {log_x.shape}")
        
        return max_val + torch.log(torch.clamp(sub_result, min=self.epsilon))

    def log_mul(self, log_x, log_y):
        """ Logarithmic multiplication using CUDA (log-space addition). """
        logging.debug(f"shape of log_x for mul {log_x.shape}")
        logging.debug(f"shape of log_y for mul {log_y.shape}")
        return log_x + log_y

    def log_div(self, log_x, log_y):
        """ Logarithmic division using CUDA (log-space subtraction). """
        logging.debug(f"shape of log_x for div {log_x.shape}")
        logging.debug(f"shape of log_y for div {log_y.shape}")
        return log_x - log_y
    
    def log_add_einsum(self, equation, log_x, log_y):
        """
        Implements log-space einsum operation by applying log-sum-exp trick.
        """
        # Ensure tensors have same shape
        assert log_x.shape == log_y.shape, f"Shape mismatch: {log_x.shape} vs {log_y.shape}"

        max_val = torch.max(log_x, log_y)
        logging.debug(f"shape of max_val for einsum {max_val.shape}")
        logging.debug(f"shape of log_x for einsum {log_x.shape}")
        logging.debug(f"shape of log_y for einsum {log_y.shape}")
        log_x_adj = log_x - max_val
        log_y_adj = log_y - max_val
        logging.debug(f"Einsum equation: {equation}")
        logging.debug(f"log_x_adj shape: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape: {log_y_adj.shape}")
        log_x_adj = log_sum_exp(log_x_adj, dim=-1)
        #log_x_adj = log_x_adj.expand(-1,-1,128, -1)
        log_y_adj = log_sum_exp(log_y_adj, dim=-1)
        #log_y_adj = log_y_adj.expand(-1,-1,128, -1)
        logging.debug(f"log_x_adj shape after log_sum_exp: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape after log_sum_exp: {log_y_adj.shape}")
        einsum_tensor = torch.einsum(equation, [log_x_adj, log_y_adj])
        logging.debug(f"einsum_tenspr shape: {einsum_tensor.shape}")
        einsum_tensor = einsum_tensor.unsqueeze(-1)
        # ✅ Ensure max_val reduces along the last dim before logsumexp
        max_val, _ = torch.max(einsum_tensor, dim=-1, keepdim=True)  
        logging.debug(f"Shape of max_val: {max_val.shape}")  # Should be [batch, seq_len, seq_len, 1]
        einsum_tensor_adj = einsum_tensor - max_val

        logging.debug(f"Shape of einsum t after max subtraction: {einsum_tensor_adj.shape}")
        einsum_tensor_adj = torch.logsumexp(einsum_tensor_adj, dim=-1)
        logging.debug(f"Shape einsum t before sum: {einsum_tensor_adj.shape}")
        # ✅ Apply logsumexp only across the correct dimension
        output = torch.einsum('bkd,bkdt->bkd', einsum_tensor_adj, max_val)
        logging.debug(f"Shape einsum output: {output.shape}")

        return  output


def log_sum_exp(tensor, dim=-1, keepdim=True):
    """
    Optimized Log-Sum-Exp function for stable summation in log-space.
    Prevents overflow and underflow issues by normalizing.
    """
    logging.debug(f"shape of tensor for log_sum_exp {tensor.shape}")
    
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)  # Find max value
    return max_val + torch.log(torch.sum(torch.exp(tensor - max_val), dim=dim, keepdim=keepdim))

class QuaternionTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quaternion Transformer GUI")

        # Transformer Parameters
        self.layers = []
        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Quaternion LNS Relational")
        self.num_parameters = tk.IntVar(value=1024)
        self.vocab_size = tk.IntVar(value=10000)
        self.hidden_size = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=8)

        self.pad_token_id = 0  # Default value, adjust based on your tokenizer setup

        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")

        # Dynamically calculate parameters based on other inputs
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()

        # Training Parameters
        self.dataset_path = ""
        self.vocab_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=2)
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
        self.accuracy_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()

        # Model and Data Variables
        self.model = None
        self.tokenizer = None
        self.dataset_path = None
        self.vocab_path = None
        self.tokenizer_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.tokenized_data_path = None  # To store the tokenized data file path
        self.use_genetic_algo = "Genetic Algorithm"  # default to optim

        # Device (CPU or GPU) - Initially set based on device_option
        self.device = torch.device(self.map_device(self.device_option.get()))

        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")

        self.create_widgets()

    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def create_widgets(self):
        # Transformer Construction Frame
        transformer_frame = ttk.LabelFrame(self.root, text="Transformer Construction", padding=(10, 10))
        transformer_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(transformer_frame, text="Number of Parameters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_parameters, state="readonly").grid(row=0, column=1)

        ttk.Label(transformer_frame, text="Vocabulary Size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.vocab_size).grid(row=2, column=1)

        ttk.Label(transformer_frame, text="Hidden Size:").grid(row=3, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.hidden_size).grid(row=3, column=1)

        ttk.Label(transformer_frame, text="Number of Layers:").grid(row=2, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_layers).grid(row=2, column=5)

        # Device Selection
        ttk.Label(transformer_frame, text="Select Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        device_options = ["CPU"]
        if torch.cuda.is_available():
            device_options.append("GPU")
        device_combo = ttk.Combobox(transformer_frame, textvariable=self.device_option, values=device_options, state="readonly")
        device_combo.grid(row=4, column=1, sticky="w", pady=(10, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        # Attach parameter calculation to variable updates
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="Quaternion LNS Transformer")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["Quaternion LNS Transformer", "Quaternion LNS Relational"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)
        self.genetic_algo_var = tk.StringVar(value="Genetic Algorithm")
        ttk.Label(transformer_frame, text="Algo:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.genetic_algo_var, values=["GHR Optim", "Genetic Algorithm", "Firefly", "NEAT"], state="readonly").grid(row=0, column=4)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)
        self.use_chunked_dataset = tk.BooleanVar(value=False)
        self.test_bool = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(data_frame, text="Use Chunked Dataset", variable=self.use_chunked_dataset).pack(pady=5)
        ttk.Checkbutton(data_frame, text="Use Std/bert Model", variable=self.test_bool).pack(pady=5)
        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Save Dataset as Text File", command=self.save_dataset_as_text).pack(pady=5)
        ttk.Button(data_frame, text="Select Vocabulary File", command=self.select_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Create Tokenizer from Vocab", command=self.create_tokenizer_from_vocab).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Test Tokenizer", command=self.test_tokenizer).pack(pady=5)


        # New buttons for tokenized data
        ttk.Button(data_frame, text="Select/Create Tokenized Data", command=self.select_or_create_tokenized_data).pack(pady=5)
        ttk.Button(data_frame, text="Tokenize Data", command=self.tokenize_data).pack(pady=5)

        # Training Configuration Frame
        train_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.batch_size).grid(row=0, column=1)

        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.learning_rate).grid(row=1, column=1)

        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.epochs).grid(row=2, column=1)

        ttk.Button(train_frame, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Stop Training", command=self.stop_training_command).grid(row=4, column=0, pady=5)
        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(data_frame, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)

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

    def calculate_parameters(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        embedding_params = vocab_size * embedding_dim * 4  # Quaternion embeddings (4x normal embedding size)
        transformer_params = num_layers * (
            (embedding_dim * hidden_dim * 4) +  # Attention layers
            (hidden_dim * hidden_dim * 4) +  # Feed-forward layers
            (hidden_dim * 4 * embedding_dim * 4)  # Output layers
        )
        output_projection_params = embedding_dim * 4 * vocab_size  # Final projection
        return embedding_params + transformer_params + output_projection_params

    def update_num_parameters(self):
        vocab_size = self.vocab_size.get()
        embed_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()

        total_params = self.calculate_parameters(vocab_size, embed_size, num_layers, hidden_size)
        self.num_parameters.set(total_params)

    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")

    def resize_checkpoint_weights(self, state_dict, new_vocab_size, embed_size):
        """
        Resize checkpoint weights to match the current model's dimensions.
        """
        # This method may need to be updated depending on the model's state_dict keys
        return state_dict

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def save_transformer_and_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Please initialize the model first.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Please load a tokenizer first.")
            return

        transformer_data = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            # Save configuration
            config_path = os.path.join(directory, "model_config.json")
            with open(config_path, "w") as file:
                json.dump(transformer_data, file, indent=4)

            # Save weights
            if self.architecture.get() == "Quaternion LNS Transformer":
                model_file_name = 'lns_quaternion_transformer_model.pth'
            elif self.architecture.get() == "Quaternion LNS Relational":
                model_file_name = 'quaternion_lns_relational.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(directory)

            messagebox.showinfo("Success", "Model, tokenizer, and configuration saved successfully!")
            logging.info("Model, tokenizer, and configuration saved successfully.")

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")

    def select_vocab(self):
        self.vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if self.vocab_path:
            messagebox.showinfo("Success", f"Vocabulary file selected: {self.vocab_path}")

    def select_tokenizer(self):
        self.tokenizer_path = filedialog.askopenfilename(
            title="Select Tokenizer File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if self.tokenizer_path:
            messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")

    def test_tokenizer(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        sample_text = simpledialog.askstring("Test Tokenizer", "Enter a sample text to tokenize:")
        if sample_text:
            tokens = self.tokenizer.tokenize(sample_text)
            token_ids = self.tokenizer.encode(sample_text)
            logging.info(f"Sample Text: {sample_text}")
            logging.info(f"Tokens: {tokens}")
            logging.info(f"Token IDs: {token_ids}")
            messagebox.showinfo("Tokenizer Test", f"Tokens: {tokens}\nToken IDs: {token_ids}")

    def save_dataset_as_text(self):
        if not hasattr(self, 'text_data') or not self.text_data:
            messagebox.showerror("Error", "No dataset loaded or processed to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Dataset as Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    for line in self.text_data:
                        f.write(line + '\n')
                messagebox.showinfo("Success", f"Dataset saved to {save_path}")
                logging.info(f"Dataset saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save dataset: {e}")
                messagebox.showerror("Error", f"Failed to save dataset: {e}")



    def create_tokenizer_from_vocab(self):
        try:
            # Ask the user to select the vocabulary file
            vocab_path = filedialog.askopenfilename(
                title="Select Vocabulary File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not vocab_path:
                messagebox.showerror("Error", "No vocabulary file selected.")
                return

            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)

            # Create a word-level tokenizer
            #tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
            #tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            # Initialize a BPE tokenizer with an unknown token.
            tokenizer = Tokenizer(models.BPE(vocab=vocab, unk_token="<UNK>"))
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            tokenizer.decoder = ByteLevelDecoder()

            # Set normalizer to remove unwanted whitespaces
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFKC(),
                normalizers.Replace('\n', ''),
                normalizers.Replace('\r', ''),
                normalizers.Replace('\t', ''),
            ])
            
            # Wrap with PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token='<UNK>',
                pad_token='<PAD>',
                bos_token='<BOS>',
                eos_token='<EOS>',
                model_max_length=seq_len,
            )

            # Ensure special tokens are added
            self.tokenizer.add_special_tokens({
                'unk_token': '<UNK>',
                'pad_token': '<PAD>',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })

            # Save the tokenizer
            save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                self.tokenizer.save_pretrained(save_directory)
                self.tokenizer_path = os.path.join(save_directory, 'tokenizer.json')
                messagebox.showinfo("Success", f"Tokenizer saved to {self.tokenizer_path}")
                logging.info(f"Tokenizer saved to {self.tokenizer_path}")
            else:
                messagebox.showerror("Error", "No save directory selected for tokenizer.")
                return

            # Test the tokenizer
            test_text = "Hello World!\nThis is a test.\tLet's remove line breaks and tabs."
            tokens = self.tokenizer.tokenize(test_text)
            logging.info(f"Test tokenization of '{test_text}': {tokens}")

            tokenizer_vocab = self.tokenizer.get_vocab()
            sorted_vocab = dict(sorted(tokenizer_vocab.items(), key=lambda item: item[1]))
            logging.info(f"Sorted Tokenizer Vocabulary: {sorted_vocab}")

            logging.info("Tokenizer created and saved successfully")
        except Exception as e:
            logging.error(f"Failed to create tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")
            raise

    def load_tokenizer(self):
        try:
            self.tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not self.tokenizer_path or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
            logging.info(f"Tokenizer loaded from {self.tokenizer_path}")

            # Load special tokens map
            special_tokens_path = os.path.join(os.path.dirname(self.tokenizer_path), "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, "r") as file:
                    special_tokens = json.load(file)

                for key, value in special_tokens.items():
                    if isinstance(value, dict):
                        special_tokens[key] = AddedToken(value["content"], lstrip=value.get("lstrip", False),
                                                         rstrip=value.get("rstrip", False))
                    elif not isinstance(value, (str, AddedToken)):
                        raise ValueError(f"Invalid token format for key {key}: {value}")

                self.tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration
            tokenizer_config_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r") as file:
                    tokenizer_config = json.load(file)
                    self.tokenizer.init_kwargs.update(tokenizer_config)

                    # Check and set model_max_length
                    if "model_max_length" in tokenizer_config:
                        self.tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Explicitly set model_max_length if still unset or unreasonable
            if not hasattr(self.tokenizer, "model_max_length") or self.tokenizer.model_max_length > 1024 * 1024:
                self.tokenizer.model_max_length = seq_len  # Default to seq_len

            # Check consistency
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)

            # Ensure special tokens are correctly set
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not self.tokenizer.unk_token:
                self.tokenizer.unk_token = "<UNK>"
                self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not self.tokenizer.bos_token:
                self.tokenizer.bos_token = "<BOS>"
                self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not self.tokenizer.eos_token:
                self.tokenizer.eos_token = "<EOS>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")
            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)
            

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")

    def select_or_create_tokenized_data(self):
        use_chunked = self.use_chunked_dataset.get()
        answer = messagebox.askyesno("Select or Create Tokenized Data", "Do you want to use existing tokenized data?")
        
        if answer:
            if use_chunked:
                # User wants to use existing chunked tokenized data, select a directory
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Tokenized Data Directory",
                    mustexist=True
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data directory selected: {self.tokenized_data_path}")
            else:
                # User wants to use existing single tokenized data file, select a file
                self.tokenized_data_path = filedialog.askopenfilename(
                    title="Select Tokenized Data File",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    # Attempt to load the file to validate its content
                    try:
                        with open(self.tokenized_data_path, 'r', encoding='utf-8') as f:
                            self.input_ids, self.labels = [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
                        messagebox.showinfo("Success", f"Tokenized data file loaded: {self.tokenized_data_path}")
                        logging.info(f"Tokenized data file loaded successfully with {len(self.input_ids)} entries.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load tokenized data file: {str(e)}")
        else:
            if use_chunked:
                # User wants to create new chunked tokenized data, select a directory to save
                self.tokenized_data_path = filedialog.askdirectory(
                    title="Select Directory to Save Tokenized Data"
                )
                if self.tokenized_data_path:
                    os.makedirs(self.tokenized_data_path, exist_ok=True)  # Ensure directory is created
                    messagebox.showinfo("Success", f"Tokenized data will be saved to directory: {self.tokenized_data_path}")
            else:
                # User wants to create new single tokenized data file, select a file path
                self.tokenized_data_path = filedialog.asksaveasfilename(
                    title="Save Tokenized Data As",
                    defaultextension=".jsonl",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.tokenized_data_path:
                    messagebox.showinfo("Success", f"Tokenized data will be saved to file: {self.tokenized_data_path}")


            
    def tokenize_data(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        if not hasattr(self, 'query_target_pairs') or not self.query_target_pairs:
            messagebox.showerror("Error", "No query-target pairs loaded. Please load the dataset first.")
            return
        if not self.tokenized_data_path:
            messagebox.showerror("Error", "Tokenized data path not set. Please select or create tokenized data.")
            return

        # Select training mode
        training_mode = self.training_mode.get()  # "imitation", "completion", "response"
        self.input_ids = []  # Initialize for unchunked dataset
        self.labels = []  # Initialize for unchunked dataset
        
        try:
            use_chunked = self.use_chunked_dataset.get()
            if use_chunked:
                #create path if none
                os.makedirs(self.tokenized_data_path, exist_ok=True)
                chunk_size = 32
                num_chunks = (len(self.query_target_pairs) + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    chunk_pairs = self.query_target_pairs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                    chunk_file_path = os.path.join(self.tokenized_data_path, f'chunk_{chunk_idx}.jsonl')

                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        for query, target in chunk_pairs:
                            input_ids, labels = self._generate_training_pairs(query, target, training_mode)
                            if input_ids and labels:
                                record = {'input_ids': input_ids, 'labels': labels}
                                f.write(json.dumps(record) + '\n')
                logging.info(f"Chunk {chunk_idx} tokenized and saved to {chunk_file_path}")

                messagebox.showinfo("Success", f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized into {num_chunks} chunks and saved successfully to {self.tokenized_data_path}.")
            else:
                with open(self.tokenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target in self.query_target_pairs:
                        input_ids, labels = self._generate_training_pairs(query, target, training_mode)

                        if input_ids and labels:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels}


                            f.write(json.dumps(record) + '\n')
                logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                messagebox.showinfo("Success", f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
                logging.info(f"Data tokenized and saved successfully to {self.tokenized_data_path}.")
        except Exception as e:
            logging.error(f"Tokenization failed: {str(e)}")
            messagebox.showerror("Error", f"Tokenization failed: {str(e)}")

    def _generate_training_pairs(self, query, target, training_mode):
        # Tokenize query and target
        query_ids = self.tokenizer.encode(query, truncation=True, max_length=seq_len)
        target_ids = self.tokenizer.encode(target, truncation=True, max_length=seq_len)
        # Convert tokens to integers
        query_ids = [int(token) for token in query_ids]
        target_ids = [int(token) for token in target_ids]


        if training_mode == "imitation":
            input_ids = query_ids + [self.tokenizer.eos_token_id] 
            labels = query_ids + [self.tokenizer.eos_token_id] 
        elif training_mode == "completion":
            partial_length = len(query_ids) // 2
            partial_input = query_ids[:partial_length]
            #completion = query_ids[partial_length:] + [self.tokenizer.eos_token_id]

            input_ids = partial_input + [self.tokenizer.eos_token_id]
            # For completion, we want labels to represent the entire query, not just completion
            labels = query_ids + [self.tokenizer.eos_token_id]  
        else:  # response
            input_ids = query_ids + [self.tokenizer.eos_token_id]
            labels = target_ids + [self.tokenizer.eos_token_id]

        return input_ids, labels

    def add_layer(self):
        layer_type = simpledialog.askstring("Layer Type", "Enter layer type (e.g., attention, feed_forward)")
        if layer_type:
            layer_config = {
                "type": layer_type,
                "parameters": {}  # Placeholder for future parameter configuration
            }
            self.layers.append(layer_config)
            messagebox.showinfo("Layer Added", f"Layer of type '{layer_type}' added.")

    def save_transformer(self):
        transformer_data = {
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(transformer_data, file, indent=4)
            messagebox.showinfo("Save", "Transformer saved successfully!")
            logging.info(f"Number of layers in the model: {len(self.model.transformer_encoder.layers)}")

    def load_transformer(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                transformer_data = json.load(file)
            self.num_parameters.set(transformer_data["num_parameters"])
            self.layers = transformer_data["layers"]
            messagebox.showinfo("Success", "Transformer loaded successfully")

    def load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
            else:
                vocab_size = len(self.tokenizer)

            # Log and validate vocab size
            logging.info(f"Tokenizer vocabulary size: {vocab_size}")
            self.vocab_size.set(vocab_size)

            # Initialize the model based on architecture
            if self.architecture.get() == "Quaternion LNS Transformer":
                self.model = QuaternionTransformerLNS(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    hidden_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    seq_length=seq_len
                )
            elif self.architecture.get() == "Quaternion LNS Relational":
                self.model = QuaternionTransformerRelationalLNS(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    hidden_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),

                    seq_length=seq_len
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device
            self.model.to(self.device)
            logging.info(f"Model moved to device: {self.device}")

            # Load checkpoint if a model file is selected
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                logging.info("Model weights loaded and resized successfully.")

            logging.info(f"Model initialized on device: {self.device}")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")


    def calculate_learning_rate(self, total_params):
        # Calculate learning rate based on total parameters using the derived formula
        # LR = 17.38 * (Model Size)^-0.424
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def save_checkpoint(self, model, optimizer, epoch, path):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"Expected path to be str or os.PathLike, got {type(path).__name__}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        


    def validate_training_parameters(self):
        # Validate batch size
        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {self.batch_size.get()}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {self.epochs.get()}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False

        if not self.tokenized_data_path or not os.path.exists(self.tokenized_data_path):
            logging.error("Tokenized data path is invalid or does not exist.")
            messagebox.showerror("Error", "Tokenized data is not selected or does not exist.")
            return False

        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

    def training_loop(self):
        if not self.validate_training_parameters():
            return

        logging.info("All training parameters and data are properly initialized.")
        if not self.model:
            logging.error("Model not initialized before training")
            return
        self.use_genetic_algo = self.genetic_algo_var.get()

        try:
            if self.use_chunked_dataset.get():
                # Initialize the ChunkedDataset
                dataset = ChunkedDataset(
                    tokenized_data_path=self.tokenized_data_path,
                    tokenizer=self.tokenizer,
                    max_length=seq_len
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size.get(),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            else:
                # Initialize the standard dataset and dataloader

                # Ensure the tokenizer is loaded and has a valid pad_token_id
                pad_token_id = tokenizer.pad_token_id if tokenizer else 0  # Default to 1 if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths
                input_ids, seq_lengths = zip(*[
                    (
                        torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=self.device)[:max_length],
                        min(len(tokens), max_length)
                    )
                    for tokens in self.input_ids
                ])
                logging.info("input ids torched to tensor")

                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=self.device)[:max_length]
                    for tokens in self.labels
                ]
                logging.info("labels torched to tensor")

                # Stack tensors
                input_ids = torch.stack(input_ids)
                labels = torch.stack(labels)
                logging.debug(f"Shape of input_ids after stack: {input_ids.shape}")
                logging.debug(f"Shape of labels after stack: {labels.shape}")

                seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
                logging.info("datas stacked and seq lengths torched")

                # Perform assertions to validate tensors
                assert isinstance(input_ids, torch.Tensor), "input_ids should be a tensor"
                assert isinstance(labels, torch.Tensor), "labels should be a tensor"
                assert input_ids.dtype == torch.long, "input_ids should be of type torch.long"
                assert labels.dtype == torch.long, "labels should be of type torch.long"
                assert input_ids.size(1) == max_length, "input_ids should be padded to max_length"
                assert labels.size(1) == max_length, "labels should be padded to max_length"

                dataset = torch.utils.data.TensorDataset(input_ids, labels, seq_lengths)
                logging.info("dataset torched")
                dataloader = DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=True,
                    num_workers=0,  # Set to 0 to prevent multiple workers from loading chunks simultaneously
                    pin_memory=False,
                    collate_fn=collate_fn
                )
                logging.info("dataloader defined")
            ##chunked vs. standard else complete
            # Log dataset samples


            # Adjust learning rate based on architecture
            total_params = self.num_parameters.get()
            lr = self.learning_rate.get()
            logging.info(f"Learning Rate: {lr} for total parameters: {total_params}")

            # Learning rate scheduler
            total_steps = self.epochs.get() * len(dataloader)
            logging.info(f"Total training steps: {total_steps}")
            # Separate parameters based on their shape.
            quaternion_params = []
            standard_params = []
            for name, param in self.model.named_parameters():
                # Check if the parameter has at least one dimension and its last dimension is 4.
                if param.data.ndim >= 1 and param.data.shape[-1] == 4:
                    quaternion_params.append(param)
                else:
                    standard_params.append(param)

            # Create two optimizers:
            #Enable for standard optimizer/scheduler
            #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
            optimizer_std = torch.optim.AdamW(standard_params, lr=lr, weight_decay=1e-5)
            #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

            ##Quaternion Gradient-Real Hamilton Calculus backpropagaiton optimizer
            optimizer_q = QuaternionGHROptimizer(quaternion_params, lr=lr)

            scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=total_steps, eta_min=lr * 0.1)

            scheduler_std = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_std, T_max=total_steps, eta_min=lr * 0.1)
            logging.info("Scheduler defined")
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            self.model.train()
            logging.info("Model set to training mode")
            progress_step = 0
            n = 0
            
            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break

                epoch_loss = 0
                logging.info(f"Epoch {epoch+1} started")
                
          
                # Training loop
                for batch_idx, (batch_input_ids, batch_labels, seq_lengths) in enumerate(dataloader):
                    if self.stop_training.is_set():
                        logging.info("Training stopped by user.")
                        messagebox.showinfo("Info", "Training stopped by user.")
                        return          
                    #optimizer_q.zero_grad()
                    optimizer_std.zero_grad()
                    logging.debug("Optimizer gradients zeroed")

                    # Move batches and targets to the correct device 
                    batch_input_ids = batch_input_ids.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    seq_lengths = seq_lengths.to(self.device)
                    logging.debug("Batch moved to device")

                    # Logging epoch and batch info
                    logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')
                    logging.debug(f'Batch input_ids shape: {batch_input_ids.shape}')  # (batch_size, seq_len)
                    logging.debug(f'Using device: {self.device}')

                    logging.debug(f"Shape of batch_input_ids before generate_square_subsequent_mask: {batch_input_ids.shape}")
                    logging.debug(f"Shape of batch_labels: {batch_labels.shape}")

                    # Log the shapes before combining
                    # Expand src_mask to match batch size and number of heads 

                    # Combine masks without slicing (corrected)
                    batch_size, seq_length = batch_input_ids.size()
                    # Assuming batch_input_ids and pad_token_id are defined.
                    combined_mask = create_combined_mask(batch_input_ids, self.tokenizer.pad_token_id)

                    # Log the shape of the combined mask
                    logging.debug(f"Shape of combined_mask: {combined_mask.shape}")
                                            
                    targets_reshaped = batch_labels.contiguous().view(-1)  # Ensures proper reshaping
                    logging.debug(f"Targets reshaped for loss: {targets_reshaped.shape}")  
                    logging.debug(f"Shape of batch_input_ids being passed to model: {batch_input_ids.shape}")
                    logging.debug(f"Shape of targets_reshaped being passed to model: {batch_input_ids.shape}")
                    logging.debug(f"Shape of combined_mask being passed to model: {combined_mask.shape}")
        
                    if self.model is None:
                        raise ValueError("Error: Model is not initialized properly.")

                    if optimizer_q is None:
                        raise ValueError("Error: Optimizer is not initialized properly.")

                    if not hasattr(self.model, "parameters") or self.model.parameters() is None:
                        raise ValueError("Error: Model has no accessible parameters.")

                    #check model for switchable model call
                    architecture = self.architecture.get()
                    # Check the flag and run evolution once per epoch if requested:
                    if self.use_genetic_algo == "Genetic Algorithm":
                        logging.info("Applying genetic algorithm evolution step...")
                        qga = QuaternionGeneticAlgorithm(self.model, lr)
                        # Evolve using the same loss function and dataloader (or a validation subset)
                        self.model = qga.evolve(loss_fn, batch_input_ids, targets_reshaped, combined_mask, architecture)
                        #Remove optimizer steps and gradient code enable this for Quaternion NeuroEvolution of Augmenting Topologies (NEAT)
                    elif self.use_genetic_algo == "NEAT":
                        neat = QuaternionNEAT(self.model)
                        self.model = neat.evolve(F.cross_entropy, dataloader)
                    elif self.use_genetic_algo == "Firefly":
                        #Remove optimizer steps and gradient lines to enable this for Quaternion Firefly Algo
                        firefly_optimizer = QuaternionFireflyOptimizer(self.model)
                        self.model = firefly_optimizer.optimize(F.cross_entropy, dataloader)
                    else:
                    # Forward pass
                        try:
                            if architecture == "Quaternion LNS Transformer":
                                logits = self.model(batch_input_ids, mask=combined_mask)
                                logging.debug(f"Shape of outputs: {logits.shape}")

                                # Flatten logits and targets:
                                logits_real = self.model.quat_to_real(logits)  # Convert quaternions to real values (2, 128, 10000)
                                logging.debug(f"Logits after real conversion: {logits_real.shape}")  

                                logits_flat = logits_real.view(-1, logits_real.size(-1))  # (batch * seq_len, vocab_size)
                                logging.debug(f"Logits reshaped for loss: {logits_flat.shape}")  
                            else:
                                logits = self.model(batch_input_ids)
                                if torch.isnan(logits).any() or torch.isinf(logits).any():
                                        logging.error("Model outputs contain NaN or Inf values.")
                                logging.debug(f"Model output shape: {logits.shape}")

                                logits_flat = logits.reshape(-1, logits.size(-1))
                        except Exception as e:
                            raise ValueError(f"forward pass failed for {str(e)}")

                        loss = loss_fn(logits_flat, targets_reshaped)

                        logging.info(f"Loss computed: {loss.item()}")

                        # Backward pass and optimization
                        loss.backward(retain_graph=True)
                        logging.info("Loss backward computed")

                        for param_group in optimizer_std.param_groups:
                            logging.debug(f"Learning rate: {param_group['lr']}")
                        
                        ###Clifford Backpropagation, disable for use with QGA, NEAT, or Genetic Algo    
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                # Check if the parameter is quaternion-structured.
                                # (Assuming quaternion parameters have at least 2 dimensions and the last dimension equals 4.)
                                if param.data.dim() >= 2 and param.data.shape[-1] == 4:
                                    new_grad = CliffordBackprop()(param.grad, param.data)
                                    logging.debug(f"Applying CliffordBackprop on parameter {name}, grad shape: {new_grad.shape}")
                                    param.grad = new_grad
                                else:
                                    logging.debug(f"Regular gradient for non-quaternion parameter {name} with shape: {param.data.shape}")

                        # Check for NaN or Inf in gradients
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    logging.error(f"Gradient for {name} contains NaN or Inf.")
                                    continue
                                
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                logging.debug(f"Gradient for {name}: mean={param.grad.mean().item():.4f}, max={param.grad.max().item():.4f}, min={param.grad.min().item():.4f}")
                            else:
                                logging.debug(f"Gradient for {name} is None")


                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        logging.info(f"Gradient norm: {total_norm}")

                        ###Uncomment these for gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5
                        logging.info(f"Gradient norm after clipping: {total_norm}")

                        # Log gradients for debugging
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                logging.debug(f"Gradients for {name}: {param.grad}")
                            else:
                                logging.warning(f"No gradients found for {name}.")
                                                    
                        #optimizer_q.step()
                        optimizer_std.step()
                        optimizer_q.step()


                        n+=1
                        print(f"Iteration {n}, Loss: {loss.item()}, LR_std: {optimizer_std.param_groups[0]['lr']}")
                        

                                                
                        # Before optimizer step
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                logging.debug(f"Before step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")


                        # After optimizer step
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                logging.debug(f"After step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")


                        logging.info("Optimizer step update completed")
                        with torch.no_grad():
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    param.grad = param.grad.detach()

                        scheduler_std.step()
                        scheduler_q.step()
                        logging.debug("Scheduler step completed")

                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                    # Save checkpoint at specified intervals
                    save_interval = 5  # Save every 25%
                    progress_percentage = (batch_idx + 1) / len(dataloader) * 100
                    if abs(progress_percentage % save_interval) < 1e-6:  # Avoid floating-point issues
                        checkpoint_path = f"checkpoints/epoch_{epoch}_batch_{batch_idx}.pth"
                        self.save_checkpoint(self.model, optimizer_std, epoch, checkpoint_path)
                        logging.info(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}, progress: {progress_percentage:.2f}%")



                # Log epoch loss
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed. Current LR = {scheduler_std.get_last_lr()}")

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")

    def improved_collate_fn(self, batch):
        input_ids, attention_masks, labels, seq_lengths = zip(*batch)
        
        # Convert sequences to tensors if they aren't already
        input_ids = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in input_ids]
        attention_masks = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in attention_masks]
        labels = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in labels]
        
        max_len = seq_len
        
        # Pad sequences using torch operations
        def pad_sequence(sequences, max_len, pad_value):
            return torch.stack([
                torch.cat([
                    seq,
                    torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype, device=seq.device)
                ]) if len(seq) < max_len else seq[:max_len]
                for seq in sequences
            ])
        
        # Pad all sequences
        padded_input_ids = pad_sequence(input_ids, max_len, self.tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, max_len, 0)
        padded_labels = pad_sequence(labels, max_len, self.tokenizer.pad_token_id)
        
        # Convert sequence lengths to tensor
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        
        return padded_input_ids, padded_attention_masks, padded_labels, seq_lengths

    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Cannot save.")
            logging.error("Attempted to save model but tokenizer was not initialized.")
            return

        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            config = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Ensure embeddings match tokenizer
            tokenizer_vocab_size = len(self.tokenizer)

            # Save the model state dictionary
            if self.architecture.get() == "Quaternion LNS Transformer":
                model_file_name = 'lns_quaternion_transformer_model.pth'
            elif self.architecture.get() == "Quaternion LNS Relational":
                model_file_name = 'quaternion_lns_relational.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(save_directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")

    def stop_training_command(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

    def expand_transformer(self):
        # Placeholder method; not used in current implementation
        pass

    
    def load_dataset(self):
        if self.use_chunked_dataset.get():
            # Load data from chunked files
            self.tokenized_data_path = filedialog.askdirectory(
                title="Select Tokenized Data Directory"
            )
            if not self.tokenized_data_path:
                messagebox.showerror("Error", "No tokenized data directory selected.")
                return

            # Check if directory contains chunked data files
            chunk_files = [f for f in os.listdir(self.tokenized_data_path) if f.startswith('chunk_') and f.endswith('.jsonl')]
            if not chunk_files:
                messagebox.showerror("Error", "No chunked data files found in the selected directory.")
                return

            self.chunked_files = [os.path.join(self.tokenized_data_path, f) for f in chunk_files]
            messagebox.showinfo("Success", f"Loaded chunked dataset with {len(self.chunked_files)} files.")
            logging.info(f"Loaded chunked dataset with {len(self.chunked_files)} files.")
        else:
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.json') or file.endswith('.jsonl'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]


                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")

    def extract_query_target_pairs(self, data):
        query_target_pairs = []
        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                        query = messages[i]["content"].replace('\n', ' ').strip()
                        target = messages[i + 1]["content"].replace('\n', ' ').strip()
                        query_target_pairs.append((query, target))
            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i]["from"] == "user" and messages[i + 1]["from"] == "assistant":
                        query = messages[i]["value"].replace('\n', ' ').strip()
                        target = messages[i + 1]["value"].replace('\n', ' ').strip()
                        query_target_pairs.append((query, target))
            else:
                messages = conversation.get("user",[])
                messages = conversation.get("assistant",[])
                for i in range(data):
                            if messages[i] == "user" and messages[i + 1] == "assistant":
                                query = messages[i].replace('\n', ' ').strip()
                                target = messages[i + 1].replace('\n', ' ').strip()
                                query_target_pairs.append((query, target))
                            elif messages == "user" and messages == "assistant":
                                query = messages[i].replace('\n', ' ').strip()
                                target = messages[i + 1].replace('\n', ' ').strip()
                                query_target_pairs.append((query, target))     
              
        return query_target_pairs



# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()

    app = QuaternionTransformerGUI(root)
    root.mainloop()
