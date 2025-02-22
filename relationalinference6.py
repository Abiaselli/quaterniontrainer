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
import math
import psutil
import threading 
import copy
import random

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
seq_len = 128

# Global tokenizer variable for multiprocessing
tokenizer = None


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")
#debug for cuda
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
max_length = seq_len

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
                loss = 0

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
                
            if isinstance(model, QuaternionTransformerLNS):
                
                outputs = model(input_tensor, mask=combined_mask)
            else:
                outputs = model(input_tensor)
            logging.info(f"outputs shape: {outputs.shape}")
                                # Flatten logits and targets:
            outputs = model.quat_to_real(outputs)  # Convert quaternions to real values (2, 128, 10000)
            logging.info(f"Logits after real conversion: {outputs.shape}")  

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

            if next_token == tokenizer.eos_token_id:
                break

            if len(generated_tokens) == 50:
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
        architecture = model_parameters.get('architecture', "Quaternion LNS Transformer" or 'Quaternion LNS Relational')

        if architecture == 'Quaternion LNS Transformer':
            model = QuaternionTransformerLNS(
                vocab_size=model_parameters['vocab_size'],
                embedding_dim=model_parameters['embed_size'],
                hidden_dim=model_parameters['hidden_size'],
                num_layers=model_parameters['num_layers'],
                seq_length=seq_len
            )
            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'lns_quaternion_transformer_model.pth')
        elif architecture =='Quaternion LNS Relational':
            model = QuaternionTransformerRelationalLNS(
                vocab_size=model_parameters['vocab_size'],
                embedding_dim=model_parameters['embed_size'],
                hidden_dim=model_parameters['hidden_size'],
                num_layers=model_parameters['num_layers'],
                seq_length=seq_len
            )

            # Adjust model path if needed
            model_path = os.path.join(os.path.dirname(model_path), 'quaternion_lns_relational.pth')
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
