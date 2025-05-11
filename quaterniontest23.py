import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random
import os
import pandas as pd
import copy
import gc
import torch.utils.checkpoint as cp
from torch.autograd import Function
from typing import Any, Callable, Optional, Union


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


seq_len = 128

########################################
# Tokenizer
########################################

class RawPairDataset(torch.utils.data.Dataset):
    def __init__(self, query_target_pairs):
            self.pairs = query_target_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        if isinstance(sample, dict):
            return sample['query'], sample['target']
        return sample  # assume it's already a tuple

# Global tokenizer reference
global_tokenizer = None
seq_len_for_collate = seq_len

def init_collate_globals(tokenizer, seq_len):
    global global_tokenizer, seq_len_for_collate
    global_tokenizer = tokenizer
    seq_len_for_collate = seq_len



class TokenizerWrapper:
    def __init__(self, tokenizer, seq_len=seq_len, add_bos=True, add_eos=True, pad_to_max=True, shift_decoder=False, device="cuda"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_to_max = pad_to_max
        self.shift_decoder = shift_decoder
        self.device = device

        self.bos_token = tokenizer.bos_token or "<BOS>"
        self.eos_token = tokenizer.eos_token or "<EOS>"
        self.pad_token_id = tokenizer.pad_token_id or 0

    def format(self, text):
        if isinstance(text, list):
            return [self.format(t) for t in text]
        return f"{self.bos_token} {text} {self.eos_token}" if self.add_bos and self.add_eos else text

    def encode(self, text_batch, truncate=True):
        if isinstance(text_batch[0], str):
            text_batch = self.format(text_batch)

        encoded = [self.tokenizer.encode(t, add_special_tokens=False) for t in text_batch]
        result = []
        for tokens in encoded:
            if truncate and len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len - 1] + [self.tokenizer.eos_token_id]
            result.append(tokens)
        return result if not self.pad_to_max else torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, device=self.device) for seq in result],
            batch_first=True,
            padding_value=self.pad_token_id
        )

    def encode_shifted_pair(self, text_batch):
        """Returns (decoder_input_ids, labels), both padded"""
        full = self.encode(text_batch)  # [B, T]
        decoder_input = full[:, :-1]
        labels = full[:, 1:]
        return decoder_input, labels



class SemanticQuaternionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.scalar = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.vector_i = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.vector_j = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.vector_k = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        r = self.scalar(input_ids)
        i = self.vector_i(input_ids)
        j = self.vector_j(input_ids)
        k = self.vector_k(input_ids)
        return torch.stack([r, i, j, k], dim=-1)

    def initialize_from_semantic(self, semantic_json_path, tokenizer):
        with open(semantic_json_path, "r") as f:
            sem = json.load(f)

        for token, data in sem.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id >= self.scalar.num_embeddings:
                continue
            vec = torch.tensor(data["vector"], dtype=torch.float32)
            norm = torch.norm(vec)
            if norm == 0:
                continue
            vec = vec / norm
            self.scalar.weight.data[token_id] = vec[0]
            self.vector_i.weight.data[token_id] = vec[1]
            self.vector_j.weight.data[token_id] = vec[2]
            self.vector_k.weight.data[token_id] = vec[3]

def quaternion_fft_compress(q_vectors):
    #print(f"fft_input shape: {q_vectors.shape}")
    fft_input = torch.fft.fft(q_vectors.float(), dim=-1)
    #print(f"fft_input shape2: {fft_input.shape}")
    return fft_input.real

def quantum_collapse_logits(logits, temperature=1.0):
    probs = nn.functional.softmax(logits / temperature, dim=-1).squeeze(0)
    print(f"probs shape: {probs.shape}")
    probs = torch.nan_to_num(probs, nan=0.001, posinf=10.0, neginf=-10.0)
    collapsed_index = torch.multinomial(probs, num_samples=1)
    print(f"collapsed_index shape: {collapsed_index.shape}")
    return collapsed_index.squeeze(-1), probs




########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################

from transformers import PreTrainedTokenizerFast

# 🔹 Change this to the actual path where your BPE tokenizer files are stored
tokenizer_path = r"C:\Users\abias\.cursor-tutor\vccdoe\mhlamodel\mhlatest-main"  

# 🔹 Load a BPE tokenizer from local files
base_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

print(f"✅ Loaded custom BPE tokenizer from: {tokenizer_path}")
print(f"📏 Vocabulary size: {base_tokenizer.vocab_size}")

# Wrap it with the hierarchical tokenizer
tokenizer = base_tokenizer


########################################
# 2. Data Extraction
########################################

def extract_data(json_data):
    """Extracts training data from JSON file and tokenizes it."""
    input_ids_list = []
    target_ids_list = []

    for item in json_data:
        conversations = item.get("conversations", [])

        if not isinstance(conversations, list) or len(conversations) < 2:
            print(f"⚠️ Skipping entry with no valid conversation: {item}")
            continue

        for i in range(len(conversations) - 1):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            # Ensure we only process valid user-assistant exchanges
            if user_turn.get("from") in ["user", "human"] and assistant_turn.get("from") in ["assistant", "gpt"]:
                query = user_turn.get("value", "").strip()
                target = assistant_turn.get("value", "").strip()

                # 🔹 Ensure valid text exists before tokenizing
                if not query or not target:
                    print(f"⚠️ Skipping empty user/assistant exchange: {user_turn} -> {assistant_turn}")
                    continue  

                input_ids = tokenizer.tokenize(query)
                target_ids = tokenizer.tokenize(target)

                # 🔹 Ensure tokenized output isn't empty
                if not input_ids or not target_ids:
                    print(f"⚠️ Skipping invalid tokenized entry: {query} -> {input_ids}")
                    continue

                input_ids_list.append(input_ids)
                target_ids_list.append(target_ids)
    

    return list(zip(input_ids_list, target_ids_list))  # Ensure format is (input, target)

def load_dataset(dataset_path):

            dataset_files = os.listdir(dataset_path)
            query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(dataset_path, file)
                if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                            else:
                                data = json.load(f)
                                query_target_pairs.extend(extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]

                elif file.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                else:
                    print("errpr")
            if not query_target_pairs:
                print("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            text_data = []
            for query, target in query_target_pairs:
                text_data.append(f"User: {query}\nAssistant: {target}")

            logging.info(f"Loaded dataset with {len(query_target_pairs)} query/target pairs.")
            return query_target_pairs


def extract_query_target_pairs( data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

        return query_target_pairs

def tokenize_data(query_target_pairs):

        # Select training mode
        input_ids_list = []  # Initialize for unchunked dataset
        labels_list = []  # Initialize for unchunked dataset

        for query, target in query_target_pairs:
                        input_ids, labels = _generate_training_pairs(query, target)

                        if input_ids and labels:
                            input_ids_list.append(input_ids)  # Store for training
                            labels_list.append(labels)  # Store for training
                            #print (input_ids)
                            #print(labels)
        return input_ids_list, labels_list


def _generate_training_pairs(query, target):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)

        input_ids = [tokenizer.bos_token_id] + query_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

        return input_ids, labels

def prepare_batch(input_ids, labels, seq_len):
                pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths

                #input_ids = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in input_ids]
                #labels = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in labels]

                # ✅ Compute correct padding lengths
                #input_ids = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in input_ids]
                #labels = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in labels]
                
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in input_ids
                ]
                logging.info("input ids torched to tensor")
                print(input_ids)
                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels
                ]
                logging.info("labels torched to tensor")
                print(labels)
                # Stack tensors
                input_ids = torch.stack(input_ids).to(device)
                labels = torch.stack(labels).to(device)
                data = torch.utils.data.TensorDataset(input_ids, labels)
                return data


########################################
# 3. Dataset and Collate Function
########################################

def collate_fn(batch, max_length, tokenizer):
    src_batch, tgt_batch = zip(*batch)
    pad_token_id = tokenizer.pad_token_id or 0  # Ensure pad token is valid
    src_batch, seq_lengths = zip(*[
                    (
                        torch.tensor(seq + (pad_token_id * (max_length - len(seq))), dtype=torch.int64, device=device)[:max_length],
                        min(len(seq), max_length)
                    )
                    for seq in src_batch
                ])
    tgt_batch = [
                    torch.tensor(seq + (pad_token_id * (max_length - len(seq))), dtype=torch.int64, device=device)[:max_length]
                    for seq in tgt_batch
                ]
    # ✅ Compute correct padding lengths

    return torch.stack(src_batch), torch.stack(tgt_batch),seq_lengths


def collate_fn1(batch):
    global global_tokenizer, seq_len_for_collate

    BOS = global_tokenizer.bos_token or "<BOS>"
    EOS = global_tokenizer.eos_token or "<EOS>"
    PAD_ID = global_tokenizer.pad_token_id or 0  # Fallback if pad_token not set

    def encode_and_fix(texts):
        fixed_seqs = []
        for t in texts:
            tokens = global_tokenizer.encode(BOS + " " + t + " " + EOS, add_special_tokens=False)
            if len(tokens) > seq_len_for_collate:
                tokens = tokens[:seq_len_for_collate - 1] + [global_tokenizer.eos_token_id]  # truncate and force EOS
            padded = tokens + [PAD_ID] * (seq_len_for_collate - len(tokens))
            fixed_seqs.append(padded)
        return torch.tensor(fixed_seqs, dtype=torch.long)

    if isinstance(batch[0], str):
        input_ids = encode_and_fix(batch)
        return input_ids, input_ids

    elif isinstance(batch[0], tuple):
        queries, targets = zip(*batch)
        input_ids = encode_and_fix(queries)
        target_ids = encode_and_fix(targets)
        return input_ids, target_ids


########################################
#Base Model
########################################

class GeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=10):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = None
        best_loss = float('inf')
        n=0
        loss = 0
        if architecture == "Reasoning Model LNS":

            output = self.model(inputs, decoder_input)

        else:
            output = self.model(inputs, target_labels)          
                
        output = output.reshape(-1, output.shape[-1])
        logging.debug(f"output reshaped Shape: {output.shape}")
        target_labels_reshaped = target_labels.reshape(-1)
        logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
        loss = loss_fn(output, target_labels_reshaped)
        best_loss = loss
        print(f"Original model iteration {n}, Loss: {loss.item()}")
        best_model = self.model
        for model in self.population:
            loss = 0
            if architecture == "Reasoning Model LNS":

                output = model(inputs, decoder_input)

            else:
                output = model(inputs, target_labels)          
                
            output = output.reshape(-1, output.shape[-1])
            logging.debug(f"output reshaped Shape: {output.shape}")
            target_labels_reshaped = target_labels.reshape(-1)
            logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
            loss = loss_fn(output, target_labels_reshaped)
            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss.item()}")
                    best_model = model
            
            else:
                loss = 0

                if architecture == "Reasoning Model LNS":

                    output = model(inputs, decoder_input)

                else:
                    output = model(inputs, target_labels)
                # Flatten logits and targets:
                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                target_labels_reshaped = target_labels.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
                loss = loss_fn(output, target_labels_reshaped)
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss.item()}")
                        best_model = model
        return best_model

    def evolve(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        self.model = self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
        self.population = [copy.deepcopy(self.model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]

        return self.dropout(x + pe)

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, sinusoidal_emb):
    return (x * sinusoidal_emb.cos()) + (rotate_half(x) * sinusoidal_emb.sin())

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)[None, :, :]  # [1, seq_len, dim]
        return apply_rotary(x, emb)


# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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
        r = self.scalar(x)
        i = self.vector_i(x)
        j = self.vector_j(x)
        k = self.vector_k(x)

        # Apply RoPE-style scaling
        i = i * self.scale_factor
        j = j * self.scale_factor
        k = k * self.scale_factor

        # Normalize quaternion embeddings to unit norm
        norm = torch.sqrt(r ** 2 + i ** 2 + j ** 2 + k ** 2 + 1e-6)
        r, i, j, k = r / norm, i / norm, j / norm, k / norm

    
        # Combine into quaternion format: a + bi + cj + dk
        return torch.stack([r, i, j, k], dim=-1)  # Shape: (batch_size, seq_length, embedding_dim, 4)


class QuaternionRotationalEncoding(nn.Module):
    def __init__(self, seq_length, embedding_dim):
        """
        Quaternion Rotational Encoding with RoPE-style enhancements.

        Args:
            seq_length (int): Maximum sequence length.
            embedding_dim (int): Number of quaternion embedding features.
        """
        super(QuaternionRotationalEncoding, self).__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        # Generate position indices
        positions = torch.arange(seq_length).unsqueeze(-1).float()

        # RoPE-style scaling decay per dimension
        #scale_factor = 1 / (10000 ** (torch.arange(embedding_dim) / embedding_dim))
        #self.theta = nn.Parameter(positions * (torch.pi / seq_length) * scale_factor)  #trainable parameter

        scale_factor = 1 / (10000 ** (torch.arange(embedding_dim, dtype=torch.float32) / embedding_dim))
        # This will result in a tensor of shape (seq_length, embedding_dim)
        theta = positions * (torch.pi / seq_length) * scale_factor
        # This means theta is not trainable and moves with the module.
        self.register_buffer("theta", theta)

    def forward(self, x):
        """
        Apply quaternion rotational encoding to input embeddings.

        Args:
            x (torch.Tensor): Input quaternion tensor (batch, seq_len, embedding_dim, 4)

        Returns:
            torch.Tensor: Rotated quaternion embeddings
        """
        # Compute rotation angles
        cos_theta = torch.cos(self.theta).unsqueeze(-1)  # (seq_len, embedding_dim, 1)
        sin_theta = torch.sin(self.theta).unsqueeze(-1)

        # Pairwise quaternion rotations:
        # - (i, j) undergo independent rotation
        # - (k, scalar) undergo a different rotation
        r_ij = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
        r_k = torch.cat([cos_theta, sin_theta, -sin_theta, cos_theta], dim=-1)
        ##r_k = torch.cat([cos_theta, sin_theta, sin_theta, sin_theta], dim=-1) ##asymmetric rotation

        # Apply quaternion multiplication
        rotated_ij = self.quaternion_multiply(r_ij, x)
        rotated_k = self.quaternion_multiply(r_k, rotated_ij)

        return rotated_k

    def quaternion_multiply(self, q1, q2):
        """
        Performs quaternion multiplication.

        Args:
            q1, q2: Quaternion tensors of shape (..., 4)

        Returns:
            Quaternion tensor of shape (..., 4)
        """
        a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        scalar = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        return torch.stack([scalar, i, j, k], dim=-1)


class QuaternionMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = QuaternionLinear(embed_dim, embed_dim)
        self.k_linear = QuaternionLinear(embed_dim, embed_dim)
        self.v_linear = QuaternionLinear(embed_dim, embed_dim)
        self.out_proj = QuaternionLinear(embed_dim, embed_dim)

    def split_heads(self, x):
        B, T, D, Q = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim, Q)
        return x.permute(0, 2, 1, 3, 4)  # (B, heads, T, head_dim, 4)

    def combine_heads(self, x):
        B, H, T, D, Q = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, H * D, Q)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.q_linear(q))
        k = self.split_heads(self.k_linear(k))
        v = self.split_heads(self.v_linear(v))

        attn_scores = torch.einsum("bhqfd,bhkfd->bhqk", q, k) 
        #attn_scores = torch.einsum('bhqfd,bhkfd->bqk', q, k)/ math.sqrt(self.head_dim * 4)
        if mask is not None:
            #print(mask.shape)
            #print(attn_scores.shape)
            mask = mask.unsqueeze(1).unsqueeze(2)
            #print(mask.shape)
            mask =mask.expand(attn_scores.size())
            #print(mask.shape)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = attn_scores / math.sqrt((self.head_dim * 4+1e-6)+1e-6)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhqk,bhkfd->bhqfd", attn_weights, v)
        #attn_output = torch.einsum('bqk,bhkfd->bhqfd', attn_weights, v)

        return self.out_proj(self.combine_heads(attn_output))



class QuaternionFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(QuaternionFeedForward, self).__init__()
        # Replace nn.Linear with QuaternionLinear. Here, in_features=embedding_dim and out_features=hidden_dim.
        self.fc1 = QuaternionLinear(embedding_dim, hidden_dim)
        self.activation = ModReLU(hidden_dim)
        self.fc2 = QuaternionLinear(hidden_dim, embedding_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, embedding_dim, 4)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class QuaternionEncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attn = QuaternionMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = QuaternionLayerNorm(embed_dim)
        self.ff = QuaternionFeedForward(embed_dim, hidden_dim)
        self.norm2 = QuaternionLayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class QuaternionDecoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.self_attn = QuaternionMultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = QuaternionMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = QuaternionLayerNorm(embed_dim)
        self.norm2 = QuaternionLayerNorm(embed_dim)
        self.norm3 = QuaternionLayerNorm(embed_dim)
        self.ff = QuaternionFeedForward(embed_dim, hidden_dim)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, memory, memory, memory_mask))
        x = self.norm3(x + self.ff(x))
        return x

class QuaternionRotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, base=10000):
        super().__init__()
        self.embed_dim = embed_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, embed_dim, 2).float() / embed_dim+1e-6))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim, 4)
        seq_len = x.shape[1]
        freqs = torch.einsum("i,j->ij", torch.arange(seq_len, device=x.device).float(), self.inv_freq)
        angles = torch.cat((freqs.sin(), freqs.cos()), dim=-1).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, embed_dim, 1)
        return x * angles  # Applies element-wise RoPE approximation


class QuaternionLayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(QuaternionLayerNorm, self).__init__()
        # Learnable scale and shift per quaternion feature
        self.gamma = nn.Parameter(torch.ones(embedding_dim, 4))
        self.beta = nn.Parameter(torch.zeros(embedding_dim, 4))
        self.eps = eps

    def forward(self, x):
        # x shape: (batch, seq_len, embedding_dim, 4)
        norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / (norm+1e-6)
        return self.gamma.unsqueeze(0).unsqueeze(0) * x_norm + self.beta.unsqueeze(0).unsqueeze(0)

class QuaternionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, tokenizer, device):
        super().__init__()
        # --- PATCHED: Semantic Quaternion Embedding ---
        self.embedding = SemanticQuaternionEmbedding(vocab_size, embed_dim)
        self.pos_encoder = QuaternionRotaryPositionalEncoding(embed_dim)
        self.encoder_layers = nn.ModuleList([QuaternionEncoderLayer(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([QuaternionDecoderLayer(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim*4, vocab_size)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer)
        self.device = device

    def generate(self, prompt, max_new_tokens=50, seq_len=32, repetition_penalty=1.2, top_p=0.5, return_text=True):
        self.eval()

        if isinstance(prompt, str):
            input_ids = self.tokenizer_wrapper.encode([prompt], truncate=False)[0]
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            input_ids = self.tokenizer_wrapper.encode(prompt, truncate=False)[0]
        else:
            input_ids = prompt  # already tokenized

        input_ids = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
        generated = input_ids
        generated_2 = input_ids[:]

        for _ in range(max_new_tokens):
            # Get current window
            print(f"generated shape: {len(generated)}")
            print(f"generated_2 shape: {len(generated_2)}")
            print(f"generated_2: {generated_2}")
            window = generated_2[-seq_len:] if len(generated_2) > seq_len else generated_2
            window_tensor = torch.tensor([window], dtype=torch.long, device=self.device)

            # Use as both src and initial tgt (autoprompted decoder)
            src = window_tensor
            tgt = torch.tensor([window[-1:]], dtype=torch.long, device=self.device)
            #tgt = window_tensor
            
            with torch.no_grad():
                #src=src.view(1, -1)
                #tgt=tgt.view(1, -1)
                print(f"src shape: {src.shape}, tgt shape: {tgt.shape}")
                logits, _ = self.forward(src, tgt)
                print(f"logits shape: {logits.shape}")
            logits = logits[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt[0].tolist()):
                logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')
            filtered_logits = torch.nan_to_num(filtered_logits, nan=0.001, posinf=10.0, neginf=-10.0)

            print(filtered_logits)
            filtered_softmax = F.softmax(filtered_logits, dim=-1)
            print(f"filtered softmax: {filtered_softmax}")
            next_token_id = torch.multinomial(filtered_softmax, num_samples=1)


            # With this:
            next_token, token_probs = quantum_collapse_logits(logits)

            generated.append(next_token)

            # Choose top-1 (greedy) or sample (optional)
            generated_2.append(next_token_id.item())

            # Optional early stop
            if next_token == self.tokenizer_wrapper.eos_token:
                break

            # Optional early stop
            if next_token == self.tokenizer_wrapper.eos_token:
                break
            tgt = torch.cat([tgt, next_token_id], dim=1)
            src = torch.cat([src, next_token_id], dim=1)
        if return_text:
            generated = torch.tensor(generated, dtype=torch.long, device=self.device) 
            return self.tokenizer_wrapper.tokenizer.decode(generated, skip_special_tokens=True), self.tokenizer_wrapper.tokenizer.decode(generated_2, skip_special_tokens=True)
        else:
            return generated
    def generate_mask(self, src, tgt):
        # Padding mask: (batch_size, seq_len) with True for padding tokens
        src_pad_mask = (src == 0)  # Shape: [batch, src_len]
        tgt_pad_mask = (tgt == 0)  # Shape: [batch, tgt_len]

        # Causal mask for decoder (no peeking into the future)
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)  # Shape: [tgt_len, tgt_len]

        return src_pad_mask, tgt_pad_mask, causal_mask

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


    def forward(self, src, tgt=None, mode="eval"):
        if isinstance(src[0], str):
            src = self.tokenizer_wrapper.encode(src).to(self.device)
        if tgt is not None and isinstance(tgt[0], str):
            tgt = self.tokenizer_wrapper.encode(tgt).to(self.device)
        elif tgt is not None and mode == 'train':
            tgt = tgt
            #tgt_ids = tgt_ids[:, 1:]
        #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt is not None and tgt.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        src_pad_mask, tgt_pad_mask, _ = self.generate_mask(src, tgt if tgt is not None else src)
        src_embed = self.pos_encoder(self.embedding(src))
        for layer in self.encoder_layers:
            src_embed = layer(src_embed, src_pad_mask)

        if tgt is None:
            tgt = src[:, :1]
        tgt_embed = self.pos_encoder(self.embedding(tgt))

        for layer in self.decoder_layers:
            tgt_embed = layer(tgt_embed, src_embed, tgt_pad_mask, src_pad_mask)
 
        #out = self.quat_to_real(tgt_embed)
        #out = tgt_embed.view(tgt_embed.size(0), tgt_embed.size(1), -1, 4).permute(0, 2, 1, 3).contiguous().view(tgt_embed.size(0), -1, tgt_embed.size(1) * 4)
        #print(f"tgt_embed shape: {tgt_embed.shape}")
        compressed_out = torch.stack([quaternion_fft_compress(tgt_embed[b]) for b in range(tgt_embed.size(0))])  # shape: [batch, dim, 4]
        #print(f"compressed_out shape: {compressed_out.shape}")
                # Optionally flatten or project before output
        out = compressed_out.view(compressed_out.size(0),compressed_out.size(1), -1)  # shape: [batch, dim*4]
        
        #print(f"out shape: {out.shape}")
        with torch.no_grad():
            if tgt is not None:
                predicted_targets = self.embedding(tgt)
                # Apply FFT composition across sequence
                compressed = torch.stack([quaternion_fft_compress(predicted_targets[b]) for b in range(predicted_targets.size(0))])  # shape: [batch, dim, 4]
                #print(f"compressed shape: {compressed.shape}")
                # Optionally flatten or project before output
                flat = compressed.view(compressed.size(0),compressed.size(1), -1)  # shape: [batch, dim*4]
                #print(f"flat shape: {flat.shape}")
                predicted_targets = self.fc_out(flat)  # logits over vocab

        return self.fc_out(out), predicted_targets

########################################
# 5. Training Loop
########################################


class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Define separate weights for the four quaternion components.
        self.r_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.i_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.j_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.k_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # Bias is a quaternion for each output unit.
            self.bias = nn.Parameter(torch.Tensor(out_features, 4))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Use an initialization adapted for quaternion weights.
        nn.init.xavier_uniform_(self.r_weight)
        nn.init.xavier_uniform_(self.i_weight)
        nn.init.xavier_uniform_(self.j_weight)
        nn.init.xavier_uniform_(self.k_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: (batch, seq_len, in_features, 4)
        # Separate input components.
        r, i, j, k = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        # Apply the Hamilton product rules:
        r_out = F.linear(r, self.r_weight) - F.linear(i, self.i_weight) - F.linear(j, self.j_weight) - F.linear(k, self.k_weight)
        i_out = F.linear(r, self.i_weight) + F.linear(i, self.r_weight) + F.linear(j, self.k_weight) - F.linear(k, self.j_weight)
        j_out = F.linear(r, self.j_weight) - F.linear(i, self.k_weight) + F.linear(j, self.r_weight) + F.linear(k, self.i_weight)
        k_out = F.linear(r, self.k_weight) + F.linear(i, self.j_weight) - F.linear(j, self.i_weight) + F.linear(k, self.r_weight)
        out = torch.stack([r_out, i_out, j_out, k_out], dim=-1)
        if self.bias is not None:
            # Broadcast bias along batch and seq dimensions.
            out = out + self.bias.unsqueeze(0).unsqueeze(0)
        return out

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
                q_star = (q0 - q1 * 1j - q2 * 1j - q3 * 1j) / (4  +1e-6)
                
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

def get_quaternion_parameters(model):
    quaternion_params = []
    for name, module in model.named_modules():
        if isinstance(module, QuaternionLinear):
            quaternion_params.extend(list(module.parameters()))
    return quaternion_params


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
    loss = -torch.mean(torch.log(torch.exp(similarity+1e-6) / (1e-6+torch.sum(torch.exp(similarity+1e-6), dim=-1, keepdim=True))))
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

def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output


def train_model(batch_size, model, dataset, optimizer_std, optimizer_q, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        if not batch:
            continue
        def build_training_tokens_batch(batch):
            bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
            sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
            eos = sep

            src = []
            targets= []
            for query, target in batch:
                query_ids = tokenizer.encode(query, add_special_tokens=False)
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                query_ids = [bos] + query_ids + [eos]
                target_ids = [bos] + target_ids + [eos]
                src.append(torch.tensor(query_ids, dtype=torch.long))
                targets.append(torch.tensor(target_ids, dtype=torch.long))
            return pad_sequence(src, batch_first=True, padding_value=tokenizer.pad_token_id), pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
        src, target = build_training_tokens_batch(batch)  # [B, max_len]

        try:
            src = src.to(device)
            target = target.to(device)
            optimizer_std.zero_grad()
            optimizer_q.zero_grad()

            # 🔹 Get predictions & rule-modified embeddings
            output, target_labels = model(src, target)
            #output = model(src, target_labels)
            # 🔹 Ensure `output` and `target_labels` have the same sequence length
            seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
            output = output[:, :seq_len, :]  # Truncate logits if too long
            predicted_targets = target_labels[:, :seq_len, :]  # Truncate targets if too long

            # 🔹 Flatten for cross_entropy()
            #loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1, target_labels.shape[-1]))
            logits_filtered = output.reshape(-1, output.shape[2])                    # [B*T, V]
                #targets_filtered = tgt_ids.reshape(-1)                                    # [B*T]
                #active_mask_flat = (targets_flat != pad_token_id)                     # [B*T]

                #logits_filtered = logits_filtered[active_mask_flat]                      # [N, V]
                #targets_filtered = targets_filtered[active_mask_flat]                    # [N]

            if logits_filtered.size(0) == 0:
                    continue  # skip if nothing to train on this step
            logits_filtered = torch.nan_to_num(logits_filtered, nan=0.01, posinf=10.0, neginf=-10.0)
                #print(f"logits_filtered shape: {logits_filtered.shape}")
                #print(targets_filtered.shape)
            predicted_targets = predicted_targets.reshape(predicted_targets.shape[0]*predicted_targets.shape[1],predicted_targets.shape[2])  # [B*T, V]
                #print(f"Predicted targets reshaped: {predicted_targets.shape}")
                #step_loss = loss_fn(logits_filtered, targets_filtered)
            loss = quaternion_angular_loss(logits_filtered, predicted_targets)  # Use quaternion loss function
                #step_loss = quaternion_mse_loss(logits_filtered, predicted_targets)  # Use quaternion loss function
            if not torch.isfinite(loss):
                    print("🚨 Warning: NaN or Inf detected in loss. Skipping update.")
                    optimizer_std.zero_grad()
                    optimizer_q.zero_grad()
                    continue
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        ###Clifford Backpropagation, disable for use with QGA, NEAT, or Genetic Algo    
            for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Check if the parameter is quaternion-structured.
                         # (Assuming quaternion parameters have at least 2 dimensions and the last dimension equals 4.)
                        if param.data.dim() >= 2 and param.data.shape[-1] == 4:
                            new_grad = CliffordBackprop()(param.grad, param.data)
                            #new_grad = param.grad.clone()   
                            logging.debug(f"Applying CliffordBackprop on parameter {name}, grad shape: {new_grad.shape}")
                            param.grad = torch.nan_to_num(new_grad, nan=0.01, posinf=10.0, neginf=-10.0)
                        else:
                            param.grad = torch.nan_to_num(param.grad, nan=-0.01, posinf=10.0, neginf=-10.0)
                            logging.debug(f"Regular gradient for non-quaternion parameter {name} with shape: {param.data.shape}")

            src = src.detach()
            target_labels = target_labels.detach()
            
                # check for exploding grads
            for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"🚨 Non-finite grad in {name}, skipping step")
                        optimizer_std.zero_grad()
                        optimizer_q.zero_grad()
                        continue

            optimizer_std.step()
            optimizer_q.step()
            optimizer_std.zero_grad()
            optimizer_q.zero_grad()
            
            n+=1
            print(f"Batch: {i} Iteration {n}, Loss: {loss.item()}")
            if torch.isnan(loss) or torch.isinf(loss):
                print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
                return
            gc.collect()
            torch.cuda.empty_cache()

            total_loss += loss.item()
            if n % 50 ==0:
                    prompt = "Find the slope of the line $3x+5y=20$."
                    #generated_text = generate_autoregressive(model, prompt, tokenizer, seq_length, device)

                    #generated_text = generate_2(model,prompt, base_tokenizer, seq_length, device)
                    generated_text, generated_text_2 = model.generate(prompt, max_new_tokens=100, seq_len=32)


                    print("Generated text:")
                    print(generated_text)
                    print("Generated text 2:")
                    print(generated_text_2)
        except Exception as e:
                print(f"🚨 Error during forward pass: {e}")
                optimizer_std.zero_grad()
                optimizer_q.zero_grad()
                continue



    return total_loss / len(dataset)


def build_training_tokens(query, target, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
    eos = sep

    query_ids = tokenizer.encode(query, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    # Construct full sequence: [BOS] query [SEP] target [EOS]
    full_seq = [bos] + query_ids + [sep] + target_ids + [eos]

    return torch.tensor(full_seq, dtype=torch.long)


def build_training_tokens_batch(batch, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
    eos = sep

    full_seqs = []
    for query, target in batch:
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        full_seq = [bos] + query_ids + [sep] + target_ids + [eos]
        full_seqs.append(torch.tensor(full_seq, dtype=torch.long))

    padded = pad_sequence(full_seqs, batch_first=True, padding_value=tokenizer.pad_token_id or 0)
    return padded  # [batch, padded_len]



def train_decoder_autoregressive(model, dataset, tokenizer, optimizer_std, optimizer_q, loss_fn, batch_size, seq_len, device):
    model.train()
    total_loss = 0
    pad_token_id = tokenizer.pad_token_id or 0

    def build_training_tokens_batch(batch):
        bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
        eos = sep

        full_seqs = []
        for query, target in batch:
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            full_seq = [bos] + query_ids + [sep] + target_ids + [eos]
            full_seqs.append(torch.tensor(full_seq, dtype=torch.long))
        return pad_sequence(full_seqs, batch_first=True, padding_value=pad_token_id)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        if not batch:
            continue

        full_tokens = build_training_tokens_batch(batch).to(device)  # [B, max_len]
        batch_size, max_len = full_tokens.shape

        optimizer_std.zero_grad()
        optimizer_q.zero_grad()
        batch_loss = 0
        step_count = 0

        for t in range(2, max_len):
            start = max(0, t - seq_len)
            src = full_tokens[:, start:t]                # decoder input
            tgt_ids = full_tokens[:, start + 1:t + 1]    # target input

            # Clip if lengths don’t match due to short edges
            min_len = min(src.size(1), tgt_ids.size(1))
            src = src[:, -min_len:]
            tgt_ids = tgt_ids[:, -min_len:]

            if src.size(1) == 0 or tgt_ids.size(1) == 0:
                continue

            active_mask = (tgt_ids[:, -1] != pad_token_id)
            if active_mask.sum().item() == 0:
                continue



            try:
                logits, predicted_targets= model(src, tgt_ids)

                #print(f"Logits shape: {logits.shape}")
                #print(f"Predicted targets shape: {predicted_targets.shape}")
                # Reshape to [batch * seq_len, vocab] and filter by mask
                logits_filtered = logits.reshape(-1, logits.shape[2])                    # [B*T, V]
                #targets_filtered = tgt_ids.reshape(-1)                                    # [B*T]
                #active_mask_flat = (targets_flat != pad_token_id)                     # [B*T]

                #logits_filtered = logits_filtered[active_mask_flat]                      # [N, V]
                #targets_filtered = targets_filtered[active_mask_flat]                    # [N]

                if logits_filtered.size(0) == 0:
                    continue  # skip if nothing to train on this step
                logits_filtered = torch.nan_to_num(logits_filtered, nan=0.00001, posinf=10.0, neginf=-10.0)
                #print(f"logits_filtered shape: {logits_filtered.shape}")
                #print(targets_filtered.shape)
                predicted_targets = predicted_targets.reshape(predicted_targets.shape[0]*predicted_targets.shape[1],predicted_targets.shape[2])  # [B*T, V]
                #print(f"Predicted targets reshaped: {predicted_targets.shape}")
                #step_loss = loss_fn(logits_filtered, targets_filtered)
                #step_loss = F.smooth_l1_loss(logits_filtered, predicted_targets)
                #step_loss = quaternion_cross_entropy(logits_filtered, predicted_targets)  # Use quaternion loss function
                step_loss = quaternion_angular_loss(logits_filtered, predicted_targets)  # Use quaternion loss function
                #step_loss += quaternion_mse_loss(logits_filtered, predicted_targets)  # Use quaternion loss function
                if not torch.isfinite(step_loss):
                    print("🚨 Warning: NaN or Inf detected in loss. Skipping update.")
                    optimizer_std.zero_grad()
                    optimizer_q.zero_grad()
                    continue
                step_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        ###Clifford Backpropagation, disable for use with QGA, NEAT, or Genetic Algo    
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Check if the parameter is quaternion-structured.
                         # (Assuming quaternion parameters have at least 2 dimensions and the last dimension equals 4.)
                        if param.data.dim() >= 2 and param.data.shape[-1] == 4:
                            new_grad = CliffordBackprop()(param.grad, param.data)
                            #new_grad = param.grad.clone()   
                            logging.debug(f"Applying CliffordBackprop on parameter {name}, grad shape: {new_grad.shape}")
                            param.grad = torch.nan_to_num(new_grad, nan=0.01, posinf=10.0, neginf=-10.0)
                        else:
                            param.grad = torch.nan_to_num(param.grad, nan=-0.01, posinf=10.0, neginf=-10.0)
                            logging.debug(f"Regular gradient for non-quaternion parameter {name} with shape: {param.data.shape}")

                src = src.detach()
                tgt_ids = tgt_ids.detach()
                print(f"Batch: {i} Iteration {t}, Loss: {step_loss.item()}")
                # check for exploding grads
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"🚨 Non-finite grad in {name}, skipping step")
                        optimizer_std.zero_grad()
                        optimizer_q.zero_grad()
                        continue
                step_count += 1
                optimizer_std.step()
                optimizer_q.step()
                optimizer_std.zero_grad()
                optimizer_q.zero_grad()
                batch_loss += step_loss.item()

                if step_count % 100 ==0:
                    avg_loss = batch_loss / step_count
                    total_loss += avg_loss
                    print(f"📦 Batch {i // batch_size + 1}: Avg loss {avg_loss:.4f} over {step_count} steps")
                    prompt = "Find the slope of the line $3x+5y=20$."
                    #generated_text = generate_autoregressive(model, prompt, tokenizer, seq_length, device)

                    #generated_text = generate_2(model,prompt, base_tokenizer, seq_length, device)
                    generated_text, generated_text_2 = model.generate(prompt, max_new_tokens=100, seq_len=32)


                    print("Generated text:")
                    print(generated_text)
                    print("Generated text 2:")
                    print(generated_text_2)
            except Exception as e:
                print(f"🚨 Error during forward pass: {e}")
                optimizer_std.zero_grad()
                optimizer_q.zero_grad()
                continue


            gc.collect()
            torch.cuda.empty_cache()

            #print(f"  💥 Loss: {step_loss.item():.4f}")
            #print(f"  🧠 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")



    return total_loss / (len(dataset) // batch_size + 1)


########################################
#6. inference
########################################

def generate_autoregressive(model, prompt, tokenizer, max_tokens=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        input_ids = model.tokenizer_wrapper.encode([prompt], truncate=True)
        src_tokens = input_ids[0]
        if isinstance(src_tokens, torch.Tensor):
            src_tokens = src_tokens.tolist()
        src_tokens = src_tokens[:model.tokenizer_wrapper.seq_len]

        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        memory = model.encode_src(src_tensor)

        bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 1

        decoder_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        generated_tokens = [bos_id]

        for step in range(max_tokens):
            logits = model.decode_tgt(decoder_tokens, memory)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            generated_tokens.append(next_token)

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            decoder_tokens = torch.cat([decoder_tokens, next_token_tensor], dim=1)

            # Sliding window context
            context_window = 2
            decoder_tokens = decoder_tokens[:, -context_window:]
            decoder_tokens = decoder_tokens.detach()

            print(f"[{step}] Input: {tokenizer.decode(decoder_tokens[0])}, Next: {tokenizer.decode([next_token])}")

            if next_token == eos_id:
                break

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def load_json_file(file_path):
    """Load the JSON dataset file properly."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 🔹 Ensure it's properly parsed
            if not isinstance(data, list):
                raise ValueError("🚨 Loaded data is not a list of dictionaries.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"🚨 Failed to parse JSON: {e}")

def generate_2(model, prompt, tokenizer, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []

    with torch.no_grad():
        # Tokenize prompt → fixed encoder input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        encoder_input_len = input_ids.size(1)

        # Pad encoder input to max model length
        if encoder_input_len < seq_len:
            pad_len = seq_len - encoder_input_len
            pad_token_id = tokenizer.pad_token_id or 0
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :seq_len]

        # Encoder is static throughout generation
        encoder_input_ids = input_ids

        # Setup initial decoder input
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        tgt_ids = torch.tensor([[bos_token_id]], device=device)

        for _ in range(max_generated):
            # Forward pass through model
            batch_size, seq_lengths = encoder_input_ids.size()
            outputs, _ = model(encoder_input_ids, seq_lengths)
            logits = outputs[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt_ids[0].tolist()):
                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                    logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return tokenizer.decode(generated_tokens)



########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\abias\.cursor-tutor\vccdoe\mhlamodel\mhla\KANseriesNeuralNetwork-main\inhibitorynetwork\data", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=seq_len, help='Fixed maximum sequence length')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    # Load dataset correctly
    #json_data = load_json_file(args.data)

    # Pass parsed JSON instead of raw file path
    data = load_dataset(args.data)
    dataset = RawPairDataset(data)
    

    # 🔹 Ensure dataset isn't empty
    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty after filtering invalid entries! Check your dataset.")

    # Use a lambda to pass the fixed length to collate_fn.

    dataloader = dataset  # since we train token-wise without batching
    lr = 0.001
    embed_size = 256
    num_heads = 4
    num_layers = 2
    seq_length = args.max_seq_length
    # Initialize the integrated model with desired module toggles.
    #model = Transformer_Model(vocab_size, embed_size, num_layers, num_heads, seq_length=args.max_seq_length, device=device, tokenizer=base_tokenizer).to(device)
    model =  QuaternionTransformer( vocab_size, embed_size, embed_size, num_heads,num_layers, tokenizer, device)
    #model.embedding.initialize_from_semantic("semantic_vectors.json", tokenizer)
    #print("semantic_vectors copied")
    model = model.to(device)
    #optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    quaternion_params = []
    quaternion_params = [p for p in model.parameters() if p.requires_grad and p.shape[-1] == 4]
    standard_params = []
    standard_params = [p for p in model.parameters() if p.requires_grad and p.shape[-1] != 4]
    #quaternion_params = get_quaternion_parameters(model)

    optimizer_std = torch.optim.AdamW(standard_params, lr=lr)
    ##Quaternion Gradient-Real Hamilton Calculus backpropagaiton optimizer
    optimizer_q = QuaternionGHROptimizer(quaternion_params, lr=lr)
    #optimizer_q = torch.optim.AdamW(quaternion_params, lr=lr)
    # Create two optimizers:
    #Enable for standard optimizer/scheduler
    #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
    #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)


    #scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=total_steps, eta_min=lr * 0.1)

    #scheduler_std = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_std, T_max=total_steps, eta_min=lr * 0.1)
    #logging.info("Scheduler defined")
    # Set the model to evaluation mode and perform inference.
    prompt = "Find the slope of the line $3x+5y=20$."
    #generated_text = generate_autoregressive(model, prompt, tokenizer, seq_length, device)

    #generated_text = generate_2(model,prompt, base_tokenizer, seq_length, device)
    generated_text, generated_text_2 = model.generate(prompt, max_new_tokens=100, seq_len=32)


    print("Generated text:")
    print(generated_text)
    print("Generated text 2:")
    print(generated_text_2)


    for epoch in range(1, args.epochs + 1):

        avg_loss = train_model(args.batch_size, model, dataset, optimizer_std, optimizer_q, criterion, device)
        #avg_loss = train_decoder_autoregressive(
          #  model, dataset, tokenizer, optimizer_std, optimizer_q, criterion,
           # args.batch_size, args.max_seq_length, device
        #)
        if epoch % 5 == 0:        
            generated_text, generated_text_2 = model.generate(prompt, max_new_tokens=100, seq_len=32)


            print("Generated text:")
            print(generated_text)
            print("Generated text 2:")
            print(generated_text_2)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    prompt = "Find the slope of the line $3x+5y=20$."
    #generated_text = generate_autoregressive(model, prompt, tokenizer, seq_length, device)

    #generated_text = generate_2(model,prompt, base_tokenizer, seq_length, device)
    generated_text, generated_text_2 = model.generate(prompt, max_new_tokens=100, seq_len=32)


    print("Generated text:")
    print(generated_text)
    print("Generated text 2:")
    print(generated_text_2)

if __name__ == '__main__':
    main()
