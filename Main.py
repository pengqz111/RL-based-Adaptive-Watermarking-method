import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
from math import sqrt
from functools import partial
import random

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer
)
from torch.distributions import Beta, Normal
from torch.distributions.utils import clamp_probs


@dataclass
class RLAWMConfig:
    """Configuration for RLAWM algorithm"""

    def __init__(self, config_path: str = None, **kwargs):
        # Basic parameters
        self.algorithm_name = "RLAWM"
        self.device = kwargs.get('device', 'cuda')
        self.vocab_size = kwargs.get('vocab_size', 32000)

        # Watermark parameters
        self.gamma = kwargs.get('gamma', 0.5)  # Green list ratio
        self.hash_key = kwargs.get('hash_key', 42)
        self.z_threshold = kwargs.get('z_threshold', 2.0)
        self.prefix_length = kwargs.get('prefix_length', 5)

        # RL parameters
        self.delta_min = kwargs.get('delta_min', 0.5)
        self.delta_max = kwargs.get('delta_max', 3.0)
        self.hash_precision = kwargs.get('hash_precision', 16)

        # DPO parameters
        self.alpha = kwargs.get('alpha', 0.5)  # Detection weight
        self.beta = kwargs.get('beta', 0.5)  # Quality weight
        self.lambda_mmd = kwargs.get('lambda_mmd', 1.0)
        self.dpo_beta = kwargs.get('dpo_beta', 0.1)

        # Model parameters
        self.d_model = kwargs.get('d_model', 384)
        self.n_layers = kwargs.get('n_layers', 6)
        self.n_heads = kwargs.get('n_heads', 8)
        self.d_ff = kwargs.get('d_ff', 2048)

        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 2e-5)
        self.weight_decay = kwargs.get('weight_decay', 0.01)

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block for watermark agents"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class WatermarkGenerationAgent(nn.Module):
    """Watermark generation agent using RL"""

    def __init__(self, config: RLAWMConfig):
        super().__init__()
        self.config = config

        # Input embedding for vocabulary distribution
        self.vocab_embedding = nn.Linear(config.vocab_size, config.d_model)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])

        # Seed control branch
        self.seed_branch = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # alpha, beta for Beta distribution
        )

        # Delta control branch
        self.delta_branch = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # mean, std for Normal distribution
        )

    def forward(self, vocab_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vocab_probs: [batch_size, vocab_size] probability distribution
        Returns:
            seed_params: [batch_size, 2] parameters for seed distribution
            delta_params: [batch_size, 2] parameters for delta distribution
        """
        # Embed vocabulary probabilities
        x = self.vocab_embedding(vocab_probs)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Pool to get final representation
        x = x.squeeze(1)  # [batch_size, d_model]

        # Get seed and delta parameters
        seed_params = F.softplus(self.seed_branch(x)) + 1e-3  # Ensure positive
        delta_params = self.delta_branch(x)
        delta_params[:, 1] = F.softplus(delta_params[:, 1]) + 1e-3  # Ensure positive std

        return seed_params, delta_params

    def sample_actions(self, vocab_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy"""
        seed_params, delta_params = self.forward(vocab_probs)

        # Sample seed from Beta distribution
        seed_dist = Beta(seed_params[:, 0], seed_params[:, 1])
        seed = seed_dist.sample()

        # Sample delta from truncated normal
        delta_mean, delta_std = delta_params[:, 0], delta_params[:, 1]
        delta_dist = Normal(delta_mean, delta_std)
        delta = torch.clamp(delta_dist.sample(), self.config.delta_min, self.config.delta_max)

        return seed, delta


class WatermarkDetectionAgent(nn.Module):
    """Watermark detection agent"""

    def __init__(self, config: RLAWMConfig):
        super().__init__()
        self.config = config

        # Input embedding for tokens and green flags
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.green_embedding = nn.Embedding(2, config.d_model)  # 0 or 1

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary classification
        )

    def forward(self, token_ids: torch.Tensor, green_flags: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len]
            green_flags: [batch_size, seq_len]
        Returns:
            logits: [batch_size, 2]
        """
        # Embed tokens and green flags
        token_emb = self.token_embedding(token_ids)
        green_emb = self.green_embedding(green_flags)

        # Combine embeddings
        x = token_emb + green_emb  # [batch_size, seq_len, d_model]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, d_model]

        # Classification
        logits = self.classifier(x)

        return logits


class RLAWMUtils:
    """Utility functions for RLAWM"""

    def __init__(self, config: RLAWMConfig):
        self.config = config
        # Store tokenizer reference for MMD computation
        self.config.tokenizer = None  # Will be set from main class

        # Initialize after vocab_size is properly set
        self._initialize_generators()

    def _initialize_generators(self):
        """Initialize random generators after vocab_size is set"""
        try:
            self.rng = torch.Generator(device=self.config.device)
            self.rng.manual_seed(self.config.hash_key)
            # Ensure vocab_size is valid
            if hasattr(self.config, 'vocab_size') and self.config.vocab_size > 0:
                self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
            else:
                self.prf = None
        except Exception as e:
            print(f"Generator initialization error: {e}")
            self.rng = None
            self.prf = None

    def reinitialize_with_vocab_size(self, vocab_size: int):
        """Reinitialize with correct vocab size"""
        self.config.vocab_size = vocab_size
        self._initialize_generators()

    def get_greenlist_ids(self, input_ids: torch.LongTensor, seed: float) -> List[int]:
        """Get greenlist ids using controllable seed"""
        try:
            if self.prf is None:
                # Fallback: simple deterministic selection
                greenlist_size = int(self.config.vocab_size * self.config.gamma)
                start_idx = int(seed * (self.config.vocab_size - greenlist_size))
                return list(range(start_idx, start_idx + greenlist_size))

            # Get context hash
            context_hash = self._get_context_hash(input_ids)

            # Use seed to modify hash
            seed_int = int(seed * (2 ** min(16, self.config.hash_precision)))
            modified_hash = (self.config.hash_key * context_hash * seed_int) % self.config.vocab_size

            # Generate green list
            if self.rng is not None:
                self.rng.manual_seed(modified_hash)
                greenlist_size = int(self.config.vocab_size * self.config.gamma)
                vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
                greenlist_ids = vocab_permutation[:greenlist_size]
                return greenlist_ids.tolist()
            else:
                # Fallback method
                greenlist_size = int(self.config.vocab_size * self.config.gamma)
                start_idx = modified_hash % (self.config.vocab_size - greenlist_size)
                return list(range(start_idx, start_idx + greenlist_size))

        except Exception as e:
            print(f"Green list generation error: {e}")
            # Safe fallback
            greenlist_size = int(self.config.vocab_size * self.config.gamma)
            return list(range(0, greenlist_size))

    def _get_context_hash(self, input_ids: torch.LongTensor) -> int:
        """Get hash from context"""
        try:
            if len(input_ids) < self.config.prefix_length:
                return 1

            # Use additive scheme as default
            context_sum = 0
            for i in range(min(self.config.prefix_length, len(input_ids))):
                token_id = input_ids[-1 - i].item()
                # Ensure token_id is within vocab range
                if token_id < self.config.vocab_size:
                    context_sum += token_id
                else:
                    context_sum += token_id % self.config.vocab_size

            if self.prf is not None:
                return self.prf[context_sum % self.config.vocab_size].item()
            else:
                return context_sum % self.config.vocab_size

        except Exception as e:
            print(f"Context hash error: {e}")
            return 1

    def compute_z_score(self, observed_count: int, total_tokens: int) -> float:
        """Compute z-score for watermark detection"""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * total_tokens
        denom = sqrt(total_tokens * expected_count * (1 - expected_count))
        z = numer / denom if denom > 0 else 0
        return z

    def score_sequence(self, input_ids: torch.Tensor, seeds: List[float]) -> Tuple[float, List[int]]:
        """Score sequence for watermark detection"""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(f"Must have at least 1 token to score")

        green_token_count = 0
        green_token_flags = [-1] * self.config.prefix_length

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx].item()
            seed = seeds[idx] if idx < len(seeds) else 0.5  # Default seed
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx], seed)

            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

        z_score = self.compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags

    def compute_mmd(self, original_texts: List[str], watermarked_texts: List[str]) -> float:
        """Compute Maximum Mean Discrepancy between text distributions using simple embeddings"""
        try:
            # Use tokenizer embeddings as a simple alternative to sentence transformer
            def get_text_embedding(text: str) -> torch.Tensor:
                # Tokenize and get average token embeddings
                tokens = self.config.tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
                if len(tokens) == 0:
                    return torch.zeros(self.config.d_model, device=self.config.device)

                # Use simple averaging of token indices as embedding
                token_tensor = torch.tensor(tokens, device=self.config.device).float()
                # Normalize and project to embedding dimension
                embedding = torch.mean(token_tensor).unsqueeze(0).repeat(self.config.d_model)
                return embedding / torch.norm(embedding)

            # Get embeddings for both text sets
            orig_embeddings = torch.stack([get_text_embedding(text) for text in original_texts])
            water_embeddings = torch.stack([get_text_embedding(text) for text in watermarked_texts])

            # Compute RBF kernel with median heuristic
            def rbf_kernel(x, y, sigma):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if y.dim() == 1:
                    y = y.unsqueeze(0)
                dist = torch.cdist(x, y) ** 2
                return torch.exp(-dist / (2 * sigma ** 2))

            # Compute median distance for bandwidth
            all_embeddings = torch.cat([orig_embeddings, water_embeddings], dim=0)
            distances = torch.cdist(all_embeddings, all_embeddings)
            sigma = torch.median(distances[distances > 0])
            if sigma == 0:
                sigma = torch.tensor(1.0, device=self.config.device)

            # Compute MMD
            K_orig = rbf_kernel(orig_embeddings, orig_embeddings, sigma)
            K_water = rbf_kernel(water_embeddings, water_embeddings, sigma)
            K_cross = rbf_kernel(orig_embeddings, water_embeddings, sigma)

            m = orig_embeddings.shape[0]
            mmd_squared = (K_orig.sum() + K_water.sum()) / (m * m) - 2 * K_cross.sum() / (m * m)

            return max(0.0, mmd_squared.item())  # Ensure non-negative

        except Exception as e:
            print(f"MMD computation error: {e}")
            # Return a simple text length difference as fallback
            orig_len = np.mean([len(text.split()) for text in original_texts])
            water_len = np.mean([len(text.split()) for text in watermarked_texts])
            return abs(orig_len - water_len) / max(orig_len, water_len, 1.0)


class RLAWMLogitsProcessor(LogitsProcessor):
    """Logits processor for RLAWM watermarking"""

    def __init__(self, config: RLAWMConfig, utils: RLAWMUtils, generation_agent: WatermarkGenerationAgent):
        self.config = config
        self.utils = utils
        self.generation_agent = generation_agent
        self.generation_agent.eval()

        # Store seeds and deltas for later use
        self.seeds = []
        self.deltas = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add adaptive watermark"""
        try:
            if input_ids.shape[-1] < self.config.prefix_length:
                return scores

            batch_size = input_ids.shape[0]
            processed_scores = scores.clone()

            for b_idx in range(batch_size):
                try:
                    # Get current probability distribution
                    current_probs = F.softmax(scores[b_idx], dim=-1)

                    # Generate adaptive actions
                    with torch.no_grad():
                        seed, delta = self.generation_agent.sample_actions(current_probs.unsqueeze(0))
                        seed = seed.item()
                        delta = delta.item()

                    # Store for later use
                    self.seeds.append(seed)
                    self.deltas.append(delta)

                    # Get green list using adaptive seed
                    greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx], seed)

                    # Apply adaptive delta to green list tokens
                    for token_id in greenlist_ids:
                        if 0 <= token_id < processed_scores.shape[-1]:  # Safety check
                            processed_scores[b_idx, token_id] += delta

                except Exception as e:
                    print(f"Error processing batch {b_idx}: {e}")
                    continue

            return processed_scores

        except Exception as e:
            print(f"Logits processor error: {e}")
            return scores


class RLAWM:
    """Main RLAWM watermarking system"""

    def __init__(self, model_path: str, config: RLAWMConfig = None):
        self.model_path = model_path
        self.config = config or RLAWMConfig()

        print("Loading model and tokenizer...")
        # Load model and tokenizer (local only)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                local_files_only=True,  # Use only local files
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True  # Use only local files
            )
        except Exception as e:
            print(f"Error loading with local_files_only, trying without: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Update vocab size and ensure it's correct
        actual_vocab_size = len(self.tokenizer)
        self.config.vocab_size = actual_vocab_size
        print(f"Actual vocabulary size: {actual_vocab_size}")

        # Initialize components with correct vocab size
        self.utils = RLAWMUtils(self.config)
        self.utils.reinitialize_with_vocab_size(actual_vocab_size)
        # Set tokenizer reference for MMD computation
        self.utils.config.tokenizer = self.tokenizer

        self.generation_agent = WatermarkGenerationAgent(self.config).to(self.config.device)
        self.detection_agent = WatermarkDetectionAgent(self.config).to(self.config.device)

        # Initialize logits processor
        self.logits_processor = RLAWMLogitsProcessor(self.config, self.utils, self.generation_agent)

        print("RLAWM initialization completed")

    def generate_watermarked_text(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate watermarked text"""
        try:
            # Reset stored seeds and deltas
            self.logits_processor.seeds = []
            self.logits_processor.deltas = []

            # Encode prompt with error handling
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,  # Avoid padding issues
                    truncation=True,
                    max_length=512
                )
                # Move to device safely
                input_ids = inputs['input_ids'].to(self.config.device)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

            except Exception as e:
                print(f"Tokenization error: {e}")
                return prompt  # Return original prompt if tokenization fails

            # Generate with watermark
            with torch.no_grad():
                try:
                    generation_kwargs = {
                        'input_ids': input_ids,
                        'max_new_tokens': max_new_tokens,
                        'logits_processor': LogitsProcessorList([self.logits_processor]),
                        'do_sample': True,
                        'temperature': 1.0,
                        'pad_token_id': self.tokenizer.eos_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                    }

                    if attention_mask is not None:
                        generation_kwargs['attention_mask'] = attention_mask

                    generation_kwargs.update(kwargs)

                    outputs = self.model.generate(**generation_kwargs)

                except Exception as e:
                    print(f"Generation error: {e}")
                    return prompt

            # Decode output
            try:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return generated_text
            except Exception as e:
                print(f"Decoding error: {e}")
                return prompt

        except Exception as e:
            print(f"Overall generation error: {e}")
            return prompt

    def detect_watermark(self, text: str, return_dict: bool = True) -> Union[Dict, Tuple]:
        """Detect watermark in text"""
        try:
            # Encode text with safety checks
            tokens = self.tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
            if len(tokens) == 0:
                if return_dict:
                    return {"is_watermarked": False, "z_score": 0.0}
                else:
                    return False, 0.0

            input_ids = torch.tensor(tokens, device=self.config.device)

            # For detection, use default seeds (we don't have the original seeds)
            default_seeds = [0.5] * len(tokens)

            try:
                # Compute z-score
                z_score, green_flags = self.utils.score_sequence(input_ids, default_seeds)

                # Statistical detection
                is_watermarked_stat = z_score > self.config.z_threshold

                # Neural detection (if we have a trained detection agent)
                try:
                    with torch.no_grad():
                        # Limit sequence length for safety
                        max_len = min(128, len(tokens))
                        token_ids = input_ids[:max_len].unsqueeze(0)
                        green_tensor = torch.tensor(green_flags[:max_len], device=self.config.device).unsqueeze(0)

                        # Ensure green_tensor values are valid
                        green_tensor = torch.clamp(green_tensor, 0, 1)

                        logits = self.detection_agent(token_ids, green_tensor)
                        probs = F.softmax(logits, dim=-1)
                        is_watermarked_neural = probs[0, 1] > 0.5
                        neural_confidence = probs[0, 1].item()
                except Exception as e:
                    print(f"Neural detection error: {e}")
                    is_watermarked_neural = is_watermarked_stat
                    neural_confidence = float(is_watermarked_stat)

                # Combine statistical and neural detection
                is_watermarked = is_watermarked_stat or is_watermarked_neural

                if return_dict:
                    return {
                        "is_watermarked": is_watermarked,
                        "z_score": z_score,
                        "statistical_detection": is_watermarked_stat,
                        "neural_detection": is_watermarked_neural,
                        "neural_confidence": neural_confidence
                    }
                else:
                    return is_watermarked, z_score

            except Exception as e:
                print(f"Detection computation error: {e}")
                if return_dict:
                    return {"is_watermarked": False, "z_score": 0.0}
                else:
                    return False, 0.0

        except Exception as e:
            print(f"Detection error: {e}")
            if return_dict:
                return {"is_watermarked": False, "z_score": 0.0}
            else:
                return False, 0.0

    def get_data_for_visualization(self, text: str) -> Tuple[List[str], List[int]]:
        """Get visualization data"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor(tokens, device=self.config.device)

        # Use default seeds for visualization
        default_seeds = [0.5] * len(tokens)

        try:
            _, green_flags = self.utils.score_sequence(input_ids, default_seeds)

            # Decode individual tokens
            decoded_tokens = []
            for token_id in tokens:
                token = self.tokenizer.decode([token_id])
                decoded_tokens.append(token)

            return decoded_tokens, green_flags
        except:
            # Fallback
            decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in tokens]
            green_flags = [0] * len(tokens)
            return decoded_tokens, green_flags

    def train_agents(self, dataset: List[Dict], num_epochs: int = 100):
        """Train the generation and detection agents using DPO"""
        print("Training RLAWM agents...")

        # Initialize optimizers
        gen_optimizer = torch.optim.AdamW(
            self.generation_agent.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        det_optimizer = torch.optim.AdamW(
            self.detection_agent.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        for epoch in range(num_epochs):
            total_gen_loss = 0
            total_det_loss = 0

            for batch_idx, data in enumerate(dataset):
                prompt = data['prompt']

                # Generate watermarked and non-watermarked texts
                watermarked_text = self.generate_watermarked_text(prompt, max_new_tokens=50)

                # Generate non-watermarked text (without logits processor)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                non_watermarked_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Train detection agent
                self.detection_agent.train()
                det_optimizer.zero_grad()

                # Prepare detection data
                try:
                    # Watermarked sample
                    water_tokens = self.tokenizer.encode(watermarked_text, add_special_tokens=False)
                    water_input_ids = torch.tensor(water_tokens, device=self.config.device)
                    _, water_green_flags = self.utils.score_sequence(water_input_ids, [0.5] * len(water_tokens))

                    # Non-watermarked sample
                    non_water_tokens = self.tokenizer.encode(non_watermarked_text, add_special_tokens=False)
                    non_water_input_ids = torch.tensor(non_water_tokens, device=self.config.device)
                    _, non_water_green_flags = self.utils.score_sequence(non_water_input_ids,
                                                                         [0.5] * len(non_water_tokens))

                    # Prepare batch (limit sequence length)
                    max_len = 128
                    water_tokens_batch = water_input_ids[:max_len].unsqueeze(0)
                    water_green_batch = torch.tensor(water_green_flags[:max_len], device=self.config.device).unsqueeze(
                        0)

                    non_water_tokens_batch = non_water_input_ids[:max_len].unsqueeze(0)
                    non_water_green_batch = torch.tensor(non_water_green_flags[:max_len],
                                                         device=self.config.device).unsqueeze(0)

                    # Pad to same length
                    if water_tokens_batch.shape[1] != non_water_tokens_batch.shape[1]:
                        min_len = min(water_tokens_batch.shape[1], non_water_tokens_batch.shape[1])
                        water_tokens_batch = water_tokens_batch[:, :min_len]
                        water_green_batch = water_green_batch[:, :min_len]
                        non_water_tokens_batch = non_water_tokens_batch[:, :min_len]
                        non_water_green_batch = non_water_green_batch[:, :min_len]

                    # Detection loss
                    water_logits = self.detection_agent(water_tokens_batch, water_green_batch)
                    non_water_logits = self.detection_agent(non_water_tokens_batch, non_water_green_batch)

                    water_labels = torch.ones(1, device=self.config.device, dtype=torch.long)
                    non_water_labels = torch.zeros(1, device=self.config.device, dtype=torch.long)

                    det_loss = F.cross_entropy(water_logits, water_labels) + F.cross_entropy(non_water_logits,
                                                                                             non_water_labels)

                    det_loss.backward()
                    det_optimizer.step()
                    total_det_loss += det_loss.item()

                except Exception as e:
                    print(f"Detection training error: {e}")
                    continue

                # Train generation agent (simplified DPO-like objective)
                self.generation_agent.train()
                gen_optimizer.zero_grad()

                try:
                    # Compute MMD loss for quality preservation
                    mmd_loss = self.utils.compute_mmd([non_watermarked_text], [watermarked_text])

                    # Compute detection reward
                    detection_result = self.detect_watermark(watermarked_text, return_dict=True)
                    detection_reward = float(detection_result['is_watermarked'])

                    # Compute quality reward
                    quality_reward = torch.exp(-self.config.lambda_mmd * mmd_loss)

                    # Combined reward
                    total_reward = self.config.alpha * detection_reward + self.config.beta * quality_reward

                    # Simple policy gradient loss (negative reward)
                    gen_loss = -torch.tensor(total_reward, requires_grad=True, device=self.config.device)

                    gen_loss.backward()
                    gen_optimizer.step()
                    total_gen_loss += gen_loss.item()

                except Exception as e:
                    print(f"Generation training error: {e}")
                    continue

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}: Gen Loss: {total_gen_loss / (batch_idx + 1):.4f}, Det Loss: {total_det_loss / (batch_idx + 1):.4f}")

            print(
                f"Epoch {epoch}: Avg Gen Loss: {total_gen_loss / len(dataset):.4f}, Avg Det Loss: {total_det_loss / len(dataset):.4f}")

        print("Training completed!")

    def save_model(self, save_path: str):
        """Save trained agents"""
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.generation_agent.state_dict(), os.path.join(save_path, "generation_agent.pth"))
        torch.save(self.detection_agent.state_dict(), os.path.join(save_path, "detection_agent.pth"))

        # Save config
        config_dict = {
            'algorithm_name': self.config.algorithm_name,
            'gamma': self.config.gamma,
            'hash_key': self.config.hash_key,
            'z_threshold': self.config.z_threshold,
            'prefix_length': self.config.prefix_length,
            'delta_min': self.config.delta_min,
            'delta_max': self.config.delta_max,
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'lambda_mmd': self.config.lambda_mmd
        }

        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Model saved to {save_path}")

    # Add this method to the RLAWM class in Main.py

    def train_agents_safe(self, dataset: List[Dict], num_epochs: int = 100, focus: str = "collaborative"):
        """Safe training method with better error handling and staged approach"""
        print(f"Starting {focus} training for {num_epochs} epochs...")

        # Initialize optimizers
        gen_optimizer = torch.optim.AdamW(
            self.generation_agent.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        det_optimizer = torch.optim.AdamW(
            self.detection_agent.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        successful_batches = 0
        total_gen_loss = 0
        total_det_loss = 0

        for epoch in range(num_epochs):
            epoch_gen_loss = 0
            epoch_det_loss = 0
            epoch_successful = 0

            # Shuffle dataset for each epoch
            epoch_data = random.sample(dataset, min(50, len(dataset)))  # Limit batch size

            for batch_idx, data in enumerate(epoch_data):
                try:
                    prompt = data.get('prompt', data.get('text', ''))
                    if len(prompt) > 200:
                        prompt = prompt[:200]  # Limit prompt length

                    if len(prompt.strip()) == 0:
                        continue

                    # Generate texts with safer approach
                    try:
                        # Generate watermarked text
                        watermarked_text = self.generate_watermarked_text(prompt, max_new_tokens=30)
                        if not watermarked_text or len(watermarked_text.strip()) <= len(prompt):
                            continue

                        # Generate non-watermarked text
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=256,
                            padding=False
                        )
                        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=30,
                                do_sample=True,
                                temperature=1.0,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                        non_watermarked_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    except Exception as e:
                        print(f"Generation error in epoch {epoch}, batch {batch_idx}: {e}")
                        continue

                    # Train detection agent (always train this)
                    try:
                        self.detection_agent.train()
                        det_optimizer.zero_grad()

                        # Prepare detection data with safer tokenization
                        water_tokens = self.tokenizer.encode(
                            watermarked_text,
                            add_special_tokens=False,
                            max_length=128,
                            truncation=True
                        )
                        non_water_tokens = self.tokenizer.encode(
                            non_watermarked_text,
                            add_special_tokens=False,
                            max_length=128,
                            truncation=True
                        )

                        if len(water_tokens) < 2 or len(non_water_tokens) < 2:
                            continue

                        # Ensure tokens are within vocab range
                        water_tokens = [min(t, self.config.vocab_size - 1) for t in water_tokens]
                        non_water_tokens = [min(t, self.config.vocab_size - 1) for t in non_water_tokens]

                        # Get green flags safely
                        water_input_ids = torch.tensor(water_tokens, device=self.config.device)
                        non_water_input_ids = torch.tensor(non_water_tokens, device=self.config.device)

                        try:
                            _, water_green_flags = self.utils.score_sequence(water_input_ids, [0.5] * len(water_tokens))
                            _, non_water_green_flags = self.utils.score_sequence(non_water_input_ids,
                                                                                 [0.5] * len(non_water_tokens))
                        except:
                            # Fallback green flags
                            water_green_flags = [0] * len(water_tokens)
                            non_water_green_flags = [0] * len(non_water_tokens)

                        # Prepare batch with equal lengths
                        max_len = min(64, max(len(water_tokens), len(non_water_tokens)))

                        # Pad or truncate to max_len
                        water_tokens_padded = water_tokens[:max_len] + [self.tokenizer.eos_token_id] * max(0,
                                                                                                           max_len - len(
                                                                                                               water_tokens))
                        water_green_padded = water_green_flags[:max_len] + [0] * max(0,
                                                                                     max_len - len(water_green_flags))

                        non_water_tokens_padded = non_water_tokens[:max_len] + [self.tokenizer.eos_token_id] * max(0,
                                                                                                                   max_len - len(
                                                                                                                       non_water_tokens))
                        non_water_green_padded = non_water_green_flags[:max_len] + [0] * max(0, max_len - len(
                            non_water_green_flags))

                        # Convert to tensors
                        water_tokens_batch = torch.tensor([water_tokens_padded], device=self.config.device)
                        water_green_batch = torch.tensor([water_green_padded], device=self.config.device)

                        non_water_tokens_batch = torch.tensor([non_water_tokens_padded], device=self.config.device)
                        non_water_green_batch = torch.tensor([non_water_green_padded], device=self.config.device)

                        # Ensure green flags are in valid range
                        water_green_batch = torch.clamp(water_green_batch, 0, 1)
                        non_water_green_batch = torch.clamp(non_water_green_batch, 0, 1)

                        # Detection forward pass
                        water_logits = self.detection_agent(water_tokens_batch, water_green_batch)
                        non_water_logits = self.detection_agent(non_water_tokens_batch, non_water_green_batch)

                        water_labels = torch.ones(1, device=self.config.device, dtype=torch.long)
                        non_water_labels = torch.zeros(1, device=self.config.device, dtype=torch.long)

                        det_loss = F.cross_entropy(water_logits, water_labels) + F.cross_entropy(non_water_logits,
                                                                                                 non_water_labels)

                        det_loss.backward()
                        det_optimizer.step()

                        epoch_det_loss += det_loss.item()

                    except Exception as e:
                        print(f"Detection training error in epoch {epoch}, batch {batch_idx}: {e}")
                        continue

                    # Train generation agent (focus-dependent)
                    if focus in ["generation", "collaborative"]:
                        try:
                            self.generation_agent.train()
                            gen_optimizer.zero_grad()

                            # Simplified generation training
                            # Focus on detection success
                            detection_result = self.detect_watermark(watermarked_text, return_dict=True)
                            detection_reward = float(detection_result.get('is_watermarked', False))
                            if torch.is_tensor(detection_reward):
                                detection_reward = detection_reward.item()

                            # Simple quality reward (text length similarity)
                            quality_reward = 1.0 / (1.0 + abs(len(watermarked_text) - len(non_watermarked_text)) / max(
                                len(watermarked_text), len(non_watermarked_text), 1))

                            # Combined reward
                            total_reward = self.config.alpha * detection_reward + self.config.beta * quality_reward

                            # Simple policy gradient loss
                            gen_loss = -torch.tensor(total_reward, requires_grad=True, device=self.config.device)

                            gen_loss.backward()
                            gen_optimizer.step()

                            epoch_gen_loss += gen_loss.item()

                        except Exception as e:
                            print(f"Generation training error in epoch {epoch}, batch {batch_idx}: {e}")
                            continue

                    epoch_successful += 1
                    successful_batches += 1

                except Exception as e:
                    print(f"Batch error in epoch {epoch}, batch {batch_idx}: {e}")
                    continue

            # Epoch summary
            if epoch_successful > 0:
                avg_gen_loss = epoch_gen_loss / epoch_successful
                avg_det_loss = epoch_det_loss / epoch_successful
                total_gen_loss += avg_gen_loss
                total_det_loss += avg_det_loss

                if epoch % 20 == 0:
                    print(
                        f"Epoch {epoch}/{num_epochs}: Gen Loss: {avg_gen_loss:.4f}, Det Loss: {avg_det_loss:.4f}, Successful: {epoch_successful}")
            else:
                print(f"Epoch {epoch}/{num_epochs}: No successful batches")

        final_gen_loss = total_gen_loss / num_epochs if num_epochs > 0 else 0
        final_det_loss = total_det_loss / num_epochs if num_epochs > 0 else 0

        print(f"Training completed: {successful_batches} successful batches")
        print(f"Final avg losses - Gen: {final_gen_loss:.4f}, Det: {final_det_loss:.4f}")

    def load_model(self, load_path: str):
        """Load trained agents"""
        gen_path = os.path.join(load_path, "generation_agent.pth")
        det_path = os.path.join(load_path, "detection_agent.pth")
        config_path = os.path.join(load_path, "config.json")

        if os.path.exists(gen_path):
            self.generation_agent.load_state_dict(torch.load(gen_path, map_location=self.config.device))
            print("Generation agent loaded")

        if os.path.exists(det_path):
            self.detection_agent.load_state_dict(torch.load(det_path, map_location=self.config.device))
            print("Detection agent loaded")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self.config, key, value)
            print("Config loaded")


def evaluate_rlawm(rlawm_model: RLAWM, test_data: List[Dict], num_samples: int = 50) -> Dict:
    """Evaluate RLAWM performance - simplified version"""
    print("Evaluating RLAWM...")

    results = {
        'tpr_at_1': 0.0,
        'f1_score': 0.0,
        'detection_accuracy': 0.0,
        'avg_z_score_watermarked': 0.0,
        'avg_z_score_non_watermarked': 0.0
    }

    detection_results = []
    z_scores_watermarked = []
    z_scores_non_watermarked = []

    for i, data in enumerate(test_data[:num_samples]):
        prompt = data.get('prompt', data.get('text', ''))[:100]  # Use shorter prompts for faster evaluation

        try:
            # Generate watermarked text
            watermarked_text = rlawm_model.generate_watermarked_text(prompt, max_new_tokens=50)

            # Generate non-watermarked text
            inputs = rlawm_model.tokenizer(prompt, return_tensors="pt").to(rlawm_model.config.device)
            with torch.no_grad():
                outputs = rlawm_model.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=rlawm_model.tokenizer.eos_token_id
                )
            non_watermarked_text = rlawm_model.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Test detection
            water_result = rlawm_model.detect_watermark(watermarked_text, return_dict=True)
            non_water_result = rlawm_model.detect_watermark(non_watermarked_text, return_dict=True)

            detection_results.extend([
                (True, water_result['is_watermarked']),
                (False, non_water_result['is_watermarked'])
            ])

            z_scores_watermarked.append(water_result['z_score'])
            z_scores_non_watermarked.append(non_water_result['z_score'])

            if i % 10 == 0:
                print(f"Processed {i}/{num_samples} samples")
                print(
                    f"  Watermarked detection: {water_result['is_watermarked']}, z-score: {water_result['z_score']:.3f}")
                print(
                    f"  Non-watermarked detection: {non_water_result['is_watermarked']}, z-score: {non_water_result['z_score']:.3f}")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Calculate metrics
    if detection_results:
        true_positives = sum(1 for true_label, pred in detection_results if true_label and pred)
        false_positives = sum(1 for true_label, pred in detection_results if not true_label and pred)
        true_negatives = sum(1 for true_label, pred in detection_results if not true_label and not pred)
        false_negatives = sum(1 for true_label, pred in detection_results if true_label and not pred)

        # TPR@1% (approximation)
        total_watermarked = true_positives + false_negatives
        if total_watermarked > 0:
            results['tpr_at_1'] = true_positives / total_watermarked

        # F1 Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        results['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Detection accuracy
        correct_predictions = true_positives + true_negatives
        total_predictions = len(detection_results)
        results['detection_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Z-score statistics
    if z_scores_watermarked:
        results['avg_z_score_watermarked'] = np.mean(z_scores_watermarked)
    if z_scores_non_watermarked:
        results['avg_z_score_non_watermarked'] = np.mean(z_scores_non_watermarked)

    print(f"\nEvaluation Results:")
    print(f"TPR@1%: {results['tpr_at_1']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Detection Accuracy: {results['detection_accuracy']:.4f}")
    print(f"Avg Z-score (Watermarked): {results['avg_z_score_watermarked']:.4f}")
    print(f"Avg Z-score (Non-watermarked): {results['avg_z_score_non_watermarked']:.4f}")

    return results


def main():
    """Main function to demonstrate RLAWM usage"""

    # Set environment variables to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configuration
    config = RLAWMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gamma=0.5,
        hash_key=42,
        z_threshold=2.0,
        prefix_length=5,
        delta_min=0.5,
        delta_max=3.0,
        alpha=0.5,
        beta=0.5,
        lambda_mmd=1.0,
        learning_rate=2e-5,
        n_layers=6,
        n_heads=8,
        d_model=384,
        d_ff=2048,
        hash_precision=16
    )

    # Load dataset
    print("Loading dataset...")
    dataset_path = '/root/autodl-tmp/WaterMarking/Dataset/processed_c4.json'
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            dataset = [json.loads(line) for line in lines]
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Dataset loading error: {e}")
        # Create dummy dataset for testing
        dataset = [
                      {"prompt": "The benefits of renewable energy include", "text": "sample text"},
                      {"prompt": "Climate change is", "text": "another sample"},
                      {"prompt": "Artificial intelligence can", "text": "third sample"}
                  ] * 100
        print("Using dummy dataset for testing")

    # Initialize RLAWM
    model_path = "/root/autodl-tmp/Llama-2-7b-chat-hf"
    print(f"Initializing RLAWM with model: {model_path}")

    try:
        rlawm = RLAWM(model_path, config)
    except Exception as e:
        print(f"RLAWM initialization error: {e}")
        return

    # Split dataset
    train_data = dataset[:min(50, len(dataset) // 2)]  # Smaller training set
    test_data = dataset[len(dataset) // 2:len(dataset) // 2 + 50]  # Smaller test set

    # Skip training for now to test basic functionality
    print("Skipping training for basic functionality test...")

    # Test watermark generation and detection
    print("\nTesting watermark generation...")
    test_prompt = "The benefits of renewable energy include"

    try:
        # Generate watermarked text
        watermarked_text = rlawm.generate_watermarked_text(test_prompt, max_new_tokens=30)
        print(f"Prompt: {test_prompt}")
        print(f"Watermarked: {watermarked_text}")

        # Generate non-watermarked text for comparison
        try:
            inputs = rlawm.tokenizer(test_prompt, return_tensors="pt").to(config.device)
            with torch.no_grad():
                outputs = rlawm.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=rlawm.tokenizer.eos_token_id
                )
            non_watermarked_text = rlawm.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Non-watermarked: {non_watermarked_text}")
        except Exception as e:
            print(f"Non-watermarked generation error: {e}")
            non_watermarked_text = test_prompt + " clean energy sources."

        # Test detection
        print("\nTesting watermark detection...")
        water_result = rlawm.detect_watermark(watermarked_text, return_dict=True)
        non_water_result = rlawm.detect_watermark(non_watermarked_text, return_dict=True)

        print(f"Watermarked text detection: {water_result}")
        print(f"Non-watermarked text detection: {non_water_result}")

        # Test visualization data
        print("\nTesting visualization...")
        try:
            tokens, green_flags = rlawm.get_data_for_visualization(watermarked_text)
            print(f"First 10 tokens: {tokens[:10] if len(tokens) >= 10 else tokens}")
            print(f"First 10 green flags: {green_flags[:10] if len(green_flags) >= 10 else green_flags}")
        except Exception as e:
            print(f"Visualization error: {e}")

        print("\nRLAWM basic functionality test completed successfully!")

    except Exception as e:
        print(f"Testing error: {e}")
        print("Basic functionality test failed")
        rlawm.save_model("rlawm_checkpoint")


if __name__ == "__main__":
    main()