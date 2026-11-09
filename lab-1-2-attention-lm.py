"""Lab 1-2: Attention language models.

This lab builds two small language models on TinyStories:

1. A uniform-attention language model, where each position represents its
   context by averaging previous token embeddings.
2. A learned-attention language model, where each position computes causal
   scaled dot-product attention over previous positions.

The tokenizer sits behind a tiny interface, so that none of the
language-modeling code depends on tokenizer internals.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sympy import denom
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.sparse import softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Config:
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_vocab_size: int = 5_000

    max_train_examples: int = 5_000
    max_valid_examples: int = 1_000
    block_size: int = 64
    batch_size: int = 32

    embedding_size: int = 512
    hidden_size: int = 512
    num_epochs: int = 3
    learning_rate: float = 3e-4

    # Which model to train and sample from: "uattnlm", "attnlm" or "both".
    models: str = "both"

    sample_prompt: str = "Once upon a time "
    sample_new_tokens: int = 60
    temperature: float = 0.8
    top_k: Optional[int] = 50


class TinyStoriesBPETokenizer:
    """Small BPE tokenizer trained on the lab corpus.

    This uses an existing tokenizer implementation while keeping the language
    model code independent of tokenizer internals.
    """

    def __init__(self, texts: list[str], vocab_size: int):
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tinystories_texts(config: Config) -> tuple[list[str], list[str]]:
    """Load a small TinyStories subset for quick in-class training."""
    train = load_dataset(
        config.dataset_name, split=f"train[:{config.max_train_examples}]"
    )
    valid = load_dataset(
        config.dataset_name, split=f"validation[:{config.max_valid_examples}]"
    )
    return list(train["text"]), list(valid["text"])


def load_tokenizer(config: Config, train_texts: list[str]) -> TinyStoriesBPETokenizer:
    """Load a tokenizer with the interface used by all language-modeling code."""
    return TinyStoriesBPETokenizer(train_texts, vocab_size=config.tokenizer_vocab_size)


def tokenize_texts(texts: list[str], tokenizer: TinyStoriesBPETokenizer) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        if tokenizer.bos_id is not None:
            token_ids.append(tokenizer.bos_id)
        token_ids.extend(tokenizer.encode(text))
        if tokenizer.eos_id is not None:
            token_ids.append(tokenizer.eos_id)
    return token_ids


class TokenBlockDataset(Dataset):
    """Fixed-length next-token prediction examples.

    Each item returns:
      input_ids: token_ids[i : i + block_size]
      target_ids: token_ids[i + 1 : i + block_size + 1]

    We use non-overlapping blocks to keep this lab fast. A production training
    loop would usually sample many more overlapping or randomly shifted blocks.
    """

    def __init__(self, token_ids: list[int], block_size: int):
        if len(token_ids) <= block_size:
            raise ValueError("Need more tokens than block_size.")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.num_blocks = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        x = self.token_ids[start : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y


def make_dataloader(
    token_ids: list[int],
    block_size: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TokenBlockDataset(token_ids, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def causal_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
    """Build the mask that keeps a position from attending to later positions.

    Args:
        seq_len: the number of positions in the sequence.
        device: the device the mask should live on.

    Returns:
        A boolean tensor of shape (seq_len, seq_len) whose entry [i, j] is True
        exactly when position i is allowed to attend to position j, which is the
        case when j <= i.

    Example:
        causal_mask(3, "cpu").tolist() ->
            [[True, False, False], [True, True, False], [True, True, True]]
    """
    mask = torch.full((seq_len, seq_len), False)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                mask[i, j] = True
    return mask


def attention(
    batched_Q: torch.Tensor,
    batched_K: torch.Tensor,
    batched_V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    The attention weights of a specific query are:
        softmax(Q K^T / sqrt(hidden_size))
    over the key positions, and its context vector is those weights applied to V.
    A position the mask disallows gets a weight of exactly zero, and shouldn't
    affect the the other scores.

    Args:
        batched_Q: queries of shape (batch_size, query_len, hidden_size).
        batched_K: keys of shape (batch_size, key_len, hidden_size).
        batched_V: values of shape (batch_size, key_len, hidden_size).
        mask: optional boolean tensor of shape (batch_size, query_len, key_len),
            where True means attention is allowed.

    Returns:
        batched_A: attention weights of shape (batch_size, query_len, key_len).
        batched_C: context vectors of shape (batch_size, query_len, hidden_size).

    Example:
        for Q, K and V of shape (2, 4, 8), returns weights of shape (2, 4, 4)
        and context vectors of shape (2, 4, 8).
    """
    hidden_size = batched_Q.size(-1)
    batch_size = batched_Q.size(0)
    query_len = batched_Q.size(1)
    key_len = batched_K.size(1)

    assert batched_K.size(-1) == hidden_size
    assert batched_V.size(-1) == hidden_size
    assert batched_K.size(0) == batch_size
    assert batched_V.size(0) == batch_size
    assert batched_V.size(1) == key_len
    if mask is not None:
        assert mask.shape == torch.Size([batch_size, query_len, key_len])

    attn = batched_Q @ batched_K.transpose(-1, -2) / math.sqrt(hidden_size)
    if mask is not None:
        attn = attn.masked_fill(~mask, float('-inf'))

    batched_A = torch.softmax(attn, dim=-1)
    batched_C = batched_A @ batched_V


    # TODO COMPLETE (4)
    return batched_A, batched_C


class UATTNLM(nn.Module):
    """Language model with uniform attention over previous token embeddings."""

    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.hidden2output = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the next-token logits (for each entry in the batch, for each position t),
        with uniform causal attention.

        The context at position i is the mean of the token embeddings at
        positions 0 to i. The logits are a linear map of this context:

            c_i      = (1 / (i + 1)) * sum_{j=0..i} e_j
            logits_i = hidden2output(c_i)

        Args:
            input_ids: token ids of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        embedded = self.embed(input_ids)
        attentioned = torch.cumsum(embedded, dim=1)
        divisor = torch.arange(1, input_ids.size(1) + 1)
        attentioned = attentioned/ divisor.view(1,-1,1)
        print(self.hidden2output(attentioned))
        return self.hidden2output(attentioned)


class ATTNLM(nn.Module):
    """Single-layer causal dot-product attention language model."""

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        """Create the layers of the model.

        `embed` and `hidden2output` are given.
        You need to fill in the lines of the three projections that turn a token embedding
         into a query, a key and a value vector.

        Args:
            vocab_size: the number of tokens the model scores.
            embedding_size: the size of a token embedding.
            hidden_size: the size of the queries, keys and values.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        # TODO COMPLETE (5)
        self.q_proj = ...
        self.k_proj = ...
        self.v_proj = ...
        self.hidden2output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the next-token logits (for each entry in the batch, for each position t),
        with causal attention.

        Args:
            input_ids: token ids of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size). The context at
            position i is a scaled dot-product attention over the queries, keys
            and values of positions 0 to i only, so the logits at position i do
            not change when a later token changes.

        Example:
            for input_ids of shape (2, 5) and a vocabulary of 100 tokens,
            returns a tensor of shape (2, 5, 100).
        """
        # TODO COMPLETE (6)
        raise NotImplementedError("TODO 6")


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in tqdm(dataloader, desc="train", leave=False):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = loss_fn(logits, target_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> tuple[float, float]:
    """Return average loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(input_ids)
        total_loss += loss_fn(logits, target_ids).item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


@torch.no_grad()
def evaluate_with_shuffled_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> tuple[float, float]:
    """Evaluate after shuffling target tokens inside each batch.

    This breaks the alignment between contexts and next tokens. A trained
    language model should do worse here than on the real validation targets.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        shuffled_targets = target_ids.reshape(-1)
        shuffled_targets = shuffled_targets[
            torch.randperm(shuffled_targets.numel(), device=device)
        ]
        shuffled_targets = shuffled_targets.reshape_as(target_ids)

        logits = model(input_ids)
        total_loss += loss_fn(logits, shuffled_targets).item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def print_evaluation_sanity_checks(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device | str,
    label: str,
) -> None:
    valid_loss, valid_ppl = evaluate(model, valid_loader, device)
    shuffled_loss, shuffled_ppl = evaluate_with_shuffled_targets(
        model, valid_loader, device
    )
    print(
        f"{label}: valid loss {valid_loss:.3f}, valid ppl {valid_ppl:.1f}; "
        f"shuffled-target loss {shuffled_loss:.3f}, shuffled-target ppl {shuffled_ppl:.1f}"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Config,
    label: str,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print(f"\nTraining {label} model")

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config.device)
        valid_loss, valid_ppl = evaluate(model, valid_loader, config.device)
        print(
            f"epoch {epoch}: "
            f"train loss {train_loss:.3f}, "
            f"valid loss {valid_loss:.3f}, "
            f"valid ppl {valid_ppl:.1f}"
        )


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: TinyStoriesBPETokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device | str,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    """Generate text autoregressively by feeding model samples back as context.

    The context starts with the tokenizer's BOS (beginning of sequence) token,
    followed by the encoded prompt. Every new token is drawn at random from the
    distribution the model predicts for the last position of the context so far,
    and is then appended to the context. Sampling is done proportional to the probabilites
    outputted by the model (see `torch.multinomial`).

    Generation stops after `max_new_tokens` new tokens, or earlier if the sampled
    token is the tokenizer's EOS (end of sequence) token.

    Args:
        model: a language model mapping token ids of shape
            (batch_size, seq_len) to logits of shape
            (batch_size, seq_len, vocab_size).
        tokenizer: the tokenizer the model was trained with. Its bos_id and
            eos_id can be None, in which case there is no token to prepend and
            no token to stop on.
        prompt: the text to continue.
        max_new_tokens: the largest number of tokens to add to the prompt.
        device: the device the model lives on.
        temperature: rescales the logits before they become probabilities, so
            that the distribution a token is sampled from is
            p = softmax(logits / temperature). Values below 1 concentrate the
            distribution on the most likely tokens, values above 1 flatten it.
        top_k: if given, only the k highest-scoring tokens can be sampled and
            every other token gets probability zero. If None, any token in the
            vocabulary can be sampled. Applied after the temperature rescaling,
            and k is never larger than the vocabulary.

    Returns:
        The decoded text, holding the prompt followed by everything generated
        after it. Special tokens are not part of the returned string.

    Example:
        for a model that always scores the token "b" highest:
        generate(model, tokenizer, "a", 3, "cpu", top_k=1) -> "abbb".
    """
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if tokenizer.bos_id is not None:
        token_ids = [tokenizer.bos_id] + token_ids
    for i in range(0, max_new_tokens):
        distributions: torch.Tensor = model.forward(torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0))
        last_dist = distributions[-1][-1]
        dist = torch.softmax(last_dist/temperature, dim=-1)
        if top_k is not None:
            topk_result = torch.topk(dist, k=top_k)
            dist = torch.full(dist.size(), 0, dtype=dist.dtype, device=dist.device)
            dist.scatter_(dim=-1, index=topk_result.indices, src=topk_result.values)
        next_token = torch.multinomial(dist, num_samples=1, replacement=True).tolist()[0]
        token_ids.append(next_token)
        if next_token == tokenizer.eos_id:
            break
    output_str = tokenizer.decode(token_ids)

    return output_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tiny attention language models."
    )
    parser.add_argument(
        "--max-train-examples", type=int, default=Config.max_train_examples
    )
    parser.add_argument(
        "--max-valid-examples", type=int, default=Config.max_valid_examples
    )
    parser.add_argument("--num-epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--block-size", type=int, default=Config.block_size)
    parser.add_argument(
        "--tokenizer-vocab-size", type=int, default=Config.tokenizer_vocab_size
    )
    parser.add_argument(
        "--sample-new-tokens", type=int, default=Config.sample_new_tokens
    )
    parser.add_argument(
        "--models", choices=["uattnlm", "attnlm", "both"], default=Config.models
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        max_train_examples=args.max_train_examples,
        max_valid_examples=args.max_valid_examples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        sample_new_tokens=args.sample_new_tokens,
        models=args.models,
    )
    set_seed(config.seed)

    print(f"Using device: {config.device}")

    train_texts, valid_texts = load_tinystories_texts(config)
    tokenizer = load_tokenizer(config, train_texts)

    train_ids = tokenize_texts(train_texts, tokenizer)
    valid_ids = tokenize_texts(valid_texts, tokenizer)
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Valid tokens: {len(valid_ids):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    train_loader = make_dataloader(
        train_ids, config.block_size, config.batch_size, shuffle=True
    )
    valid_loader = make_dataloader(
        valid_ids, config.block_size, config.batch_size, shuffle=False
    )

    if config.models in ("uattnlm", "both"):
        uniform_model = UATTNLM(
            vocab_size=tokenizer.vocab_size,
            embedding_size=config.embedding_size,
        ).to(config.device)
        print(f"Uniform attention parameters: {count_parameters(uniform_model):,}")
        print_evaluation_sanity_checks(
            uniform_model,
            valid_loader,
            config.device,
            "uniform attention before training",
        )
        train_model(
            uniform_model, train_loader, valid_loader, config, "uniform attention"
        )
        print_evaluation_sanity_checks(
            uniform_model,
            valid_loader,
            config.device,
            "uniform attention after training",
        )
        print("\nUniform attention sample:")
        print(
            generate(
                uniform_model,
                tokenizer,
                config.sample_prompt,
                config.sample_new_tokens,
                config.device,
                temperature=config.temperature,
                top_k=config.top_k,
            )
        )

    if config.models in ("attnlm", "both"):
        attention_model = ATTNLM(
            vocab_size=tokenizer.vocab_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
        ).to(config.device)
        print(f"Learned attention parameters: {count_parameters(attention_model):,}")
        print_evaluation_sanity_checks(
            attention_model,
            valid_loader,
            config.device,
            "learned attention before training",
        )
        train_model(
            attention_model, train_loader, valid_loader, config, "learned attention"
        )
        print_evaluation_sanity_checks(
            attention_model,
            valid_loader,
            config.device,
            "learned attention after training",
        )
        print("\nLearned attention sample:")
        print(
            generate(
                attention_model,
                tokenizer,
                config.sample_prompt,
                config.sample_new_tokens,
                config.device,
                temperature=config.temperature,
                top_k=config.top_k,
            )
        )


if __name__ == "__main__":
    main()
