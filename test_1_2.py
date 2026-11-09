#!/usr/bin/env python3
"""Tests for lab 1-2. Usage: python test_1_2.py [todo_number ...]"""

import importlib.util
import math
import sys
from pathlib import Path

import torch

LAB_PATH = Path(__file__).resolve().parent / "lab-1-2-attention-lm.py"


def load_lab():
    spec = importlib.util.spec_from_file_location("lab", LAB_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab"] = module
    spec.loader.exec_module(module)
    return module


def check(name, got_fn, expected, compare=None):
    """Run got_fn(), compare against expected, print a PASS/FAIL line."""
    try:
        got = got_fn()
    except NotImplementedError:
        print(f"  [FAIL] {name}\n         not implemented yet")
        return False
    except Exception as error:
        print(f"  [FAIL] {name}\n         raised {type(error).__name__}: {error}")
        return False
    ok = compare(got, expected) if compare else got == expected
    if ok:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}\n         expected {expected!r}\n         got      {got!r}")
    return ok


def check_raises(name, fn, exc_type, message_substring):
    try:
        fn()
    except exc_type as error:
        if message_substring in str(error):
            print(f"  [PASS] {name}")
            return True
        print(f"  [FAIL] {name}\n         expected message containing {message_substring!r}\n         got      {str(error)!r}")
        return False
    except Exception as error:
        print(f"  [FAIL] {name}\n         expected {exc_type.__name__}, got {type(error).__name__}: {error}")
        return False
    print(f"  [FAIL] {name}\n         expected {exc_type.__name__}, nothing raised")
    return False


def nested_close(got, expected, tolerance=1e-6):
    """Compare nested lists of floats elementwise."""
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return False
        return all(
            nested_close(g, e, tolerance) for g, e in zip(got, expected)
        )
    if not isinstance(got, (int, float)):
        return False
    return math.isclose(got, expected, rel_tol=tolerance, abs_tol=tolerance)


def make_uniform_model(lab):
    """A UATTNLM with fixed weights, so its output is fully determined."""
    model = lab.UATTNLM(vocab_size=4, embedding_size=3)
    with torch.no_grad():
        model.embed.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]
            )
        )
        model.hidden2output.weight.copy_(torch.eye(4, 3))
        model.hidden2output.bias.zero_()
    return model


def make_attention_model(lab):
    """An ATTNLM with fixed weights, so its output is fully determined."""
    model = lab.ATTNLM(vocab_size=4, embedding_size=3, hidden_size=2)
    with torch.no_grad():
        model.embed.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
            )
        )
        for index, projection in enumerate([model.q_proj, model.k_proj, model.v_proj]):
            projection.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) + 0.1 * index
            )
            projection.bias.zero_()
        model.hidden2output.weight.copy_(torch.eye(4, 2))
        model.hidden2output.bias.zero_()
    return model


class FakeTokenizer:
    """Three-letter tokenizer with the interface `generate` relies on."""

    def __init__(self):
        self.bos_id = 1
        self.eos_id = 2
        self.vocab_size = 6
        self.pieces = {3: "a", 4: "b", 5: "c"}

    def encode(self, text):
        return [{"a": 3, "b": 4, "c": 5}[character] for character in text]

    def decode(self, token_ids):
        return "".join(self.pieces.get(token_id, "") for token_id in token_ids)


class TableModel(torch.nn.Module):
    """Deterministic model: the next token is a fixed function of the last one.

    Records every input it was called with, so a test can see what context the
    generation loop fed back in.
    """

    NEXT = {1: 3, 3: 4, 4: 5, 5: 2}

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, input_ids):
        self.seen.append(input_ids[0].tolist())
        seq_len = input_ids.size(1)
        logits = torch.zeros(1, seq_len, 6)
        for position in range(seq_len):
            token = int(input_ids[0, position])
            logits[0, position, self.NEXT.get(token, 2)] = 10.0
        return logits


class FlatModel(torch.nn.Module):
    """Context-independent logits that grow with the token id."""

    def forward(self, input_ids):
        base = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        return base.expand(1, input_ids.size(1), 6).clone()


def todo_1(lab):
    """TODO 1: UATTNLM.forward"""
    model = make_uniform_model(lab)
    return [
        check("returns one logit vector per position",
              lambda: list(model(torch.tensor([[0, 1, 2], [3, 3, 3]])).shape),
              [2, 3, 4]),
        check("averages the embeddings of the current and previous positions",
              lambda: model(torch.tensor([[0, 1, 2]])).tolist(),
              [[[1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.3333333333, 0.6666666667, 1.0, 0.0]]],
              compare=nested_close),
        check("each sequence in a batch is handled independently",
              lambda: model(torch.tensor([[3, 3], [0, 3]])).tolist(),
              [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
               [[1.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.5, 0.0]]],
              compare=nested_close),
    ]


def todo_2(lab):
    """TODO 2: generate"""
    tokenizer = FakeTokenizer()
    short_model = TableModel()
    eos_model = TableModel()
    prompt_model = TableModel()

    def generate_short():
        return lab.generate(
            short_model, tokenizer, "", 2, "cpu", temperature=1.0, top_k=1
        )

    def generate_until_eos():
        return lab.generate(
            eos_model, tokenizer, "", 20, "cpu", temperature=1.0, top_k=1
        )

    def generate_from_prompt():
        return lab.generate(
            prompt_model, tokenizer, "c", 5, "cpu", temperature=1.0, top_k=1
        )

    def low_temperature_sample():
        torch.manual_seed(0)
        return lab.generate(
            FlatModel(), tokenizer, "", 30, "cpu", temperature=0.02, top_k=None
        )

    def top_k_sample():
        torch.manual_seed(0)
        text = lab.generate(
            FlatModel(), tokenizer, "", 40, "cpu", temperature=5.0, top_k=2
        )
        return sorted(set(text))

    results = [
        check("generates the requested number of tokens",
              generate_short,
              "ab"),
        check("starts from the BOS token and feeds its own samples back in",
              lambda: short_model.seen,
              [[1], [1, 3]]),
        check("stops once the EOS token is sampled",
              generate_until_eos,
              "abc"),
        check("does not keep generating after EOS",
              lambda: eos_model.seen,
              [[1], [1, 3], [1, 3, 4], [1, 3, 4, 5]]),
        check("keeps the prompt in the returned text",
              generate_from_prompt,
              "c"),
        check("the prompt is encoded after the BOS token",
              lambda: prompt_model.seen,
              [[1, 5]]),
        check("a low temperature makes sampling near-greedy",
              low_temperature_sample,
              "c" * 30),
        check("top_k restricts sampling to the k most likely tokens",
              top_k_sample,
              ["b", "c"]),
    ]
    return results


def todo_3(lab):
    """TODO 3: causal_mask"""
    return [
        check("lower triangular for a length-3 sequence",
              lambda: lab.causal_mask(3, "cpu").tolist(),
              [[True, False, False], [True, True, False], [True, True, True]]),
        check("length-1 sequence attends to itself",
              lambda: lab.causal_mask(1, "cpu").tolist(),
              [[True]]),
        check("is a boolean tensor",
              lambda: lab.causal_mask(2, "cpu").dtype,
              torch.bool),
    ]


def todo_4(lab):
    """TODO 4: attention"""
    Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    K = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    V = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
    mask = torch.tensor(
        [[[True, False, False], [True, True, False], [True, True, True]]]
    )
    return [
        check("unmasked attention weights",
              lambda: lab.attention(Q, K, V)[0].tolist(),
              [[[0.4011120796, 0.1977758110, 0.4011120796],
                [0.1400292516, 0.5759753585, 0.2839954197],
                [0.1977758110, 0.4011120796, 0.4011120796]]],
              compare=nested_close),
        check("unmasked context vectors",
              lambda: lab.attention(Q, K, V)[1].tolist(),
              [[[1.2033362389, 1.0],
                [0.7080200911, 1.1439661980],
                [1.0, 1.2033362389]]],
              compare=nested_close),
        check("attention weights sum to one per query",
              lambda: lab.attention(Q, K, V)[0].sum(dim=-1).tolist(),
              [[1.0, 1.0, 1.0]],
              compare=nested_close),
        check("masked positions get zero weight",
              lambda: lab.attention(Q, K, V, mask)[0].tolist(),
              [[[1.0, 0.0, 0.0],
                [0.1955703199, 0.8044297099, 0.0],
                [0.1977758110, 0.4011120796, 0.4011120796]]],
              compare=nested_close),
        check("masked context vectors",
              lambda: lab.attention(Q, K, V, mask)[1].tolist(),
              [[[1.0, 0.0],
                [0.1955703199, 0.8044297099],
                [1.0, 1.2033362389]]],
              compare=nested_close),
    ]


def todo_5(lab):
    """TODO 5: ATTNLM.__init__"""
    model = lab.ATTNLM(vocab_size=4, embedding_size=3, hidden_size=2)
    projections = [model.q_proj, model.k_proj, model.v_proj]
    return [
        check("q_proj, k_proj and v_proj are linear layers",
              lambda: [isinstance(p, torch.nn.Linear) for p in projections],
              [True, True, True]),
        check("they map an embedding to a hidden vector",
              lambda: [(p.in_features, p.out_features) for p in projections],
              [(3, 2), (3, 2), (3, 2)]),
        check("they are three separate layers",
              lambda: len({id(p) for p in projections}),
              3),
    ]


def todo_6(lab):
    """TODO 6: ATTNLM.forward"""
    model = make_attention_model(lab)
    return [
        check("returns one logit vector per position",
              lambda: list(model(torch.tensor([[0, 1, 2], [3, 3, 3]])).shape),
              [2, 3, 4]),
        check("causal attention over the fixed-weight model",
              lambda: model(torch.tensor([[0, 1, 2]])).tolist(),
              [[[1.2, 0.2, 0.0, 0.0],
                [0.5302384496, 0.8697615266, 0.0, 0.0],
                [0.5333333611, 0.5333333611, 0.0, 0.0]]],
              compare=nested_close),
        check("changing the last token leaves earlier logits untouched",
              lambda: torch.allclose(
                  model(torch.tensor([[0, 1, 2]]))[:, :2],
                  model(torch.tensor([[0, 1, 3]]))[:, :2],
              ),
              True),
        check("the last position does depend on the last token",
              lambda: torch.allclose(
                  model(torch.tensor([[0, 1, 2]]))[:, 2],
                  model(torch.tensor([[0, 1, 3]]))[:, 2],
              ),
              False),
    ]


TODOS = {
    1: ("UATTNLM.forward", todo_1),
    2: ("generate", todo_2),
    3: ("causal_mask", todo_3),
    4: ("attention", todo_4),
    5: ("ATTNLM.__init__", todo_5),
    6: ("ATTNLM.forward", todo_6),
}


def main():
    requested = sys.argv[1:]
    numbers = [int(n) for n in requested] if requested else sorted(TODOS)
    lab = load_lab()
    passed = total = 0
    for number in numbers:
        title, runner = TODOS[number]
        print(f"TODO {number}: {title}")
        try:
            results = runner(lab)
        except NotImplementedError:
            print("  [FAIL] not implemented yet")
            results = [False]
        except Exception as error:
            print(f"  [FAIL] raised {type(error).__name__}: {error}")
            results = [False]
        passed += sum(results)
        total += len(results)
        print()
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
