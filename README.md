# Lab 1-2: Attention language models

## Goal

In this lab you build your first two language models. Both of them are **autoregressive decoders**: they read a
sequence of tokens and predict the next token at every position.
The two models differ in how they **attend** to the context (the input sequence of tokens) when decoding the next-token.
By the end you will have trained both on TinyStories and sampled stories out of the models.

Before the lab, you should read [chapter 7.1](https://web.stanford.edu/~jurafsky/slp3/7.pdf) from Jurafsky's "Speech and Language Processing", or watch one of the million introductory videos on the topic from YouTube.

## Setup
You have been given multiple files:
```
 agents.md 		            # A guidelines file for coding agents - keep it included with the code at all times
 README.md                  # This file, containing instructions and questions
 lab-1-2-attention-lm.py    # The code file which you will edit
 test_1_2.py                # A test file to run after each TODO for validation
```

You are also given a ready-to-use BPE tokenizer class, `TinyStoriesBPETokenizer`, which is pre-trained on the TinyStories dataset. You do not need to edit it.

To run the test file, run it as a python script, either with a TODO number or without arguments to run all tests:

```
python test_1_2.py 1      # check TODO 1
python test_1_2.py 1 2    # check several TODOs at once
python test_1_2.py        # check everything
python lab-1-2-attention-lm.py
```

The tests run in a few seconds. 
Note that these tests are basic and non-comprehensive.
The exercise will be graded using unseen tests.

The full script downloads a slice of TinyStories, trains a tokenizer, and then trains both model types for three epochs.

## TODOs

The TODOs are marked `# TODO COMPLETE (N)` in the lab file. Each one is specified by the docstring of the function it sits in, so read that docstring before you write anything. Do them in order: later TODOs build on earlier ones.

### TODO 1: Uniform attention over the context

The first language modeling function you will implement will apply a simple mechanism of uniform attention: at each step of decoding, each past token will contribute the same amount to the context summary, which the model predicts the next token from.

In a `UATTNLM` model, you are given an embedding layer (`self.embed`, converts tokens to embedding vectors) and an unembedding layer (`self.hidden2output`, converts the final hidden representations to output logits of each vocabulary token).

Your task is to implement `UATTNLM.forward`. In this function, you receive a batch of token sequences. 
You should convert them to their embeddings. 
Then, compute the hidden state at a position `t` as the average of all embedding tokens up to position `t` (inclusive).
Finally, convert that hidden state (the context) to a log-prob distribution over the output vocabulary.

**Note:** You should use PyTorch batched computations (Implement the function without writing any loop).

```
python test_1_2.py 1
```

### TODO 2: Generating text

A single forward pass through the model (as you just implemented) gives you a distribution over the next token at each position, but turning that into text is a separate job: you have to sample a token fom the distribution of the last position, append it to the context, and run the model again on the longer context. This is **autoregressive generation**.

Implement the `generate` function. The function implementation should handle and use all of the given parameters (including `temprature`, `topk` and `max_tokens`).

**Note:** You might want to read on the functions `torch.topk`, `torch.softmax`, `torch.multinomial`.

```
python test_1_2.py 2
```

With TODOs 1 and 2 done, you have everything you need to train and sample from the uniform-attention model:

```
python lab-1-2-attention-lm.py --models uattnlm
```

Check the output makes sense:
* The training and validation loss should go down from epoch to epoch, and the validation perplexity with them.
* After training, the loss on the real validation targets should be clearly better than the loss on the shuffled targets. If the two are equal, the model has learned nothing about which token follows which.
* The sample should be made of real words, even though it will *still* not be a coherent story.

### TODO 3: The causal mask

The uniform-attention model you just built has a problem: every token in the context gets exactly the same weight, no matter which token it is or how far back in the context it appears. 
Ideally, you would want to weight each past token differently - according to its relevance to the next-token prediction.
The **attention** mechanism does exactly that - it learns a function that weights each token in the context and how it should affect the next-token prediction.
Before implementing the attention function, note that an attention-based language model predicting the token at position i+1 is only allowed to look at positions up to i (if it could see later tokens, it would just read off the answer). The **causal mask** is what enforces that.

Implement the `causal_mask` function, and test that it works:

```
python test_1_2.py 3
```

### TODO 4: Scaled dot-product attention

Now that you have the causal mask in place, its time to implement your attention mechanism.
Start with the function `attention`, which receives the pre-calculated inputs (the query, key, and value tensors) and the causal mask as a parameter.
The function should compute the attention scores (according to the attention equation), guided by the causal mask; it should also return the final context tensors (i.e. the attention scores multiplied by the value tensors).

Test your function implementation:
```
python test_1_2.py 4
```

### TODO 5 & 6: The attention language model

Now that you have the `attention` and `causal_mask` functions, combine them both to create your full attention-based LM.
Start by implementing `ATTNLM.__init__`: complete the definitions of the query, key and value projection matrices (keep the variable names).
Then, implement `ATTNLM.forward`, by calling `causal_mask`, `attention`, and utilizing the query, key and value projections.
Note that the `forward` function returns the vocabulary logits, so you will have to project the final post-attention context to the vocabulary space (do so with `hidden2output`).

After implementing the language model, run the basic tests:

```
python test_1_2.py 5 6
```

Once all tests pass, train both models, and answer the questions that come after:

```
python lab-1-2-attention-lm.py --models both
```

### TODO 7: The cost of attention

**Question:** What is the runtime (compute) and memory complexity of an attention forward pass? Which of the two do you think is more limiting at large scales (sequences longer than 100k tokens) and why?

**Answer:**
Time complexity is O(N^2*dk + N*d*dk + N*dk*V) with N being the length of the token sequence, d is the embedding dimension, and dk is the hidden dimension size and V being the vocabulary size.
Under the same naming scheme the memory complexity is O(N^2 + N*dk + dk*V).
Our estimation is that a 100k long token sequence will demand an order of 10 GB RAM (under the assumption that a number is represented with one byte) which is achievable even with a private laptop so in our opinion the more limiting factor is the runtime

### TODO 8: Scaling the scores

**Question:** Part of the attention function requires dividing by the square root of the hidden dimension. Why is this useful?

**Answer:**
When ones has large values in the attention one also get weak gradients, so in order to avoid that issue we attempt to "normalize" the values

### TODO 9: Counting forward passes

**Question:** Training and inference (post-training text generation) both run a model over an input sequence, but they do not need the same number of forward passes.
During training, how many forward passes does it take to compute the loss over a single example of 63 tokens?
During generation, how many forward passes does `generate` need to produce 63 **new** tokens? Explain what causes the difference.

**Answer:**
Training only requires one pass assuming Q is of length of at least 63. This is because the calculation is done in parallel
Inference requires 63 passes with each one generating one new token and appending it to the context

## Lab debrief
**Question:** We're interested in any thoughts you have about this lab so that we can improve this lab for later years, and to inform later labs for this year. Please list any issues that arose or comments you have to improve the lab. Useful things to comment on include the following, but you're not restricted to these:
* Was the lab too long or too short?
* Were the instructions clear?
* Was it clear (at least after you completed the lab) what the points of the exercises were?

**Answer:**
We got stuck for a long time on technical dimension "play", this made the lab quite long. All in all it took us 4 hours
Must of the instruction were clear, ToDo 1 was not completely clear that we need to create a cumulative sum
The point of the lab was clear
## Submission
Collect all of the files in the given directory into a **flat** zip file (**NO** internal directories) and submit it for grading via Gradescope.


## Further reading

- Cheaper attention: [Linear attention](https://arxiv.org/abs/2006.16236)
