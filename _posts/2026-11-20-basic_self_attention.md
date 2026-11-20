---
layout: post
title: "Introduction to Self-Attention"
date: 2026-11-20
tags: math LLM
categories: AI-Architecture
---

<p style="opacity: 0.5;"><strong>Disclaimer:</strong> There are hundreds of blog posts about self-attention. This one presents my interpretation of the mechanism. Sources are linked at the bottom of the post.</p>

## Inspiration

MLPs alone are not well suited to next-token prediction. They require an entire sequence as input, and because sequences have variable lengths, matching the dimensions becomes difficult. Even if an entire sequence could be passed to an MLP, each input-token embedding would be processed separately, and the path to the final output would contain only matrix multiplications and nonlinear functions. The token vectors could not interact with one another, so reordering the sequence would not change the result. To address these limitations, earlier models used a sliding window to keep the sequence length fixed and padded the window when necessary.

Before attention became popular, architectures such as RNNs made significant advances. Attention takes inspiration from them, but it offers a more effective way to generate contextual embeddings. The general goal is to update each embedding

$$
x = x + \Delta_{context}
$$

## Attention Core

The framework described in (3) presents an important idea: attention is really a two-stage process. The first section is computing how each token attends to each other token and the second process is taking the "attention scores" and using them to actually change the embeddings.

### Scoring

We establish two parameter matrices that the model uses to represent relationships between tokens. They are commonly called the query and key matrices: the query matrix represents what the current token is looking for, while the key matrix represents what it contains.

$$
\begin{aligned}
x &\in \mathbb{R}^{n \times d}, \quad W_q \in \mathbb{R}^{d \times d_k}, \quad W_k \in \mathbb{R}^{d \times d_k} \\
Q &= xW_q \qquad K = xW_k
\end{aligned}
$$

```python
x = model.encode(["AMIR", "wants", "to", "learn", "about", "attention"])  # Embedding shape: (6, 384)
x = torch.from_numpy(x) # converting numpy array to tensor
#initialize matrices with proper
W_q = torch.randn(384,128)
W_k = torch.randn(384,128)
#matrix multiplication
Q = torch.matmul(x,W_q)
K = torch.matmul(x,W_k)
```

These two matrices are two different representations of the same x vector and remember the questions that they tend to represent. We now want to examine how each token correlates with each other token. We can do this by matrix multiplying them together getting raw scores

$$
R = QK^{\mathsf{T}}
$$

For a small $d_k$, the raw score matrix $R$ is usually manageable. As $d_k$ increases—for example, to 128—large variances in $R$ can cause problems in later steps. We address this by scaling the matrix multiplication by the reciprocal square root of the inner dimension, $\frac{1}{\sqrt{d_k}}$. To illustrate the effect on variance, we can take the trace of the covariance matrix and divide it by the number of elements.

```python
scaledR = R / (d_k**0.5)

sigma = torch.cov(R)  # covariance matrix
scalar_cov = torch.trace(sigma) / 6
# 124.06

scaled_sigma = torch.cov(scaledR)
scaled_scalar_cov = torch.trace(scaled_sigma) / 6
# 0.9693
```

This means that if averaged at 0, the highest we could see in $R$ is approximately 124 and the lowest value we can see is -124. In scaled $R$, the highest value we could see is approximately 1 and lowest is -1. This will help in terms of the next step.

For the next step, we need to consider our goal: adding a bounded update to the original embeddings. We first use another matrix, commonly called the value matrix, to project the original embeddings into a new space. What traditional attention uses is a convex combination which is just a special type of linear combination where all the coefficients sum to 1. One useful property of this approach is that it bounds the contextual change to the space of the value matrix $V$. The way that regular self attention does it is by performing a softmax over the attention scores.

### Causal Masking

Softmax creates a probability distribution by enforcing nonnegative weights that sum to 1. Before applying softmax, we usually apply a mask that turns the attention scores into a lower-triangular matrix. This prevents a token from attending to tokens that come after it. Most language models use causal masks because their goal is next-token prediction: each prediction should depend only on earlier tokens. Models such as BERT did not use a causal mask and achieved strong results on text-classification tasks.

```python
import torch.nn.functional as F

# Create a lower-triangular mask.
mask = torch.tril(torch.ones(6, 6))

# Fill all positions above the diagonal with negative infinity.
masked_R = R.masked_fill(mask == 0, float('-inf'))
masked_scaled = scaledR.masked_fill(mask == 0, float('-inf'))

softmax_scaledR = F.softmax(masked_scaled, dim=1)
softmax_R = F.softmax(masked_R, dim=1)
```

The final row of the scaled result is approximately:

```text
[0.0849, 0.0537, 0.2433, 0.1107, 0.1819, 0.3256]
```

The corresponding row without scaling is approximately:

```text
[2.3888e-07, 1.3470e-09, 3.5630e-02, 4.8308e-06, 1.3261e-03, 9.6304e-01]
```

These rows show the difference scaling makes: without scaling, most of the probability mass is concentrated on a few tokens; with scaling, the probability distribution is more evenly spread.

In the unscaled $R$, the probabilities are pushed onto a few elements—in this case, the last and third items. When we scale $R$ by $d_k$, the probabilities are more spread out, and we don't have to deal with tiny numbers.

This final matrix is usually called the attention matrix and overall this is how the first phase of scoring is done. We can grasp these concepts in a very nice and simple equation

$$
Attention = softmax(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}} + M)
$$

## Contextual Vectors

The second and final phase of basic self-attention is the value matrix that we actually use to add to our token embeddings. We talked about when trying to understand why we do softmax. Remember $W_v$ is another matrix the model can learn and similarly to $Q$ and $K$ we multiply with $x$ to make $V$. $V$ can be interpreted as how valuable each token is. We take our attention matrix which states how much attention should be placed on a token and our $V$ which states how valuable those tokens are and we matrix multiply that to get what we call an output matrix.

$$
\begin{aligned}
A \in \mathbb{R}^{n \times n},& \quad W_v \in \mathbb{R}^{d \times d_k} \\
V &= xW_v \\
O &= AV \\
O &\in \mathbb{R}^{n \times d_k}
\end{aligned}
$$

```python
W_v = torch.randn(384,128)
V = torch.matmul(x,W_v)
O = torch.matmul(attention, V)  # attention replaces softmax_scaledR
```

Notice that the dimensions of $O$ do not match the dimensions of $x$: $x \in \mathbb{R}^{n \times d}$, so the matrices cannot yet be added. What is done is yet another projection using a matrix we call $W_o$ that will change the dimensions of $O$ so that it can be added to $x$

$$
\begin{aligned}
O \in \mathbb{R}^{n \times d_k}&, \quad W_o \in \mathbb{R}^{d_k \times d} \\
\Delta_{context} &= OW_o \\
\Delta_{context} &\in \mathbb{R}^{n \times d}
\end{aligned}
$$

```python
W_o = torch.randn(128,384)
change = torch.matmul(O,W_o)
```

Our final attention formulation looks like

$$
\Delta_{context} = softmax(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}} + M)VW_o
$$

## Complexity Analysis

The importance of the KV cache becomes clear when analyzing algorithmic complexity. The core computation consists of softmax and matrix multiplications. The complexity of matrix multiplication is dependent on its dimensions.

$$
Q \in \mathbb{R}^{n \times d_k}, \qquad K^{\mathsf{T}} \in \mathbb{R}^{d_k \times n}
$$

Multiplying these two matrices is $O(n^2d_k)$. Combining the attention matrix with the values has the same complexity. Softmax is an $O(n^2)$ operation. Matrix multiplication dominates, and the operation grows quadratically with the input size which is not ideal when scaling up to large matrices.

## KV Cache

The KV cache trades memory for computation during LLM inference and next-token prediction. Suppose we have predicted token $x_{t+1}$; recomputing the entire attention matrix is unnecessary. Instead, we can cache the $K$ and $V$ matrices and compute only the new attention row.

We multiply $x_{t+1}$ by $W_q$ to obtain $q_{t+1}$, then compute that row using the cached $K_{1:t}$ together with the newly generated $k_{t+1}$. We do not need to store past query vectors. The causal mask also ensures that each new token cannot affect earlier tokens, so the past $q$ vectors remain unchanged.

After the new row is scaled and passed through softmax, we multiply it by the full $V_{1:t+1}$ matrix to obtain $o_{t+1}$. We then project $o_{t+1}$ with $W_o$ and add it to $x_{t+1}$. Every generated token adds its $k_{t+1}$ and $v_{t+1}$ vectors to the cache.

## Conclusion

This blog was about fundamental ideas behind self attention. Many things were excluded that I want to address in future blog posts. If there's anything wrong feel free to email me and I will correct it.

## Sources

1. [Peter Bloem's Blog](https://peterbloem.nl/blog/transformers)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
4. [Stanford CS224N: Self-Attention and Transformers](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)
5. [nanoGPT](https://youtu.be/kCc8FmEb1nY?si=DyTgpLSI4e4_6S-W)

## Appendix

### A.1 Attention Example Code

```python
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def attention(inputs: list[str], d_k: int):
    # The embedding dimension for this model is 384.
    d = 384
    x = model.encode(inputs)
    x = torch.from_numpy(x)
    n = len(inputs)

    W_q = torch.randn(d, d_k)
    W_k = torch.randn(d, d_k)
    W_v = torch.randn(d, d_k)
    W_o = torch.randn(d_k, d)

    Q = torch.matmul(x, W_q)
    K = torch.matmul(x, W_k)
    V = torch.matmul(x, W_v)

    R = torch.matmul(Q, torch.transpose(K, 0, 1))
    scaled_R = R / (d_k**0.5)

    mask = torch.tril(torch.ones(n, n))
    masked_scaled_R = scaled_R.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(masked_scaled_R, dim=1)

    out = torch.matmul(attention_weights, V)
    context_change = torch.matmul(out, W_o)

    return context_change
```
