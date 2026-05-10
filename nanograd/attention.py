"""
Phase 6 — Single-head self-attention on top of Tensor.

This is the heart of a Transformer: each token decides "which other tokens should
I pay attention to?" by computing a similarity score with every other token,
softmaxing into weights, and using those weights to pool a value vector.

Built entirely on Tensor + Value primitives, so autograd flows through automatically.
"""

import math
from nanograd.tensor import Tensor


class SelfAttentionHead:
    """
    A single attention head, no batching, no causal mask, no positional encoding.

    Inputs (per forward call):
        X — Tensor of shape (T, n_embed) where T = number of tokens

    Internal weights:
        W_q, W_k, W_v — each (n_embed, head_size)

    Output:
        out     — Tensor (T, head_size), the attended representations
        weights — Tensor (T, T), the attention pattern
    """

    def __init__(self, n_embed, head_size):
        self.W_q = Tensor.randn(n_embed, head_size)
        self.W_k = Tensor.randn(n_embed, head_size)
        self.W_v = Tensor.randn(n_embed, head_size)
        self.head_size = head_size
        
    def __call__(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = (Q @ K.T) / math.sqrt(self.head_size)
        weights = scores.softmax(axis=-1)
        out = weights @ V
        return (out, weights)

    def parameters(self):
        return self.W_q.parameters() + self.W_k.parameters() + self.W_v.parameters()
